"""Trade executor for managing trading operations"""

import os
import sys
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import codecs
import matplotlib.pyplot as plt
import mplfinance as mpf
import pytz
from datetime import datetime
import logging
import uuid
from typing import Optional

from .exchanges.base_exchange import BaseExchange
from .strategies.smc_strategy_logic import MatteGreenStrategy
from .utils.telegram_bot import TelegramBot, configure_logging
# from .utils.performance import get_trading_performance_summary

# Configure UTF-8 encoding
codecs.register_error('strict', codecs.replace_errors)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
elif hasattr(sys, 'stdout'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')

load_dotenv()
class TradeExecutor:
    def __init__(self, exchange: BaseExchange, risk_per_trade: float = 0.01, 
                 lookback_period: int = 40, fvg_threshold: float = 0.0003, rr :float = 0.01, symbol = 'SOLUSD', 
                 telegram_token: Optional[str] = None, telegram_chat_id: Optional[str] = None, tolarance :float = 0.005, 
                 log: Optional[logging.Logger] = None):
        """Initialize trade executor with exchange and strategy"""
        self.exchange = exchange
        self.tolarance = tolarance
        self.risk_per_trade = risk_per_trade
        self.logger, self.bot = configure_logging(telegram_token, telegram_chat_id) if telegram_token and telegram_chat_id else (log, None)
        self.symbol = symbol 
        self.df = pd.DataFrame() 
        if not self.logger:
            self.logger = logging.getLogger(__name__)
            logging.basicConfig(level=logging.INFO)
        
        # Initialize strategy
        self.strategy = MatteGreenStrategy(
            lookback_period=lookback_period,
            fvg_threshold=fvg_threshold,
            risk_per_trade=risk_per_trade,
            rr_ratio=rr
        )
        
        # Initialize state
        profile = self.exchange.get_profile_info()
        self.initial_balance = self.current_balance = profile['balance']['bitmex_usd'] if profile else 10000
        self.current_trades = []
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.api = exchange
        self.logger.info(f"TradeExecutor initialized with {self.exchange.__class__.__name__}")

    def sync_open_orders(self):
        """Synchronize local state with exchange orders"""
        try:
            open_orders = self.api.get_open_orders() or []
            positions = self.api.get_positions() or []
            self.logger.info(f"🔁Syncing: Found {len(open_orders)} open orders and {len(positions)} positions.")
            
            position_map = {pos['symbol']: pos for pos in positions if pos.get('currentQty', 0) != 0}
            position = position_map.get(self.symbol.replace('-', ''), {})
            has_position = position.get('currentQty', 0) != 0
            position_direction = 'long' if position.get('currentQty', 0) > 0 else 'short' if position.get('currentQty', 0) < 0 else None
            
            exchange_trades = {}
            
            for order in open_orders:
                clord_id = order.get('clOrdID')
                if not clord_id or clord_id in [None, 'No strings attached']:
                    self.logger.debug(f"Skipping order with no clOrdID: {order}")
                    continue
                text = order.get('text', '')
                self.logger.debug(f"Processing order: clOrdID='{clord_id}', text='{text}'")
                try:
                    clOrderID = clOrderID_string(clord_id, text)
                    if clOrderID['symbol'].replace('-', '') != self.symbol.replace('-', ''):
                        self.logger.debug(f"Skipping order for different symbol: {clOrderID['symbol']} vs {self.symbol}")
                        continue
                    if has_position and clOrderID['side'] != position_direction:
                        pass
                    exchange_trades[clord_id] = (
                        order.get('orderID', clord_id),
                        int(clOrderID['entry_idx']),
                        float(clOrderID['price']),
                        clOrderID['side'],
                        float(clOrderID['stop_loss']),
                        float(clOrderID['take_profit']),
                        int(clOrderID['position_size']),
                        clord_id,
                        text
                    )
                except ValueError as e:
                    self.logger.warning(f"Invalid clOrdID format or data: {clord_id} - {str(e)}")
                    continue
            
            local_clord_ids = {trade[7] for trade in self.current_trades if trade[7]}
            exchange_clord_ids = set(exchange_trades.keys())

            for trade in list(self.current_trades):
                clord_id = trade[7]
                sl_orderID = trade[9]
                tp_orderID = trade[10]
                if clord_id and clord_id not in exchange_clord_ids and not has_position:
                    trade_id, entry_idx, entry_price, direction, stop_loss, take_profit, size, _, text = trade[:9]
                    sl_executions = self.api.get_executions(orderID=sl_orderID) if sl_orderID else []
                    tp_executions = self.api.get_executions(orderID=tp_orderID) if tp_orderID else []
                    
                    if sl_executions:
                        exit_price = sum(exec_['price'] * exec_['execQty'] for exec_ in sl_executions) / sum(exec_['execQty'] for exec_ in sl_executions)
                        reason = 'stoploss'
                    elif tp_executions:
                        exit_price = sum(exec_['price'] * exec_['execQty'] for exec_ in tp_executions) / sum(exec_['execQty'] for exec_ in tp_executions)
                        reason = 'takeprofit'
                    else:
                        exit_price = self.df['close'].iloc[-1]  # Fallback if no executions found
                        reason = 'unknown'
                    
                    pl = (exit_price - entry_price) * size if direction == 'long' else (entry_price - exit_price) * size
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': exit_price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': clord_id, 'reason': reason
                    })
                    self.current_balance += pl
                    self.current_trades.remove(trade)
                    self.equity_curve.append(self.current_balance)
                    self.logger.info(f"Closed trade {clord_id} at {exit_price} due to {reason}")
                elif clord_id in exchange_trades and has_position and trade[3] != position_direction:
                    self.logger.warning(f"Correcting direction mismatch for clOrdID {clord_id}: local={trade[3]}, exchange={position_direction}")
                    trade_id, entry_idx, entry_price, _, stop_loss, take_profit, size, clord_id, text, sl_orderID, tp_orderID = trade
                    self.current_trades.remove(trade)
                    self.current_trades.append((trade_id, entry_idx, entry_price, position_direction, stop_loss, take_profit, size, clord_id, text, sl_orderID, tp_orderID))

            for clord_id, trade_data in exchange_trades.items():
                if clord_id not in local_clord_ids:
                    self.current_trades.append(trade_data + (None, None))  # Append None for sl_orderID, tp_orderID
            
            self.logger.debug(f"Current trades after sync: {self.current_trades}")
        except Exception as e:
            self.logger.error(f"Failed to sync open orders: {str(e)}")
        self.logger.info('🔄Done Sync....... ')

    def execute_entry(self, signal):
        """Execute entry order based on signal"""
        side = signal['side']
        desired_price = signal['price']  # 5m closing price
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        position_size = signal['position_size']
        entry_idx = signal['entry_idx']
        fvg_confirmed = signal['fvg_confirmed']
        signal_type = signal['signal_type']

        sast_now = get_sast_time()
        # Fetch the latest 1m candle to check current price
        current_1m_data = self.api.get_candle(timeframe='1m', count=1)
        if current_1m_data is None or current_1m_data.empty:
            self.logger.error("Failed to fetch 1m candle for price validation")
            return
        current_1m_price = current_1m_data['close'].iloc[-1]
        price_threshold = desired_price * self.tolarance  # 0.5% tolerance

        self.logger.info(f"Desired 5m entry price: {desired_price}, Current 1m price: {current_1m_price}, Threshold: ±{price_threshold}")
        pos_side = "Sell" if side.lower() in ['short', 'sell'] else "Buy"

        # Check if 1m price is within acceptable range of 5m price
        if abs(current_1m_price - desired_price) > price_threshold:
            self.logger.warning(f"1m price ({current_1m_price}) deviates from 5m desired price ({desired_price}) beyond threshold. Adjusting to limit order.")
            order_type = "Limit"
            entry_price = desired_price  # Use the 5m price as a limit order
        else:
            self.logger.info("1m price within threshold of 5m price. Proceeding with market order.")
            order_type = "Market"
            entry_price = current_1m_price  # Use current 1m price for market order
            stop_loss = stop_loss + price_threshold if pos_side == "Sell" else stop_loss - price_threshold 
            take_profit = take_profit + price_threshold if pos_side == "Buy" else take_profit - price_threshold  
        date_str = sast_now.strftime("%Y%m%d%H%M")

        
        if fvg_confirmed or signal_type == "BOS":  # Either FVG confirmation or BOS 
            uid = str(uuid.uuid4())[:6]
            clord_id = f"({self.symbol});({date_str});({uid})"
            text = f"('open');('entry');({entry_price}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"

            # Generate unique clOrdIDs for SL and TP without prefixes
            # Note: Assuming BitMEXApi.open_position may add 'SL;' or 'TP;' prefixes internally.
            # To prevent this, we pass distinct clOrdIDs for SL and TP orders and rely on BitMEXApi to use them as provided.
            sl_clord_id = f"({self.symbol});({date_str});(SL{uid[:4]})"
            tp_clord_id = f"({self.symbol});({date_str});(TP{uid[:4]})"
            sl_text = f"('open');('stoploss');({stop_loss}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"
            tp_text = f"('open');('takeprofit');({take_profit}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"

            self.logger.info(f"Opening position with clOrdID: '{clord_id}' (length: {len(clord_id)}), text: '{text}', order_type: {order_type}")
            self.logger.info(f"SL clOrdID: '{sl_clord_id}', TP clOrdID: '{tp_clord_id}'")

            if len(clord_id) > 36 or len(sl_clord_id) > 36 or len(tp_clord_id) > 36:
                self.logger.error(f"clOrdID exceeds 36 characters: main={clord_id}, SL={sl_clord_id}, TP={tp_clord_id}")
                raise ValueError(f"clOrdID exceeds 36 characters")

            pos_quantity = max(1, int(position_size))
            profile = self.api.get_profile_info()
            if not profile or 'balance' not in profile:
                self.logger.error("Failed to fetch profile info for margin check")
                return

            available_margin = profile['balance']['bitmex_usd']
            required_margin = (position_size * entry_price) / 5  # Assuming 5x leverage

            try:
                orders = self.api.open_position(
                    side=pos_side,
                    quantity=pos_quantity,
                    order_type=order_type,
                    price=entry_price if order_type == "Limit" else None,
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                    clOrdID=clord_id,
                    text=text,
                )
                if orders and orders.get('entry'):
                    trade_id = orders['entry']['orderID']
                    sl_orderID = orders['stop_loss']['orderID'] if orders.get('stop_loss') else None
                    tp_orderID = orders['take_profit']['orderID'] if orders.get('take_profit') else None
                    # Replace the last appended trade with full details
                    self.current_trades[-1] = (trade_id, entry_idx, entry_price, side, stop_loss, take_profit, position_size, clord_id, text, sl_orderID, tp_orderID)
                    self.logger.info(f"📈🎉Opened {pos_side} at {entry_price}, SL: {stop_loss}, TP: {take_profit}, ID: {trade_id}, clOrdID: {clord_id}")
                else:
                    self.logger.warning(f"Order failed, tracking locally with clOrdID: {clord_id}")
            except Exception as e:
                self.logger.error(f"Error opening position: {str(e)}")
                raise
        else:
            self.logger.info(f"Signal type not BOS or fvg_confirmed")
            uid = str(uuid.uuid4())[:6]
            clord_id = f"({self.symbol});({date_str});({uid})"
            text = f"('open');('entry');({entry_price}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"

            # Generate unique clOrdIDs for SL and TP without prefixes
            # Note: Assuming BitMEXApi.open_position may add 'SL;' or 'TP;' prefixes internally.
            # To prevent this, we pass distinct clOrdIDs for SL and TP orders and rely on BitMEXApi to use them as provided.
            sl_clord_id = f"({self.symbol});({date_str});(SL{uid[:4]})"
            tp_clord_id = f"({self.symbol});({date_str});(TP{uid[:4]})"
            sl_text = f"('open');('stoploss');({stop_loss}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"
            tp_text = f"('open');('takeprofit');({take_profit}, {position_size}, {side}, {entry_idx});({take_profit}, {stop_loss})"

            self.logger.info(f"Opening position with clOrdID: '{clord_id}' (length: {len(clord_id)}), text: '{text}', order_type: {order_type}")
            self.logger.info(f"SL clOrdID: '{sl_clord_id}', TP clOrdID: '{tp_clord_id}'")

            if len(clord_id) > 36 or len(sl_clord_id) > 36 or len(tp_clord_id) > 36:
                self.logger.error(f"clOrdID exceeds 36 characters: main={clord_id}, SL={sl_clord_id}, TP={tp_clord_id}")
                raise ValueError(f"clOrdID exceeds 36 characters")

            pos_quantity = max(1, int(position_size))
            profile = self.api.get_profile_info()
            if not profile or 'balance' not in profile:
                self.logger.error("Failed to fetch profile info for margin check")
                return

            available_margin = profile['balance']['bitmex_usd']
            required_margin = (position_size * entry_price) / 5  # Assuming 5x leverage

            try:
                orders = self.api.open_position(
                    side=pos_side,
                    quantity=pos_quantity,
                    order_type=order_type,
                    price=entry_price if order_type == "Limit" else None,
                    take_profit_price=take_profit,
                    stop_loss_price=stop_loss,
                    clOrdID=clord_id,
                    text=text,
                )
                if orders and orders.get('entry'):
                    trade_id = orders['entry']['orderID']
                    sl_orderID = orders['stop_loss']['orderID'] if orders.get('stop_loss') else None
                    tp_orderID = orders['take_profit']['orderID'] if orders.get('take_profit') else None
                    # Replace the last appended trade with full details
                    self.current_trades[-1] = (trade_id, entry_idx, entry_price, side, stop_loss, take_profit, position_size, clord_id, text, sl_orderID, tp_orderID)
                    self.logger.info(f"📈🎉Opened {pos_side} at {entry_price}, SL: {stop_loss}, TP: {take_profit}, ID: {trade_id}, clOrdID: {clord_id}")
                else:
                    self.logger.warning(f"Order failed, tracking locally with clOrdID: {clord_id}")
            except Exception as e:
                self.logger.error(f"Error opening position: {str(e)}")
                raise

    def execute_exit(self, signal):
        """Execute exit order based on signal"""
        reason = signal['reason']
        price = signal['price']
        direction = signal['direction']
        entry_idx = signal['entry_idx']
        trade_id = signal.get('trade_id')
        sast_now = get_sast_time()

        for trade in list(self.current_trades):
            stored_trade_id, idx, entry_price, trade_direction, stop_loss, take_profit, size, clord_id, text, sl_orderID, tp_orderID = trade
            if idx == entry_idx and trade_direction == direction:
                try:
                    positions = self.api.get_positions() or []
                    position_qty = next((pos['currentQty'] for pos in positions if pos['symbol'] == self.symbol.replace('-', '')), 0)
                    if position_qty == 0:
                        pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                        self.current_balance += pl
                        self.equity_curve.append(self.current_balance)
                        self.trades.append({
                            'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                            'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                            'trade_id': trade_id, 'clord_id': clord_id
                        })
                        self.current_trades.remove(trade)
                        continue

                    if trade_id and trade_id == stored_trade_id:
                        if clord_id is None or text is None:
                            self.logger.warning(f"Invalid clOrdID ({clord_id}) or text ({text}) for trade {trade_id}, skipping API close.")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        side = 'Sell' if position_qty > 0 else 'Buy' if position_qty < 0 else None
                        if side is None:
                            self.logger.warning(f"Position quantity is 0 but trade exists locally, skipping API close for {clord_id}")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        if len(new_clord_id) > 36:
                            self.logger.error(f"clOrdID exceeds 36 characters: {new_clord_id}")
                            raise ValueError(f"clOrdID exceeds 36 characters: {new_clord_id}")
                        self.api.close_position(side=side, quantity=size, order_type="Market", 
                                                clOrdID=new_clord_id, text=new_text)
                        self.logger.info(f"Closed position via API: {new_clord_id}")
                    else:
                        if clord_id is None:
                            self.logger.warning(f"No valid clOrdID for trade, skipping manual close.")
                            pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                            self.current_balance += pl
                            self.equity_curve.append(self.current_balance)
                            self.trades.append({
                                'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                                'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                                'trade_id': trade_id, 'clord_id': clord_id
                            })
                            self.current_trades.remove(trade)
                            continue
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        self.logger.warning(f"No valid trade_id, closing manually with clOrdID: {new_clord_id}")

                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': new_clord_id
                    })
                    self.current_trades.remove(trade)
                    self.logger.info(f"Closed {direction} at {price}, Reason: {reason}, PnL: {pl}, clOrdID: {new_clord_id}")
                except Exception as e:
                    self.logger.error(f"Failed to close position {clord_id}: {str(e)}")
                    if clord_id is None:
                        self.logger.warning(f"No valid clOrdID for failed trade, removing locally.")
                    else:
                        new_clord_id, new_text = update_clOrderID_string(clord_id, text, status='closed')
                        self.api.close_all_positions(clOrdID=new_clord_id)
                    pl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    self.current_balance += pl
                    self.equity_curve.append(self.current_balance)
                    self.trades.append({
                        'entry_idx': entry_idx, 'exit_idx': len(self.df) - 1, 'entry_price': entry_price,
                        'exit_price': price, 'direction': direction, 'pl': pl, 'result': 'win' if pl > 0 else 'loss',
                        'trade_id': trade_id, 'clord_id': clord_id if clord_id is None else new_clord_id
                    })
                    self.current_trades.remove(trade)
                break
        pass

    def manage_positions(self, current_data):
        """Check and manage existing positions"""
        signals = []
        current_idx = len(current_data) - 1
        for trade in list(self.current_trades):
            trade_id, idx, entry_price, direction, stop_loss, take_profit, size, clord_id, text, sl_orderID, tp_orderID = trade
            if (direction == 'long' and self.df['low'].iloc[current_idx] <= stop_loss) or \
               (direction == 'short' and self.df['high'].iloc[current_idx] >= stop_loss):
                pl = (stop_loss - entry_price) * size if direction == 'long' else (entry_price - stop_loss) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': entry_price,
                                    'exit_price': round(stop_loss, 4), 'direction': direction, 'pl': pl, 'result': 'loss', 'trade_id': trade_id})
                signals.append({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                self.execute_exit({'action': 'exit', 'price': stop_loss, 'reason': 'stoploss', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
            elif (direction == 'long' and self.df['high'].iloc[current_idx] >= take_profit) or \
                 (direction == 'short' and self.df['low'].iloc[current_idx] <= take_profit):
                pl = (take_profit - entry_price) * size if direction == 'long' else (entry_price - take_profit) * size
                self.current_balance += pl
                self.trades.append({'entry_idx': idx, 'exit_idx': current_idx, 'entry_price': round(entry_price, 2),
                                    'exit_price': round(take_profit, 4), 'direction': direction, 'pl': pl, 'result': 'win', 'trade_id': trade_id})
                signals.append({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
                self.execute_exit({'action': 'exit', 'price': take_profit, 'reason': 'takeprofit', 'direction': direction, 'entry_idx': idx, 'trade_id': trade_id})
        return signals

    def calculate_position_size(self, entry_price, stop_loss):
        """Calculate position size based on risk parameters"""
        risk_amount = self.current_balance * self.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            self.logger.warning(f"Risk per unit is zero (entry_price={entry_price}, stop_loss={stop_loss}), defaulting position size to 1.")
            return 1
        return max(1, int(risk_amount / risk_per_unit))

    def visualize_results(self, start_idx=0, end_idx=None):
        """Generate visualization of trading results"""
        if not hasattr(self.strategy, 'df') or self.strategy.df.empty:
            return None
        
        if end_idx is None:
            end_idx = len(self.strategy.df)
        subset = self.strategy.df.iloc[start_idx:end_idx]

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        mpf.plot(subset, type='candle', style='charles', ax=ax1, ylabel='Price', show_nontrading=False,
                 datetime_format='%H:%M')
        
        # Plot technical analysis markers
        for start, end, high, low, fvg_type in self.strategy.fvg_areas:
            if start_idx <= end < end_idx:
                color = 'green' if fvg_type == 'bullish' else 'red'
                ax1.fill_between(range(max(0, start - start_idx), min(end - start_idx + 1, len(subset))),
                                 high, low, color=color, alpha=0.2, label=f"{fvg_type.capitalize()} FVG" if start == self.strategy.fvg_areas[0][0] else "")
       
        ax1.legend(loc='upper left')

        # Plot equity curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(range(len(self.equity_curve)), self.equity_curve, label='Equity', color='blue')
        ax2.set_title("Equity Curve")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig

    def run(self, scan_interval=5*60, max_runtime_minutes=40, sleep_interval_minutes=1, iterations_before_sleep=2):
        """Main trading loop"""
        start_time = time.time()
        sast_now = get_sast_time()
        
        # Display welcome banner
        welcome_banner = """
        ┏━━━✦❘༻༺❘✦━━━┓
                  Welcome to
                           💵
                 ᙏα𝜏𝜏ҽGɾҽҽɳ
        ┗━━━✦❘༻༺❘✦━━━┛
        
        ~'Where the green💸💸 make u happy😊' 
        """
        self.logger.info(welcome_banner)
        self.logger.info(f"Starting TradeExecutor at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        iteration = 0
        while (time.time() - start_time) < max_runtime_minutes * 60:
            sast_now = get_sast_time()
            self.logger.info(f"🤔📈🕵🏽‍♂️🔍🔎Scan {iteration + 1} started at {sast_now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                # Get market data
                market_data = self.exchange.get_candle(timeframe="5m", count=self.strategy.lookback_period + 20)
                # Ensure market_data index is a DatetimeIndex
                if market_data is not None and not isinstance(market_data.index, pd.DatetimeIndex):
                    # Try common timestamp columns
                    for col in ["timestamp", "date", "datetime", "time"]:
                        if col in market_data.columns:
                            market_data[col] = pd.to_datetime(market_data[col], errors='coerce')
                            market_data = market_data.set_index(col)
                            break
                if market_data is None or len(market_data) < self.strategy.lookback_period:
                    self.logger.warning(f"Insufficient data: {len(market_data) if market_data is not None else 0} candles")
                    time.sleep(scan_interval)
                    iteration += 1
                    continue
                self.df = market_data
                # Sync open orders and positions
                self.sync_open_orders()
                
                # Get strategy signals
                strategy_signals = self.strategy.analyze_market_data(market_data, self.current_balance)
                
                # Manage existing positions
                exit_signals = self.manage_positions(market_data)
                
                # Process exit signals first
                for signal in exit_signals:
                    self.execute_exit(signal)
                
                # Process entry signals
                if len(self.current_trades) <= 3 :
                    for signal in strategy_signals:
                        if signal['action'] == 'entry' and len(self.current_trades) < 3:
                            # Calculate position size
                            size = self.calculate_position_size(signal['price'], signal['stop_loss'])
                            signal['position_size'] = size
                            self.execute_entry(signal)
                
                # Update performance metrics
                profile = self.exchange.get_profile_info()
                if profile:
                    self.current_balance = profile['balance']['bitmex_usd']
                    self.equity_curve.append(self.current_balance)
                
                # Send visualization if telegram bot is configured
                if self.bot and iteration % 2 == 0:
                    fig = self.visualize_results(start_idx=max(0, len(market_data) - 48))
                    if fig:
                        caption = (f"📸Scan {iteration+1}\n"
                                f"Timestamp: {sast_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Balance: ${self.current_balance:.2f}\n"
                                f"Price @ ${market_data['close'].iloc[-1]}\n"
                                f"Market Bias: {self.strategy.market_bias}")
                        self.bot.send_photo(fig=fig, caption=caption)
                
                self.logger.info(f"Price @ ${market_data['close'].iloc[-1]} \n\n 😪😪😪 Sleeping for {scan_interval/60} minutes....")
            
            except Exception as e:
                self.logger.error(f"An error occurred during trading: {str(e)}")
            
            time.sleep(scan_interval)
            iteration += 1
            
            if iteration % iterations_before_sleep == 0 and iteration > 0:
                self.logger.info(f"Pausing for {sleep_interval_minutes} minutes...")
                time.sleep(sleep_interval_minutes * 60)
                self.logger.info("🙋🏾‍♂️Resuming...")
        
        self.logger.info(f"TradeExecutor run completed. Final balance: ${self.current_balance:.2f}")





def get_sast_time():
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

def clOrderID_string(clord_id, text=None):
    try:
        if clord_id is None:
            raise ValueError("clOrdID cannot be None")
        if not isinstance(clord_id, str):
            raise ValueError(f"clOrdID must be a string, got {type(clord_id)}")
        parts = clord_id.split(';')
        if len(parts) < 3:
            raise IndexError(f"clOrdID has fewer than 3 parts: {clord_id}")
        elif len(parts) == 4:
            reason = parts[0] #SL or TP
            parts.pop(0)
        symbol = parts[0].strip('(').strip(')')
        date = parts[1].strip('(').strip(')')
        uid = parts[2].strip('(').strip(')')
        
        if text is not None:
            if not isinstance(text, str):
                raise ValueError(f"text must be a string or None, got {type(text)}")
            text_parts = text.split(';')
            if len(text_parts) < 4:
                raise IndexError(f"text has fewer than 4 parts: {text}")
            status = text_parts[0].strip('(').strip(')')
            action = text_parts[1].strip("'").strip('(').strip(')').replace("'", '').strip()
            entry_data = text_parts[2].strip('(').strip(')').split(', ')
            if len(entry_data) < 4:
                raise IndexError(f"entry_data has fewer than 4 values: {text_parts[2]}")
            entry_price = entry_data[0]
            position_size = entry_data[1]
            direction = entry_data[2]
            current_idx = entry_data[3]
            tp_sl_data = text_parts[3].strip('(').strip(')').split(', ')
            if len(tp_sl_data) < 2:
                raise IndexError(f"tp_sl_data has fewer than 2 values: {text_parts[3]}")
            take_profit = tp_sl_data[0]
            stop_loss = tp_sl_data[1]
        else:
            status = 'open'
            action = 'entry'
            entry_price = position_size = direction = current_idx = take_profit = stop_loss = '0'

        return {
            'action': action,
            'symbol': symbol,
            'date': date,   
            'side': direction,
            'price': round(float(entry_price), 2),
            'stop_loss': round(float(stop_loss), 4),
            'take_profit': round(float(take_profit), 4),
            'position_size': int(position_size),
            'entry_idx': current_idx,
            'status': status,
            'uuid': uid
        }
    except Exception as e:
        raise ValueError(f"Failed to parse clOrdID '{clord_id}' or text '{text}': {str(e)}")

def update_clOrderID_string(clord_id, text=None, **updates):
    clOrderID_dict = clOrderID_string(clord_id, text)
    temp = {
        'action': clOrderID_dict['action'],
        'symbol': clOrderID_dict['symbol'],
        'side': clOrderID_dict['side'],
        'price': clOrderID_dict['price'],
        'stop_loss': clOrderID_dict['stop_loss'],
        'take_profit': clOrderID_dict['take_profit'],
        'position_size': clOrderID_dict['position_size'],
        'entry_idx': clOrderID_dict['entry_idx'],
        'status': clOrderID_dict['status'],
        'date': clOrderID_dict['date'],
        'uuid': clOrderID_dict['uuid']
    }
    
    if updates:
        for key, value in updates.items():
            temp[key] = value
    
    new_clord_id = f"({temp['symbol']});({temp['date']});({temp['uuid'][:6]})"
    if len(new_clord_id) > 36:
        excess = len(new_clord_id) - 36
        temp['uuid'] = temp['uuid'][:6 - excess]
        new_clord_id = f"({temp['symbol']});({temp['date']});({temp['uuid']})"
        if len(new_clord_id) > 36:
            raise ValueError(f"clOrdID still exceeds 36 characters after truncation: {new_clord_id}")

    new_text = f"({temp['status']});({temp['action']});({temp['price']}, {temp['position_size']}, {temp['side']}, {temp['entry_idx']});({temp['take_profit']}, {temp['stop_loss']})"
    
    return new_clord_id, new_text



