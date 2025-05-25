from typing import Optional, Dict, Any, List
import logging
import pandas as pd
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from .base_exchange import BaseExchange

class AlpacaExchange(BaseExchange):
    def __init__(self, api_key: str, api_secret: str, symbol: str = "SOL/USD",
                 paper: bool = True, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.symbol = symbol.replace("-", "/")
        
        # Initialize clients
        self.trading_client = TradingClient(api_key, api_secret, paper=paper)
        self.data_client = StockHistoricalDataClient(api_key, api_secret)
        
        self._timeframe_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame.Minute * 5,
            '15m': TimeFrame.Minute * 15,
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }

    def get_profile_info(self) -> Dict[str, Any]:
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            
            return {
                "user": {
                    "id": account.account_number,
                    "username": account.account_number,
                    "email": None,
                    "account": account.account_number
                },
                "balance": {
                    "wallet_balance": float(account.cash),
                    "margin_balance": float(account.buying_power),
                    "available_margin": float(account.buying_power),
                    "unrealized_pnl": float(account.unrealized_pl),
                    "realized_pnl": float(account.realized_pl),
                    "bitmex_usd": float(account.equity)  # Using equity as USD equivalent
                },
                "positions": [{
                    "symbol": pos.symbol,
                    "current_qty": int(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "unrealized_pnl": float(pos.unrealized_pl),
                    "realized_pnl": float(pos.cost_basis)
                } for pos in positions]
            }
        except Exception as e:
            self.logger.error(f"Error getting profile information: {str(e)}")
            return None

    def get_candle(self, timeframe: str = "5m", count: int = 100) -> Optional[pd.DataFrame]:
        try:
            if timeframe not in self._timeframe_map:
                raise ValueError(f"Invalid timeframe. Supported: {list(self._timeframe_map.keys())}")
            
            # Calculate start time based on count and timeframe
            tf_minutes = int(timeframe[:-1]) if 'm' in timeframe else \
                        int(timeframe[:-1]) * 60 if 'h' in timeframe else \
                        int(timeframe[:-1]) * 1440  # days
            start_time = datetime.now() - timedelta(minutes=tf_minutes * count)
            
            request = StockBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=self._timeframe_map[timeframe],
                start=start_time
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if not bars:
                raise ValueError("No candle data returned")
            
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume)
            } for bar in bars[self.symbol]])
            
            return df.sort_values('timestamp').reset_index(drop=True)
        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'symbol': pos.symbol,
                'currentQty': int(pos.qty),
                'avgEntryPrice': float(pos.avg_entry_price),
                'unrealisedPnl': float(pos.unrealized_pl),
                'realisedPnl': float(pos.cost_basis)
            } for pos in positions]
        except Exception as e:
            self.logger.error(f"Error retrieving positions: {str(e)}")
            return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            orders = self.trading_client.get_orders()
            return [{
                'orderID': order.client_order_id,
                'clOrdID': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'orderQty': float(order.qty),
                'price': float(order.limit_price) if order.limit_price else None,
                'stopPx': float(order.stop_price) if order.stop_price else None,
                'ordType': order.type,
                'ordStatus': order.status,
                'text': order.client_order_id
            } for order in orders]
        except Exception as e:
            self.logger.error(f"Error retrieving open orders: {str(e)}")
            return []

    def open_position(self, side: str, quantity: int, order_type: str = "Market", **kwargs) -> Dict[str, Any]:
        try:
            # Prepare base order parameters
            order_data = {
                "symbol": self.symbol,
                "qty": quantity,
                "side": OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                "time_in_force": TimeInForce.GTC,
                "client_order_id": kwargs.get("clOrdID")
            }

            # Create appropriate order request based on type
            if order_type.lower() == "market":
                order_data["type"] = OrderType.MARKET
                order = MarketOrderRequest(**order_data)
            else:  # Limit order
                order_data["type"] = OrderType.LIMIT
                order_data["limit_price"] = kwargs.get("price")
                order = LimitOrderRequest(**order_data)

            # Submit the order
            result = self.trading_client.submit_order(order)
            
            response = {
                "entry": {
                    "orderID": result.id,
                    "clOrdID": result.client_order_id,
                    "symbol": result.symbol,
                    "side": result.side,
                    "orderQty": float(result.qty),
                    "price": float(result.limit_price) if result.limit_price else None,
                    "ordType": result.type,
                    "ordStatus": result.status
                }
            }

            # Handle stop loss if provided
            if "stop_loss_price" in kwargs:
                sl_order = self.trading_client.submit_order(
                    symbol=self.symbol,
                    qty=quantity,
                    side=OrderSide.SELL if side.lower() == "buy" else OrderSide.BUY,
                    type=OrderType.STOP,
                    time_in_force=TimeInForce.GTC,
                    stop_price=kwargs["stop_loss_price"]
                )
                response["stop_loss"] = {
                    "orderID": sl_order.id,
                    "clOrdID": sl_order.client_order_id
                }

            # Handle take profit if provided
            if "take_profit_price" in kwargs:
                tp_order = self.trading_client.submit_order(
                    symbol=self.symbol,
                    qty=quantity,
                    side=OrderSide.SELL if side.lower() == "buy" else OrderSide.BUY,
                    type=OrderType.LIMIT,
                    time_in_force=TimeInForce.GTC,
                    limit_price=kwargs["take_profit_price"]
                )
                response["take_profit"] = {
                    "orderID": tp_order.id,
                    "clOrdID": tp_order.client_order_id
                }

            return response
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, side: str, quantity: int, order_type: str = "Market", **kwargs) -> Dict[str, Any]:
        try:
            order_data = {
                "symbol": self.symbol,
                "qty": quantity,
                "side": OrderSide.SELL if side.lower() == "buy" else OrderSide.BUY,
                "time_in_force": TimeInForce.GTC,
                "type": OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT,
            }
            
            if order_type.lower() != "market":
                order_data["limit_price"] = kwargs.get("price")
            
            if "clOrdID" in kwargs:
                order_data["client_order_id"] = kwargs["clOrdID"]

            order = self.trading_client.submit_order(**order_data)
            
            return {
                "orderID": order.id,
                "clOrdID": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "orderQty": float(order.qty),
                "price": float(order.limit_price) if order.limit_price else None,
                "ordType": order.type,
                "ordStatus": order.status
            }
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None

    def close_all_positions(self, **kwargs) -> bool:
        try:
            self.trading_client.close_all_positions()
            return True
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return False

    def set_leverage(self, leverage: int) -> Dict[str, Any]:
        # Alpaca doesn't support setting leverage directly
        self.logger.warning("Leverage setting not supported in Alpaca")
        return {"message": "Leverage setting not supported in Alpaca", "status": "warning"}