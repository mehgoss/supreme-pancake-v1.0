from typing import Optional, Dict, Any, List
import json
import logging
import pandas as pd
import bitmex
from datetime import datetime
from .base_exchange import BaseExchange

class BitMEXExchange(BaseExchange):
    def __init__(self, api_key: str, api_secret: str, test: bool = True, symbol: str = 'SOL-USD', 
                 timeframe: str = '5m', logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

        try:
            self.client = bitmex.bitmex(
                test=test,
                api_key=api_key,
                api_secret=api_secret
            )
            self.symbol = symbol.replace('-', '')
            self.timeframe = timeframe
            self.max_balance_usage = 0.20
            network_type = 'testnet' if test else 'mainnet'
            self.logger.info(f"BitMEXExchange initialized for {self.symbol} on {network_type}")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    def get_profile_info(self) -> Dict[str, Any]:
        try:
            user_info = self.client.User.User_get().result()[0]
            margin = self.client.User.User_getMargin().result()[0]
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]

            btc_price_data = self.client.Trade.Trade_getBucketed(
                symbol="XBTUSD",
                binSize=self.timeframe,
                count=1,
                reverse=True
            ).result()[0]
            btc_usd_price = btc_price_data[-1]['close'] if btc_price_data else 90000

            wallet_balance_btc = margin.get('walletBalance', 0) / 100000000
            wallet_balance_usd = wallet_balance_btc * btc_usd_price

            return {
                "user": {
                    "id": user_info.get('id'),
                    "username": user_info.get('username'),
                    "email": user_info.get('email'),
                    "account": user_info.get('account')
                },
                "balance": {
                    "wallet_balance": margin.get('walletBalance', 0),
                    "margin_balance": margin.get('marginBalance', 0),
                    "available_margin": margin.get('availableMargin', 0),
                    "unrealized_pnl": margin.get('unrealisedPnl', 0),
                    "realized_pnl": margin.get('realisedPnl', 0),
                    "bitmex_usd": wallet_balance_usd
                },
                "positions": positions
            }
        except Exception as e:
            self.logger.error(f"Error getting profile information: {str(e)}")
            return None

    def get_candle(self, timeframe: Optional[str] = None, count: int = 100) -> Optional[pd.DataFrame]:
        timeframe = timeframe or self.timeframe
        try:
            valid_timeframes = ['1m', '5m', '1h', '1d']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}")

            candles = self.client.Trade.Trade_getBucketed(
                symbol=self.symbol,
                binSize=timeframe,
                count=count,
                reverse=True
            ).result()[0]

            if not candles:
                raise ValueError("No candle data returned")

            formatted_candles = [{
                'timestamp': candle['timestamp'],
                'open': candle['open'],
                'high': candle['high'],
                'low': candle['low'],
                'close': candle['close'],
                'volume': candle['volume']
            } for candle in candles if all(key in candle for key in ['open', 'high', 'low', 'close'])]

            df = pd.DataFrame(formatted_candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df
        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            positions = self.client.Position.Position_get(
                filter=json.dumps({"symbol": self.symbol})
            ).result()[0]
            return positions
        except Exception as e:
            self.logger.error(f"Error retrieving positions: {str(e)}")
            return []

    def get_open_orders(self) -> List[Dict[str, Any]]:
        try:
            open_orders = self.client.Order.Order_getOrders(
                filter=json.dumps({"symbol": self.symbol, "open": True})
            ).result()[0]
            return open_orders
        except Exception as e:
            self.logger.error(f"Error retrieving open orders: {str(e)}")
            return []

    def open_position(self, side: str, quantity: int, order_type: str = "Market", **kwargs) -> Dict[str, Any]:
        try:
            order_params = {
                "symbol": self.symbol,
                "side": side,
                "orderQty": quantity,
                "ordType": order_type,
            }

            if order_type == "Limit" and "price" in kwargs:
                order_params["price"] = round(kwargs["price"], 2)

            if "exec_inst" in kwargs:
                order_params["execInst"] = kwargs["exec_inst"]

            if "clOrdID" in kwargs:
                order_params["clOrdID"] = kwargs["clOrdID"]

            if "text" in kwargs:
                order_params["text"] = kwargs["text"]

            order = self.client.Order.Order_new(**order_params).result()[0]

            result = {"entry": order}

            # Handle stop loss if provided
            if "stop_loss_price" in kwargs:
                sl_params = order_params.copy()
                sl_params["stopPx"] = kwargs["stop_loss_price"]
                sl_params["ordType"] = "Stop"
                sl_params["execInst"] = "Close"
                sl_params["side"] = "Sell" if side == "Buy" else "Buy"
                result["stop_loss"] = self.client.Order.Order_new(**sl_params).result()[0]

            # Handle take profit if provided
            if "take_profit_price" in kwargs:
                tp_params = order_params.copy()
                tp_params["stopPx"] = kwargs["take_profit_price"]
                tp_params["ordType"] = "MarketIfTouched"
                tp_params["side"] = "Sell" if side == "Buy" else "Buy"
                result["take_profit"] = self.client.Order.Order_new(**tp_params).result()[0]

            return result
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, side: str, quantity: int, order_type: str = "Market", **kwargs) -> Dict[str, Any]:
        try:
            order_params = {
                "symbol": self.symbol,
                "side": side,
                "orderQty": quantity,
                "ordType": order_type,
                "execInst": "Close"
            }

            if "clOrdID" in kwargs:
                order_params["clOrdID"] = kwargs["clOrdID"]

            if "text" in kwargs:
                order_params["text"] = kwargs["text"]

            order = self.client.Order.Order_new(**order_params).result()[0]
            return order
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None

    def close_all_positions(self, **kwargs) -> bool:
        try:
            positions = self.get_positions()
            for pos in positions:
                if pos['currentQty'] != 0:
                    side = "Sell" if pos['currentQty'] > 0 else "Buy"
                    qty = abs(pos['currentQty'])
                    self.close_position(
                        side=side,
                        quantity=qty,
                        order_type="Market",
                        clOrdID=kwargs.get("clOrdID"),
                        text=kwargs.get("text", "Close all positions")
                    )
            return True
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return False

    def set_leverage(self, leverage: int) -> Dict[str, Any]:
        try:
            if not 0 <= leverage <= 100:
                raise ValueError("Leverage must be between 0 and 100")

            response = self.client.Position.Position_updateLeverage(
                symbol=self.symbol,
                leverage=leverage
            ).result()[0]

            return response
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            return None