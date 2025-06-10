from typing import Optional, Dict, Any, List
import json
import logging
import pandas as pd
import bitmex
import asyncio
import websockets
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
            self.symbol = symbol.replace('-', '')  # e.g., SOL-USD -> SOLUSD
            self.timeframe = timeframe
            self.test = test
            self.max_balance_usage = 0.20
            network_type = 'testnet' if test else 'mainnet'
            self.logger.info(f"BitMEXExchange initialized for {self.symbol} on {network_type}")
        except Exception as e:
            self.logger.error(f"Initialization error: {str(e)}")
            raise

    async def _build_candles(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
        """Build candles from WebSocket trade stream."""
        base_url = 'testnet.bitmex.com' if self.test else 'www.bitmex.com'
        uri = f"wss://{base_url}/realtime?subscribe=trade:{symbol}"
        candles = []
        batch = []
        last_ts = None
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '1h': 3600,
            '1d': 86400
        }.get(timeframe)
        if not timeframe_seconds:
            raise ValueError(f"Invalid timeframe. Supported: {', '.join(timeframe_seconds.keys())}")

        try:
            async with websockets.connect(uri, ping_interval=30, ping_timeout=60) as ws:
                self.logger.info(f"Connected to WebSocket: {uri}")
                while len(candles) < count:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(msg)
                        for trade in data.get('data', []):
                            try:
                                ts = datetime.strptime(trade['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
                                # Align timestamp to timeframe boundary
                                ts = ts.replace(second=0, microsecond=0)
                                if timeframe == '1h':
                                    ts = ts.replace(minute=0)
                                elif timeframe == '1d':
                                    ts = ts.replace(hour=0, minute=0)

                                if last_ts is None:
                                    last_ts = ts

                                if ts > last_ts:
                                    # Finalize candle
                                    if batch:
                                        candle = {
                                            'timestamp': last_ts,
                                            'open': batch[0]['price'],
                                            'high': max(t['price'] for t in batch),
                                            'low': min(t['price'] for t in batch),
                                            'close': batch[-1]['price'],
                                            'volume': sum(t['size'] for t in batch)
                                        }
                                        candles.append(candle)
                                        self.logger.debug(f"Built candle: {candle}")
                                        batch = []
                                        last_ts = ts

                                batch.append(trade)
                            except Exception as e:
                                self.logger.error(f"Error processing trade: {str(e)}")

                        # Trim candles to desired count
                        if len(candles) >= count:
                            candles = candles[-count:]
                            break
                    except asyncio.TimeoutError:
                        self.logger.warning("WebSocket receive timeout, continuing...")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("WebSocket connection closed, attempting to reconnect...")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
            return []

        return candles

    def get_candle(self, timeframe: Optional[str] = None, count: int = 100) -> Optional[pd.DataFrame]:
        """Retrieve candlestick data using WebSocket trade stream."""
        timeframe = timeframe or self.timeframe
        try:
            valid_timeframes = ['1m', '5m', '1h', '1d']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}")

            # Run async candle builder
            candles = asyncio.run(self._build_candles(self.symbol, timeframe, count))

            if not candles:
                self.logger.warning("No candle data retrieved")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

    # Other methods (get_profile_info, get_positions, etc.) remain unchanged
    # ... [Include the rest of your original class methods here] ...
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

    async def _build_candles(self, symbol: str, timeframe: str, count: int) -> List[Dict[str, Any]]:
        """Build candles from WebSocket trade stream."""
        base_url = 'testnet.bitmex.com' if self.test else 'www.bitmex.com'
        uri = f"wss://{base_url}/realtime?subscribe=trade:{symbol}"
        candles = []
        batch = []
        last_ts = None
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '1h': 3600,
            '1d': 86400
        }.get(timeframe)
        if not timeframe_seconds:
            raise ValueError(f"Invalid timeframe. Supported: {', '.join(timeframe_seconds.keys())}")

        try:
            async with websockets.connect(uri, ping_interval=30, ping_timeout=60) as ws:
                self.logger.info(f"Connected to WebSocket: {uri}")
                while len(candles) < count:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                        data = json.loads(msg)
                        for trade in data.get('data', []):
                            try:
                                ts = datetime.strptime(trade['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ")
                                # Align timestamp to timeframe boundary
                                ts = ts.replace(second=0, microsecond=0)
                                if timeframe == '1h':
                                    ts = ts.replace(minute=0)
                                elif timeframe == '1d':
                                    ts = ts.replace(hour=0, minute=0)

                                if last_ts is None:
                                    last_ts = ts

                                if ts > last_ts:
                                    # Finalize candle
                                    if batch:
                                        candle = {
                                            'timestamp': last_ts,
                                            'open': batch[0]['price'],
                                            'high': max(t['price'] for t in batch),
                                            'low': min(t['price'] for t in batch),
                                            'close': batch[-1]['price'],
                                            'volume': sum(t['size'] for t in batch)
                                        }
                                        candles.append(candle)
                                        self.logger.debug(f"Built candle: {candle}")
                                        batch = []
                                        last_ts = ts

                                batch.append(trade)
                            except Exception as e:
                                self.logger.error(f"Error processing trade: {str(e)}")

                        # Trim candles to desired count
                        if len(candles) >= count:
                            candles = candles[-count:]
                            break
                    except asyncio.TimeoutError:
                        self.logger.warning("WebSocket receive timeout, continuing...")
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self.logger.error("WebSocket connection closed, attempting to reconnect...")
                        break

        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
            return []

        return candles

    def get_candle(self, timeframe: Optional[str] = None, count: int = 100) -> Optional[pd.DataFrame]:
        """Retrieve candlestick data using WebSocket trade stream."""
        timeframe = timeframe or self.timeframe
        try:
            valid_timeframes = ['1m', '5m', '1h', '1d']
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe. Supported: {', '.join(valid_timeframes)}")

            # Run async candle builder
            candles = asyncio.run(self._build_candles(self.symbol, timeframe, count))

            if not candles:
                self.logger.warning("No candle data retrieved")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            self.logger.error(f"Error retrieving candle data: {str(e)}")
            return None

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
                sl_params["stopPx"] = round(kwargs["stop_loss_price"],2)
                sl_params["ordType"] = "Stop"
                sl_params["execInst"] = "Close"
                sl_params["side"] = "Sell" if side == "Buy" else "Buy"
                result["stop_loss"] = self.client.Order.Order_new(**sl_params).result()[0]

            # Handle take profit if provided
            if "take_profit_price" in kwargs:
                tp_params = order_params.copy()
                tp_params["stopPx"] = round(kwargs["take_profit_price"], 2)
                tp_params["ordType"] = "MarketIfTouched"
                tp_params["side"] = "Sell" if side == "Buy" else "Buy"
                result["take_profit"] = self.client.Order.Order_new(**tp_params).result()[0]

            return result
        except Exception as e:
            self.logger.error(f"Error opening position: {str(e)}")
            return None

    def close_position(self, side: str, quantity: int, order_type: str = "Market", **kwargs) -> Dict[str, Any]:
        positions = self.get_positions()
        if not positions:
            self.logger.info("No positions to close")
            return None

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
            self.logger.info(f"Closed order :\n{order}\n")
            return order
        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            return None

    def close_all_positions(self, **kwargs) -> bool:
        positions = self.get_positions()
        if not positions:
            self.logger.info("No positions to close")
            return None

        try:
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