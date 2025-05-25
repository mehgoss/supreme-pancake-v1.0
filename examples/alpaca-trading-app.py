"""
Alpaca Algorithmic Trading Application

This script implements a simple moving average crossover strategy
with risk management features using the Alpaca API.

Usage:
1. Set your Alpaca API credentials in environment variables or in the script
2. Configure your trading parameters
3. Run the script: python alpaca_trading_app.py
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("alpaca_trading")

class AlpacaTradingApp:
    """
    A class that implements a trading strategy using the Alpaca API.
    Current implementation: Moving Average Crossover Strategy
    """
    
    def __init__(self, config):
        """
        Initialize the trading app with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary containing:
                - api_key: Alpaca API key
                - api_secret: Alpaca API secret
                - paper_trading: Boolean to indicate if using paper trading
                - symbols: List of stock symbols to trade
                - max_position_size: Maximum position size in dollars
                - risk_per_trade_pct: Percentage of account to risk per trade
                - short_window: Short-term moving average window
                - long_window: Long-term moving average window
        """
        self.config = config
        
        # Initialize API clients
        self.trading_client = TradingClient(
            config['api_key'],
            config['api_secret'],
            paper=config['paper_trading']
        )
        
        self.data_client = StockHistoricalDataClient(
            config['api_key'],
            config['api_secret']
        )
        
        # Trading parameters
        self.symbols = config['symbols']
        self.max_position_size = config['max_position_size']
        self.risk_per_trade_pct = config['risk_per_trade_pct']
        self.short_window = config['short_window']
        self.long_window = config['long_window']
        
        # Track positions and signals
        self.positions = {}
        self.signals = {}
        
        logger.info(f"AlpacaTradingApp initialized with symbols: {self.symbols}")
        logger.info(f"Using {'paper' if config['paper_trading'] else 'live'} trading")
    
    def get_account_info(self):
        """Get Alpaca account information."""
        account = self.trading_client.get_account()
        logger.info(f"Account: {account.id}")
        logger.info(f"Cash: ${float(account.cash):.2f}")
        logger.info(f"Portfolio value: ${float(account.portfolio_value):.2f}")
        logger.info(f"Buying power: ${float(account.buying_power):.2f}")
        return account
    
    def get_historical_data(self, symbol, timeframe=TimeFrame.DAY, limit=100):
        """
        Get historical bar data for a symbol.
        
        Args:
            symbol (str): The stock symbol
            timeframe (TimeFrame): The timeframe for bars
            limit (int): Number of bars to retrieve
            
        Returns:
            pandas.DataFrame: DataFrame containing historical data
        """
        try:
            # Calculate start and end times
            end = datetime.now()
            start = end - timedelta(days=limit * 2)  # Request more days to ensure we get enough bars
            
            # Define the request
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment='all'
            )
            
            # Get the bars
            bars = self.data_client.get_stock_bars(request_params)
            
            # Convert to dataframe
            if symbol in bars:
                df = bars[symbol].df
                logger.info(f"Retrieved {len(df)} bars for {symbol}")
                return df
            else:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_signals(self, symbol):
        """
        Calculate trading signals based on moving average crossover.
        
        Args:
            symbol (str): The stock symbol
            
        Returns:
            dict: Signal information including:
                - signal: 1 for buy, -1 for sell, 0 for hold
                - price: Current price
                - stop_loss: Calculated stop loss price
                - take_profit: Calculated take profit price
        """
        try:
            # Get historical data
            data = self.get_historical_data(symbol, TimeFrame.DAY, max(self.long_window * 2, 40))
            
            if data.empty:
                logger.warning(f"No data for {symbol}, skipping signal calculation")
                return {'signal': 0, 'price': 0, 'stop_loss': 0, 'take_profit': 0}
            
            # Calculate moving averages
            data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
            data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
            
            # Calculate ATR for stop loss
            data['high_low'] = data['high'] - data['low']
            data['high_close'] = abs(data['high'] - data['close'].shift())
            data['low_close'] = abs(data['low'] - data['close'].shift())
            data['tr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
            data['atr'] = data['tr'].rolling(window=14).mean()
            
            # Generate signal
            data['signal'] = 0
            # Crossover condition (short MA crosses above long MA)
            data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
            # Crossunder condition (short MA crosses below long MA)
            data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
            
            # Get the latest data point
            latest = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Determine signal based on crossover
            signal = 0
            if previous['short_ma'] <= previous['long_ma'] and latest['short_ma'] > latest['long_ma']:
                signal = 1  # Buy signal
            elif previous['short_ma'] >= previous['long_ma'] and latest['short_ma'] < latest['long_ma']:
                signal = -1  # Sell signal
            
            # Current price
            current_price = latest['close']
            
            # Calculate stop loss and take profit levels
            atr = latest['atr']
            stop_loss = current_price - (2 * atr) if signal == 1 else current_price + (2 * atr)
            take_profit = current_price + (3 * atr) if signal == 1 else current_price - (3 * atr)
            
            signal_info = {
                'signal': signal,
                'price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            }
            
            logger.info(f"Signal for {symbol}: {signal_info}")
            return signal_info
        except Exception as e:
            logger.error(f"Error calculating signals for {symbol}: {str(e)}")
            return {'signal': 0, 'price': 0, 'stop_loss': 0, 'take_profit': 0}
    
    def calculate_position_size(self, entry_price, stop_loss, account_value):
        """
        Calculate position size based on risk management rules.
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            account_value (float): Current account value
            
        Returns:
            int: Number of shares to trade
        """
        # Risk amount in dollars
        risk_amount = account_value * (self.risk_per_trade_pct / 100)
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate number of shares
        if risk_per_share == 0:
            return 0
            
        shares = int(risk_amount / risk_per_share)
        
        # Apply maximum position size limit
        position_value = shares * entry_price
        if position_value > self.max_position_size:
            shares = int(self.max_position_size / entry_price)
        
        return shares
    
    def place_market_order(self, symbol, side, qty, stop_loss=None, take_profit=None):
        """
        Place a market order with optional stop loss and take profit.
        
        Args:
            symbol (str): The stock symbol
            side (OrderSide): Buy or sell
            qty (int): Quantity of shares
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            
        Returns:
            Order: The order object
        """
        try:
            # Place the market order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_data)
            logger.info(f"Placed {side.name} order for {qty} shares of {symbol}")
            
            # Wait for the order to fill
            filled_order = self._wait_for_order_fill(order.id)
            if not filled_order:
                logger.warning(f"Order for {symbol} did not fill within timeout")
                return None
            
            # Place stop loss if provided
            if stop_loss:
                self._place_stop_loss(symbol, side, qty, stop_loss)
            
            # Place take profit if provided
            if take_profit:
                self._place_take_profit(symbol, side, qty, take_profit)
            
            return filled_order
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return None
    
    def _wait_for_order_fill(self, order_id, timeout=60):
        """Wait for an order to be filled."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            order = self.trading_client.get_order(order_id)
            if order.status == OrderStatus.FILLED:
                return order
            elif order.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
                logger.warning(f"Order {order_id} was {order.status}")
                return None
            time.sleep(2)
        return None
    
    def _place_stop_loss(self, symbol, entry_side, qty, stop_price):
        """Place a stop loss order."""
        try:
            # Determine the side for the stop loss
            stop_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
            
            # Create stop loss order
            stop_loss_order = self.trading_client.submit_order(
                symbol=symbol,
                qty=qty,
                side=stop_side,
                type='stop',
                stop_price=stop_price,
                time_in_force=TimeInForce.GTC
            )
            
            logger.info(f"Placed stop loss for {symbol} at {stop_price}")
            return stop_loss_order
        except Exception as e:
            logger.error(f"Error placing stop loss for {symbol}: {str(e)}")
            return None
    
    def _place_take_profit(self, symbol, entry_side, qty, take_profit_price):
        """Place a take profit order."""
        try:
            # Determine the side for the take profit
            take_profit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
            
            # Create take profit order
            take_profit_order = self.trading_client.submit_order(
                symbol=symbol,
                qty=qty,
                side=take_profit_side,
                type='limit',
                limit_price=take_profit_price,
                time_in_force=TimeInForce.GTC
            )
            
            logger.info(f"Placed take profit for {symbol} at {take_profit_price}")
            return take_profit_order
        except Exception as e:
            logger.error(f"Error placing take profit for {symbol}: {str(e)}")
            return None
    
    def cancel_open_orders(self, symbol=None):
        """Cancel all open orders for a symbol or all symbols."""
        try:
            if symbol:
                orders = self.trading_client.get_orders(GetOrdersRequest(
                    status='open',
                    symbols=[symbol]
                ))
                for order in orders:
                    self.trading_client.cancel_order(order.id)
                logger.info(f"Canceled all open orders for {symbol}")
            else:
                self.trading_client.cancel_orders()
                logger.info("Canceled all open orders")
        except Exception as e:
            logger.error(f"Error canceling orders: {str(e)}")
    
    def get_position(self, symbol):
        """Get current position for a symbol."""
        try:
            position = self.trading_client.get_open_position(symbol)
            logger.info(f"Current position for {symbol}: {position.qty} shares at avg price ${float(position.avg_entry_price):.2f}")
            return position
        except Exception as e:
            # If position doesn't exist, a 404 error is raised
            logger.debug(f"No position for {symbol}: {str(e)}")
            return None
    
    def execute_strategy(self):
        """
        Execute the trading strategy for all symbols.
        This is the main method that runs the trading logic.
        """
        logger.info("Starting strategy execution")
        
        # Get account information
        account = self.get_account_info()
        account_value = float(account.portfolio_value)
        
        for symbol in self.symbols:
            try:
                # Check current position
                position = self.get_position(symbol)
                position_qty = int(float(position.qty)) if position else 0
                
                # Calculate signals
                signal_info = self.calculate_signals(symbol)
                signal = signal_info['signal']
                current_price = signal_info['price']
                stop_loss = signal_info['stop_loss']
                take_profit = signal_info['take_profit']
                
                # Track signals
                self.signals[symbol] = signal_info
                
                # Execute trades based on signals
                if signal == 1 and position_qty <= 0:  # Buy signal and no long position
                    # Cancel any existing orders
                    self.cancel_open_orders(symbol)
                    
                    # Close short position if exists
                    if position_qty < 0:
                        abs_qty = abs(position_qty)
                        logger.info(f"Closing short position of {abs_qty} shares for {symbol}")
                        self.place_market_order(symbol, OrderSide.BUY, abs_qty)
                    
                    # Calculate position size
                    qty = self.calculate_position_size(current_price, stop_loss, account_value)
                    
                    # Only place order if we have shares to buy
                    if qty > 0:
                        logger.info(f"BUY signal for {symbol} - placing order for {qty} shares")
                        self.place_market_order(
                            symbol, 
                            OrderSide.BUY, 
                            qty, 
                            stop_loss, 
                            take_profit
                        )
                
                elif signal == -1 and position_qty >= 0:  # Sell signal and no short position
                    # Cancel any existing orders
                    self.cancel_open_orders(symbol)
                    
                    # Close long position if exists
                    if position_qty > 0:
                        logger.info(f"Closing long position of {position_qty} shares for {symbol}")
                        self.place_market_order(symbol, OrderSide.SELL, position_qty)
                    
                    # If shorting is enabled, calculate and place short position
                    # Note: Shorting may not be available for all accounts or symbols
                    # Un-comment this section if you want to enable shorting
                    
                    # qty = self.calculate_position_size(current_price, stop_loss, account_value)
                    # if qty > 0:
                    #     logger.info(f"SELL signal for {symbol} - placing short order for {qty} shares")
                    #     self.place_market_order(
                    #         symbol, 
                    #         OrderSide.SELL, 
                    #         qty, 
                    #         stop_loss, 
                    #         take_profit
                    #     )
                
                else:
                    logger.info(f"No new signals for {symbol}, current signal: {signal}, position: {position_qty}")
            
            except Exception as e:
                logger.error(f"Error executing strategy for {symbol}: {str(e)}")
    
    def run(self, interval=60):
        """
        Run the trading app in a loop.
        
        Args:
            interval (int): Time interval between iterations in seconds
        """
        try:
            logger.info(f"Starting trading app with {len(self.symbols)} symbols")
            
            while True:
                try:
                    self.execute_strategy()
                    logger.info(f"Waiting {interval} seconds until next iteration")
                    time.sleep(interval)
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, exiting")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(interval)
        
        finally:
            logger.info("Shutting down trading app")


def main():
    """Main function to run the trading app."""
    # Get API credentials from environment variables or specify directly
    api_key = os.environ.get("ALPACA_API_KEY", "your_api_key_here")
    api_secret = os.environ.get("ALPACA_API_SECRET", "your_api_secret_here")
    
    # Configuration
    config = {
        'api_key': api_key,
        'api_secret': api_secret,
        'paper_trading': True,  # Set to False for live trading
        'symbols': ['AAPL', 'MSFT', 'GOOG'],  # Symbols to trade
        'max_position_size': 5000,  # Maximum position size in dollars
        'risk_per_trade_pct': 1.0,  # Risk 1% of account per trade
        'short_window': 10,  # Short-term moving average window
        'long_window': 30,  # Long-term moving average window
    }
    
    # Create and run the trading app
    app = AlpacaTradingApp(config)
    app.run()


if __name__ == "__main__":
    main()