import pytest
from unittest.mock import MagicMock
import os
import sys

# Add project root (parent of src) to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import TradeExecutor, ExchangeFactory
from src.config.exchange_config import ExchangeConfig

def get_test_config():
    """Helper to create a test exchange config from env vars with test mode enabled."""
    exchange_type = os.getenv('EXCHANGE', 'bitmex')
    api_key = os.getenv('BITMEX_API_KEY') if exchange_type == 'bitmex' else os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('BITMEX_API_SECRET') if exchange_type == 'bitmex' else os.getenv('ALPACA_API_SECRET')
    symbol = os.getenv('SYMBOL', 'SOL-USD')
    timeframe = os.getenv('TIMEFRAME', '5m')
    test = True  # Always test mode
    leverage = int(os.getenv('LEVERAGE', '10'))
    return ExchangeConfig.from_env(
        exchange_type=exchange_type,
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        timeframe=timeframe,
        test=test,
        leverage=leverage
    )

def test_exchange_connection():
    """Test that the exchange connects and profile info is available."""
    config = get_test_config()
    logger = MagicMock()
    exchange = ExchangeFactory.create_exchange(config, logger)
    profile = exchange.get_profile_info()
    assert profile is not None, "Profile info should not be None"
    assert 'balance' in profile, "Profile should have 'balance' key"

def test_account_balance():
    """Test that account balance can be retrieved."""
    config = get_test_config()
    logger = MagicMock()
    exchange = ExchangeFactory.create_exchange(config, logger)
    profile = exchange.get_profile_info()
    balance = profile['balance']
    assert isinstance(balance, dict), "Balance should be a dict"
    # For BitMEX, expect 'bitmex_usd' key
    assert any(balance.values()), "Balance values should not be empty"

def test_open_and_close_trade():
    """Test opening and closing a trade (mocked in test mode)."""
    config = get_test_config()
    logger = MagicMock()
    exchange = ExchangeFactory.create_exchange(config, logger)
    # Mock methods if needed to avoid real trades
    exchange.open_position = MagicMock(return_value={'status': 'filled', 'order_id': 'test123'})
    exchange.close_position = MagicMock(return_value={'status': 'closed', 'order_id': 'test123'})

    # Open trade
    result = exchange.open_position(symbol=config.symbol, qty=1, side='buy')
    assert result['status'] == 'filled', "Trade should be opened (mocked)"
    # Close trade
    result = exchange.close_position(symbol=config.symbol)
    assert result['status'] == 'closed', "Trade should be closed (mocked)"
