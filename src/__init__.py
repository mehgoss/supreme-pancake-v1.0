"""Supreme Pancake Trading Bot - A modular crypto trading bot supporting multiple exchanges"""

from .executor import TradeExecutor
from .exchanges import (
    BaseExchange,
    BitMEXExchange,
    AlpacaExchange,
    ExchangeFactory
)
from .strategies import (
    BaseStrategy,
    MatteGreenStrategy
)
from .utils import (
    TelegramBot,
    configure_logging,
    get_trading_performance_summary
)

__version__ = "1.0.0"

__all__ = [
    'TradeExecutor',
    'BaseExchange',
    'BitMEXExchange',
    'AlpacaExchange',
    'ExchangeFactory',
    'BaseStrategy',
    'MatteGreenStrategy',
    'TelegramBot',
    'configure_logging',
    'get_trading_performance_summary'
]