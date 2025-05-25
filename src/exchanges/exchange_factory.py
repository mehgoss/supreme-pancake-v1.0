from typing import Optional
import logging
from .base_exchange import BaseExchange
from .bitmex_exchange import BitMEXExchange
from .alpaca_exchange import AlpacaExchange
from ..config.exchange_config import ExchangeConfig, ExchangeType

class ExchangeFactory:
    @staticmethod
    def create_exchange(config: ExchangeConfig, logger: Optional[logging.Logger] = None) -> BaseExchange:
        """Create an exchange instance based on configuration"""
        if config.exchange_type == ExchangeType.BITMEX:
            return BitMEXExchange(
                api_key=config.api_key,
                api_secret=config.api_secret,
                test=config.test,
                symbol=config.symbol,
                timeframe=config.timeframe,
                logger=logger
            )
        elif config.exchange_type == ExchangeType.ALPACA:
            return AlpacaExchange(
                api_key=config.api_key,
                api_secret=config.api_secret,
                symbol=config.symbol,
                paper=config.test
            )
        else:
            raise ValueError(f"Unsupported exchange type: {config.exchange_type}")