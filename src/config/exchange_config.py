from dataclasses import dataclass
from typing import Optional
from enum import Enum

class ExchangeType(Enum):
    BITMEX = "bitmex"
    ALPACA = "alpaca"

@dataclass
class ExchangeConfig:
    exchange_type: ExchangeType
    api_key: str
    api_secret: str
    symbol: str
    timeframe: str
    test: bool = True
    leverage: int = 1
    
    @classmethod
    def from_env(cls, exchange_type: str, api_key: str, api_secret: str, 
                 symbol: str = "SOL-USD", timeframe: str = "5m", 
                 test: bool = True, leverage: int = 1) -> "ExchangeConfig":
        try:
            ex_type = ExchangeType(exchange_type.lower())
        except ValueError:
            raise ValueError(f"Unsupported exchange type: {exchange_type}. "
                           f"Supported types: {[e.value for e in ExchangeType]}")
        
        return cls(
            exchange_type=ex_type,
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol,
            timeframe=timeframe,
            test=test,
            leverage=leverage
        )