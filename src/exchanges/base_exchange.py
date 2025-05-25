from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Dict, Any, List

class BaseExchange(ABC):
    """Base class for all exchange implementations"""
    
    @abstractmethod
    def get_profile_info(self) -> Dict[str, Any]:
        """Get account profile information"""
        pass

    @abstractmethod
    def get_candle(self, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get candlestick data"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get open orders"""
        pass

    @abstractmethod
    def open_position(self, side: str, quantity: int, order_type: str, **kwargs) -> Dict[str, Any]:
        """Open a new position"""
        pass

    @abstractmethod
    def close_position(self, side: str, quantity: int, order_type: str, **kwargs) -> Dict[str, Any]:
        """Close an existing position"""
        pass

    @abstractmethod
    def close_all_positions(self, **kwargs) -> bool:
        """Close all open positions"""
        pass

    @abstractmethod
    def set_leverage(self, leverage: int) -> Dict[str, Any]:
        """Set leverage for trading"""
        pass