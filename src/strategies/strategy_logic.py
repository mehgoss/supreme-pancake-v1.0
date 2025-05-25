import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import logging

class BaseStrategy:
    """Base class for trading strategies"""
    def analyze_market_data(self, df: pd.DataFrame) -> list:
        """Analyze market data and return trading signals"""
        raise NotImplementedError("Subclasses must implement analyze_market_data")

class MatteGreenStrategy(BaseStrategy):
    def __init__(self, lookback_period=20, fvg_threshold=0.003):
        self.lookback_period = lookback_period
        self.fvg_threshold = fvg_threshold
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.market_bias = 'neutral'
        self.logger = logging.getLogger(__name__)

    def analyze_market_data(self, df):
        """Update market data and run analysis"""
        self.df = df
        self.identify_swing_points()
        self.detect_market_structure()
        return self.generate_signals()

    def identify_swing_points(self):
        """Identify swing high and low points"""
        window = min(self.lookback_period // 2, 3)
        self.swing_highs = np.zeros(len(self.df))
        self.swing_lows = np.zeros(len(self.df))
        
        for i in range(window, len(self.df) - window):
            if all(self.df['high'].iloc[i] >= self.df['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['high'].iloc[i] >= self.df['high'].iloc[i+j] for j in range(1, window+1)):
                self.swing_highs[i] = 1
            if all(self.df['low'].iloc[i] <= self.df['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(self.df['low'].iloc[i] <= self.df['low'].iloc[i+j] for j in range(1, window+1)):
                self.swing_lows[i] = 1

    def detect_market_structure(self):
        """Detect market structure including CHoCH and BoS points"""
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        recent_highs = deque(maxlen=self.lookback_period)
        recent_lows = deque(maxlen=self.lookback_period)

        for i in range(self.lookback_period, len(self.df)):
            if self.swing_highs[i]:
                recent_highs.append((i, self.df['high'].iloc[i]))
            if self.swing_lows[i]:
                recent_lows.append((i, self.df['low'].iloc[i]))

            # Detect Change of Character (CHoCH)
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (self.market_bias in ['bullish', 'neutral']) and \
                   recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bearish'))
                    self.market_bias = 'bearish'
                elif (self.market_bias in ['bearish', 'neutral']) and \
                     recent_lows[-1][1] > recent_lows[-2][1] and recent_highs[-1][1] > recent_highs[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bullish'))
                    self.market_bias = 'bullish'

            # Detect Break of Structure (BoS)
            if self.market_bias == 'bearish' and recent_highs and self.df['high'].iloc[i] > recent_highs[-1][1]:
                self.bos_points.append((i, self.df['high'].iloc[i], 'bullish'))
            elif self.market_bias == 'bullish' and recent_lows and self.df['low'].iloc[i] < recent_lows[-1][1]:
                self.bos_points.append((i, self.df['low'].iloc[i], 'bearish'))

            # Detect Fair Value Gaps (FVG)
            if i > 1:
                if (self.df['low'].iloc[i] - self.df['high'].iloc[i-2]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i-2], self.df['low'].iloc[i], 'bullish'))
                if (self.df['low'].iloc[i-2] - self.df['high'].iloc[i]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i], self.df['low'].iloc[i-2], 'bearish'))

    def calculate_position_params(self, current_price, direction):
        """Calculate entry, stop loss, and take profit levels"""
        lookback_start = max(0, len(self.df) - self.lookback_period)
        if direction == 'long':
            stop_dist = current_price - min(self.df['low'].iloc[lookback_start:])
            stop_loss = current_price - stop_dist * 0.4
            take_profit = current_price + stop_dist * 0.5
        else:  # short
            stop_dist = max(self.df['high'].iloc[lookback_start:]) - current_price
            stop_loss = current_price + stop_dist * 0.4
            take_profit = current_price - stop_dist * 0.5
        
        return {
            'entry_price': current_price,
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_distance': stop_dist
        }

    def generate_signals(self):
        """Generate trading signals based on market structure"""
        current_idx = len(self.df) - 1
        current_price = self.df['close'].iloc[current_idx]
        
        signals = []
        
        if current_idx >= self.lookback_period:
            direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
            
            if direction:
                params = self.calculate_position_params(current_price, direction)
                signals.append({
                    'action': 'entry',
                    'side': direction,
                    'price': round(params['entry_price'], 2),
                    'stop_loss': params['stop_loss'],
                    'take_profit': params['take_profit'],
                    'entry_idx': current_idx
                })
        
        return signals

    def get_market_state(self):
        """Return current market state information"""
        return {
            'market_bias': self.market_bias,
            'structure': {
                'swing_highs': self.swing_highs,
                'swing_lows': self.swing_lows,
                'choch_points': self.choch_points,
                'bos_points': self.bos_points,
                'fvg_areas': self.fvg_areas
            }
        }