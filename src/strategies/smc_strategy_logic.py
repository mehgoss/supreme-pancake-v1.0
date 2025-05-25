import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import logging

class BaseStrategy:
    """Base class for trading strategies"""
    def analyze_market_data(self, df: pd.DataFrame, c_balance:float) -> list:
        """Analyze market data and return trading signals"""
        raise NotImplementedError("Subclasses must implement analyze_market_data")

class MatteGreenStrategy(BaseStrategy):
    def __init__(self, lookback_period=40, fvg_threshold=0.0003, rr_ratio=0.25, risk_per_trade:float=0.01):
        self.lookback_period = lookback_period
        self.fvg_threshold = fvg_threshold
        self.df = pd.DataFrame()
        self.swing_highs = []
        self.swing_lows = []
        self.choch_points = []
        self.bos_points = []
        self.fvg_areas = []
        self.market_bias = 'neutral'
        self.rr_ratio = rr_ratio
        self.risk_per_trade = risk_per_trade
        self.logger = logging.getLogger(__name__)

    def analyze_market_data(self, df, c_balance):
        """Update market data and run analysis"""
        self.df = df
        self.current_balance = c_balance
        self.identify_swing_points()
        self.detect_market_structure()
        return self.generate_signals()

    def identify_swing_points(self):
        """Identify swing high and low points"""
        window = min(5, self.lookback_period // 2)
        self.swing_highs = np.zeros(len(self.df['high']))
        self.swing_lows = np.zeros(len(self.df['low']))
        
        for i in range(window, len(self.df['high']) - window):
            if all(self.df['high'].iloc[i] >= self.df['high'].iloc[i-window:i]) and \
               all(self.df['high'].iloc[i] >= self.df['high'].iloc[i+1:i+window+1]):
                self.swing_highs[i] = 1
            if all(self.df['low'].iloc[i] <= self.df['low'].iloc[i-window:i]) and \
               all(self.df['low'].iloc[i] <= self.df['low'].iloc[i+1:i+window+1]):
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

            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if (self.market_bias in ['bullish', 'neutral']) and \
                   recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bearish'))
                    self.market_bias = 'bearish'
                elif (self.market_bias in ['bearish', 'neutral']) and \
                     recent_lows[-1][1] > recent_lows[-2][1] and recent_highs[-1][1] > recent_highs[-2][1]:
                    self.choch_points.append((i, self.df['close'].iloc[i], 'bullish'))
                    self.market_bias = 'bullish'

            if self.market_bias == 'bearish' and recent_highs and self.df['high'].iloc[i] > recent_highs[-1][1]:
                self.bos_points.append((i, self.df['high'].iloc[i], 'bullish'))
            elif self.market_bias == 'bullish' and recent_lows and self.df['low'].iloc[i] < recent_lows[-1][1]:
                self.bos_points.append((i, self.df['low'].iloc[i], 'bearish'))

            if i > 1:
                if (self.df['low'].iloc[i] - self.df['high'].iloc[i-2]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i-2], self.df['low'].iloc[i], 'bullish'))
                if (self.df['low'].iloc[i-2] - self.df['high'].iloc[i]) > self.fvg_threshold * self.df['close'].iloc[i]:
                    self.fvg_areas.append((i-2, i, self.df['high'].iloc[i], self.df['low'].iloc[i-2], 'bearish'))

    def calculate_position_params(self, current_price, direction):
        """Calculate entry, stop loss, and take profit levels"""
        # lookback_start = max(0, len(self.df) - self.lookback_period)
        if direction == 'long':
            _lookback_start = max(0, current_price - 5)
            stop_dist = current_price - min(self.df['low'].iloc[_lookback_start:])
            stop_loss = current_price - stop_dist * 0.6
            take_profit = current_price + stop_dist * self.rr_ratio
        else:  # short
            _lookback_start = max(0, current_price - 5)
            stop_dist = max(self.df['high'].iloc[_lookback_start:]) - current_price
            stop_loss = current_price + stop_dist * 0.6
            take_profit = current_price - stop_dist * self.rr_ratio
        
        return {
            'entry_price': current_price,
            'stop_loss': round(stop_loss, 4),
            'take_profit': round(take_profit, 4),
            'risk_distance': stop_dist
        }

    def generate_signals(self):
        """Generate trading signals based on market structure"""
        
        signals = []
        current_idx = len(self.df) - 1
        current_price = self.df['close'].iloc[current_idx]  # 5m closing price
    
        total_risk_amount = sum(abs(entry_price - stop_loss) * size for _, _, entry_price, _, stop_loss, _, size, _, _, _, _ in self.current_trades)
        max_total_risk = self.current_balance * 0.20
        
        if current_idx >= self.lookback_period:
            direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
            if direction:
                fvg_confirmed = False
                # Add CHoCH points as potential entries
                for idx, price, choch_type in self.choch_points:
                    if idx == current_idx:
                        if choch_type == 'bullish':
                            # if current_idx >= self.lookback_period:
                            # direction = 'long' if self.market_bias == 'bullish' else 'short' if self.market_bias == 'bearish' else None
                            for fvg_start, fvg_end, fvg_min, fvg_max, fvg_type in self.fvg_areas:
                                # FVG should align with trade direction
                                if (direction == 'long' and fvg_type == 'bullish') or \
                                    (choch_type == 'bullish' and fvg_type == 'bullish'):
                                    # FVG should be recent
                                    if current_idx - fvg_end <= 10:
                                        fvg_confirmed = True
                                        break
                                        
                            entry_price = current_price  # Intended 5m entry price
                            lookback_start = max(0, current_idx - self.lookback_period)
                            stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                                        max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                            stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                            take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                            size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                            risk_of_new_trade = abs(entry_price - stop_loss) * size
    
                            if total_risk_amount + risk_of_new_trade <= max_total_risk:
                                signals.append(
                                    {'action': 'entry', 
                                     'side': direction, 
                                     'price': round(entry_price, 2), 
                                     'stop_loss': round(stop_loss, 4),
                                    'take_profit': round(take_profit, 4), 
                                    'position_size': int(size) if size < 1 else 1, 
                                    'entry_idx': current_idx,
                                    'signal_type': 'CHoCH',
                                    'fvg_confirmed': fvg_confirmed
                                    })
                                # self.current_trades.append((None, current_idx, entry_price, direction, stop_loss, take_profit, size, None, None, None, None))
                                self.logger.info(f"Entry signal: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")

                                # potential_entries.append((i, self.close[i], 'long', 'CHoCH'))
                            else:
                                for fvg_start, fvg_end, fvg_min, fvg_max, fvg_type in self.fvg_areas:
                                    # FVG should align with trade direction
                                    if (direction == 'short' and fvg_type == 'bearish') or \
                                        (choch_type == 'bearish' and fvg_type == 'bearish'):
                                        # FVG should be recent
                                        if current_idx - fvg_end <= 10:
                                            fvg_confirmed = True
                                            break

                                entry_price = current_price  # Intended 5m entry price
                                lookback_start = max(0, current_idx - self.lookback_period)
                                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                                            max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                                stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                                risk_of_new_trade = abs(entry_price - stop_loss) * size

                                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                                    signals.append(
                                        {'action': 'entry', 
                                         'side': direction, 
                                         'price': round(entry_price, 2), 
                                         'stop_loss': round(stop_loss, 4),
                                        'take_profit': round(take_profit, 4), 
                                        'position_size': int(size) if size < 1 else 1, 
                                        'entry_idx': current_idx,
                                        'signal_type': 'CHoCH',
                                        'fvg_confirmed': fvg_confirmed
                                        })
                                # pass
                                # potential_entries.append((i, self.close[i], 'short', 'CHoCH'))
                # Add BOS points as potential entries
                for idx, price, bos_type in self.bos_points:
                        if idx == current_idx:
                            if bos_type == 'bullish':                                        
                                entry_price = current_price  # Intended 5m entry price
                                lookback_start = max(0, current_idx - self.lookback_period)
                                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                                        max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                                stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                                risk_of_new_trade = abs(entry_price - stop_loss) * size

                                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                                    signals.append({
                                        'action': 'entry', 
                                        'side': direction, 
                                        'price': round(entry_price, 2), 
                                        'stop_loss': round(stop_loss, 4),
                                        'take_profit': round(take_profit, 4), 
                                        'position_size': int(size) if size < 1 else 1, 
                                        'entry_idx': current_idx,
                                        'signal_type': 'BOS',
                                        'fvg_confirmed': fvg_confirmed
                                        })
                                    # self.current_trades.append((None, current_idx, entry_price, direction, stop_loss, take_profit, size, None, None, None, None))
                                    self.logger.info(f"Entry signal: {direction} at {entry_price}, SL: {stop_loss}, TP: {take_profit}")
                                    # potential_entries.append((i, self.close[i], 'long', 'BOS'))
                                else:                          
                                    entry_price = current_price  # Intended 5m entry price
                                    lookback_start = max(0, current_idx - self.lookback_period)
                                    stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                                            max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                                    stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                                    take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                                    size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                                    risk_of_new_trade = abs(entry_price - stop_loss) * size

                                    if total_risk_amount + risk_of_new_trade <= max_total_risk:
                                        signals.append({
                                            'action': 'entry', 
                                            'side': direction, 
                                            'price': round(entry_price, 2), 
                                            'stop_loss': round(stop_loss, 4),
                                            'take_profit': round(take_profit, 4), 
                                            'position_size': int(size) if size < 1 else 1, 
                                            'entry_idx': current_idx,
                                            'signal_type': 'BOS',
                                            'fvg_confirmed': fvg_confirmed
                                            })

                                    # pass
                                    # potential_entries.append((i, self.close[i], 'short', 'BO  S'))
                entry_price = current_price  # Intended 5m entry price
                lookback_start = max(0, current_idx - self.lookback_period)
                stop_dist = entry_price - min(self.df['low'].iloc[lookback_start:current_idx+1]) if direction == 'long' else \
                        max(self.df['high'].iloc[lookback_start:current_idx+1]) - entry_price
                stop_loss = entry_price - stop_dist * 0.25 if direction == 'long' else entry_price + stop_dist * 0.25
                take_profit = entry_price + stop_dist * self.rr_ratio if direction == 'long' else entry_price - stop_dist * self.rr_ratio
                size = (self.current_balance * self.risk_per_trade) / abs(entry_price - stop_loss)
                risk_of_new_trade = abs(entry_price - stop_loss) * size

                if total_risk_amount + risk_of_new_trade <= max_total_risk:
                    signals.append({
                        'action': 'entry', 
                        'side': direction, 
                        'price': round(entry_price, 2), 
                        'stop_loss': round(stop_loss, 4),
                        'take_profit': round(take_profit, 4), 
                        'position_size': int(size) if size < 1 else 1, 
                        'entry_idx': current_idx,
                        'signal_type': 'Direction',
                        'fvg_confirmed': fvg_confirmed
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