"""Performance calculation utilities for trading strategies"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

def get_trading_performance_summary(wallet_history: List[Dict], positions: List[Dict]) -> Dict[str, Any]:
    """Calculate trading performance metrics"""
    try:
        # Convert wallet history to DataFrame
        wallet_df = pd.DataFrame(wallet_history) if wallet_history else pd.DataFrame()
        positions_df = pd.DataFrame(positions) if positions else pd.DataFrame()
        
        # Calculate basic metrics
        total_trades = len(wallet_df)
        winning_trades = len(wallet_df[wallet_df['realized_pnl'] > 0]) if not wallet_df.empty else 0
        losing_trades = len(wallet_df[wallet_df['realized_pnl'] < 0]) if not wallet_df.empty else 0
        
        # Calculate win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate profit metrics
        total_profit = wallet_df['realized_pnl'].sum() if not wallet_df.empty else 0
        avg_profit = wallet_df[wallet_df['realized_pnl'] > 0]['realized_pnl'].mean() if not wallet_df.empty else 0
        avg_loss = wallet_df[wallet_df['realized_pnl'] < 0]['realized_pnl'].mean() if not wallet_df.empty else 0
        
        # Calculate risk metrics
        risk_reward_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        max_drawdown = wallet_df['realized_pnl'].min() if not wallet_df.empty else 0
        
        # Calculate current positions value
        open_positions_value = positions_df['unrealisedPnl'].sum() if not positions_df.empty else 0
        
        return {
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2),
                'average_profit': round(avg_profit, 2) if not pd.isna(avg_profit) else 0,
                'average_loss': round(avg_loss, 2) if not pd.isna(avg_loss) else 0,
                'risk_reward_ratio': round(risk_reward_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'open_positions_value': round(open_positions_value, 2),
                'timestamp': datetime.now().isoformat()
            },
            'positions': positions,
            'history': wallet_history
        }
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }