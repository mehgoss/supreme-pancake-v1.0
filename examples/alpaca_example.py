#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv(dotenv_path=project_root / '.env')

from src import (
    TradeExecutor,
    ExchangeFactory,
    configure_logging
)
from src.config.exchange_config import ExchangeConfig

def main():
    # Configure logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Get Alpaca credentials from environment
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    
    if not all([api_key, api_secret]):
        logger.error("Missing Alpaca API credentials")
        sys.exit(1)

    try:
        # Create exchange configuration for Alpaca
        config = ExchangeConfig.from_env(
            exchange_type="alpaca",
            api_key=api_key,
            api_secret=api_secret,
            symbol="SPY",  # Using SPY as an example
            timeframe="5m",
            test=True  # Use paper trading
        )
        
        # Create Alpaca exchange instance
        exchange = ExchangeFactory.create_exchange(config, logger)
        
        # Initialize trader
        trader = TradeExecutor(
            exchange=exchange,
            risk_per_trade=0.01,  # 1% risk per trade
            log=logger
        )
        
        # Run trading strategy
        trader.run(
            scan_interval=2*60,  # Scan every 2 minutes
            max_runtime_minutes=40,
            sleep_interval_minutes=1,
            iterations_before_sleep=2
        )
    
    except Exception as e:
        logger.error(f"Error running Alpaca example: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()