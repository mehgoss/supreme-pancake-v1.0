from src.executor import TradeExecutor
from src.config.exchange_config import ExchangeConfig 
from src.exchanges.exchange_factory import ExchangeFactory
from src.utils.telegram_bot import configure_logging
import os 
from dotenv import load_dotenv
load_dotenv()

def get_config():
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

def main():
    config = get_config()
    logger , _ = configure_logging(os.getenv('TOKEN'),os.getenv('CHAT_ID'))
    exchange = ExchangeFactory.create_exchange(config, logger)
    Bot_Trader = TradeExecutor(
        exchange =exchange, 
        risk_per_trade = 0.01, 
        lookback_period= 40, 
        fvg_threshold= 0.0003, 
        rr = 0.01,
        telegram_token = os.getenv('TOKEN'), 
        telegram_chat_id= os.getenv('CHAT_ID'), 
        tolarance  = 0.005
        )
    Bot_Trader.run(
    scan_interval=5*60, 
    max_runtime_minutes=30, 
    sleep_interval_minutes=1, 
    iterations_before_sleep=2
    )
    logger.info('Done running bot stopped all process')

if __name__ == "__main__":
    main()