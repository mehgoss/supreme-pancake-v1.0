# Supreme Pancake

**Supreme Pancake V1.0** is a Python-based trading bot designed to execute automated trading strategies on the BitMEX testnet. It leverages the **MatteGreen** trading strategy to analyze the **SOL-USD** market on a **5-minute timeframe**, with configurable risk management and **Telegram logging integration**.

## Features

- **BitMEX Testnet Integration**: Uses the **BitMEXTestAPI** for safe, simulated trading.
- **MatteGreen Strategy**: Custom trading logic for signal detection and execution.
- **Time Zone Support**: Operates in **South African Standard Time (SAST)**.
- **Logging**: Detailed logging with optional Telegram notifications via **TeleLogBot**.
- **Configurable**: Adjust **risk per trade, symbol, and timeframe** via parameters.

## Prerequisites

- **Python 3.10+**
- Required libraries: `pytz`, `requests` (assumed for API calls)
- Custom modules: `TeleLogBot`, `MatteGreen`, `BitMEXApi` (ensure these are available or included)
- Environment variables (.env):
  ```bash
  # BitMEX API Credentials
  BITMEX_API_KEY=
  BITMEX_API_SECRET=

  # Alpaca API Credentials 
  ALPACA_API_KEY=your_alpaca_key_here
  ALPACA_API_SECRET=your_alpaca_secret_here

  # Telegram Bot Credentials
  TOKEN=
  CHAT_ID=

  # Trading Configuration
  EXCHANGE=bitmex  # or alpaca
  SYMBOL=SOL-USD
  TIMEFRAME=5m
  RISK_PER_TRADE=0.02
  INITIAL_CAPITAL=10000
  LEVERAGE=10

  # Environment
  TEST_MODE=True
  LOG_LEVEL=INFO
  ```

## Installation

### Clone the repository:
```bash
git clone https://github.com/mehgoss/supreme-pancake-v1.0.git
cd supreme-pancake-v1.0
```

### Install dependencies:
```bash
pip install -r requirements.txt
```


### Set up environment variables:
```bash
export TOKEN="your-telegram-token"
export CHAT_ID="your-chat-id"
export API_KEY="your-bitmex-api-key"
export API_SECRET="your-bitmex-api-secret"
```

### Usage:
```bash
python App.py
```

## Badge

[![BitMEX v Strategy Backtest](https://github.com/mehgoss/supreme-pancake/actions/workflows/Green.yml/badge.svg?branch=main&event=schedule)](https://github.com/mehgoss/supreme-pancake/actions/workflows/Green.yml)
