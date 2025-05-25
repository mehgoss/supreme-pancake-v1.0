"""Telegram bot utilities for trading notifications"""

import asyncio
import logging
import sys
import time 
from io import BytesIO
import pytz
from datetime import datetime
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import TelegramError
from httpx import AsyncClient, Limits

# Set default encoding for all file operations
import codecs
codecs.register_error('strict', codecs.replace_errors)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def get_sast_time():
    """Get current time in South African Standard Time (SAST)"""
    utc_now = datetime.utcnow()
    sast = pytz.timezone('Africa/Johannesburg')
    return utc_now.replace(tzinfo=pytz.utc).astimezone(sast)

class TelegramBot:
    """Telegram bot for sending trading notifications"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._bot = None
        self._client = AsyncClient(
            limits=Limits(max_connections=100, max_keepalive_connections=20),
            timeout=30.0
        )

    def _ensure_bot(self):
        """Ensure bot is initialized"""
        if not self._bot:
            if not self.token or "your_bot_token" in self.token:
                raise ValueError("Invalid bot token.")
            trequest = HTTPXRequest(connection_pool_size=20)
            self._bot = Bot(token=self.token, request=trequest)

    async def _async_send_message(self, message=None):
        """Send text message asynchronously"""
        try:
            self._ensure_bot()
            if message is None:
                current_time = get_sast_time()
                message = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - This is a test message"
            await self._bot.send_message(chat_id=self.chat_id, text=message)
        except TelegramError as e:
            logging.error(f"Telegram error sending message: {e}")
        except Exception as e:
            logging.error(f"Error sending message: {e}")

    async def _async_send_photo(self, photo_buffer, caption=None):
        """Send photo asynchronously"""
        try:
            self._ensure_bot()
            photo_buffer.seek(0)
            await self._bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_buffer,
                caption=caption if caption else f"Chart at {get_sast_time().strftime('%Y-%m-%d %H:%M:%S')}"
            )
        except TelegramError as e:
            logging.error(f"Telegram error sending photo: {e}")
        except Exception as e:
            logging.error(f"Error sending photo: {e}")

    def send_message(self, message=None):
        """Synchronous wrapper for sending text messages"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_send_message(message))
        except Exception as e:
            logging.error(f"Error in send_message: {e}")
        finally:
            loop.stop()
            loop.close()

    def send_photo(self, fig=None, caption=None):
        """Synchronous wrapper for sending a matplotlib figure"""
        try:
            import matplotlib.pyplot as plt
            buffer = BytesIO()
            if fig is None:
                plt.savefig(buffer, format='png', bbox_inches='tight')
            else:
                fig.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._async_send_photo(buffer, caption))
            
            buffer.close()
            if fig is not None:
                plt.close(fig)
        except Exception as e:
            logging.error(f"Error in send_photo: {e}")
        finally:
            loop.stop()
            loop.close()

class CustomLoggingHandler(logging.Handler):
    """Custom logging handler that sends important messages to Telegram"""
    
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self._emitting = False
        # Define important message patterns
        self.important_patterns = [
            'Entry signal',
            'Closed trade',
            'Error',
            'Opening position',
            'Failed to',
            'Closed position',
            'Balance:',
            'Position',
            'Initial balance'
        ]

    def emit(self, record):
        if self._emitting:
            return

        try:
            self._emitting = True
            log_message = self.format(record)
            
            # Only emit if it's an important message or error/warning
            is_important = (
                record.levelno >= logging.WARNING or  # Always log warnings and errors
                any(pattern in log_message for pattern in self.important_patterns)
            )
            
            if is_important:
                self.bot.send_message(log_message)
                time.sleep(1.5)  # Rate limiting for important messages
        except Exception as e:
            print(f"Error in custom logging handler: {e}", file=sys.stderr)
        finally:
            self._emitting = False

def configure_logging(token: str, chat_id: str) -> tuple:
    """Configure logging with Telegram integration"""
    logger = logging.getLogger(__name__)
    bot = TelegramBot(token, chat_id)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Add custom handler for Telegram
        custom_handler = CustomLoggingHandler(bot)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        custom_handler.setFormatter(formatter)
        custom_handler.setLevel(logging.INFO)
        
        # Add file handler for log file output
        file_handler = logging.FileHandler('trading_bot.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Add stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        
        logger.addHandler(custom_handler)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
    
    return logger, bot