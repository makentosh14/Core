# error_handler.py

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import aiohttp

BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

async def send_telegram_message(message):
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(BOT_URL, data=payload)
    except Exception as e:
        print(f"[ErrorHandler] Failed to send message: {e}")

async def send_error_to_telegram(error_text):
    error_msg = f"❗️<b>Bot Error</b>\n<pre>{error_text}</pre>"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": error_msg[:4096],
        "parse_mode": "HTML"
    }
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(BOT_URL, data=payload)
    except Exception as e:
        print(f"[ErrorHandler] Failed to send error to Telegram: {e}")
