# logger.py

import datetime
import asyncio
import os
import sys
import aiohttp
from config import TELEGRAM_ASSISTANT_CHAT_ID, TELEGRAM_BOT_TOKEN

# ✅ Unified log path (used across bot)
LOG_PATH = "/mnt/data/bot_logs"
LOG_FILE = os.path.join(LOG_PATH, "trading_bot_activity.log")

# Ensure log directory exists
os.makedirs(LOG_PATH, exist_ok=True)

def log(msg, level="INFO"):
    """
    Main logger that prints to console, writes to file, and optionally sends to Telegram.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Console log
    try:
        print(f"[{timestamp}] [{level.upper()}] {msg}")
    except UnicodeEncodeError:
        print(f"[{timestamp}] [{level.upper()}] {msg.encode('utf-8', 'ignore').decode('utf-8')}", file=sys.stderr)

    # File log
    write_log(msg, level)

    # Optional Telegram alert for high-severity
    if TELEGRAM_ASSISTANT_CHAT_ID and level.upper() in ["ERROR", "ALERT"]:
        asyncio.create_task(send_assistant_log(msg))

def write_log(message, level="INFO"):
    """
    Writes logs to persistent file.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [{level.upper()}] {message}\n")

async def send_assistant_log(message):
    """
    Sends error logs to assistant Telegram channel.
    """
    if not message.strip():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_ASSISTANT_CHAT_ID,
        "text": f"📋 <b>Log</b>:\n<code>{message}</code>",
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload) as resp:
                if resp.status != 200:
                    print(f"[Logger] Telegram response status: {resp.status}")
    except Exception as e:
        print(f"[Logger] Failed to send assistant log: {e}")
