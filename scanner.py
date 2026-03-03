# scanner.py

import aiohttp
import asyncio
from config import BYBIT_API_URL
from logger import log
from error_handler import send_error_to_telegram  # ✅ instead of telegram_bot

symbol_category_map = {}

async def fetch_symbols():
    symbols = set()
    futures_url = f"{BYBIT_API_URL}/v5/market/instruments-info?category=linear"

    try:
        async with aiohttp.ClientSession() as session:
            url = futures_url
            category = "linear"
            cursor = None
            while True:
                params = {"cursor": cursor} if cursor else {}
                log(f"🌐 Fetching {category} symbols | URL: {url} | Cursor: {cursor}")
                async with session.get(url, params=params) as resp:
                    raw = await resp.text()
                    try:
                        data = await resp.json()
                    except Exception:
                        msg = f"❌ Failed to parse {category} JSON.\nRaw:\n{raw}"
                        log(msg)
                        await send_error_to_telegram(msg)
                        break

                    if data.get("retCode") != 0:
                        msg = f"❌ Error fetching {category} symbols:\n{data}"
                        log(msg)
                        await send_error_to_telegram(msg)
                        break

                    instruments = data.get("result", {}).get("list", [])
                    for instrument in instruments:
                        symbol = instrument.get("symbol", "")
                        status = instrument.get("status", "")
                        if symbol.endswith("USDT") and status == "Trading":
                            symbols.add(symbol)
                            symbol_category_map[symbol] = category

                    next_cursor = data.get("result", {}).get("nextPageCursor")
                    if not next_cursor or next_cursor == cursor:
                        break
                    cursor = next_cursor
                    await asyncio.sleep(0.1)

        log(f"✅ Total symbols fetched: {len(symbols)}")
        return list(symbols)

    except Exception as e:
        error_msg = f"❌ Exception while fetching symbols: {e}"
        log(error_msg)
        await send_error_to_telegram(error_msg)
        return []

async def ensure_symbol_mapped(symbol: str, category: str = 'linear'):
    """Add symbol to category map if missing — handles new listings."""
    if symbol not in symbol_category_map:
        symbol_category_map[symbol] = category
        log(f"📍 Auto-mapped new symbol: {symbol} → {category}")

