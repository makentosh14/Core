# symbol_info.py
import aiohttp
import asyncio

symbol_precisions = {}

def count_decimal_places(number: float) -> int:
    s = f"{number:.10f}".rstrip("0")
    if '.' in s:
        return len(s.split(".")[1])
    return 0

async def fetch_symbol_info():
    url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            for item in data.get("result", {}).get("list", []):
                symbol = item["symbol"]
                tick_size = float(item["lotSizeFilter"]["qtyStep"])
                min_qty = float(item["lotSizeFilter"]["minOrderQty"])
                precision = count_decimal_places(tick_size)
                symbol_precisions[symbol] = {
                    "min_qty": min_qty,
                    "step": tick_size,
                    "precision": precision
                }

def get_precision(symbol):
    info = symbol_precisions.get(symbol, {})
    return info.get("precision", 0)

def round_qty(symbol, qty):
    precision = get_precision(symbol)
    return round(qty, precision)
