import asyncio
import json
import websockets
from collections import defaultdict, deque
from scanner import symbol_category_map
from logger import log
from error_handler import send_error_to_telegram
from bybit_api import signed_request

live_candles = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
SUPPORTED_INTERVALS = ['1', '3', '5', '15', '30', '60', '240']

RECONNECT_DELAY = 5

async def handle_stream(url, symbols, category, interval):
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                args = [f"kline.{interval}.{symbol}" for symbol in symbols if symbol_category_map.get(symbol) == category]
                if not args:
                    return

                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                log(f"📡 Subscribed to {len(args)} {category.upper()} @ {interval}m")

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(message)

                        topic = data.get("topic", "")
                        symbol = topic.split(".")[-1] if topic else None
                        interval_from_topic = topic.split(".")[1] if topic else None

                        if not symbol or "data" not in data or not interval_from_topic:
                            continue

                        candles = data["data"]

                        if isinstance(candles, list):
                            for k in candles:
                                candle = {
                                    "timestamp": int(k["start"]),
                                    "open": k["open"],
                                    "high": k["high"],
                                    "low": k["low"],
                                    "close": k["close"],
                                    "volume": k["volume"]
                                }
                                live_candles[symbol][interval_from_topic].append(candle)

                        elif isinstance(candles, dict):
                            k = candles
                            candle = {
                                "timestamp": int(k["start"]),
                                "open": k["open"],
                                "high": k["high"],
                                "low": k["low"],
                                "close": k["close"],
                                "volume": k["volume"]
                            }
                            live_candles[symbol][interval_from_topic].append(candle)

                        log(f"📈 {symbol} [{category}] @{interval_from_topic} updated | total: {len(live_candles[symbol][interval_from_topic])}")

                    except asyncio.TimeoutError:
                        warning = f"⚠️ No data for {category} {interval}m in 30s — reconnecting..."
                        log(warning, level="WARNING")
                        await send_error_to_telegram(warning)
                        break

                    except websockets.exceptions.ConnectionClosedError as e:
                        warning = f"❌ WebSocket closed unexpectedly for {category} {interval}m: {e}"
                        log(warning, level="ERROR")
                        await send_error_to_telegram(warning)
                        break

                    except Exception as e:
                        error = f"❌ WebSocket stream error in {category} {interval}m: {e}"
                        log(error, level="ERROR")
                        await send_error_to_telegram(error)
                        break

        except Exception as e:
            error = f"❌ Connection failed for {category} {interval}m: {e}"
            log(error, level="ERROR")
            await send_error_to_telegram(error)

        reconnect_msg = f"🔁 Reconnecting {category} {interval}m in {RECONNECT_DELAY}s..."
        log(reconnect_msg)
        await send_error_to_telegram(reconnect_msg)
        await asyncio.sleep(RECONNECT_DELAY)

async def fetch_candles_rest(symbol: str, interval: str = '1', limit: int = 200, category: str = 'linear'):
    """
    Fetch historical candles via Bybit REST v5.
    Returns a list[dict] shaped like websocket 'live_candles' entries:
      { "timestamp": int(ms), "open": str, "high": str, "low": str, "close": str, "volume": str }
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Unsupported interval: {interval}")

    # Clamp limit to Bybit's max 200 and min 1
    safe_limit = max(1, min(int(limit), 200))

    params = {
        "category": category,          # "linear" for perps (your default)
        "symbol": symbol,
        "interval": interval,          # minutes as string (e.g., '1', '5', '60')
        "limit": str(safe_limit)
    }

    resp = await signed_request("GET", "/v5/market/kline", params)

    if resp.get("retCode") != 0:
        # Surface a helpful error so the caller can log it
        raise RuntimeError(f"Bybit kline error for {symbol} ({interval}m): {resp.get('retMsg')}")

    raw = resp.get("result", {}).get("list", []) or []

    # Bybit returns newest-first; normalize to oldest->newest to match how you append WebSocket candles
    candles = []
    for c in reversed(raw):
        # Bybit format: [start, open, high, low, close, volume, turnover]
        candles.append({
            "timestamp": int(c[0]),
            "open": str(c[1]),
            "high": str(c[2]),
            "low":  str(c[3]),
            "close": str(c[4]),
            "volume": str(c[5]),
        })
    return candles

async def stream_candles(symbols):
    futures_url = "wss://stream.bybit.com/v5/public/linear"
    tasks = []

    for interval in SUPPORTED_INTERVALS:
        tasks.append(asyncio.create_task(handle_stream(futures_url, symbols, "linear", interval)))

    await asyncio.gather(*tasks)
