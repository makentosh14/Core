#!/usr/bin/env python3
"""
data_fetcher.py - Bybit API Data Fetcher
==========================================
Fetches klines, symbols, and market data from Bybit v5 API.
Handles rate limiting, pagination, and error recovery.
Designed for Hetzner cloud deployment.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone

from config import (
    BYBIT_API, RATE_DELAY, API_TIMEOUT, MAX_RETRIES,
    BATCH_SIZE, TOP_SYMBOLS, MIN_24H_VOLUME_USDT,
    TIMEFRAMES, WARMUP_CANDLES,
)
from indicators import Candle


# ============================================================
# SYMBOL DISCOVERY
# ============================================================

async def fetch_all_usdt_symbols(session: aiohttp.ClientSession) -> List[str]:
    """Fetch all active USDT perpetual symbols from Bybit."""
    symbols = []
    url = f"{BYBIT_API}/v5/market/instruments-info"
    cursor = None

    while True:
        params = {"category": "linear", "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    break
                for inst in data.get("result", {}).get("list", []):
                    if (inst.get("symbol", "").endswith("USDT") and
                            inst.get("status") == "Trading"):
                        symbols.append(inst["symbol"])
                nc = data.get("result", {}).get("nextPageCursor")
                if not nc or nc == cursor:
                    break
                cursor = nc
                await asyncio.sleep(RATE_DELAY)
        except Exception as e:
            print(f"[FETCH] Error fetching symbols: {e}")
            break

    return sorted(symbols)


async def fetch_top_symbols_by_volume(
    session: aiohttp.ClientSession,
    top_n: int = TOP_SYMBOLS,
    min_volume: float = MIN_24H_VOLUME_USDT,
) -> List[Dict]:
    """
    Fetch top N symbols sorted by 24h turnover.
    Returns list of dicts with symbol info.
    """
    url = f"{BYBIT_API}/v5/market/tickers"
    params = {"category": "linear"}

    try:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
    except Exception as e:
        print(f"[FETCH] Error fetching tickers: {e}")
        return []

    tickers = data.get("result", {}).get("list", [])
    usdt_tickers = []

    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        turnover = float(t.get("turnover24h") or 0)
        if turnover < min_volume:
            continue
        usdt_tickers.append({
            "symbol": sym,
            "turnover24h": turnover,
            "lastPrice": float(t.get("lastPrice") or 0),
            "price24hPcnt": float(t.get("price24hPcnt") or 0),
            "volume24h": float(t.get("volume24h") or 0),
        })

    usdt_tickers.sort(key=lambda x: x["turnover24h"], reverse=True)
    return usdt_tickers[:top_n]


# ============================================================
# KLINE / CANDLE FETCHING
# ============================================================

async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int = 200,
    category: str = "linear",
) -> List[Candle]:
    """
    Fetch historical klines via Bybit REST v5.
    Returns list of Candle objects sorted oldest→newest.
    """
    url = f"{BYBIT_API}/v5/market/kline"
    safe_limit = max(1, min(int(limit), 200))
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(safe_limit),
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    print(f"[FETCH] Bybit error {symbol} {interval}m: {data.get('retMsg')}")
                    return []
                raw = data.get("result", {}).get("list", []) or []
                candles = []
                for c in reversed(raw):  # Bybit returns newest-first
                    candles.append(Candle(
                        timestamp=int(c[0]),
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[5]),
                    ))
                return candles
        except asyncio.TimeoutError:
            print(f"[FETCH] Timeout {symbol} {interval}m (attempt {attempt + 1})")
            await asyncio.sleep(RATE_DELAY * 2)
        except Exception as e:
            print(f"[FETCH] Error {symbol} {interval}m: {e}")
            await asyncio.sleep(RATE_DELAY)

    return []


async def fetch_extended_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    total_candles: int = 500,
    category: str = "linear",
) -> List[Candle]:
    """
    Fetch more candles than the 200 limit by paginating backwards.
    Used for warm-up data (need 200+ candles for EMA200).
    """
    all_candles = []
    end_ts = int(time.time() * 1000)
    max_pages = (total_candles // 200) + 2

    for page in range(max_pages):
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "end": str(end_ts),
            "limit": "200",
        }
        try:
            async with session.get(
                f"{BYBIT_API}/v5/market/kline", params=params
            ) as resp:
                data = await resp.json()
                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break
                for c in klines:
                    all_candles.append(Candle(
                        timestamp=int(c[0]),
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[5]),
                    ))
                oldest = int(klines[-1][0])
                if oldest >= end_ts:
                    break
                end_ts = oldest - 1
                await asyncio.sleep(RATE_DELAY)
        except Exception as e:
            print(f"[FETCH] Pagination error {symbol} {interval}m: {e}")
            break

        if len(all_candles) >= total_candles:
            break

    # Deduplicate and sort oldest→newest
    seen = set()
    unique = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)
    unique.sort(key=lambda x: x.timestamp)
    return unique


# ============================================================
# BATCH SYMBOL DATA FETCH
# ============================================================

async def fetch_symbol_all_timeframes(
    session: aiohttp.ClientSession,
    symbol: str,
) -> Dict[str, List[Candle]]:
    """
    Fetch candles for all configured timeframes for a single symbol.
    Returns dict: {"5": [candles], "15": [...], "60": [...], "240": [...]}
    """
    result = {}

    for tf_key, tf_config in TIMEFRAMES.items():
        interval = tf_config["interval"]
        needed = tf_config["candles"] + WARMUP_CANDLES

        if needed <= 200:
            candles = await fetch_klines(session, symbol, interval, limit=needed)
        else:
            candles = await fetch_extended_klines(session, symbol, interval, total_candles=needed)

        result[tf_key] = candles
        await asyncio.sleep(RATE_DELAY)

    return result


async def fetch_batch_symbols(
    session: aiohttp.ClientSession,
    symbols: List[str],
    batch_size: int = BATCH_SIZE,
    progress_callback=None,
) -> Dict[str, Dict[str, List[Candle]]]:
    """
    Fetch all timeframe data for multiple symbols in batches.
    Returns: {symbol: {tf: [candles]}}
    """
    all_data = {}
    total = len(symbols)

    for i in range(0, total, batch_size):
        batch = symbols[i:i + batch_size]
        tasks = [fetch_symbol_all_timeframes(session, sym) for sym in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sym, res in zip(batch, results):
            if isinstance(res, Exception):
                print(f"[FETCH] Failed {sym}: {res}")
                continue
            all_data[sym] = res

        done = min(i + batch_size, total)
        pct = (done / total) * 100
        print(f"  [{done}/{total}] ({pct:.0f}%) Fetched data...")

        if progress_callback:
            progress_callback(done, total)

        await asyncio.sleep(RATE_DELAY * 2)

    return all_data
