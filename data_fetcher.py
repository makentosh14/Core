#!/usr/bin/env python3
"""
data_fetcher.py - Bybit API Data Fetcher
==========================================
Fetches klines, symbols, and market data from Bybit v5 API.
Handles rate limiting, pagination, and error recovery.

WARMUP ALIGNMENT FIX:
  fetch_symbol_all_timeframes() now fetches exactly
  WARMUP_CANDLES + tf_config["candles"] bars per timeframe.
  This guarantees EMA200 and all other slow indicators have
  enough history to produce valid values.
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
    url     = f"{BYBIT_API}/v5/market/instruments-info"
    cursor  = None

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
    """Fetch top N symbols sorted by 24h turnover."""
    url    = f"{BYBIT_API}/v5/market/tickers"
    params = {"category": "linear"}
    try:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if data.get("retCode") != 0:
                return []
            tickers = []
            for t in data.get("result", {}).get("list", []):
                sym = t.get("symbol", "")
                if not sym.endswith("USDT"):
                    continue
                try:
                    vol = float(t.get("turnover24h", 0))
                    price = float(t.get("lastPrice", 0))
                except (ValueError, TypeError):
                    continue
                if vol >= min_volume and price > 0:
                    tickers.append({
                        "symbol":      sym,
                        "price":       price,
                        "turnover24h": vol,
                        "volume24h":   float(t.get("volume24h", 0)),
                    })
            tickers.sort(key=lambda x: x["turnover24h"], reverse=True)
            return tickers[:top_n]
    except Exception as e:
        print(f"[FETCH] Error fetching tickers: {e}")
        return []


# ============================================================
# CANDLE FETCHER
# ============================================================

async def fetch_klines(
    session:    aiohttp.ClientSession,
    symbol:     str,
    interval:   str,
    limit:      int,
    retries:    int = MAX_RETRIES,
) -> List[Candle]:
    """
    Fetch klines from Bybit v5.
    interval: "5", "15", "60", "240" etc.
    limit: total candles to fetch (includes warmup bars).
    """
    url    = f"{BYBIT_API}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol":   symbol,
        "interval": interval,
        "limit":    str(min(limit, 1000)),  # Bybit max = 1000
    }
    for attempt in range(retries):
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    await asyncio.sleep(RATE_DELAY * (attempt + 1))
                    continue
                raw = data.get("result", {}).get("list", [])
                if not raw:
                    return []
                candles = []
                for row in raw:
                    # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
                    candles.append(Candle(
                        timestamp = int(row[0]),
                        open      = float(row[1]),
                        high      = float(row[2]),
                        low       = float(row[3]),
                        close     = float(row[4]),
                        volume    = float(row[5]),
                    ))
                # Sort chronologically (oldest first)
                candles.sort(key=lambda c: c.timestamp)
                return candles
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(RATE_DELAY * (attempt + 1))
            else:
                print(f"[FETCH] {symbol} {interval}: {e}")
    return []


async def fetch_symbol_all_timeframes(
    session: aiohttp.ClientSession,
    symbol:  str,
) -> Dict[str, List[Candle]]:
    """
    Fetch all 4 timeframes for a symbol.

    WARMUP FIX: Each timeframe fetches WARMUP_CANDLES + tf_config["candles"]
    so that slow indicators (EMA200, Ichimoku) always have enough history.
    This aligns with the strict validation in process_symbol().
    """
    result = {}
    for tf_key, tf_config in TIMEFRAMES.items():
        # Total bars needed = warmup + analysis window
        total_bars = WARMUP_CANDLES + tf_config["candles"]
        candles    = await fetch_klines(session, symbol, tf_key, total_bars)
        if candles:
            result[tf_key] = candles
        await asyncio.sleep(RATE_DELAY)
    return result


# ============================================================
# BATCH FETCHER
# ============================================================

async def fetch_batch_symbols(
    session:  aiohttp.ClientSession,
    symbols:  List[str],
    batch_sz: int = BATCH_SIZE,
) -> Dict[str, Dict[str, List[Candle]]]:
    """
    Fetch all timeframes for a list of symbols in parallel batches.
    Returns {symbol: {tf_key: [Candle, ...]}}
    """
    result = {}
    for i in range(0, len(symbols), batch_sz):
        batch = symbols[i : i + batch_sz]
        tasks = {sym: fetch_symbol_all_timeframes(session, sym) for sym in batch}
        coros = list(tasks.values())
        syms  = list(tasks.keys())
        try:
            results = await asyncio.gather(*coros, return_exceptions=True)
            for sym, res in zip(syms, results):
                if isinstance(res, Exception):
                    print(f"[BATCH] {sym}: {res}")
                elif res and len(res) >= 3:  # need at least 3 timeframes
                    result[sym] = res
        except Exception as e:
            print(f"[BATCH] Batch error: {e}")
        # Small pause between batches
        await asyncio.sleep(RATE_DELAY * 2)
    return result
