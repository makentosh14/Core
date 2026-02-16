#!/usr/bin/env python3
"""
indicator_backscan.py - Historical Crypto Indicator Backscan Tool
================================================================
Scans ALL Bybit USDT perpetual coins for 6 months of historical data.
Detects pumps/dumps on 15m, 1h, 4h timeframes.
Analyzes which indicators were signaling BEFORE each move.
Outputs optimal settings for TradingView Pine Script.

Compatible with your existing Bybit perpetual scanner setup.

Usage:
    python indicator_backscan.py
    python indicator_backscan.py --symbols BTCUSDT,ETHUSDT --days 90
    python indicator_backscan.py --top 50 --min-move 5.0
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
import os
import csv
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ============================================================
# CONFIGURATION
# ============================================================

BYBIT_API_URL = "https://api.bybit.com"
RATE_LIMIT_DELAY = 0.12  # seconds between API calls (Bybit allows ~10/sec)

# Timeframes to scan
TIMEFRAMES = {
    "15":  {"interval": "15",  "label": "15m", "candles_per_day": 96,  "lookback_candles": 8},
    "60":  {"interval": "60",  "label": "1h",  "candles_per_day": 24,  "lookback_candles": 6},
    "240": {"interval": "240", "label": "4h",  "candles_per_day": 6,   "lookback_candles": 4},
}

# Pump/Dump detection thresholds (percentage move within N candles)
MOVE_THRESHOLDS = {
    "15":  {"pump": 3.0, "dump": -3.0, "window": 4},   # 3% in 4 candles (1 hour)
    "60":  {"pump": 5.0, "dump": -5.0, "window": 4},   # 5% in 4 candles (4 hours)
    "240": {"pump": 8.0, "dump": -8.0, "window": 3},   # 8% in 3 candles (12 hours)
}

# Default indicator parameters (matching your bot's settings)
DEFAULT_RSI_PERIOD = 14
DEFAULT_EMA_FAST = 9
DEFAULT_EMA_MID = 21
DEFAULT_EMA_SLOW = 55
DEFAULT_MACD_FAST = 12
DEFAULT_MACD_SLOW = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0
DEFAULT_SUPERTREND_PERIOD = 10
DEFAULT_SUPERTREND_MULT = 3.0
DEFAULT_STOCH_RSI_PERIOD = 14
DEFAULT_STOCH_RSI_K = 3
DEFAULT_STOCH_RSI_D = 3
DEFAULT_ATR_PERIOD = 14
DEFAULT_VOLUME_MA = 20


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def datetime_str(self) -> str:
        return datetime.utcfromtimestamp(self.timestamp / 1000).strftime("%Y-%m-%d %H:%M")


@dataclass
class MoveEvent:
    """Represents a detected pump or dump event"""
    symbol: str
    timeframe: str
    move_type: str          # "pump" or "dump"
    move_pct: float         # percentage move
    start_time: str
    end_time: str
    start_price: float
    end_price: float
    volume_ratio: float     # volume vs average
    indicators_before: Dict  # indicator states BEFORE the move


@dataclass
class IndicatorStats:
    """Tracks how often an indicator fires before pumps/dumps"""
    total_pumps: int = 0
    total_dumps: int = 0
    # Each key = indicator condition, value = count of times it appeared before the move
    pump_signals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    dump_signals: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


# ============================================================
# INDICATOR CALCULATIONS (matching your bot's logic)
# ============================================================

def calc_ema(prices: List[float], period: int) -> List[float]:
    """Calculate EMA - matches your ema.py logic"""
    if len(prices) < period:
        return []
    ema_values = []
    # Initialize with SMA (matching your trend_filters.py fix)
    sma = np.mean(prices[:period])
    ema_values.append(sma)
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        ema_val = (prices[i] - ema_values[-1]) * multiplier + ema_values[-1]
        ema_values.append(ema_val)
    return ema_values


def calc_rsi(candles: List[Candle], period: int = DEFAULT_RSI_PERIOD) -> List[float]:
    """Calculate RSI using Wilder's method - matches your rsi.py"""
    if len(candles) < period + 1:
        return []
    closes = [c.close for c in candles]
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]

    gains = [max(0, d) for d in deltas]
    losses = [max(0, -d) for d in deltas]

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi_values


def calc_macd(candles: List[Candle], fast=DEFAULT_MACD_FAST,
              slow=DEFAULT_MACD_SLOW, signal=DEFAULT_MACD_SIGNAL
              ) -> Tuple[List[float], List[float], List[float]]:
    """Calculate MACD line, signal line, histogram - matches your macd.py"""
    closes = [c.close for c in candles]
    if len(closes) < slow + signal:
        return [], [], []
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    # Align lengths
    offset = len(ema_fast) - len(ema_slow)
    ema_fast_aligned = ema_fast[offset:]
    macd_line = [f - s for f, s in zip(ema_fast_aligned, ema_slow)]
    if len(macd_line) < signal:
        return [], [], []
    signal_line = calc_ema(macd_line, signal)
    offset2 = len(macd_line) - len(signal_line)
    macd_trimmed = macd_line[offset2:]
    histogram = [m - s for m, s in zip(macd_trimmed, signal_line)]
    return macd_trimmed, signal_line, histogram


def calc_bollinger(candles: List[Candle], period=DEFAULT_BB_PERIOD,
                   std_mult=DEFAULT_BB_STD) -> Tuple[List[float], List[float], List[float]]:
    """Calculate Bollinger Bands - matches your bollinger.py"""
    closes = [c.close for c in candles]
    if len(closes) < period:
        return [], [], []
    upper, middle, lower = [], [], []
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        sma = np.mean(window)
        std = np.std(window)
        middle.append(sma)
        upper.append(sma + std_mult * std)
        lower.append(sma - std_mult * std)
    return upper, middle, lower


def calc_supertrend(candles: List[Candle], period=DEFAULT_SUPERTREND_PERIOD,
                    multiplier=DEFAULT_SUPERTREND_MULT) -> List[str]:
    """Calculate Supertrend signal - matches your supertrend.py"""
    if len(candles) < period + 1:
        return []
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]

    # ATR
    tr_values = []
    for i in range(1, len(candles)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_values.append(tr)

    if len(tr_values) < period:
        return []

    atr_values = []
    atr = np.mean(tr_values[:period])
    atr_values.append(atr)
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period
        atr_values.append(atr)

    signals = []
    prev_upper = 0
    prev_lower = 0
    prev_trend = 1  # 1 = bullish, -1 = bearish

    start_idx = period
    for i in range(len(atr_values)):
        ci = start_idx + i
        if ci >= len(candles):
            break
        hl2 = (highs[ci] + lows[ci]) / 2.0
        upper_band = hl2 + multiplier * atr_values[i]
        lower_band = hl2 - multiplier * atr_values[i]

        if prev_upper > 0:
            upper_band = min(upper_band, prev_upper) if closes[ci-1] <= prev_upper else upper_band
            lower_band = max(lower_band, prev_lower) if closes[ci-1] >= prev_lower else lower_band

        if closes[ci] > upper_band:
            trend = 1
        elif closes[ci] < lower_band:
            trend = -1
        else:
            trend = prev_trend

        signals.append("bullish" if trend == 1 else "bearish")
        prev_upper = upper_band
        prev_lower = lower_band
        prev_trend = trend

    return signals


def calc_stoch_rsi(candles: List[Candle], rsi_period=DEFAULT_STOCH_RSI_PERIOD,
                   stoch_period=DEFAULT_STOCH_RSI_PERIOD,
                   k_period=DEFAULT_STOCH_RSI_K,
                   d_period=DEFAULT_STOCH_RSI_D) -> Tuple[List[float], List[float]]:
    """Calculate Stochastic RSI - matches your rsi.py calc_stoch_rsi"""
    rsi_values = calc_rsi(candles, rsi_period)
    if len(rsi_values) < stoch_period:
        return [], []

    stoch_values = []
    for i in range(stoch_period - 1, len(rsi_values)):
        window = rsi_values[i - stoch_period + 1:i + 1]
        min_rsi = min(window)
        max_rsi = max(window)
        if max_rsi - min_rsi > 0:
            stoch = ((rsi_values[i] - min_rsi) / (max_rsi - min_rsi)) * 100
        else:
            stoch = 50.0
        stoch_values.append(stoch)

    # K line (SMA of stoch)
    k_values = []
    for i in range(k_period - 1, len(stoch_values)):
        k_values.append(np.mean(stoch_values[i - k_period + 1:i + 1]))

    # D line (SMA of K)
    d_values = []
    for i in range(d_period - 1, len(k_values)):
        d_values.append(np.mean(k_values[i - d_period + 1:i + 1]))

    return k_values, d_values


def calc_atr(candles: List[Candle], period=DEFAULT_ATR_PERIOD) -> List[float]:
    """Calculate ATR"""
    if len(candles) < period + 1:
        return []
    tr_values = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i-1].close),
            abs(candles[i].low - candles[i-1].close)
        )
        tr_values.append(tr)

    atr_vals = []
    atr = np.mean(tr_values[:period])
    atr_vals.append(atr)
    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period
        atr_vals.append(atr)
    return atr_vals


def calc_volume_ma(candles: List[Candle], period=DEFAULT_VOLUME_MA) -> List[float]:
    """Calculate volume moving average"""
    volumes = [c.volume for c in candles]
    if len(volumes) < period:
        return []
    result = []
    for i in range(period - 1, len(volumes)):
        result.append(np.mean(volumes[i - period + 1:i + 1]))
    return result


def detect_ema_crossover(candles: List[Candle]) -> str:
    """Detect EMA crossover state - matches your ema.py detect_ema_crossover"""
    closes = [c.close for c in candles]
    ema_fast = calc_ema(closes, DEFAULT_EMA_FAST)
    ema_slow = calc_ema(closes, DEFAULT_EMA_SLOW)
    if len(ema_fast) < 2 or len(ema_slow) < 2:
        return "neutral"
    # Align
    offset = len(ema_fast) - len(ema_slow)
    if offset < 0:
        return "neutral"
    f1, f2 = ema_fast[-1 + offset] if offset else ema_fast[-1], ema_fast[-2 + offset] if offset else ema_fast[-2]
    # Simpler: just use last values
    fast_last = ema_fast[-1]
    slow_last = ema_slow[-1]
    fast_prev = ema_fast[-2]
    slow_prev = ema_slow[-2]
    if fast_prev <= slow_prev and fast_last > slow_last:
        return "bullish_cross"
    elif fast_prev >= slow_prev and fast_last < slow_last:
        return "bearish_cross"
    elif fast_last > slow_last:
        return "bullish"
    elif fast_last < slow_last:
        return "bearish"
    return "neutral"


def detect_ema_ribbon_state(candles: List[Candle]) -> str:
    """Analyze EMA ribbon alignment"""
    closes = [c.close for c in candles]
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema55 = calc_ema(closes, 55)
    if not ema9 or not ema21 or not ema55:
        return "neutral"
    if ema9[-1] > ema21[-1] > ema55[-1]:
        return "bullish_aligned"
    elif ema9[-1] < ema21[-1] < ema55[-1]:
        return "bearish_aligned"
    return "mixed"


# ============================================================
# INDICATOR STATE ANALYZER (reads state BEFORE a move)
# ============================================================

def analyze_indicators_before_move(candles: List[Candle], lookback: int) -> Dict:
    """
    Analyze all indicator states using candles BEFORE the move started.
    candles = candles up to (but NOT including) the move start.
    lookback = how many candles before the move to check for signals.
    """
    if len(candles) < 60:
        return {}

    # Use only candles before the move
    pre_move = candles[:-1] if len(candles) > lookback else candles

    result = {}

    # --- RSI ---
    rsi = calc_rsi(pre_move)
    if rsi:
        rsi_val = rsi[-1]
        result["rsi_value"] = round(rsi_val, 1)
        result["rsi_oversold"] = rsi_val < 30
        result["rsi_overbought"] = rsi_val > 70
        result["rsi_extreme_oversold"] = rsi_val < 20
        result["rsi_extreme_overbought"] = rsi_val > 80
        result["rsi_rising"] = len(rsi) >= 3 and rsi[-1] > rsi[-3]
        result["rsi_falling"] = len(rsi) >= 3 and rsi[-1] < rsi[-3]
        # RSI divergence check (price lower low but RSI higher low = bullish div)
        if len(rsi) >= 10 and len(pre_move) >= 10:
            price_slope = pre_move[-1].close - pre_move[-5].close
            rsi_slope = rsi[-1] - rsi[-5]
            result["rsi_bullish_div"] = price_slope < 0 and rsi_slope > 0
            result["rsi_bearish_div"] = price_slope > 0 and rsi_slope < 0

    # --- Stochastic RSI ---
    k_vals, d_vals = calc_stoch_rsi(pre_move)
    if k_vals and d_vals:
        result["stoch_rsi_k"] = round(k_vals[-1], 1)
        result["stoch_rsi_d"] = round(d_vals[-1], 1)
        result["stoch_rsi_oversold"] = k_vals[-1] < 20 and d_vals[-1] < 20
        result["stoch_rsi_overbought"] = k_vals[-1] > 80 and d_vals[-1] > 80
        if len(k_vals) >= 2 and len(d_vals) >= 2:
            result["stoch_rsi_bullish_cross"] = k_vals[-2] < d_vals[-2] and k_vals[-1] > d_vals[-1]
            result["stoch_rsi_bearish_cross"] = k_vals[-2] > d_vals[-2] and k_vals[-1] < d_vals[-1]

    # --- MACD ---
    macd_line, signal_line, histogram = calc_macd(pre_move)
    if macd_line and signal_line and histogram:
        result["macd_above_signal"] = macd_line[-1] > signal_line[-1]
        result["macd_below_signal"] = macd_line[-1] < signal_line[-1]
        result["macd_histogram_positive"] = histogram[-1] > 0
        result["macd_histogram_negative"] = histogram[-1] < 0
        if len(histogram) >= 2:
            result["macd_histogram_rising"] = histogram[-1] > histogram[-2]
            result["macd_histogram_falling"] = histogram[-1] < histogram[-2]
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            result["macd_bullish_cross"] = macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]
            result["macd_bearish_cross"] = macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]
        # MACD divergence
        if len(macd_line) >= 10 and len(pre_move) >= 10:
            price_slope = pre_move[-1].close - pre_move[-5].close
            macd_slope = macd_line[-1] - macd_line[-5]
            result["macd_bullish_div"] = price_slope < 0 and macd_slope > 0
            result["macd_bearish_div"] = price_slope > 0 and macd_slope < 0

    # --- EMA ---
    ema_state = detect_ema_crossover(pre_move)
    result["ema_crossover"] = ema_state
    result["ema_bullish_cross"] = ema_state == "bullish_cross"
    result["ema_bearish_cross"] = ema_state == "bearish_cross"
    result["ema_bullish"] = ema_state in ("bullish", "bullish_cross")
    result["ema_bearish"] = ema_state in ("bearish", "bearish_cross")

    # --- EMA Ribbon ---
    ribbon = detect_ema_ribbon_state(pre_move)
    result["ema_ribbon"] = ribbon
    result["ema_ribbon_bullish"] = ribbon == "bullish_aligned"
    result["ema_ribbon_bearish"] = ribbon == "bearish_aligned"

    # --- Bollinger Bands ---
    bb_upper, bb_middle, bb_lower = calc_bollinger(pre_move)
    if bb_upper and bb_lower and bb_middle:
        close = pre_move[-1].close
        result["bb_above_upper"] = close > bb_upper[-1]
        result["bb_below_lower"] = close < bb_lower[-1]
        result["bb_near_lower"] = close < bb_lower[-1] * 1.005
        result["bb_near_upper"] = close > bb_upper[-1] * 0.995
        # Bollinger bandwidth (squeeze detection)
        bandwidth = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1] * 100
        result["bb_bandwidth"] = round(bandwidth, 2)
        result["bb_squeeze"] = bandwidth < 3.0  # tight squeeze
        if len(bb_upper) >= 5:
            prev_bw = (bb_upper[-5] - bb_lower[-5]) / bb_middle[-5] * 100
            result["bb_expanding"] = bandwidth > prev_bw
            result["bb_contracting"] = bandwidth < prev_bw

    # --- Supertrend ---
    st_signals = calc_supertrend(pre_move)
    if st_signals:
        result["supertrend_bullish"] = st_signals[-1] == "bullish"
        result["supertrend_bearish"] = st_signals[-1] == "bearish"
        if len(st_signals) >= 2:
            result["supertrend_flip_bullish"] = st_signals[-2] == "bearish" and st_signals[-1] == "bullish"
            result["supertrend_flip_bearish"] = st_signals[-2] == "bullish" and st_signals[-1] == "bearish"

    # --- Volume ---
    vol_ma = calc_volume_ma(pre_move)
    if vol_ma and len(pre_move) > 0:
        current_vol = pre_move[-1].volume
        avg_vol = vol_ma[-1]
        if avg_vol > 0:
            vol_ratio = current_vol / avg_vol
            result["volume_ratio"] = round(vol_ratio, 2)
            result["volume_spike"] = vol_ratio > 2.0
            result["volume_high"] = vol_ratio > 1.5
            result["volume_low"] = vol_ratio < 0.5
        # Volume trend (last 5 candles)
        if len(pre_move) >= 5:
            recent_vols = [c.volume for c in pre_move[-5:]]
            result["volume_increasing"] = all(recent_vols[i] >= recent_vols[i-1] for i in range(1, len(recent_vols)))

    # --- ATR / Volatility ---
    atr_vals = calc_atr(pre_move)
    if atr_vals and pre_move[-1].close > 0:
        atr_pct = (atr_vals[-1] / pre_move[-1].close) * 100
        result["atr_pct"] = round(atr_pct, 3)
        result["high_volatility"] = atr_pct > 2.0
        result["low_volatility"] = atr_pct < 0.5

    # --- Price action patterns ---
    if len(pre_move) >= 3:
        c1, c2, c3 = pre_move[-3], pre_move[-2], pre_move[-1]
        # Bullish engulfing
        result["bullish_engulfing"] = (c2.close < c2.open and
                                        c3.close > c3.open and
                                        c3.close > c2.open and
                                        c3.open < c2.close)
        # Bearish engulfing
        result["bearish_engulfing"] = (c2.close > c2.open and
                                        c3.close < c3.open and
                                        c3.close < c2.open and
                                        c3.open > c2.close)
        # Hammer (long lower wick, small body at top)
        body = abs(c3.close - c3.open)
        lower_wick = min(c3.open, c3.close) - c3.low
        upper_wick = c3.high - max(c3.open, c3.close)
        if body > 0:
            result["hammer"] = lower_wick > body * 2 and upper_wick < body * 0.5
            result["inverted_hammer"] = upper_wick > body * 2 and lower_wick < body * 0.5
        # Three consecutive green/red
        result["three_green"] = all(pre_move[-i].close > pre_move[-i].open for i in range(1, 4))
        result["three_red"] = all(pre_move[-i].close < pre_move[-i].open for i in range(1, 4))

    return result


# ============================================================
# BYBIT API - FETCH HISTORICAL DATA
# ============================================================

async def fetch_all_usdt_symbols(session: aiohttp.ClientSession) -> List[str]:
    """Fetch all active USDT perpetual symbols from Bybit"""
    symbols = []
    url = f"{BYBIT_API_URL}/v5/market/instruments-info"
    cursor = None

    while True:
        params = {"category": "linear", "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    print(f"  ❌ Error fetching symbols: {data.get('retMsg')}")
                    break
                instruments = data.get("result", {}).get("list", [])
                for inst in instruments:
                    sym = inst.get("symbol", "")
                    status = inst.get("status", "")
                    if sym.endswith("USDT") and status == "Trading":
                        symbols.append(sym)
                next_cursor = data.get("result", {}).get("nextPageCursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor
                await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"  ❌ Exception fetching symbols: {e}")
            break

    return sorted(symbols)


async def fetch_klines(session: aiohttp.ClientSession, symbol: str,
                       interval: str, days: int) -> List[Candle]:
    """
    Fetch historical klines from Bybit v5 API.
    Bybit returns max 200 candles per request, newest first.
    We paginate backwards to get the full history.
    """
    all_candles = []
    end_time = int(time.time() * 1000)
    start_time = int((time.time() - days * 86400) * 1000)
    url = f"{BYBIT_API_URL}/v5/market/kline"

    current_end = end_time
    max_requests = 200  # safety limit

    while current_end > start_time and max_requests > 0:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "end": str(current_end),
            "limit": "200"
        }
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    break
                klines = data.get("result", {}).get("list", [])
                if not klines:
                    break
                for k in klines:
                    ts = int(k[0])
                    if ts < start_time:
                        continue
                    candle = Candle(
                        timestamp=ts,
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    )
                    all_candles.append(candle)
                # Bybit returns newest first, move cursor back
                oldest_ts = int(klines[-1][0])
                if oldest_ts >= current_end:
                    break
                current_end = oldest_ts - 1
                max_requests -= 1
                await asyncio.sleep(RATE_LIMIT_DELAY)
        except Exception as e:
            print(f"    ❌ Error fetching {symbol} {interval}: {e}")
            break

    # Sort oldest first (consistent with your bot: FIX #8)
    all_candles.sort(key=lambda c: c.timestamp)

    # Remove duplicates
    seen = set()
    unique = []
    for c in all_candles:
        if c.timestamp not in seen:
            seen.add(c.timestamp)
            unique.append(c)

    return unique


# ============================================================
# PUMP / DUMP DETECTION
# ============================================================

def detect_moves(candles: List[Candle], tf_key: str) -> List[Dict]:
    """
    Detect significant pumps and dumps in the candle data.
    Returns list of move events with index positions.
    """
    if len(candles) < 20:
        return []

    threshold = MOVE_THRESHOLDS[tf_key]
    pump_pct = threshold["pump"]
    dump_pct = threshold["dump"]
    window = threshold["window"]

    moves = []
    i = 0
    cooldown = 0

    while i < len(candles) - window:
        if cooldown > 0:
            cooldown -= 1
            i += 1
            continue

        start_price = candles[i].close
        for j in range(1, window + 1):
            if i + j >= len(candles):
                break
            end_price = candles[i + j].close
            if start_price == 0:
                continue
            move_pct = ((end_price - start_price) / start_price) * 100

            if move_pct >= pump_pct:
                # Calculate volume ratio during the move
                move_vols = [candles[i + k].volume for k in range(j + 1)]
                pre_vols = [candles[i - k].volume for k in range(1, min(21, i + 1))]
                avg_pre_vol = np.mean(pre_vols) if pre_vols else 1
                vol_ratio = np.mean(move_vols) / avg_pre_vol if avg_pre_vol > 0 else 1

                moves.append({
                    "type": "pump",
                    "start_idx": i,
                    "end_idx": i + j,
                    "move_pct": round(move_pct, 2),
                    "start_price": start_price,
                    "end_price": end_price,
                    "volume_ratio": round(vol_ratio, 2),
                    "start_time": candles[i].datetime_str,
                    "end_time": candles[i + j].datetime_str,
                })
                cooldown = window * 2
                break

            elif move_pct <= dump_pct:
                move_vols = [candles[i + k].volume for k in range(j + 1)]
                pre_vols = [candles[i - k].volume for k in range(1, min(21, i + 1))]
                avg_pre_vol = np.mean(pre_vols) if pre_vols else 1
                vol_ratio = np.mean(move_vols) / avg_pre_vol if avg_pre_vol > 0 else 1

                moves.append({
                    "type": "dump",
                    "start_idx": i,
                    "end_idx": i + j,
                    "move_pct": round(move_pct, 2),
                    "start_price": start_price,
                    "end_price": end_price,
                    "volume_ratio": round(vol_ratio, 2),
                    "start_time": candles[i].datetime_str,
                    "end_time": candles[i + j].datetime_str,
                })
                cooldown = window * 2
                break
        i += 1

    return moves


# ============================================================
# MAIN SCAN LOGIC
# ============================================================

async def scan_symbol(session: aiohttp.ClientSession, symbol: str,
                      days: int, stats: Dict[str, IndicatorStats]) -> List[MoveEvent]:
    """Scan a single symbol across all timeframes"""
    events = []

    for tf_key, tf_config in TIMEFRAMES.items():
        candles = await fetch_klines(session, symbol, tf_config["interval"], days)
        if len(candles) < 60:
            continue

        moves = detect_moves(candles, tf_key)

        for move in moves:
            start_idx = move["start_idx"]
            lookback = tf_config["lookback_candles"]

            # Get candles BEFORE the move for indicator analysis
            pre_candles = candles[:start_idx]
            if len(pre_candles) < 60:
                continue

            indicators = analyze_indicators_before_move(pre_candles, lookback)

            event = MoveEvent(
                symbol=symbol,
                timeframe=tf_config["label"],
                move_type=move["type"],
                move_pct=move["move_pct"],
                start_time=move["start_time"],
                end_time=move["end_time"],
                start_price=move["start_price"],
                end_price=move["end_price"],
                volume_ratio=move["volume_ratio"],
                indicators_before=indicators,
            )
            events.append(event)

            # Update statistics
            tf_stats_key = tf_config["label"]
            if tf_stats_key not in stats:
                stats[tf_stats_key] = IndicatorStats()

            s = stats[tf_stats_key]
            if move["type"] == "pump":
                s.total_pumps += 1
                for key, val in indicators.items():
                    if isinstance(val, bool) and val:
                        s.pump_signals[key] += 1
            else:
                s.total_dumps += 1
                for key, val in indicators.items():
                    if isinstance(val, bool) and val:
                        s.dump_signals[key] += 1

    return events


async def run_full_scan(symbols: List[str] = None, days: int = 180,
                        top_n: int = 0, min_move: float = 0.0):
    """Run the complete backscan"""
    print("=" * 70)
    print("  CRYPTO INDICATOR BACKSCAN - Historical Pump/Dump Analyzer")
    print("=" * 70)
    print(f"  Lookback: {days} days")
    print(f"  Timeframes: {', '.join(tf['label'] for tf in TIMEFRAMES.values())}")
    print(f"  Thresholds: {json.dumps(MOVE_THRESHOLDS, indent=2)}")
    print("=" * 70)

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # 1. Get symbols
        if not symbols:
            print("\n📡 Fetching all USDT perpetual symbols from Bybit...")
            symbols = await fetch_all_usdt_symbols(session)
            if top_n > 0:
                # Optionally limit to top N by volume (fetch tickers)
                print(f"  Filtering to top {top_n} by 24h volume...")
                ticker_url = f"{BYBIT_API_URL}/v5/market/tickers"
                try:
                    async with session.get(ticker_url, params={"category": "linear"}) as resp:
                        data = await resp.json()
                        tickers = data.get("result", {}).get("list", [])
                        vol_map = {}
                        for t in tickers:
                            sym = t.get("symbol", "")
                            vol = float(t.get("turnover24h", 0))
                            vol_map[sym] = vol
                        symbols = sorted(symbols, key=lambda s: vol_map.get(s, 0), reverse=True)[:top_n]
                except Exception as e:
                    print(f"  ⚠️ Could not sort by volume: {e}")

        print(f"\n✅ Scanning {len(symbols)} symbols\n")

        # 2. Scan all symbols
        all_events = []
        stats = {}  # tf_label -> IndicatorStats
        total = len(symbols)

        for idx, symbol in enumerate(symbols):
            pct = ((idx + 1) / total) * 100
            print(f"  [{idx+1}/{total}] ({pct:.0f}%) Scanning {symbol}...", end="", flush=True)
            try:
                events = await scan_symbol(session, symbol, days, stats)
                if min_move > 0:
                    events = [e for e in events if abs(e.move_pct) >= min_move]
                all_events.extend(events)
                pump_count = sum(1 for e in events if e.move_type == "pump")
                dump_count = sum(1 for e in events if e.move_type == "dump")
                print(f" {pump_count} pumps, {dump_count} dumps")
            except Exception as e:
                print(f" ❌ Error: {e}")

    # 3. Analyze & output results
    print(f"\n{'=' * 70}")
    print(f"  SCAN COMPLETE: {len(all_events)} total events detected")
    print(f"{'=' * 70}\n")

    analyze_and_report(all_events, stats, days)
    export_csv(all_events)
    generate_pine_script(stats)

    return all_events, stats


# ============================================================
# ANALYSIS & REPORTING
# ============================================================

def analyze_and_report(events: List[MoveEvent], stats: Dict[str, IndicatorStats], days: int):
    """Print detailed analysis of indicator effectiveness"""
    print("=" * 70)
    print("  INDICATOR EFFECTIVENESS REPORT")
    print("=" * 70)

    for tf_label, s in sorted(stats.items()):
        print(f"\n{'─' * 50}")
        print(f"  TIMEFRAME: {tf_label}")
        print(f"  Total pumps: {s.total_pumps} | Total dumps: {s.total_dumps}")
        print(f"{'─' * 50}")

        if s.total_pumps > 0:
            print(f"\n  🟢 TOP INDICATORS BEFORE PUMPS (hit rate):")
            sorted_pump = sorted(s.pump_signals.items(), key=lambda x: x[1], reverse=True)
            for indicator, count in sorted_pump[:20]:
                hit_rate = (count / s.total_pumps) * 100
                bar = "█" * int(hit_rate / 5)
                print(f"    {indicator:<35} {count:>4}/{s.total_pumps:>4} ({hit_rate:5.1f}%) {bar}")

        if s.total_dumps > 0:
            print(f"\n  🔴 TOP INDICATORS BEFORE DUMPS (hit rate):")
            sorted_dump = sorted(s.dump_signals.items(), key=lambda x: x[1], reverse=True)
            for indicator, count in sorted_dump[:20]:
                hit_rate = (count / s.total_dumps) * 100
                bar = "█" * int(hit_rate / 5)
                print(f"    {indicator:<35} {count:>4}/{s.total_dumps:>4} ({hit_rate:5.1f}%) {bar}")

        # Best combo signals for pumps
        if s.total_pumps >= 5:
            print(f"\n  ⚡ BEST PUMP SIGNAL COMBOS (appear together >40% of the time):")
            pump_keys = [k for k, v in s.pump_signals.items() if (v / s.total_pumps) > 0.4]
            if pump_keys:
                for pk in pump_keys:
                    rate = (s.pump_signals[pk] / s.total_pumps) * 100
                    print(f"    + {pk:<35} ({rate:.1f}%)")
            else:
                print(f"    (no single indicator exceeds 40%)")

        if s.total_dumps >= 5:
            print(f"\n  ⚡ BEST DUMP SIGNAL COMBOS (appear together >40% of the time):")
            dump_keys = [k for k, v in s.dump_signals.items() if (v / s.total_dumps) > 0.4]
            if dump_keys:
                for dk in dump_keys:
                    rate = (s.dump_signals[dk] / s.total_dumps) * 100
                    print(f"    + {dk:<35} ({rate:.1f}%)")
            else:
                print(f"    (no single indicator exceeds 40%)")

    # Cross-timeframe analysis
    print(f"\n{'=' * 70}")
    print(f"  CROSS-TIMEFRAME SUMMARY")
    print(f"{'=' * 70}")

    all_pump_signals = defaultdict(int)
    all_dump_signals = defaultdict(int)
    total_pumps = sum(s.total_pumps for s in stats.values())
    total_dumps = sum(s.total_dumps for s in stats.values())

    for s in stats.values():
        for k, v in s.pump_signals.items():
            all_pump_signals[k] += v
        for k, v in s.dump_signals.items():
            all_dump_signals[k] += v

    if total_pumps > 0:
        print(f"\n  🟢 GLOBAL TOP PUMP SIGNALS (across all timeframes, {total_pumps} pumps):")
        for indicator, count in sorted(all_pump_signals.items(), key=lambda x: x[1], reverse=True)[:15]:
            rate = (count / total_pumps) * 100
            print(f"    {indicator:<35} {rate:5.1f}%")

    if total_dumps > 0:
        print(f"\n  🔴 GLOBAL TOP DUMP SIGNALS (across all timeframes, {total_dumps} dumps):")
        for indicator, count in sorted(all_dump_signals.items(), key=lambda x: x[1], reverse=True)[:15]:
            rate = (count / total_dumps) * 100
            print(f"    {indicator:<35} {rate:5.1f}%")


def export_csv(events: List[MoveEvent]):
    """Export all events to CSV for further analysis"""
    filename = "backscan_events.csv"
    if not events:
        print(f"\n  ⚠️ No events to export")
        return

    # Collect all indicator keys
    all_keys = set()
    for e in events:
        all_keys.update(e.indicators_before.keys())
    all_keys = sorted(all_keys)

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["symbol", "timeframe", "type", "move_pct", "start_time",
                  "end_time", "start_price", "end_price", "volume_ratio"] + all_keys
        writer.writerow(header)
        for e in events:
            row = [e.symbol, e.timeframe, e.move_type, e.move_pct,
                   e.start_time, e.end_time, e.start_price, e.end_price, e.volume_ratio]
            for k in all_keys:
                val = e.indicators_before.get(k, "")
                row.append(val)
            writer.writerow(row)

    print(f"\n  📄 Events exported to {filename}")


# ============================================================
# PINE SCRIPT GENERATOR
# ============================================================

def generate_pine_script(stats: Dict[str, IndicatorStats]):
    """
    Generate TradingView Pine Script based on the most effective
    indicator combinations found in the backscan.
    """
    print(f"\n{'=' * 70}")
    print(f"  GENERATING TRADINGVIEW PINE SCRIPT")
    print(f"{'=' * 70}")

    # Determine best indicators per timeframe
    best_pump_indicators = {}
    best_dump_indicators = {}

    for tf_label, s in stats.items():
        if s.total_pumps >= 3:
            sorted_pump = sorted(s.pump_signals.items(), key=lambda x: x[1], reverse=True)
            best_pump_indicators[tf_label] = [
                (k, round(v / s.total_pumps * 100, 1))
                for k, v in sorted_pump[:10]
                if (v / s.total_pumps) >= 0.3  # at least 30% hit rate
            ]
        if s.total_dumps >= 3:
            sorted_dump = sorted(s.dump_signals.items(), key=lambda x: x[1], reverse=True)
            best_dump_indicators[tf_label] = [
                (k, round(v / s.total_dumps * 100, 1))
                for k, v in sorted_dump[:10]
                if (v / s.total_dumps) >= 0.3
            ]

    # Build Pine Script
    pine = []
    pine.append('// ============================================================')
    pine.append('// AUTO-GENERATED PINE SCRIPT - Indicator Backscan Results')
    pine.append(f'// Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}')
    pine.append(f'// Backscan period: historical data')
    pine.append('// Based on analysis of pump/dump patterns across all USDT perps')
    pine.append('// ============================================================')
    pine.append('//@version=5')
    pine.append('indicator("Backscan Pump/Dump Detector", overlay=true, max_labels_count=500)')
    pine.append('')
    pine.append('// === INPUTS (tune these based on backscan results) ===')
    pine.append(f'rsi_period    = input.int({DEFAULT_RSI_PERIOD}, "RSI Period")')
    pine.append(f'rsi_oversold  = input.float(30, "RSI Oversold")')
    pine.append(f'rsi_overbought = input.float(70, "RSI Overbought")')
    pine.append(f'ema_fast      = input.int({DEFAULT_EMA_FAST}, "EMA Fast")')
    pine.append(f'ema_mid       = input.int({DEFAULT_EMA_MID}, "EMA Mid")')
    pine.append(f'ema_slow      = input.int({DEFAULT_EMA_SLOW}, "EMA Slow")')
    pine.append(f'macd_fast     = input.int({DEFAULT_MACD_FAST}, "MACD Fast")')
    pine.append(f'macd_slow     = input.int({DEFAULT_MACD_SLOW}, "MACD Slow")')
    pine.append(f'macd_signal   = input.int({DEFAULT_MACD_SIGNAL}, "MACD Signal")')
    pine.append(f'bb_period     = input.int({DEFAULT_BB_PERIOD}, "BB Period")')
    pine.append(f'bb_mult       = input.float({DEFAULT_BB_STD}, "BB StdDev")')
    pine.append(f'st_period     = input.int({DEFAULT_SUPERTREND_PERIOD}, "Supertrend Period")')
    pine.append(f'st_mult       = input.float({DEFAULT_SUPERTREND_MULT}, "Supertrend Multiplier")')
    pine.append(f'stoch_period  = input.int({DEFAULT_STOCH_RSI_PERIOD}, "Stoch RSI Period")')
    pine.append(f'stoch_k       = input.int({DEFAULT_STOCH_RSI_K}, "Stoch RSI K")')
    pine.append(f'stoch_d       = input.int({DEFAULT_STOCH_RSI_D}, "Stoch RSI D")')
    pine.append(f'vol_ma_len    = input.int({DEFAULT_VOLUME_MA}, "Volume MA Length")')
    pine.append(f'atr_period    = input.int({DEFAULT_ATR_PERIOD}, "ATR Period")')
    pine.append('')
    pine.append('// === INDICATOR CALCULATIONS ===')
    pine.append('')
    pine.append('// RSI')
    pine.append('rsi_val = ta.rsi(close, rsi_period)')
    pine.append('rsi_oversold_cond  = rsi_val < rsi_oversold')
    pine.append('rsi_overbought_cond = rsi_val > rsi_overbought')
    pine.append('rsi_rising = rsi_val > rsi_val[3]')
    pine.append('rsi_falling = rsi_val < rsi_val[3]')
    pine.append('')
    pine.append('// Stochastic RSI')
    pine.append('rsi_src = ta.rsi(close, stoch_period)')
    pine.append('stoch_k_val = ta.sma(ta.stoch(rsi_src, rsi_src, rsi_src, stoch_period), stoch_k)')
    pine.append('stoch_d_val = ta.sma(stoch_k_val, stoch_d)')
    pine.append('stoch_oversold = stoch_k_val < 20 and stoch_d_val < 20')
    pine.append('stoch_overbought = stoch_k_val > 80 and stoch_d_val > 80')
    pine.append('stoch_bull_cross = ta.crossover(stoch_k_val, stoch_d_val)')
    pine.append('stoch_bear_cross = ta.crossunder(stoch_k_val, stoch_d_val)')
    pine.append('')
    pine.append('// MACD')
    pine.append('[macd_line, signal_line, hist_line] = ta.macd(close, macd_fast, macd_slow, macd_signal)')
    pine.append('macd_bull_cross = ta.crossover(macd_line, signal_line)')
    pine.append('macd_bear_cross = ta.crossunder(macd_line, signal_line)')
    pine.append('macd_above = macd_line > signal_line')
    pine.append('macd_below = macd_line < signal_line')
    pine.append('macd_hist_rising = hist_line > hist_line[1]')
    pine.append('macd_hist_falling = hist_line < hist_line[1]')
    pine.append('')
    pine.append('// EMA')
    pine.append('ema_f = ta.ema(close, ema_fast)')
    pine.append('ema_m = ta.ema(close, ema_mid)')
    pine.append('ema_s = ta.ema(close, ema_slow)')
    pine.append('ema_bull = ema_f > ema_s')
    pine.append('ema_bear = ema_f < ema_s')
    pine.append('ema_bull_cross = ta.crossover(ema_f, ema_s)')
    pine.append('ema_bear_cross = ta.crossunder(ema_f, ema_s)')
    pine.append('ema_ribbon_bull = ema_f > ema_m and ema_m > ema_s')
    pine.append('ema_ribbon_bear = ema_f < ema_m and ema_m < ema_s')
    pine.append('')
    pine.append('// Bollinger Bands')
    pine.append('[bb_mid, bb_up, bb_low] = ta.bb(close, bb_period, bb_mult)')
    pine.append('bb_below_lower = close < bb_low')
    pine.append('bb_above_upper = close > bb_up')
    pine.append('bb_bandwidth = (bb_up - bb_low) / bb_mid * 100')
    pine.append('bb_squeeze = bb_bandwidth < 3.0')
    pine.append('bb_expanding = bb_bandwidth > bb_bandwidth[5]')
    pine.append('')
    pine.append('// Supertrend')
    pine.append('[st_val, st_dir] = ta.supertrend(st_mult, st_period)')
    pine.append('st_bull = st_dir < 0')
    pine.append('st_bear = st_dir > 0')
    pine.append('st_flip_bull = st_dir < 0 and st_dir[1] > 0')
    pine.append('st_flip_bear = st_dir > 0 and st_dir[1] < 0')
    pine.append('')
    pine.append('// Volume')
    pine.append('vol_avg = ta.sma(volume, vol_ma_len)')
    pine.append('vol_ratio = volume / vol_avg')
    pine.append('vol_spike = vol_ratio > 2.0')
    pine.append('vol_high = vol_ratio > 1.5')
    pine.append('')
    pine.append('// ATR')
    pine.append('atr_val = ta.atr(atr_period)')
    pine.append('atr_pct = atr_val / close * 100')
    pine.append('')
    pine.append('// Candlestick Patterns')
    pine.append('bullish_engulfing = close[1] < open[1] and close > open and close > open[1] and open < close[1]')
    pine.append('bearish_engulfing = close[1] > open[1] and close < open and close < open[1] and open > close[1]')
    pine.append('body = math.abs(close - open)')
    pine.append('lower_wick = math.min(open, close) - low')
    pine.append('upper_wick = high - math.max(open, close)')
    pine.append('hammer = body > 0 ? (lower_wick > body * 2 and upper_wick < body * 0.5) : false')
    pine.append('')
    pine.append('// === COMPOSITE SIGNALS (from backscan analysis) ===')
    pine.append('')

    # Generate pump signal conditions based on top performing indicators
    pump_conditions = []
    dump_conditions = []

    # Collect the best indicators across all timeframes
    all_best_pump = defaultdict(list)
    all_best_dump = defaultdict(list)

    for tf, indicators in best_pump_indicators.items():
        for ind_name, hit_rate in indicators:
            all_best_pump[ind_name].append((tf, hit_rate))

    for tf, indicators in best_dump_indicators.items():
        for ind_name, hit_rate in indicators:
            all_best_dump[ind_name].append((tf, hit_rate))

    # Map Python indicator names to Pine Script conditions
    indicator_to_pine = {
        "rsi_oversold": "rsi_oversold_cond",
        "rsi_overbought": "rsi_overbought_cond",
        "rsi_rising": "rsi_rising",
        "rsi_falling": "rsi_falling",
        "rsi_bullish_div": "rsi_val > rsi_val[5] and close < close[5]",
        "rsi_bearish_div": "rsi_val < rsi_val[5] and close > close[5]",
        "stoch_rsi_oversold": "stoch_oversold",
        "stoch_rsi_overbought": "stoch_overbought",
        "stoch_rsi_bullish_cross": "stoch_bull_cross",
        "stoch_rsi_bearish_cross": "stoch_bear_cross",
        "macd_above_signal": "macd_above",
        "macd_below_signal": "macd_below",
        "macd_bullish_cross": "macd_bull_cross",
        "macd_bearish_cross": "macd_bear_cross",
        "macd_histogram_rising": "macd_hist_rising",
        "macd_histogram_falling": "macd_hist_falling",
        "macd_histogram_positive": "hist_line > 0",
        "macd_histogram_negative": "hist_line < 0",
        "macd_bullish_div": "macd_line > macd_line[5] and close < close[5]",
        "macd_bearish_div": "macd_line < macd_line[5] and close > close[5]",
        "ema_bullish": "ema_bull",
        "ema_bearish": "ema_bear",
        "ema_bullish_cross": "ema_bull_cross",
        "ema_bearish_cross": "ema_bear_cross",
        "ema_ribbon_bullish": "ema_ribbon_bull",
        "ema_ribbon_bearish": "ema_ribbon_bear",
        "bb_below_lower": "bb_below_lower",
        "bb_above_upper": "bb_above_upper",
        "bb_squeeze": "bb_squeeze",
        "bb_expanding": "bb_expanding",
        "supertrend_bullish": "st_bull",
        "supertrend_bearish": "st_bear",
        "supertrend_flip_bullish": "st_flip_bull",
        "supertrend_flip_bearish": "st_flip_bear",
        "volume_spike": "vol_spike",
        "volume_high": "vol_high",
        "bullish_engulfing": "bullish_engulfing",
        "bearish_engulfing": "bearish_engulfing",
        "hammer": "hammer",
    }

    # Build pump score
    pine.append('// --- PUMP SCORE (higher = more likely pump incoming) ---')
    pine.append('pump_score = 0.0')
    for ind_name, tf_hits in sorted(all_best_pump.items(), key=lambda x: max(h for _, h in x[1]), reverse=True):
        pine_cond = indicator_to_pine.get(ind_name)
        if pine_cond:
            avg_hit = np.mean([h for _, h in tf_hits])
            weight = round(avg_hit / 100, 2)
            pine.append(f'pump_score += {pine_cond} ? {weight} : 0  // {ind_name} (avg hit: {avg_hit:.1f}%)')
    pine.append('')

    # Build dump score
    pine.append('// --- DUMP SCORE (higher = more likely dump incoming) ---')
    pine.append('dump_score = 0.0')
    for ind_name, tf_hits in sorted(all_best_dump.items(), key=lambda x: max(h for _, h in x[1]), reverse=True):
        pine_cond = indicator_to_pine.get(ind_name)
        if pine_cond:
            avg_hit = np.mean([h for _, h in tf_hits])
            weight = round(avg_hit / 100, 2)
            pine.append(f'dump_score += {pine_cond} ? {weight} : 0  // {ind_name} (avg hit: {avg_hit:.1f}%)')
    pine.append('')

    # Signal thresholds
    pine.append('// === SIGNAL THRESHOLDS ===')
    pine.append('pump_threshold = input.float(2.0, "Pump Signal Threshold", step=0.1)')
    pine.append('dump_threshold = input.float(2.0, "Dump Signal Threshold", step=0.1)')
    pine.append('')
    pine.append('pump_signal = pump_score >= pump_threshold')
    pine.append('dump_signal = dump_score >= dump_threshold')
    pine.append('')

    # Alerts & plotting
    pine.append('// === VISUALIZATION ===')
    pine.append('plotshape(pump_signal, title="Pump Alert", style=shape.triangleup, ')
    pine.append('          location=location.belowbar, color=color.new(color.green, 0), size=size.small)')
    pine.append('plotshape(dump_signal, title="Dump Alert", style=shape.triangledown, ')
    pine.append('          location=location.abovebar, color=color.new(color.red, 0), size=size.small)')
    pine.append('')
    pine.append('// Score display')
    pine.append('plot(pump_score, "Pump Score", color=color.green, display=display.data_window)')
    pine.append('plot(dump_score, "Dump Score", color=color.red, display=display.data_window)')
    pine.append('')
    pine.append('// Background coloring')
    pine.append('bgcolor(pump_signal ? color.new(color.green, 90) : na)')
    pine.append('bgcolor(dump_signal ? color.new(color.red, 90) : na)')
    pine.append('')
    pine.append('// EMA lines')
    pine.append('plot(ema_f, "EMA Fast", color=color.new(color.blue, 50))')
    pine.append('plot(ema_m, "EMA Mid", color=color.new(color.orange, 50))')
    pine.append('plot(ema_s, "EMA Slow", color=color.new(color.red, 50))')
    pine.append('')
    pine.append('// Supertrend')
    pine.append('plot(st_val, "Supertrend", color=st_bull ? color.green : color.red, linewidth=2)')
    pine.append('')
    pine.append('// Bollinger Bands')
    pine.append('plot(bb_up, "BB Upper", color=color.new(color.gray, 70))')
    pine.append('plot(bb_low, "BB Lower", color=color.new(color.gray, 70))')
    pine.append('')
    pine.append('// === ALERTS ===')
    pine.append('alertcondition(pump_signal, title="Pump Detected", ')
    pine.append('               message="🟢 PUMP SIGNAL: Score={{plot_0}} on {{ticker}}")')
    pine.append('alertcondition(dump_signal, title="Dump Detected", ')
    pine.append('               message="🔴 DUMP SIGNAL: Score={{plot_1}} on {{ticker}}")')

    # Write Pine Script to file
    pine_filename = "backscan_indicator.pine"
    with open(pine_filename, "w") as f:
        f.write("\n".join(pine))

    print(f"\n  📜 Pine Script saved to {pine_filename}")
    print(f"     Copy this into TradingView → Pine Editor → Add to chart")

    # Also write a summary JSON
    summary = {
        "generated": datetime.utcnow().isoformat(),
        "best_pump_indicators": {k: v for k, v in best_pump_indicators.items()},
        "best_dump_indicators": {k: v for k, v in best_dump_indicators.items()},
        "settings": {
            "rsi_period": DEFAULT_RSI_PERIOD,
            "ema_fast": DEFAULT_EMA_FAST,
            "ema_mid": DEFAULT_EMA_MID,
            "ema_slow": DEFAULT_EMA_SLOW,
            "macd": f"{DEFAULT_MACD_FAST}/{DEFAULT_MACD_SLOW}/{DEFAULT_MACD_SIGNAL}",
            "bb": f"{DEFAULT_BB_PERIOD}/{DEFAULT_BB_STD}",
            "supertrend": f"{DEFAULT_SUPERTREND_PERIOD}/{DEFAULT_SUPERTREND_MULT}",
            "stoch_rsi": f"{DEFAULT_STOCH_RSI_PERIOD}/{DEFAULT_STOCH_RSI_K}/{DEFAULT_STOCH_RSI_D}",
        }
    }
    with open("backscan_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  📊 Summary saved to backscan_summary.json")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Historical Crypto Indicator Backscan - Detect pumps/dumps & find best indicators"
    )
    parser.add_argument("--symbols", type=str, default="",
                        help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT). Empty = all USDT perps")
    parser.add_argument("--days", type=int, default=180,
                        help="Days of history to scan (default: 180 = ~6 months)")
    parser.add_argument("--top", type=int, default=0,
                        help="Only scan top N symbols by 24h volume (0 = all)")
    parser.add_argument("--min-move", type=float, default=0.0,
                        help="Minimum move %% to include in results (default: 0 = use per-TF thresholds)")
    parser.add_argument("--pump-15m", type=float, default=MOVE_THRESHOLDS["15"]["pump"],
                        help=f"Pump threshold for 15m (default: {MOVE_THRESHOLDS['15']['pump']}%%)")
    parser.add_argument("--dump-15m", type=float, default=MOVE_THRESHOLDS["15"]["dump"],
                        help=f"Dump threshold for 15m (default: {MOVE_THRESHOLDS['15']['dump']}%%)")
    parser.add_argument("--pump-1h", type=float, default=MOVE_THRESHOLDS["60"]["pump"],
                        help=f"Pump threshold for 1h (default: {MOVE_THRESHOLDS['60']['pump']}%%)")
    parser.add_argument("--dump-1h", type=float, default=MOVE_THRESHOLDS["60"]["dump"],
                        help=f"Dump threshold for 1h (default: {MOVE_THRESHOLDS['60']['dump']}%%)")
    parser.add_argument("--pump-4h", type=float, default=MOVE_THRESHOLDS["240"]["pump"],
                        help=f"Pump threshold for 4h (default: {MOVE_THRESHOLDS['240']['pump']}%%)")
    parser.add_argument("--dump-4h", type=float, default=MOVE_THRESHOLDS["240"]["dump"],
                        help=f"Dump threshold for 4h (default: {MOVE_THRESHOLDS['240']['dump']}%%)")

    args = parser.parse_args()

    # Apply custom thresholds
    MOVE_THRESHOLDS["15"]["pump"] = args.pump_15m
    MOVE_THRESHOLDS["15"]["dump"] = args.dump_15m
    MOVE_THRESHOLDS["60"]["pump"] = args.pump_1h
    MOVE_THRESHOLDS["60"]["dump"] = args.dump_1h
    MOVE_THRESHOLDS["240"]["pump"] = args.pump_4h
    MOVE_THRESHOLDS["240"]["dump"] = args.dump_4h

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    asyncio.run(run_full_scan(
        symbols=symbols,
        days=args.days,
        top_n=args.top,
        min_move=args.min_move,
    ))


if __name__ == "__main__":
    main()
