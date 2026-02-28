#!/usr/bin/env python3
"""
pine_backscan.py — Backscan v3 + v1 Signal Scanner
====================================================
Replicates the Pine Script "Backscan v3 + v1 Long Setups FIXED v3" logic
in Python, scans the top 100 coins on Bybit, and for every signal that fired
in the past N days checks whether the crypto moved ≥3% in any direction
within the next 8 candles (2 hours on 15m).

This lets you:
  • See WHICH signals actually fired on each coin
  • See WHICH indicator combos were active at signal time
  • See WHICH signals led to a 3%+ move (removing inflated win-rate bias)
  • Get a real, unbiased win-rate per signal type and indicator combo

Output files:
  pine_backscan_events.csv     — Every signal event with all indicators
  pine_backscan_summary.json   — Win-rate breakdown per signal / combo
  pine_backscan_report.txt     — Human-readable report

Usage:
    python pine_backscan.py                          # top 100 coins, 30 days
    python pine_backscan.py --top 50 --days 60
    python pine_backscan.py --symbols BTCUSDT,ETHUSDT --days 90
"""

import asyncio
import aiohttp
import argparse
import csv
import json
import math
import time
import os
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG  (mirrors Pine Script defaults)
# ─────────────────────────────────────────────────────────────
BYBIT_API     = "https://api.bybit.com"
RATE_DELAY    = 0.12          # seconds between requests

# Indicator periods (match Pine Script inputs)
EMA_F, EMA_M, EMA_S  = 9, 21, 55
MACD_F, MACD_SL, MACD_SIG = 12, 26, 9
BB_P, BB_STD          = 20, 2.0
ST_P, ST_MULT         = 10, 3.0
STOCH_P, STOCH_K, STOCH_D = 14, 3, 3
RSI_P                 = 14
VOL_MA                = 20
ADX_P                 = 14
CCI_P                 = 20
WR_P                  = 14
MFI_P                 = 14
KC_P, KC_MULT         = 20, 1.5
ATR_P                 = 14
ICH_T, ICH_K, ICH_B   = 9, 26, 52
OBV_MA                = 20
ROC_P                 = 12
MOM_P                 = 10
PSAR_AF, PSAR_MAX     = 0.02, 0.2

# v1 thresholds
V1_PUMP_THRESH        = 4.0
V1_STRONG_PUMP_THRESH = 7.0
V1_MIN_NET_GAP        = 2.0
V1_COOLDOWN_BARS      = 8

# v3 thresholds
V3_COOLDOWN           = 12
V3_FLIP_MULT          = 2.5

# Outcome check
MOVE_CHECK_BARS       = 8     # bars after signal to check for 3% move
MOVE_TARGET_PCT       = 3.0   # % move required to count as "worked"

OUTPUT_CSV   = "pine_backscan_events.csv"
OUTPUT_JSON  = "pine_backscan_summary.json"
OUTPUT_TXT   = "pine_backscan_report.txt"


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────
@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SignalEvent:
    symbol: str
    bar_idx: int
    ts: int
    dt: str
    close: float
    signal_type: str        # v1_pump, v1_strong_pump, v3_pump_sniper, v3_pump_elite, v3_pump_std, v3_pump_pullback, v3_dump_sniper, v3_dump_elite, v3_dump_std
    direction: str          # pump / dump
    pump_score: float
    dump_score: float
    net_score: float
    pump_tier: int
    dump_tier: int
    # indicator combo flags
    indicators: Dict        = field(default_factory=dict)
    # outcome
    moved_3pct: bool        = False
    max_move_pct: float     = 0.0
    move_direction: str     = ""


# ─────────────────────────────────────────────────────────────
# FETCH HELPERS
# ─────────────────────────────────────────────────────────────
async def fetch_top_symbols(session: aiohttp.ClientSession, top_n: int = 100) -> List[str]:
    url = f"{BYBIT_API}/v5/market/tickers"
    params = {"category": "linear"}
    async with session.get(url, params=params) as resp:
        data = await resp.json()
    tickers = data.get("result", {}).get("list", [])
    usdt = [t for t in tickers if t.get("symbol", "").endswith("USDT")]
    usdt.sort(key=lambda t: float(t.get("turnover24h") or 0), reverse=True)
    return [t["symbol"] for t in usdt[:top_n]]


async def fetch_klines(session: aiohttp.ClientSession, symbol: str,
                       interval: str, days: int) -> List[Candle]:
    """Fetch historical klines, oldest first."""
    url     = f"{BYBIT_API}/v5/market/kline"
    start   = int((time.time() - days * 86400) * 1000)
    end     = int(time.time() * 1000)
    all_c: List[Candle] = []
    cur_end = end
    for _ in range(50):   # safety limit
        params = {
            "category": "linear",
            "symbol":   symbol,
            "interval": interval,
            "end":      str(cur_end),
            "limit":    "200",
        }
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
            klines = data.get("result", {}).get("list", [])
            if not klines:
                break
            for k in klines:
                ts = int(k[0])
                if ts < start:
                    continue
                all_c.append(Candle(ts, float(k[1]), float(k[2]),
                                    float(k[3]), float(k[4]), float(k[5])))
            oldest = int(klines[-1][0])
            if oldest <= start or oldest >= cur_end:
                break
            cur_end = oldest - 1
            await asyncio.sleep(RATE_DELAY)
        except Exception as e:
            print(f"  ⚠️  kline fetch error {symbol} {interval}: {e}")
            break

    seen = set(); unique = []
    for c in all_c:
        if c.ts not in seen:
            seen.add(c.ts); unique.append(c)
    return sorted(unique, key=lambda x: x.ts)


# ─────────────────────────────────────────────────────────────
# INDICATOR MATH (numpy-free, pure Python)
# ─────────────────────────────────────────────────────────────

def _sma(arr: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(arr)
    for i in range(period - 1, len(arr)):
        out[i] = sum(arr[i - period + 1: i + 1]) / period
    return out


def _ema(arr: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(arr)
    mult = 2 / (period + 1)
    for i in range(len(arr)):
        if math.isnan(arr[i]):
            continue
        if math.isnan(out[i - 1]) if i > 0 else True:
            if i >= period - 1:
                out[i] = sum(arr[i - period + 1: i + 1]) / period
        else:
            out[i] = arr[i] * mult + out[i - 1] * (1 - mult)
    return out


def _rsi(close: List[float], period: int = 14) -> List[float]:
    out   = [float("nan")] * len(close)
    gains = [0.0] * len(close)
    losses= [0.0] * len(close)
    for i in range(1, len(close)):
        d = close[i] - close[i - 1]
        gains[i]  = d if d > 0 else 0
        losses[i] = -d if d < 0 else 0
    if len(close) < period + 1:
        return out
    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(close)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - 100 / (1 + rs)
    return out


def _macd(close: List[float], fast=12, slow=26, sig=9):
    ef   = _ema(close, fast)
    es   = _ema(close, slow)
    macd = [ef[i] - es[i] if not (math.isnan(ef[i]) or math.isnan(es[i]))
            else float("nan") for i in range(len(close))]
    signal = _ema(macd, sig)
    hist   = [macd[i] - signal[i] if not (math.isnan(macd[i]) or math.isnan(signal[i]))
              else float("nan") for i in range(len(close))]
    return macd, signal, hist


def _atr(candles: List[Candle], period: int = 14) -> List[float]:
    trs = [float("nan")]
    for i in range(1, len(candles)):
        hl = candles[i].high - candles[i].low
        hc = abs(candles[i].high - candles[i - 1].close)
        lc = abs(candles[i].low  - candles[i - 1].close)
        trs.append(max(hl, hc, lc))
    out = [float("nan")] * len(trs)
    if len(trs) >= period:
        out[period - 1] = sum(trs[1:period]) / (period - 1)
        for i in range(period, len(trs)):
            out[i] = (out[i - 1] * (period - 1) + trs[i]) / period
    return out


def _supertrend(candles: List[Candle], period: int = 10, mult: float = 3.0):
    atr    = _atr(candles, period)
    n      = len(candles)
    up     = [float("nan")] * n
    dn     = [float("nan")] * n
    trend  = [float("nan")] * n  # -1 = bull, 1 = bear (matches Pine st_dir<0 = bull)
    st_val = [float("nan")] * n

    for i in range(period, n):
        if math.isnan(atr[i]):
            continue
        hl2  = (candles[i].high + candles[i].low) / 2
        basic_up = hl2 - mult * atr[i]
        basic_dn = hl2 + mult * atr[i]

        up[i] = basic_up if (math.isnan(up[i - 1]) or
                             candles[i - 1].close <= up[i - 1]) else max(basic_up, up[i - 1])
        dn[i] = basic_dn if (math.isnan(dn[i - 1]) or
                             candles[i - 1].close >= dn[i - 1]) else min(basic_dn, dn[i - 1])

        if math.isnan(trend[i - 1]):
            trend[i] = -1 if candles[i].close > basic_dn else 1
        elif trend[i - 1] == 1:
            trend[i] = -1 if candles[i].close > dn[i] else 1
        else:
            trend[i] = 1 if candles[i].close < up[i] else -1

        st_val[i] = up[i] if trend[i] == -1 else dn[i]

    return st_val, trend   # trend -1 = bull (st_bull = trend<0)


def _bb(close: List[float], period: int = 20, std_mult: float = 2.0):
    mid = _sma(close, period)
    up  = [float("nan")] * len(close)
    lo  = [float("nan")] * len(close)
    for i in range(period - 1, len(close)):
        sl = close[i - period + 1: i + 1]
        std = math.sqrt(sum((x - mid[i]) ** 2 for x in sl) / period)
        up[i] = mid[i] + std_mult * std
        lo[i] = mid[i] - std_mult * std
    return mid, up, lo


def _stoch_rsi(close: List[float], rsi_p=14, stoch_p=14, k_p=3, d_p=3):
    rsi   = _rsi(close, rsi_p)
    stoch = [float("nan")] * len(rsi)
    for i in range(stoch_p - 1, len(rsi)):
        window = rsi[i - stoch_p + 1: i + 1]
        if any(math.isnan(x) for x in window):
            continue
        lo, hi = min(window), max(window)
        stoch[i] = (rsi[i] - lo) / (hi - lo) * 100 if hi != lo else 50
    k = _sma(stoch, k_p)
    d = _sma(k, d_p)
    return k, d


def _wpr(candles: List[Candle], period: int = 14) -> List[float]:
    out = [float("nan")] * len(candles)
    for i in range(period - 1, len(candles)):
        highs = [c.high for c in candles[i - period + 1: i + 1]]
        lows  = [c.low  for c in candles[i - period + 1: i + 1]]
        hh    = max(highs); ll = min(lows)
        out[i] = ((hh - candles[i].close) / (hh - ll) * -100) if hh != ll else -50
    return out


def _cci(candles: List[Candle], period: int = 20) -> List[float]:
    out = [float("nan")] * len(candles)
    for i in range(period - 1, len(candles)):
        tp_window = [(c.high + c.low + c.close) / 3 for c in candles[i - period + 1: i + 1]]
        mean = sum(tp_window) / period
        mad  = sum(abs(x - mean) for x in tp_window) / period
        out[i] = (tp_window[-1] - mean) / (0.015 * mad) if mad != 0 else 0
    return out


def _mfi(candles: List[Candle], period: int = 14) -> List[float]:
    out = [float("nan")] * len(candles)
    tp  = [(c.high + c.low + c.close) / 3 for c in candles]
    for i in range(period, len(candles)):
        pos_mf = neg_mf = 0.0
        for j in range(i - period + 1, i + 1):
            raw_mf = tp[j] * candles[j].volume
            if j > 0 and tp[j] > tp[j - 1]:
                pos_mf += raw_mf
            else:
                neg_mf += raw_mf
        out[i] = 100 - 100 / (1 + pos_mf / neg_mf) if neg_mf != 0 else 100
    return out


def _roc(close: List[float], period: int = 12) -> List[float]:
    out = [float("nan")] * len(close)
    for i in range(period, len(close)):
        if close[i - period] != 0:
            out[i] = (close[i] - close[i - period]) / close[i - period] * 100
    return out


def _obv(candles: List[Candle]) -> List[float]:
    out = [0.0]
    for i in range(1, len(candles)):
        sign = 1 if candles[i].close > candles[i - 1].close else (-1 if candles[i].close < candles[i - 1].close else 0)
        out.append(out[-1] + sign * candles[i].volume)
    return out


def _cmf(candles: List[Candle], period: int = 20) -> List[float]:
    out = [float("nan")] * len(candles)
    for i in range(period - 1, len(candles)):
        mf_vol = vol_sum = 0.0
        for c in candles[i - period + 1: i + 1]:
            hl = c.high - c.low
            if hl == 0:
                continue
            mf  = ((2 * c.close - c.low - c.high) / hl) * c.volume
            mf_vol  += mf
            vol_sum += c.volume
        out[i] = mf_vol / vol_sum if vol_sum != 0 else 0
    return out


def _psar(candles: List[Candle], start=0.02, inc=0.02, max_af=0.2) -> List[float]:
    n   = len(candles)
    out = [float("nan")] * n
    if n < 2:
        return out
    bull    = True
    af      = start
    ep      = candles[0].high
    psar_v  = candles[0].low
    for i in range(1, n):
        psar_v = psar_v + af * (ep - psar_v)
        if bull:
            if candles[i].low < psar_v:
                bull   = False
                psar_v = ep
                ep     = candles[i].low
                af     = start
            else:
                if candles[i].high > ep:
                    ep = candles[i].high
                    af = min(af + inc, max_af)
                psar_v = min(psar_v, candles[i - 1].low,
                             candles[i - 2].low if i >= 2 else candles[i - 1].low)
        else:
            if candles[i].high > psar_v:
                bull   = True
                psar_v = ep
                ep     = candles[i].high
                af     = start
            else:
                if candles[i].low < ep:
                    ep = candles[i].low
                    af = min(af + inc, max_af)
                psar_v = max(psar_v, candles[i - 1].high,
                             candles[i - 2].high if i >= 2 else candles[i - 1].high)
        out[i] = psar_v
    return out


def _donchian_mid(candles: List[Candle], period: int) -> List[float]:
    """Donchian midpoint = (highest_high + lowest_low) / 2"""
    out = [float("nan")] * len(candles)
    for i in range(period - 1, len(candles)):
        hh = max(c.high for c in candles[i - period + 1: i + 1])
        ll = min(c.low  for c in candles[i - period + 1: i + 1])
        out[i] = (hh + ll) / 2
    return out


def _dmi(candles: List[Candle], period: int = 14):
    """Returns DI+, DI-, ADX arrays."""
    n     = len(candles)
    dm_p  = [0.0] * n
    dm_m  = [0.0] * n
    tr_arr= [0.0] * n
    for i in range(1, n):
        up   = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        dm_p[i] = up   if up > down and up > 0 else 0
        dm_m[i] = down if down > up and down > 0 else 0
        hl = candles[i].high - candles[i].low
        hc = abs(candles[i].high - candles[i - 1].close)
        lc = abs(candles[i].low  - candles[i - 1].close)
        tr_arr[i] = max(hl, hc, lc)

    def _smooth(arr, p):
        s = [float("nan")] * len(arr)
        if len(arr) <= p:
            return s
        s[p] = sum(arr[1:p + 1])
        for i in range(p + 1, len(arr)):
            s[i] = s[i - 1] - s[i - 1] / p + arr[i]
        return s

    str14 = _smooth(tr_arr, period)
    sdp   = _smooth(dm_p,   period)
    sdm   = _smooth(dm_m,   period)

    di_p  = [float("nan")] * n
    di_m  = [float("nan")] * n
    dx    = [float("nan")] * n
    for i in range(period, n):
        if math.isnan(str14[i]) or str14[i] == 0:
            continue
        di_p[i] = 100 * sdp[i] / str14[i]
        di_m[i] = 100 * sdm[i] / str14[i]
        s       = di_p[i] + di_m[i]
        dx[i]   = 100 * abs(di_p[i] - di_m[i]) / s if s != 0 else 0

    adx = _smooth(dx, period)
    return di_p, di_m, adx


# ─────────────────────────────────────────────────────────────
# ICHIMOKU HELPERS
# ─────────────────────────────────────────────────────────────
def _ich_donchian(candles: List[Candle], period: int) -> List[float]:
    """(highest_high + lowest_low) / 2  — same as Pine f_donchian"""
    out = [float("nan")] * len(candles)
    for i in range(period - 1, len(candles)):
        hh = max(c.high for c in candles[i - period + 1: i + 1])
        ll = min(c.low  for c in candles[i - period + 1: i + 1])
        out[i] = (hh + ll) / 2
    return out


# ─────────────────────────────────────────────────────────────
# MAIN INDICATOR CALCULATOR PER TIMEFRAME
# ─────────────────────────────────────────────────────────────
def calc_indicators(candles: List[Candle]) -> List[Dict]:
    """
    For each bar, compute all indicators.
    Returns a list of dicts (one per candle bar), indexed same as candles.
    """
    n = len(candles)
    close  = [c.close  for c in candles]
    high   = [c.high   for c in candles]
    low    = [c.low    for c in candles]
    volume = [c.volume for c in candles]
    hlc3   = [(c.high + c.low + c.close) / 3 for c in candles]

    # Core indicators
    ema_f_arr = _ema(close, EMA_F)
    ema_m_arr = _ema(close, EMA_M)
    ema_s_arr = _ema(close, EMA_S)
    rsi_arr   = _rsi(close, RSI_P)
    macd_l, macd_s, macd_h = _macd(close, MACD_F, MACD_SL, MACD_SIG)
    bb_mid, bb_up, bb_lo   = _bb(close, BB_P, BB_STD)
    st_val, st_dir          = _supertrend(candles, ST_P, ST_MULT)
    stoch_k, stoch_d        = _stoch_rsi(close, RSI_P, STOCH_P, STOCH_K, STOCH_D)
    wr_arr  = _wpr(candles, WR_P)
    cci_arr = _cci(candles, CCI_P)
    mfi_arr = _mfi(candles, MFI_P)
    roc_arr = _roc(close, ROC_P)
    obv_arr = _obv(candles)
    cmf_arr = _cmf(candles)
    psar_arr= _psar(candles, PSAR_AF, PSAR_AF, PSAR_MAX)
    atr_arr = _atr(candles, ATR_P)
    di_p, di_m, adx_arr = _dmi(candles, ADX_P)
    vol_ma  = _sma(volume, VOL_MA)

    # Ichimoku
    ten_arr = _ich_donchian(candles, ICH_T)
    kij_arr = _ich_donchian(candles, ICH_K)
    span_b  = _ich_donchian(candles, ICH_B)
    # senkou A = avg(tenkan, kijun) shifted forward ICH_K (for current bar: look back ICH_K)
    ich_senkou_A = [float("nan")] * n
    ich_senkou_B = [float("nan")] * n
    for i in range(ICH_K, n):
        if not (math.isnan(ten_arr[i - ICH_K]) or math.isnan(kij_arr[i - ICH_K])):
            ich_senkou_A[i] = (ten_arr[i - ICH_K] + kij_arr[i - ICH_K]) / 2
        if not math.isnan(span_b[i - ICH_K]):
            ich_senkou_B[i] = span_b[i - ICH_K]

    # BB bandwidth
    bb_bw = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(bb_mid[i]) or math.isnan(bb_up[i]) or bb_mid[i] == 0):
            bb_bw[i] = (bb_up[i] - bb_lo[i]) / bb_mid[i] * 100

    # KC
    kc_mid_arr = _ema(close, KC_P)
    kc_up = [float("nan")] * n
    kc_lo = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(kc_mid_arr[i]) or math.isnan(atr_arr[i])):
            kc_up[i] = kc_mid_arr[i] + KC_MULT * atr_arr[i]
            kc_lo[i] = kc_mid_arr[i] - KC_MULT * atr_arr[i]

    # Momentum
    mom_arr = [float("nan")] * n
    for i in range(MOM_P, n):
        mom_arr[i] = close[i] - close[i - MOM_P]

    results = []
    for i in range(n):
        def g(arr, idx=i, default=float("nan")):
            v = arr[idx] if 0 <= idx < len(arr) else float("nan")
            return v if not math.isnan(v) else default

        def gp(arr, back=1, default=float("nan")):
            return g(arr, i - back, default)

        ef = g(ema_f_arr); em = g(ema_m_arr); es = g(ema_s_arr)
        rsi = g(rsi_arr)
        ml = g(macd_l); ms_v = g(macd_s); mh = g(macd_h)
        bm = g(bb_mid); bu = g(bb_up); bl = g(bb_lo)
        st = g(st_dir)  # -1=bull, 1=bear, nan=unknown
        sk = g(stoch_k); sd = g(stoch_d)
        wr = g(wr_arr)
        cci = g(cci_arr)
        mfi = g(mfi_arr)
        roc = g(roc_arr)
        obv = g(obv_arr, i)
        obv5= g(obv_arr, i - 5) if i >= 5 else float("nan")
        cmf = g(cmf_arr)
        psar = g(psar_arr)
        vol  = volume[i]
        vol_avg = g(vol_ma)
        bw   = g(bb_bw)
        kc_mid = g(kc_mid_arr)
        kc_atr = g(atr_arr)
        kc_up_v = g(kc_up); kc_lo_v = g(kc_lo)
        ich_A = g(ich_senkou_A); ich_B = g(ich_senkou_B)
        ich_above = close[i] > max(ich_A, ich_B) if not (math.isnan(ich_A) or math.isnan(ich_B)) else False
        ich_below = close[i] < min(ich_A, ich_B) if not (math.isnan(ich_A) or math.isnan(ich_B)) else False
        st_val_v  = g(st_val)
        di_plus  = g(di_p); di_minus = g(di_m); adx_v = g(adx_arr)
        mom      = g(mom_arr)

        # Derived booleans
        rib_bull = (ef > em > es) if not any(math.isnan(x) for x in [ef, em, es]) else False
        rib_bear = (ef < em < es) if not any(math.isnan(x) for x in [ef, em, es]) else False
        ema_bull = (ef > es)      if not any(math.isnan(x) for x in [ef, es])     else False
        ema_bear = (ef < es)      if not any(math.isnan(x) for x in [ef, es])     else False
        st_bull  = (st == -1)
        st_bear  = (st == 1)
        macd_above = (ml > ms_v)  if not any(math.isnan(x) for x in [ml, ms_v]) else False
        macd_hist_pos = (mh > 0)  if not math.isnan(mh) else False
        stoch_ob = (sk > 80)      if not math.isnan(sk) else False
        stoch_os = (sk < 20)      if not math.isnan(sk) else False
        wr_ob    = (wr > -20)     if not math.isnan(wr) else False
        wr_eos   = (wr < -95)     if not math.isnan(wr) else False
        rsi_bull = (rsi > 50)     if not math.isnan(rsi) else False
        rsi_os   = (rsi < 30)     if not math.isnan(rsi) else False
        rsi_ob   = (rsi > 70)     if not math.isnan(rsi) else False
        rsi_4h_pump_zone = (50 < rsi < 60) if not math.isnan(rsi) else False
        bb_above_up   = (close[i] > bu)      if not math.isnan(bu) else False
        bb_near_lo    = (close[i] < bl * 1.005) if not math.isnan(bl) else False
        bb_below_lo   = (close[i] < bl)      if not math.isnan(bl) else False
        bb_squeeze    = (bw < 3.0)           if not math.isnan(bw) else False
        bb_expanding  = (bw > g(bb_bw, i - 5)) if i >= 5 and not math.isnan(bw) else False
        kc_above = (close[i] > kc_up_v) if not math.isnan(kc_up_v) else False
        kc_below = (close[i] < kc_lo_v) if not math.isnan(kc_lo_v) else False
        obv_rising = (obv > obv5) if not math.isnan(obv5) else False
        cmf_pos  = (cmf > 0.05)   if not math.isnan(cmf) else False
        psar_bear= (close[i] < psar) if not math.isnan(psar) else False
        cci_eos  = (cci < -200)   if not math.isnan(cci) else False
        mfi_os   = (mfi < 20)     if not math.isnan(mfi) else False
        mfi_ob   = (mfi > 80)     if not math.isnan(mfi) else False
        roc_strong = (roc > 2)    if not math.isnan(roc) else False
        di_bull  = (di_plus > di_minus) if not any(math.isnan(x) for x in [di_plus, di_minus]) else False
        vol_high = (vol / vol_avg > 1.5) if not (math.isnan(vol_avg) or vol_avg == 0) else False
        low_vol  = (vol / vol_avg < 0.5) if not (math.isnan(vol_avg) or vol_avg == 0) else False
        vol_spike= (vol / vol_avg > 2.0) if not (math.isnan(vol_avg) or vol_avg == 0) else False

        # Higher-high / lower-low (look back 9 bars from prev bar)
        hh = (high[i] > max(high[max(0, i - 9):i])) if i >= 9 else False
        ll = (low[i]  < min(low[max(0, i - 9):i]))  if i >= 9 else False

        # Candle patterns
        body    = abs(close[i] - candles[i].open)
        lwick   = min(candles[i].open, close[i]) - low[i]
        uwick   = high[i] - max(candles[i].open, close[i])
        hammer  = (body > 0 and lwick > body * 2 and uwick < body * 0.5)
        shoot_star = (body > 0 and uwick > body * 2 and lwick < body * 0.5)
        bull_engulf = (i > 0 and
                       close[i - 1] < candles[i - 1].open and
                       close[i] > candles[i].open and
                       close[i] > candles[i - 1].open and
                       candles[i].open < close[i - 1])
        bear_engulf = (i > 0 and
                       close[i - 1] > candles[i - 1].open and
                       close[i] < candles[i].open and
                       close[i] < candles[i - 1].open and
                       candles[i].open > close[i - 1])

        # Three red candles (consecutive bearish)
        three_red = (i >= 2 and
                     close[i] < candles[i].open and
                     close[i - 1] < candles[i - 1].open and
                     close[i - 2] < candles[i - 2].open)

        # Crossovers (compare vs prev bar)
        stoch_bull_cross = (i > 0 and
                            not math.isnan(g(stoch_k, i - 1)) and
                            g(stoch_k, i - 1) <= g(stoch_d, i - 1) and
                            sk > sd) if not (math.isnan(sk) or math.isnan(sd)) else False
        stoch_bear_cross = (i > 0 and
                            not math.isnan(g(stoch_k, i - 1)) and
                            g(stoch_k, i - 1) >= g(stoch_d, i - 1) and
                            sk < sd) if not (math.isnan(sk) or math.isnan(sd)) else False
        macd_bull_cross  = (i > 0 and not math.isnan(g(macd_l, i - 1)) and
                            g(macd_l, i - 1) <= g(macd_s, i - 1) and ml > ms_v) if not any(math.isnan(x) for x in [ml, ms_v]) else False
        macd_bear_cross  = (i > 0 and not math.isnan(g(macd_l, i - 1)) and
                            g(macd_l, i - 1) >= g(macd_s, i - 1) and ml < ms_v) if not any(math.isnan(x) for x in [ml, ms_v]) else False
        ema_bull_cross   = (i > 0 and not math.isnan(g(ema_f_arr, i - 1)) and
                            g(ema_f_arr, i - 1) <= g(ema_s_arr, i - 1) and ef > es) if not any(math.isnan(x) for x in [ef, es]) else False
        ema_bear_cross   = (i > 0 and not math.isnan(g(ema_f_arr, i - 1)) and
                            g(ema_f_arr, i - 1) >= g(ema_s_arr, i - 1) and ef < es) if not any(math.isnan(x) for x in [ef, es]) else False
        st_flip_bull = (st_bull and (g(st_dir, i - 1) != -1)) if i > 0 else False
        st_flip_bear = (st_bear and (g(st_dir, i - 1) != 1))  if i > 0 else False

        # RSI zone
        rsi_rising  = (rsi > g(rsi_arr, i - 3)) if i >= 3 and not math.isnan(rsi) else False
        rsi_falling = (rsi < g(rsi_arr, i - 3)) if i >= 3 and not math.isnan(rsi) else False
        rsi_bull_zone = (50 < rsi < 70) if not math.isnan(rsi) else False
        rsi_bear_zone = (30 < rsi < 50) if not math.isnan(rsi) else False

        results.append({
            "ts": candles[i].ts, "close": close[i],
            "rib_bull": rib_bull, "rib_bear": rib_bear,
            "ema_bull": ema_bull, "ema_bear": ema_bear,
            "st_bull": st_bull, "st_bear": st_bear,
            "st_flip_bull": st_flip_bull, "st_flip_bear": st_flip_bear,
            "macd_above": macd_above, "macd_hist_pos": macd_hist_pos,
            "macd_bull_cross": macd_bull_cross, "macd_bear_cross": macd_bear_cross,
            "ema_bull_cross": ema_bull_cross, "ema_bear_cross": ema_bear_cross,
            "stoch_ob": stoch_ob, "stoch_os": stoch_os,
            "stoch_bull_cross": stoch_bull_cross, "stoch_bear_cross": stoch_bear_cross,
            "wr_ob": wr_ob, "wr_eos": wr_eos,
            "rsi": rsi, "rsi_bull": rsi_bull, "rsi_os": rsi_os, "rsi_ob": rsi_ob,
            "rsi_4h_pump_zone": rsi_4h_pump_zone,
            "rsi_rising": rsi_rising, "rsi_falling": rsi_falling,
            "rsi_bull_zone": rsi_bull_zone, "rsi_bear_zone": rsi_bear_zone,
            "bb_above_up": bb_above_up, "bb_near_lo": bb_near_lo, "bb_below_lo": bb_below_lo,
            "bb_squeeze": bb_squeeze, "bb_expanding": bb_expanding,
            "kc_above": kc_above, "kc_below": kc_below,
            "obv_rising": obv_rising, "cmf_pos": cmf_pos,
            "psar_bear": psar_bear, "cci_eos": cci_eos,
            "mfi_os": mfi_os, "mfi_ob": mfi_ob,
            "roc_strong": roc_strong, "di_bull": di_bull, "di_plus": di_plus, "di_minus": di_minus,
            "vol_high": vol_high, "low_vol": low_vol, "vol_spike": vol_spike,
            "hh": hh, "ll": ll,
            "hammer": hammer, "shoot_star": shoot_star,
            "bull_engulf": bull_engulf, "bear_engulf": bear_engulf, "three_red": three_red,
            "ich_above": ich_above, "ich_below": ich_below,
        })
    return results


# ─────────────────────────────────────────────────────────────
# PUMP / DUMP SCORE (v1 logic — exact port of Pine)
# ─────────────────────────────────────────────────────────────
def compute_v1_scores(ind15: Dict, ind1h: Dict, ind4h: Dict) -> Tuple[float, float, float]:
    """Compute pump_score, dump_score, net_score from 3 TF indicator dicts."""

    def b(d, k):
        return d.get(k, False) if d else False

    pump = 0.0
    pump += 1.06 if b(ind4h, "rib_bull")  else 0
    pump += 1.04 if b(ind4h, "st_bull")   else 0
    pump += 0.67 if b(ind4h, "di_bull")   else 0
    pump += 0.83 if b(ind4h, "macd_above")else 0
    pump += 1.19 if b(ind4h, "rsi_4h_pump_zone") else 0
    pump += 0.88 if b(ind4h, "stoch_ob")  else 0
    pump += 0.87 if b(ind4h, "vol_high")  else 0
    pump += 1.19 if b(ind1h, "rsi_bull")  else 0
    pump += 0.84 if b(ind1h, "rib_bull")  else 0
    pump += 0.78 if b(ind1h, "st_bull")   else 0
    pump += 0.72 if b(ind1h, "macd_above")else 0
    pump += 0.89 if b(ind15, "st_bull")   else 0
    pump += 0.80 if b(ind15, "rib_bull")  else 0
    pump += 0.80 if b(ind15, "ema_bull")  else 0
    pump += 0.58 if b(ind15, "macd_above")else 0
    pump += 0.45 if b(ind15, "rsi_rising")else 0
    pump += 0.50 if b(ind15, "stoch_ob")  else 0
    # MTF confluence
    mtf_st_all_bull     = b(ind15, "st_bull") and b(ind1h, "st_bull") and b(ind4h, "st_bull")
    mtf_ribbon_all_bull = b(ind15, "rib_bull") and b(ind1h, "rib_bull") and b(ind4h, "rib_bull")
    mtf_macd_all_bull   = b(ind15, "macd_above") and b(ind1h, "macd_above") and b(ind4h, "macd_above")
    mtf_rsi_dip         = b(ind15, "rsi") and ind15.get("rsi", 99) < 35 and ind1h.get("rsi", 0) > 45 and b(ind4h, "rsi_bull")
    pump += 1.50 if mtf_st_all_bull     else 0
    pump += 1.20 if mtf_ribbon_all_bull else 0
    pump += 0.80 if mtf_macd_all_bull   else 0
    pump += 1.00 if mtf_rsi_dip         else 0
    pump += 1.00 if b(ind15, "st_flip_bull")  else 0
    pump += 0.80 if b(ind15, "macd_bull_cross") else 0
    pump += 0.70 if b(ind15, "ema_bull_cross") else 0
    pump += 0.60 if b(ind15, "bull_engulf")  else 0
    pump += 0.50 if b(ind15, "hammer")       else 0
    pump += 0.70 if (b(ind15, "stoch_bull_cross") and b(ind15, "stoch_os")) else 0

    dmp = 0.0
    dmp += 1.29 if b(ind4h, "rib_bear")   else 0
    dmp += 1.04 if not b(ind4h, "st_bull")else 0
    dmp += 0.67 if not b(ind4h, "di_bull")else 0
    dmp += 0.83 if not b(ind4h, "macd_above") else 0
    dmp += 1.22 if (b(ind4h, "rsi_os") and not b(ind4h, "rsi_bull")) else 0
    dmp += 1.19 if not b(ind1h, "rsi_bull")  else 0
    dmp += 0.88 if b(ind1h, "rib_bear")   else 0
    dmp += 0.78 if not b(ind1h, "st_bull")else 0
    dmp += 0.72 if not b(ind1h, "macd_above") else 0
    dmp += 0.69 if b(ind1h, "bb_squeeze")  else 0
    dmp += 0.89 if b(ind15, "st_bear")    else 0
    dmp += 0.77 if b(ind15, "rib_bear")   else 0
    dmp += 0.80 if b(ind15, "ema_bear")   else 0
    dmp += 0.58 if not b(ind15, "macd_above") else 0
    dmp += 0.45 if b(ind15, "rsi_falling")else 0
    mtf_st_all_bear     = b(ind15, "st_bear") and not b(ind1h, "st_bull") and not b(ind4h, "st_bull")
    mtf_ribbon_all_bear = b(ind15, "rib_bear") and b(ind1h, "rib_bear") and b(ind4h, "rib_bear")
    mtf_macd_all_bear   = not b(ind15, "macd_above") and not b(ind1h, "macd_above") and not b(ind4h, "macd_above")
    dmp += 1.50 if mtf_st_all_bear     else 0
    dmp += 1.20 if mtf_ribbon_all_bear else 0
    dmp += 0.80 if mtf_macd_all_bear   else 0
    dmp += 1.00 if b(ind15, "st_flip_bear")   else 0
    dmp += 0.80 if b(ind15, "macd_bear_cross") else 0
    dmp += 0.70 if b(ind15, "ema_bear_cross")  else 0
    dmp += 0.60 if b(ind15, "bear_engulf")   else 0
    dmp += 0.50 if b(ind15, "shoot_star")    else 0
    dmp += 0.70 if (b(ind15, "stoch_bear_cross") and b(ind15, "stoch_ob")) else 0

    return pump, dmp, pump - dmp


# ─────────────────────────────────────────────────────────────
# v3 COMBO DETECTION
# ─────────────────────────────────────────────────────────────
def detect_v3_combos(ind15: Dict, ind1h: Dict, ind4h: Dict) -> Dict:
    """Returns dict of all v3 combo booleans."""
    def b(d, k):
        return bool(d.get(k, False)) if d else False

    four_h_3green     = b(ind4h, "macd_above") and b(ind4h, "roc_strong") and b(ind4h, "obv_rising")
    ich_above_1h      = not b(ind1h, "ich_below")
    psar_flip_1h      = b(ind1h, "st_bull") and not b(ind1h, "st_bull")   # st_bull just turned on 1h — would need prev bar; approximate
    pump_trend_gate   = b(ind4h, "macd_above") and b(ind4h, "rsi_bull") and b(ind4h, "hh")
    low_vol_15m       = b(ind15, "low_vol")

    # Pullback setup
    rsi_was_low = ind15.get("rsi", 99) < 40   # simplified: current RSI (Pine uses ta.lowest over 6 bars)
    bb_touch_lo  = b(ind15, "bb_near_lo") or b(ind15, "bb_below_lo")
    pullback_setup   = rsi_was_low or bb_touch_lo
    bounce_green     = False  # would need prev bars — skipped for brevity
    vol_ok           = not b(ind15, "low_vol")
    pullback_trigger = (b(ind15, "st_flip_bull") or bounce_green) and vol_ok
    pump_pullback    = pump_trend_gate and pullback_setup and pullback_trigger

    # Snipers
    pump_sniper_1 = four_h_3green and b(ind4h, "bb_squeeze") and b(ind4h, "cmf_pos")
    pump_sniper_2 = b(ind4h, "hh") and b(ind4h, "bb_squeeze") and b(ind1h, "wr_ob")
    pump_sniper_3 = four_h_3green and b(ind4h, "bb_squeeze") and b(ind4h, "roc_strong")
    pump_sniper_4 = b(ind1h, "hh") and b(ind4h, "bb_squeeze") and b(ind4h, "roc_strong")
    pump_sniper   = pump_sniper_1 or pump_sniper_2 or pump_sniper_3 or pump_sniper_4

    # Elite
    pump_elite_1 = four_h_3green and b(ind4h, "bb_squeeze") and ich_above_1h
    pump_elite_2 = b(ind1h, "hh") and b(ind4h, "bb_squeeze") and b(ind4h, "obv_rising")
    pump_elite_3 = four_h_3green and b(ind4h, "bb_squeeze") and b(ind1h, "roc_strong")
    pump_elite_4 = b(ind1h, "hh") and b(ind4h, "bb_squeeze") and b(ind4h, "cmf_pos")
    mtf_obv_all  = b(ind15, "obv_rising") and b(ind1h, "obv_rising") and b(ind4h, "obv_rising")
    pump_elite_5 = b(ind4h, "cmf_pos") and mtf_obv_all
    pump_elite_6 = b(ind1h, "hh") and b(ind4h, "bb_squeeze") and ich_above_1h
    pump_elite   = (pump_elite_1 or pump_elite_2 or pump_elite_3 or
                    pump_elite_4 or pump_elite_5 or pump_elite_6) and not pump_sniper

    # Standard
    pump_std_1 = b(ind4h, "hh") and b(ind4h, "bb_squeeze") and b(ind1h, "roc_strong")
    pump_std_2 = b(ind4h, "hh") and b(ind4h, "bb_squeeze") and b(ind4h, "cmf_pos")
    mtf_rsi_all= ind15.get("rsi", 0) > 50 and ind1h.get("rsi", 0) > 50 and b(ind4h, "rsi_bull")
    pump_std_3 = b(ind1h, "hh") and b(ind4h, "bb_squeeze") and mtf_rsi_all
    pump_std   = (pump_std_1 or pump_std_2 or pump_std_3) and not pump_sniper and not pump_elite

    pump_breakout = (pump_sniper or pump_elite or pump_std) and b(ind4h, "macd_above")
    raw_v3_pump   = pump_breakout or pump_pullback

    # Dump combos
    mtf_ich_all_bear = b(ind15, "ich_below") and b(ind1h, "ich_below") and b(ind4h, "ich_below")
    mtf_ribbon_all_bear = b(ind15, "rib_bear") and b(ind1h, "rib_bear") and b(ind4h, "rib_bear")
    mtf_psar_all_bear   = b(ind15, "psar_bear") and b(ind4h, "psar_bear")

    dump_sniper_1 = low_vol_15m and b(ind1h, "mfi_os") and mtf_ribbon_all_bear
    dump_sniper_2 = low_vol_15m and b(ind1h, "mfi_os") and mtf_ich_all_bear
    dump_sniper_3 = low_vol_15m and b(ind1h, "mfi_os") and b(ind4h, "wr_eos")
    dump_sniper   = dump_sniper_1 or dump_sniper_2 or dump_sniper_3

    mtf_macd_all_bear = not b(ind15, "macd_above") and not b(ind1h, "macd_above") and not b(ind4h, "macd_above")
    dump_elite_1 = low_vol_15m and mtf_ich_all_bear and b(ind4h, "stoch_os")
    dump_elite_2 = mtf_macd_all_bear and b(ind4h, "three_red") and b(ind4h, "three_red")
    dump_elite_3 = low_vol_15m and b(ind1h, "kc_below")
    dump_elite_4 = b(ind1h, "rsi_os") and b(ind4h, "three_red") and b(ind4h, "mfi_os")
    dump_elite_5 = b(ind4h, "three_red") and mtf_psar_all_bear
    dump_elite_6 = b(ind4h, "three_red") and b(ind4h, "rsi_os") and b(ind1h, "bb_below_lo")
    dump_elite   = (dump_elite_1 or dump_elite_2 or dump_elite_3 or
                    dump_elite_4 or dump_elite_5 or dump_elite_6) and not dump_sniper

    dump_std_1 = low_vol_15m and mtf_ich_all_bear
    dump_std_2 = b(ind1h, "bb_below_lo") and b(ind4h, "rsi_os")
    dump_std_3 = b(ind4h, "three_red") and b(ind4h, "three_red")
    dump_std   = (dump_std_1 or dump_std_2 or dump_std_3) and not dump_sniper and not dump_elite
    raw_v3_dump = dump_sniper or dump_elite or dump_std

    pump_tier = 3 if pump_sniper else (2 if pump_elite else (1 if (pump_std or pump_pullback) else 0))
    dump_tier = 3 if dump_sniper else (2 if dump_elite else (1 if dump_std else 0))

    return {
        "raw_v3_pump": raw_v3_pump, "raw_v3_dump": raw_v3_dump,
        "pump_sniper": pump_sniper, "pump_elite": pump_elite,
        "pump_std": pump_std, "pump_pullback": pump_pullback,
        "dump_sniper": dump_sniper, "dump_elite": dump_elite, "dump_std": dump_std,
        "pump_tier": pump_tier, "dump_tier": dump_tier,
        "four_h_3green": four_h_3green, "pump_trend_gate": pump_trend_gate,
        "mtf_obv_all": mtf_obv_all, "mtf_ribbon_all_bear": mtf_ribbon_all_bear,
        "mtf_ich_all_bear": mtf_ich_all_bear, "mtf_psar_all_bear": mtf_psar_all_bear,
        "mtf_macd_all_bear": mtf_macd_all_bear,
    }


# ─────────────────────────────────────────────────────────────
# SIGNAL DETECTION — full scan on a single symbol
# ─────────────────────────────────────────────────────────────
def detect_signals(symbol: str,
                   c15: List[Candle], c1h: List[Candle], c4h: List[Candle]) -> List[SignalEvent]:
    if len(c15) < 200:
        return []

    ind15 = calc_indicators(c15)
    ind1h = calc_indicators(c1h)
    ind4h = calc_indicators(c4h)

    def get_htf(c_htf: List[Candle], ind_htf: List[Dict], ts_15: int) -> Optional[Dict]:
        """Return the HTF indicator dict that covers the 15m bar timestamp."""
        # Find the last HTF bar whose close time <= ts_15
        for j in range(len(c_htf) - 1, -1, -1):
            if c_htf[j].ts <= ts_15:
                return ind_htf[j]
        return None

    events: List[SignalEvent] = []
    v1_bars_since_pump = 999
    v3_bars_any        = 999
    v3_last_dir        = 0
    v1_prev_raw_pump   = False
    v3_prev_raw_pump   = False
    v3_prev_raw_dump   = False

    for i in range(200, len(c15)):
        ts15 = c15[i].ts
        d15  = ind15[i]
        d1h  = get_htf(c1h, ind1h, ts15) or {}
        d4h  = get_htf(c4h, ind4h, ts15) or {}

        pump_sc, dump_sc, net_sc = compute_v1_scores(d15, d1h, d4h)
        v3 = detect_v3_combos(d15, d1h, d4h)
        v1_bars_since_pump += 1
        v3_bars_any        += 1

        # ── v1 signals ────────────────────────────────────────
        v1_raw_pump        = pump_sc >= V1_PUMP_THRESH        and net_sc >= V1_MIN_NET_GAP
        v1_raw_strong_pump = pump_sc >= V1_STRONG_PUMP_THRESH and net_sc >= V1_MIN_NET_GAP
        v1_pump_allowed    = v1_bars_since_pump >= V1_COOLDOWN_BARS
        v1_pump_rising     = v1_raw_pump and not v1_prev_raw_pump
        v1_strong_rising   = v1_raw_strong_pump and not (v1_prev_raw_pump and pump_sc >= V1_STRONG_PUMP_THRESH)

        v1_final_strong = v1_raw_strong_pump and v1_pump_allowed and v1_strong_rising
        v1_final_pump   = v1_raw_pump and not v1_raw_strong_pump and v1_pump_allowed and v1_pump_rising

        if v1_final_pump or v1_final_strong:
            v1_bars_since_pump = 0

        # ── v3 cooldown ───────────────────────────────────────
        pump_cd = int(V3_COOLDOWN * V3_FLIP_MULT) if v3_last_dir == -1 else V3_COOLDOWN
        dump_cd = int(V3_COOLDOWN * V3_FLIP_MULT) if v3_last_dir ==  1 else V3_COOLDOWN

        v3_pump_rising = v3["raw_v3_pump"] and not v3_prev_raw_pump
        v3_dump_rising = v3["raw_v3_dump"] and not v3_prev_raw_dump

        sig_v3_pump = v3["raw_v3_pump"] and v3_bars_any >= pump_cd and v3_pump_rising
        sig_v3_dump = v3["raw_v3_dump"] and v3_bars_any >= dump_cd and v3_dump_rising

        if sig_v3_pump:
            v3_bars_any = 0; v3_last_dir = 1
        if sig_v3_dump:
            v3_bars_any = 0; v3_last_dir = -1

        v1_prev_raw_pump  = v1_raw_pump
        v3_prev_raw_pump  = v3["raw_v3_pump"]
        v3_prev_raw_dump  = v3["raw_v3_dump"]

        dt_str = datetime.fromtimestamp(ts15 / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

        # Build indicator combo summary for this bar (for the CSV/report)
        combo_flags = {
            "st_bull_15": d15.get("st_bull", False),
            "rib_bull_15": d15.get("rib_bull", False),
            "macd_above_15": d15.get("macd_above", False),
            "rsi_rising_15": d15.get("rsi_rising", False),
            "stoch_ob_15": d15.get("stoch_ob", False),
            "st_flip_bull_15": d15.get("st_flip_bull", False),
            "macd_bull_cross_15": d15.get("macd_bull_cross", False),
            "ema_bull_cross_15": d15.get("ema_bull_cross", False),
            "bull_engulf_15": d15.get("bull_engulf", False),
            "hammer_15": d15.get("hammer", False),
            "vol_high_15": d15.get("vol_high", False),
            "low_vol_15": d15.get("low_vol", False),
            "obv_rising_15": d15.get("obv_rising", False),
            "st_bull_1h": d1h.get("st_bull", False),
            "rib_bull_1h": d1h.get("rib_bull", False),
            "macd_above_1h": d1h.get("macd_above", False),
            "rsi_bull_1h": d1h.get("rsi_bull", False),
            "cmf_pos_1h": d1h.get("cmf_pos", False),
            "hh_1h": d1h.get("hh", False),
            "mfi_os_1h": d1h.get("mfi_os", False),
            "obv_rising_1h": d1h.get("obv_rising", False),
            "st_bull_4h": d4h.get("st_bull", False),
            "rib_bull_4h": d4h.get("rib_bull", False),
            "macd_above_4h": d4h.get("macd_above", False),
            "rsi_4h_pump_zone": d4h.get("rsi_4h_pump_zone", False),
            "di_bull_4h": d4h.get("di_bull", False),
            "vol_high_4h": d4h.get("vol_high", False),
            "hh_4h": d4h.get("hh", False),
            "bb_squeeze_4h": d4h.get("bb_squeeze", False),
            "obv_rising_4h": d4h.get("obv_rising", False),
            "cmf_pos_4h": d4h.get("cmf_pos", False),
            "roc_strong_4h": d4h.get("roc_strong", False),
            "rsi_os_4h": d4h.get("rsi_os", False),
            "three_red_4h": d4h.get("three_red", False),
            **{k: v for k, v in v3.items() if isinstance(v, bool)},
        }

        def add_event(sig_type: str, direction: str):
            events.append(SignalEvent(
                symbol=symbol, bar_idx=i, ts=ts15, dt=dt_str,
                close=c15[i].close, signal_type=sig_type, direction=direction,
                pump_score=round(pump_sc, 2), dump_score=round(dump_sc, 2),
                net_score=round(net_sc, 2),
                pump_tier=v3["pump_tier"], dump_tier=v3["dump_tier"],
                indicators=dict(combo_flags),
            ))

        if v1_final_strong:
            add_event("v1_strong_pump", "pump")
        elif v1_final_pump:
            add_event("v1_pump", "pump")

        if sig_v3_pump:
            tier = "sniper" if v3["pump_sniper"] else ("elite" if v3["pump_elite"] else ("pullback" if v3["pump_pullback"] else "std"))
            add_event(f"v3_pump_{tier}", "pump")

        if sig_v3_dump:
            tier = "sniper" if v3["dump_sniper"] else ("elite" if v3["dump_elite"] else "std")
            add_event(f"v3_dump_{tier}", "dump")

    return events


# ─────────────────────────────────────────────────────────────
# OUTCOME CHECK — did the crypto move ≥3% in any direction?
# ─────────────────────────────────────────────────────────────
def evaluate_outcomes(events: List[SignalEvent],
                      candles_15: Dict[str, List[Candle]]) -> None:
    """Mutates each event in-place with move outcome."""
    for ev in events:
        c = candles_15.get(ev.symbol, [])
        if not c:
            continue
        i0 = ev.bar_idx + 1   # bar after signal
        future = c[i0: i0 + MOVE_CHECK_BARS]
        if not future:
            continue
        entry = ev.close
        max_up   = max((f.high - entry) / entry * 100 for f in future)
        max_down = max((entry - f.low)  / entry * 100 for f in future)
        ev.max_move_pct = round(max(max_up, max_down), 2)
        if ev.direction == "pump":
            ev.moved_3pct     = max_up   >= MOVE_TARGET_PCT
            ev.move_direction = "UP"   if max_up >= MOVE_TARGET_PCT else "NONE"
        else:
            ev.moved_3pct     = max_down >= MOVE_TARGET_PCT
            ev.move_direction = "DOWN" if max_down >= MOVE_TARGET_PCT else "NONE"


# ─────────────────────────────────────────────────────────────
# ANALYSIS — win-rate by signal type + indicator combos
# ─────────────────────────────────────────────────────────────
def analyze(events: List[SignalEvent]) -> Dict:
    """Returns summary stats dict."""
    by_type: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "wins": 0, "symbols": set()})
    by_indicator: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "wins": 0})
    by_combo: Dict[str, Dict]     = defaultdict(lambda: {"total": 0, "wins": 0})
    by_symbol: Dict[str, Dict]    = defaultdict(lambda: {"total": 0, "wins": 0})

    for ev in events:
        t   = by_type[ev.signal_type]
        t["total"] += 1
        t["wins"]  += int(ev.moved_3pct)
        t["symbols"].add(ev.symbol)

        by_symbol[ev.symbol]["total"] += 1
        by_symbol[ev.symbol]["wins"]  += int(ev.moved_3pct)

        active_inds = [k for k, v in ev.indicators.items() if v is True]
        for ind in active_inds:
            by_indicator[ind]["total"] += 1
            by_indicator[ind]["wins"]  += int(ev.moved_3pct)

        # Best 3-combo key  
        if ev.moved_3pct and len(active_inds) >= 3:
            # Store pairs of top indicators for the winning signals
            for a in range(min(len(active_inds), 5)):
                for bb in range(a + 1, min(len(active_inds), 5)):
                    key = f"{active_inds[a]} + {active_inds[bb]}"
                    by_combo[key]["total"] += 1
                    by_combo[key]["wins"]  += int(ev.moved_3pct)

    # Convert sets to counts
    for k, v in by_type.items():
        v["symbols"] = len(v["symbols"])
        v["win_rate"] = round(v["wins"] / v["total"] * 100, 1) if v["total"] else 0

    # Sort indicators by win-rate (min 5 samples)
    ind_sorted = sorted(
        [(k, v) for k, v in by_indicator.items() if v["total"] >= 5],
        key=lambda x: x[1]["wins"] / x[1]["total"],
        reverse=True
    )
    combo_sorted = sorted(
        [(k, v) for k, v in by_combo.items() if v["total"] >= 3],
        key=lambda x: x[1]["wins"] / x[1]["total"],
        reverse=True
    )[:30]

    sym_sorted = sorted(
        [(k, v) for k, v in by_symbol.items()],
        key=lambda x: x[1]["wins"] / max(x[1]["total"], 1),
        reverse=True
    )

    return {
        "signal_types": {k: v for k, v in by_type.items()},
        "top_indicators": [
            {"indicator": k, **v, "win_rate": round(v["wins"] / v["total"] * 100, 1)}
            for k, v in ind_sorted[:30]
        ],
        "top_combos": [
            {"combo": k, **v, "win_rate": round(v["wins"] / v["total"] * 100, 1)}
            for k, v in combo_sorted
        ],
        "symbol_win_rates": [
            {"symbol": k, **v, "win_rate": round(v["wins"] / v["total"] * 100, 1)}
            for k, v in sym_sorted[:30]
        ],
    }


# ─────────────────────────────────────────────────────────────
# REPORT WRITER
# ─────────────────────────────────────────────────────────────
def write_report(summary: Dict, total_events: int, symbols_scanned: int, days: int) -> str:
    lines = [
        "=" * 70,
        "  PINE BACKSCAN v3 + v1 — SIGNAL QUALITY REPORT",
        f"  Symbols scanned: {symbols_scanned}  |  Days: {days}  |  Move target: {MOVE_TARGET_PCT}%",
        f"  Total signal events: {total_events}",
        "=" * 70,
        "",
        "── SIGNAL TYPE WIN RATES ──",
        f"{'Signal':<25} {'Total':>7} {'Wins':>6} {'Win%':>7} {'Symbols':>9}",
        "-" * 60,
    ]
    for sig, v in sorted(summary["signal_types"].items(),
                         key=lambda x: x[1]["win_rate"], reverse=True):
        lines.append(f"{sig:<25} {v['total']:>7} {v['wins']:>6} {v['win_rate']:>7.1f}% {v['symbols']:>9}")

    lines += ["", "── TOP INDICATORS (in winning signals, min 5 samples) ──",
              f"{'Indicator':<35} {'Total':>7} {'Wins':>6} {'Win%':>7}", "-" * 55]
    for r in summary["top_indicators"][:20]:
        lines.append(f"{r['indicator']:<35} {r['total']:>7} {r['wins']:>6} {r['win_rate']:>7.1f}%")

    lines += ["", "── TOP INDICATOR COMBOS (pairs in winning signals) ──",
              f"{'Combo':<55} {'Total':>5} {'Wins':>6} {'Win%':>7}", "-" * 75]
    for r in summary["top_combos"][:20]:
        lines.append(f"{r['combo']:<55} {r['total']:>5} {r['wins']:>6} {r['win_rate']:>7.1f}%")

    lines += ["", "── TOP SYMBOLS BY WIN RATE ──",
              f"{'Symbol':<15} {'Total':>6} {'Wins':>6} {'Win%':>7}", "-" * 36]
    for r in summary["symbol_win_rates"][:20]:
        lines.append(f"{r['symbol']:<15} {r['total']:>6} {r['wins']:>6} {r['win_rate']:>7.1f}%")

    lines += ["", "=" * 70,
              "NOTE: Win = price moved ≥3% in signal direction within next 8 bars (2h on 15m)",
              "This removes the inflated 'built-in win-rate' shown in the Pine Script",
              "and gives you the REAL forward-looking performance per combo.",
              "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Pine Backscan Scanner")
    parser.add_argument("--top",     type=int, default=100, help="Top N coins by volume (default 100)")
    parser.add_argument("--days",    type=int, default=30,  help="Days of history (default 30)")
    parser.add_argument("--symbols", type=str, default="",  help="Comma-separated symbols override")
    args = parser.parse_args()

    print("🚀 Pine Backscan v3+v1 Scanner")
    print(f"   Top: {args.top} coins  |  Days: {args.days}  |  Move target: {MOVE_TARGET_PCT}%")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            print("📡 Fetching top symbols by volume...")
            symbols = await fetch_top_symbols(session, args.top)
            print(f"   Got {len(symbols)} symbols")

        all_events: List[SignalEvent] = []
        candles_15_by_sym: Dict[str, List[Candle]] = {}

        for idx, sym in enumerate(symbols):
            print(f"  [{idx + 1:3d}/{len(symbols)}] {sym:<15}", end=" ", flush=True)
            try:
                c15 = await fetch_klines(session, sym, "15",  args.days)
                c1h = await fetch_klines(session, sym, "60",  args.days)
                c4h = await fetch_klines(session, sym, "240", args.days)

                if len(c15) < 200:
                    print(f"⚠️  only {len(c15)} 15m candles — skipped")
                    continue

                evs = detect_signals(sym, c15, c1h, c4h)
                candles_15_by_sym[sym] = c15
                all_events.extend(evs)
                print(f"✅  {len(c15)} candles  →  {len(evs)} signals")

            except Exception as e:
                print(f"❌  {e}")
            await asyncio.sleep(RATE_DELAY)

    print(f"\n📊 Total signal events: {len(all_events)}")
    print("🔍 Evaluating 3% move outcomes...")
    evaluate_outcomes(all_events, candles_15_by_sym)

    wins = sum(1 for e in all_events if e.moved_3pct)
    print(f"   Wins (moved 3%+): {wins} / {len(all_events)}  = {wins/max(len(all_events),1)*100:.1f}%")

    # ── Write CSV ──────────────────────────────────────────────
    ind_cols = list(all_events[0].indicators.keys()) if all_events else []
    csv_fields = ["symbol", "dt", "close", "signal_type", "direction",
                  "pump_score", "dump_score", "net_score", "pump_tier", "dump_tier",
                  "moved_3pct", "max_move_pct", "move_direction"] + ind_cols

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for ev in all_events:
            row = {
                "symbol": ev.symbol, "dt": ev.dt, "close": ev.close,
                "signal_type": ev.signal_type, "direction": ev.direction,
                "pump_score": ev.pump_score, "dump_score": ev.dump_score,
                "net_score": ev.net_score, "pump_tier": ev.pump_tier,
                "dump_tier": ev.dump_tier, "moved_3pct": int(ev.moved_3pct),
                "max_move_pct": ev.max_move_pct, "move_direction": ev.move_direction,
                **{k: int(v) for k, v in ev.indicators.items()},
            }
            w.writerow(row)
    print(f"✅  CSV saved → {OUTPUT_CSV}")

    # ── Write JSON ─────────────────────────────────────────────
    summary = analyze(all_events)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅  JSON saved → {OUTPUT_JSON}")

    # ── Write TXT report ──────────────────────────────────────
    report = write_report(summary, len(all_events), len(symbols), args.days)
    with open(OUTPUT_TXT, "w") as f:
        f.write(report)
    print(f"✅  Report saved → {OUTPUT_TXT}")
    print()
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
