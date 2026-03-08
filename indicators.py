#!/usr/bin/env python3
"""
indicators.py - Complete Indicator Calculation Engine
=====================================================
All 30+ indicators with numpy-optimized calculations.
Each function takes raw OHLCV arrays and returns indicator values.
Matches your existing bot's indicator logic exactly.
"""

import numpy as np
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ============================================================
# PRIMITIVE MATH HELPERS
# ============================================================

def _sma(data: list, period: int) -> list:
    """Simple Moving Average."""
    if len(data) < period:
        return []
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(float("nan"))
        else:
            result.append(sum(data[i - period + 1:i + 1]) / period)
    return result


def _ema(data: list, period: int) -> list:
    """Exponential Moving Average."""
    if len(data) < period:
        return []
    result = [float("nan")] * len(data)
    # seed with SMA
    result[period - 1] = sum(data[:period]) / period
    mult = 2.0 / (period + 1)
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]
    return result


def _rsi(closes: list, period: int = 14) -> list:
    """Wilder's RSI."""
    n = len(closes)
    result = [float("nan")] * n
    if n < period + 1:
        return result
    gains = []
    losses = []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(abs(min(delta, 0)))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100 - (100 / (1 + rs))
    return result


def _macd(closes: list, fast: int = 12, slow: int = 26, sig: int = 9):
    """MACD line, signal, histogram."""
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    n = len(closes)
    macd_line = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(ema_f[i]) or math.isnan(ema_s[i])):
            macd_line[i] = ema_f[i] - ema_s[i]
    # signal line = EMA of macd_line
    valid_macd = [v if not math.isnan(v) else 0 for v in macd_line]
    signal = _ema(valid_macd, sig)
    histogram = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(macd_line[i]) or math.isnan(signal[i])):
            histogram[i] = macd_line[i] - signal[i]
    return macd_line, signal, histogram


def _bollinger(closes: list, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands: (upper, mid, lower, bandwidth%)."""
    n = len(closes)
    mid = _sma(closes, period)
    upper = [float("nan")] * n
    lower = [float("nan")] * n
    bw = [float("nan")] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1:i + 1]
        std = (sum((x - mid[i]) ** 2 for x in window) / period) ** 0.5
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
        if mid[i] > 0:
            bw[i] = (upper[i] - lower[i]) / mid[i] * 100
    return upper, mid, lower, bw


def _atr(candles: list, period: int = 14) -> list:
    """Average True Range."""
    n = len(candles)
    tr = [0.0] * n
    for i in range(1, n):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    tr[0] = candles[0].high - candles[0].low
    result = [float("nan")] * n
    if n < period:
        return result
    result[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def _supertrend(candles: list, period: int = 10, mult: float = 3.0):
    """Supertrend: returns (value, direction) where direction: 1=bull, -1=bear."""
    n = len(candles)
    atr = _atr(candles, period)
    st_val = [float("nan")] * n
    st_dir = [0] * n
    for i in range(period, n):
        if math.isnan(atr[i]):
            continue
        hl2 = (candles[i].high + candles[i].low) / 2
        up = hl2 + mult * atr[i]
        dn = hl2 - mult * atr[i]
        # Adjust bands
        if i > period and not math.isnan(st_val[i - 1]):
            prev_up = st_val[i - 1] if st_dir[i - 1] == -1 else up
            prev_dn = st_val[i - 1] if st_dir[i - 1] == 1 else dn
            if candles[i - 1].close <= prev_up:
                up = min(up, prev_up)
            if candles[i - 1].close >= prev_dn:
                dn = max(dn, prev_dn)
        # Direction
        if candles[i].close > up:
            st_dir[i] = 1
            st_val[i] = dn
        elif candles[i].close < dn:
            st_dir[i] = -1
            st_val[i] = up
        else:
            st_dir[i] = st_dir[i - 1] if i > period else 0
            st_val[i] = dn if st_dir[i] == 1 else up
    return st_val, st_dir


def _stoch_rsi(closes: list, rsi_p: int = 14, stoch_p: int = 14, k_p: int = 3, d_p: int = 3):
    """Stochastic RSI: returns (k_line, d_line)."""
    rsi = _rsi(closes, rsi_p)
    n = len(closes)
    stoch_rsi_raw = [float("nan")] * n
    for i in range(stoch_p - 1, n):
        window = [rsi[j] for j in range(i - stoch_p + 1, i + 1) if not math.isnan(rsi[j])]
        if len(window) < stoch_p:
            continue
        mn = min(window)
        mx = max(window)
        if mx - mn > 0:
            stoch_rsi_raw[i] = ((rsi[i] - mn) / (mx - mn)) * 100
        else:
            stoch_rsi_raw[i] = 50.0
    k_line = _sma_nan(stoch_rsi_raw, k_p)
    d_line = _sma_nan(k_line, d_p)
    return k_line, d_line


def _sma_nan(data: list, period: int) -> list:
    """SMA that skips NaN values properly."""
    n = len(data)
    result = [float("nan")] * n
    for i in range(period - 1, n):
        window = [data[j] for j in range(i - period + 1, i + 1) if not math.isnan(data[j])]
        if len(window) == period:
            result[i] = sum(window) / period
    return result


def _williams_r(candles: list, period: int = 14) -> list:
    """Williams %R."""
    n = len(candles)
    result = [float("nan")] * n
    for i in range(period - 1, n):
        hh = max(candles[j].high for j in range(i - period + 1, i + 1))
        ll = min(candles[j].low for j in range(i - period + 1, i + 1))
        if hh - ll > 0:
            result[i] = ((hh - candles[i].close) / (hh - ll)) * -100
    return result


def _cci(candles: list, period: int = 20) -> list:
    """Commodity Channel Index."""
    n = len(candles)
    tp = [(c.high + c.low + c.close) / 3 for c in candles]
    result = [float("nan")] * n
    for i in range(period - 1, n):
        window = tp[i - period + 1:i + 1]
        mean = sum(window) / period
        md = sum(abs(x - mean) for x in window) / period
        if md > 0:
            result[i] = (tp[i] - mean) / (0.015 * md)
    return result


def _mfi(candles: list, period: int = 14) -> list:
    """Money Flow Index."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period + 1:
        return result
    tp = [(c.high + c.low + c.close) / 3 for c in candles]
    mf = [tp[i] * candles[i].volume for i in range(n)]
    for i in range(period, n):
        pos = 0.0
        neg = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0 and tp[j] > tp[j - 1]:
                pos += mf[j]
            elif j > 0 and tp[j] < tp[j - 1]:
                neg += mf[j]
        if neg == 0:
            result[i] = 100.0
        else:
            result[i] = 100 - (100 / (1 + pos / neg))
    return result


def _roc(closes: list, period: int = 12) -> list:
    """Rate of Change."""
    n = len(closes)
    result = [float("nan")] * n
    for i in range(period, n):
        if closes[i - period] != 0:
            result[i] = ((closes[i] - closes[i - period]) / closes[i - period]) * 100
    return result


def _momentum(closes: list, period: int = 10) -> list:
    """Price Momentum."""
    n = len(closes)
    result = [float("nan")] * n
    for i in range(period, n):
        result[i] = closes[i] - closes[i - period]
    return result


def _obv(candles: list) -> list:
    """On-Balance Volume."""
    n = len(candles)
    result = [0.0] * n
    for i in range(1, n):
        if candles[i].close > candles[i - 1].close:
            result[i] = result[i - 1] + candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            result[i] = result[i - 1] - candles[i].volume
        else:
            result[i] = result[i - 1]
    return result


def _cmf(candles: list, period: int = 20) -> list:
    """Chaikin Money Flow."""
    n = len(candles)
    result = [float("nan")] * n
    for i in range(period - 1, n):
        mfv_sum = 0.0
        vol_sum = 0.0
        for j in range(i - period + 1, i + 1):
            hl = candles[j].high - candles[j].low
            if hl > 0:
                mfm = ((candles[j].close - candles[j].low) - (candles[j].high - candles[j].close)) / hl
            else:
                mfm = 0
            mfv_sum += mfm * candles[j].volume
            vol_sum += candles[j].volume
        if vol_sum > 0:
            result[i] = mfv_sum / vol_sum
    return result


def _vwap(candles: list, period: int = 200) -> list:
    """Rolling VWAP over N candles."""
    n = len(candles)
    result = [float("nan")] * n
    for i in range(min(period, n) - 1, n):
        start = max(0, i - period + 1)
        tp_vol = sum(
            ((candles[j].high + candles[j].low + candles[j].close) / 3) * candles[j].volume
            for j in range(start, i + 1)
        )
        vol = sum(candles[j].volume for j in range(start, i + 1))
        if vol > 0:
            result[i] = tp_vol / vol
    return result


def _keltner(candles: list, period: int = 20, mult: float = 1.5):
    """Keltner Channels: (upper, mid, lower)."""
    closes = [c.close for c in candles]
    mid = _ema(closes, period)
    atr = _atr(candles, period)
    n = len(candles)
    upper = [float("nan")] * n
    lower = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(mid[i]) or math.isnan(atr[i])):
            upper[i] = mid[i] + mult * atr[i]
            lower[i] = mid[i] - mult * atr[i]
    return upper, mid, lower


def _ichimoku(candles: list, tenkan_p: int = 9, kijun_p: int = 26, senkou_b_p: int = 52):
    """Ichimoku Cloud: returns dict with cloud info at last bar."""
    n = len(candles)
    if n < senkou_b_p + kijun_p:
        return None

    def donchian_mid(start, end):
        hh = max(candles[j].high for j in range(start, end))
        ll = min(candles[j].low for j in range(start, end))
        return (hh + ll) / 2

    i = n - 1
    tenkan = donchian_mid(max(0, i - tenkan_p + 1), i + 1)
    kijun = donchian_mid(max(0, i - kijun_p + 1), i + 1)

    # Senkou Span A = (tenkan + kijun) / 2 shifted forward kijun_p bars
    # For current analysis we look at the cloud that was projected kijun_p bars ago
    if i >= kijun_p:
        past_i = i - kijun_p
        past_tenkan = donchian_mid(max(0, past_i - tenkan_p + 1), past_i + 1)
        past_kijun = donchian_mid(max(0, past_i - kijun_p + 1), past_i + 1)
        senkou_a = (past_tenkan + past_kijun) / 2
    else:
        senkou_a = (tenkan + kijun) / 2

    if i >= kijun_p and i - kijun_p >= senkou_b_p:
        past_i = i - kijun_p
        senkou_b = donchian_mid(max(0, past_i - senkou_b_p + 1), past_i + 1)
    else:
        senkou_b = donchian_mid(max(0, i - senkou_b_p + 1), i + 1)

    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    cl = candles[i].close

    if cl > cloud_top:
        price_vs_cloud = "above"
    elif cl < cloud_bot:
        price_vs_cloud = "below"
    else:
        price_vs_cloud = "inside"

    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "cloud_top": cloud_top,
        "cloud_bot": cloud_bot,
        "price_vs_cloud": price_vs_cloud,
        "tk_cross_bull": tenkan > kijun,
        "cloud_green": senkou_a > senkou_b,
    }


def _parabolic_sar(candles: list, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2) -> list:
    """Parabolic SAR: returns list of direction (1=bull, -1=bear)."""
    n = len(candles)
    if n < 2:
        return [0] * n
    result = [0] * n
    trend = 1  # 1 = up, -1 = down
    af = af_start
    ep = candles[0].high
    sar = candles[0].low

    for i in range(1, n):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)

        if trend == 1:
            if candles[i].low < sar:
                trend = -1
                sar = ep
                ep = candles[i].low
                af = af_start
            else:
                if candles[i].high > ep:
                    ep = candles[i].high
                    af = min(af + af_step, af_max)
                sar = min(sar, candles[i - 1].low)
                if i >= 2:
                    sar = min(sar, candles[i - 2].low)
        else:
            if candles[i].high > sar:
                trend = 1
                sar = ep
                ep = candles[i].high
                af = af_start
            else:
                if candles[i].low < ep:
                    ep = candles[i].low
                    af = min(af + af_step, af_max)
                sar = max(sar, candles[i - 1].high)
                if i >= 2:
                    sar = max(sar, candles[i - 2].high)

        result[i] = trend
    return result


def _adx(candles: list, period: int = 14):
    """ADX with +DI and -DI: returns (plus_di, minus_di, adx)."""
    n = len(candles)
    plus_di = [float("nan")] * n
    minus_di = [float("nan")] * n
    adx_arr = [float("nan")] * n
    if n < period + 1:
        return plus_di, minus_di, adx_arr

    # True Range, +DM, -DM
    tr = [0.0] * n
    pdm = [0.0] * n
    mdm = [0.0] * n
    for i in range(1, n):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
        up = h - candles[i - 1].high
        dn = candles[i - 1].low - l
        pdm[i] = up if up > dn and up > 0 else 0
        mdm[i] = dn if dn > up and dn > 0 else 0

    # Smoothed TR, +DM, -DM using Wilder's
    str_val = sum(tr[1:period + 1])
    spdm = sum(pdm[1:period + 1])
    smdm = sum(mdm[1:period + 1])

    dx_vals = []
    for i in range(period, n):
        if i == period:
            pass
        else:
            str_val = str_val - str_val / period + tr[i]
            spdm = spdm - spdm / period + pdm[i]
            smdm = smdm - smdm / period + mdm[i]

        if str_val > 0:
            plus_di[i] = (spdm / str_val) * 100
            minus_di[i] = (smdm / str_val) * 100
        else:
            plus_di[i] = 0
            minus_di[i] = 0

        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx = abs(plus_di[i] - minus_di[i]) / di_sum * 100
        else:
            dx = 0
        dx_vals.append(dx)

        if len(dx_vals) == period:
            adx_arr[i] = sum(dx_vals) / period
        elif len(dx_vals) > period:
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx) / period

    return plus_di, minus_di, adx_arr


# ============================================================
# MASTER INDICATOR ANALYSIS — one call per timeframe
# ============================================================

def analyze_all_indicators(candles: List[Candle], prefix: str = "5m") -> Dict:
    """
    Calculate ALL indicators for a list of candles.
    Returns a flat dict of named indicator values and boolean flags.
    prefix: e.g., "5m", "15m", "1h", "4h"
    """
    from config import (
        RSI_PERIOD, EMA_FAST, EMA_MID, EMA_SLOW, EMA_200,
        MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD,
        SUPERTREND_PERIOD, SUPERTREND_MULT, STOCH_RSI_PERIOD,
        STOCH_K, STOCH_D, ATR_PERIOD, VOL_MA_PERIOD, ADX_PERIOD,
        CCI_PERIOD, WILLIAMS_R_PERIOD, KC_PERIOD, KC_MULT,
        ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU_B,
        PSAR_AF_START, PSAR_AF_MAX, MFI_PERIOD, CMF_PERIOD,
        OBV_MA_PERIOD, ROC_PERIOD, MOMENTUM_PERIOD,
    )

    r = {}
    n = len(candles)
    if n < 5:
        return r

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    cl = closes[-1]

    # === RSI ===
    rsi_arr = _rsi(closes, RSI_PERIOD)
    rsi = rsi_arr[-1] if not math.isnan(rsi_arr[-1]) else 50.0
    r[f"{prefix}_rsi"] = round(rsi, 1)
    r[f"{prefix}_rsi_ob"] = rsi > 70
    r[f"{prefix}_rsi_os"] = rsi < 30
    r[f"{prefix}_rsi_bull"] = rsi > 50
    r[f"{prefix}_rsi_bear"] = rsi < 50
    # RSI divergence (price vs RSI over last 10 bars)
    if n >= 10 and not math.isnan(rsi_arr[-10]):
        price_chg = cl - closes[-10]
        rsi_chg = rsi - rsi_arr[-10]
        r[f"{prefix}_rsi_bull_div"] = price_chg < 0 and rsi_chg > 0
        r[f"{prefix}_rsi_bear_div"] = price_chg > 0 and rsi_chg < 0

    # === EMA ===
    ef = _ema(closes, EMA_FAST)
    em = _ema(closes, EMA_MID)
    es = _ema(closes, EMA_SLOW)
    e200 = _ema(closes, EMA_200) if n >= EMA_200 else []
    if ef and es and not (math.isnan(ef[-1]) or math.isnan(es[-1])):
        r[f"{prefix}_ema_bull"] = ef[-1] > es[-1]
        r[f"{prefix}_ema_bear"] = ef[-1] < es[-1]
        if len(ef) >= 2 and len(es) >= 2:
            r[f"{prefix}_ema_bull_x"] = ef[-2] <= es[-2] and ef[-1] > es[-1]
            r[f"{prefix}_ema_bear_x"] = ef[-2] >= es[-2] and ef[-1] < es[-1]
    if ef and em and es and not any(math.isnan(x) for x in [ef[-1], em[-1], es[-1]]):
        r[f"{prefix}_ribbon_bull"] = ef[-1] > em[-1] > es[-1]
        r[f"{prefix}_ribbon_bear"] = ef[-1] < em[-1] < es[-1]
        if es[-1] > 0:
            spread = abs((ef[-1] - es[-1]) / es[-1] * 100)
            r[f"{prefix}_ema_squeeze"] = spread < 0.5
            r[f"{prefix}_ema_spread"] = round(spread, 2)
    if e200 and not math.isnan(e200[-1]):
        r[f"{prefix}_above_200ema"] = cl > e200[-1]
        r[f"{prefix}_below_200ema"] = cl < e200[-1]
    r[f"{prefix}_price_vs_ema9"] = round((cl / ef[-1] - 1) * 100, 2) if ef and ef[-1] > 0 else 0

    # === MACD ===
    ml, ms_line, mh = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    if not (math.isnan(ml[-1]) or math.isnan(ms_line[-1])):
        r[f"{prefix}_macd_above"] = ml[-1] > ms_line[-1]
        r[f"{prefix}_macd_below"] = ml[-1] < ms_line[-1]
        r[f"{prefix}_macd_hist_pos"] = mh[-1] > 0 if not math.isnan(mh[-1]) else False
        if len(mh) >= 2 and not math.isnan(mh[-2]):
            r[f"{prefix}_macd_hist_rising"] = mh[-1] > mh[-2]
            r[f"{prefix}_macd_cross_up"] = mh[-2] < 0 and mh[-1] >= 0
            r[f"{prefix}_macd_cross_dn"] = mh[-2] > 0 and mh[-1] <= 0

    # === Bollinger Bands ===
    bu, bm, bl, bw_arr = _bollinger(closes, BB_PERIOD, BB_STD)
    if bu and not math.isnan(bu[-1]):
        r[f"{prefix}_bb_above_up"] = cl > bu[-1]
        r[f"{prefix}_bb_below_lo"] = cl < bl[-1]
        r[f"{prefix}_bb_near_up"] = cl > bu[-1] * 0.995
        r[f"{prefix}_bb_near_lo"] = cl < bl[-1] * 1.005
        bw = bw_arr[-1] if not math.isnan(bw_arr[-1]) else 0
        r[f"{prefix}_bb_bw"] = round(bw, 2)
        r[f"{prefix}_bb_squeeze"] = bw < 3.0
        r[f"{prefix}_bb_wide"] = bw > 10
        if bu[-1] - bl[-1] > 0:
            r[f"{prefix}_bb_pctb"] = round((cl - bl[-1]) / (bu[-1] - bl[-1]), 2)

    # === Supertrend ===
    st_val, st_dir = _supertrend(candles, SUPERTREND_PERIOD, SUPERTREND_MULT)
    if st_dir[-1] != 0:
        r[f"{prefix}_st_bull"] = st_dir[-1] == 1
        r[f"{prefix}_st_bear"] = st_dir[-1] == -1
        if n >= 2:
            r[f"{prefix}_st_flip_bull"] = st_dir[-2] == -1 and st_dir[-1] == 1
            r[f"{prefix}_st_flip_bear"] = st_dir[-2] == 1 and st_dir[-1] == -1
        # Streak
        streak = 1
        for j in range(n - 2, -1, -1):
            if st_dir[j] == st_dir[-1]:
                streak += 1
            else:
                break
        r[f"{prefix}_st_streak"] = streak
        r[f"{prefix}_st_strong"] = streak >= 5

    # === Stochastic RSI ===
    sk, sd = _stoch_rsi(closes, RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D)
    if sk and not math.isnan(sk[-1]):
        r[f"{prefix}_stoch_k"] = round(sk[-1], 1)
        r[f"{prefix}_stoch_ob"] = sk[-1] > 80
        r[f"{prefix}_stoch_os"] = sk[-1] < 20

    # === ADX / DI ===
    pdi, mdi, adx_a = _adx(candles, ADX_PERIOD)
    if not math.isnan(adx_a[-1]):
        r[f"{prefix}_adx"] = round(adx_a[-1], 1)
        r[f"{prefix}_adx_strong"] = adx_a[-1] > 25
        r[f"{prefix}_adx_weak"] = adx_a[-1] < 20
    if not (math.isnan(pdi[-1]) or math.isnan(mdi[-1])):
        r[f"{prefix}_di_bull"] = pdi[-1] > mdi[-1]
        r[f"{prefix}_di_bear"] = pdi[-1] < mdi[-1]

    # === Williams %R ===
    wr = _williams_r(candles, WILLIAMS_R_PERIOD)
    if not math.isnan(wr[-1]):
        r[f"{prefix}_wr"] = round(wr[-1], 1)
        r[f"{prefix}_wr_ob"] = wr[-1] > -20
        r[f"{prefix}_wr_os"] = wr[-1] < -80
        r[f"{prefix}_wr_extreme_os"] = wr[-1] < -95

    # === CCI ===
    cci = _cci(candles, CCI_PERIOD)
    if not math.isnan(cci[-1]):
        r[f"{prefix}_cci"] = round(cci[-1], 1)
        r[f"{prefix}_cci_ob"] = cci[-1] > 100
        r[f"{prefix}_cci_os"] = cci[-1] < -100

    # === ROC ===
    roc = _roc(closes, ROC_PERIOD)
    if not math.isnan(roc[-1]):
        r[f"{prefix}_roc"] = round(roc[-1], 2)
        r[f"{prefix}_roc_strong"] = roc[-1] > 2
        r[f"{prefix}_roc_weak"] = roc[-1] < -2

    # === Momentum ===
    mom = _momentum(closes, MOMENTUM_PERIOD)
    if not math.isnan(mom[-1]):
        r[f"{prefix}_mom_pos"] = mom[-1] > 0
        r[f"{prefix}_mom_neg"] = mom[-1] < 0
        if n >= 3 and not (math.isnan(mom[-2]) or math.isnan(mom[-3])):
            r[f"{prefix}_mom_accel"] = mom[-1] > mom[-2] > mom[-3]

    # === OBV ===
    obv = _obv(candles)
    obv_ma = _sma(obv, OBV_MA_PERIOD)
    if obv_ma and not math.isnan(obv_ma[-1]):
        r[f"{prefix}_obv_above_ma"] = obv[-1] > obv_ma[-1]
    if n >= 5:
        r[f"{prefix}_obv_rising"] = obv[-1] > obv[-5]
        r[f"{prefix}_obv_falling"] = obv[-1] < obv[-5]
    # OBV divergence
    if n >= 10:
        ps = closes[-1] - closes[-5]
        os_ = obv[-1] - obv[-5]
        r[f"{prefix}_obv_bull_div"] = ps < 0 and os_ > 0
        r[f"{prefix}_obv_bear_div"] = ps > 0 and os_ < 0

    # === MFI ===
    mfi = _mfi(candles, MFI_PERIOD)
    if not math.isnan(mfi[-1]):
        r[f"{prefix}_mfi"] = round(mfi[-1], 1)
        r[f"{prefix}_mfi_os"] = mfi[-1] < 20
        r[f"{prefix}_mfi_ob"] = mfi[-1] > 80

    # === CMF ===
    cmf = _cmf(candles, CMF_PERIOD)
    if not math.isnan(cmf[-1]):
        r[f"{prefix}_cmf"] = round(cmf[-1], 3)
        r[f"{prefix}_cmf_pos"] = cmf[-1] > 0.05
        r[f"{prefix}_cmf_neg"] = cmf[-1] < -0.05

    # === VWAP ===
    vwap = _vwap(candles, 200)
    if not math.isnan(vwap[-1]):
        r[f"{prefix}_above_vwap"] = cl > vwap[-1]
        r[f"{prefix}_below_vwap"] = cl < vwap[-1]

    # === Keltner Channels ===
    kc_u, kc_m, kc_l = _keltner(candles, KC_PERIOD, KC_MULT)
    if not (math.isnan(kc_u[-1]) or math.isnan(kc_l[-1])):
        r[f"{prefix}_kc_above"] = cl > kc_u[-1]
        r[f"{prefix}_kc_below"] = cl < kc_l[-1]
        # TTM Squeeze: BB inside KC
        if bu and not math.isnan(bu[-1]):
            r[f"{prefix}_ttm_squeeze"] = bl[-1] > kc_l[-1] and bu[-1] < kc_u[-1]

    # === Ichimoku ===
    ich = _ichimoku(candles, ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU_B)
    if ich:
        r[f"{prefix}_ich_above"] = ich["price_vs_cloud"] == "above"
        r[f"{prefix}_ich_below"] = ich["price_vs_cloud"] == "below"
        r[f"{prefix}_ich_inside"] = ich["price_vs_cloud"] == "inside"
        r[f"{prefix}_ich_tk_bull"] = ich["tk_cross_bull"]

    # === Parabolic SAR ===
    psar = _parabolic_sar(candles, PSAR_AF_START, PSAR_AF_START, PSAR_AF_MAX)
    if psar[-1] != 0:
        r[f"{prefix}_psar_bull"] = psar[-1] == 1
        r[f"{prefix}_psar_bear"] = psar[-1] == -1
        if n >= 2:
            r[f"{prefix}_psar_flip_bull"] = psar[-2] == -1 and psar[-1] == 1
            r[f"{prefix}_psar_flip_bear"] = psar[-2] == 1 and psar[-1] == -1

    # === ATR ===
    atr = _atr(candles, ATR_PERIOD)
    if not math.isnan(atr[-1]) and cl > 0:
        r[f"{prefix}_atr_pct"] = round(atr[-1] / cl * 100, 2)

    # === Volume Analysis ===
    vm = _sma(volumes, VOL_MA_PERIOD)
    if vm and not math.isnan(vm[-1]) and vm[-1] > 0:
        vr = volumes[-1] / vm[-1]
        r[f"{prefix}_vol_ratio"] = round(vr, 2)
        r[f"{prefix}_vol_spike"] = vr > 2.0
        r[f"{prefix}_vol_high"] = vr > 1.5
        r[f"{prefix}_vol_low"] = vr < 0.5
    if n >= 5:
        rv = volumes[-5:]
        r[f"{prefix}_vol_increasing"] = all(rv[i] >= rv[i - 1] * 0.95 for i in range(1, len(rv)))

    # === Structure: Higher Highs / Lower Lows ===
    if n >= 5:
        r[f"{prefix}_hh"] = highs[-1] > highs[-2] and highs[-2] > highs[-3]
        r[f"{prefix}_ll"] = lows[-1] < lows[-2] and lows[-2] < lows[-3]

    # === Candle Patterns ===
    if n >= 3:
        r[f"{prefix}_three_green"] = all(candles[-j].close > candles[-j].open for j in range(1, 4))
        r[f"{prefix}_three_red"] = all(candles[-j].close < candles[-j].open for j in range(1, 4))

    return r
