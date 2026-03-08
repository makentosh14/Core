#!/usr/bin/env python3
"""
indicators.py - Complete Indicator Calculation Engine
=====================================================
All 30+ indicators with pure-Python calculations (no numpy dependency).
Each function takes raw OHLCV data via Candle dataclass.

WARMUP NOTE:
  analyze_all_indicators() requires at least WARMUP_CANDLES + window candles
  to produce valid values for all indicators. The slowest indicator is EMA200
  which needs 200 bars. Callers must pass enough history.
"""

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
    n = len(data)
    result = [float("nan")] * n
    if n < period:
        return result
    for i in range(period - 1, n):
        result[i] = sum(data[i - period + 1 : i + 1]) / period
    return result


def _ema(data: list, period: int) -> list:
    """Exponential Moving Average (Wilder-style seed = SMA)."""
    n = len(data)
    result = [float("nan")] * n
    if n < period:
        return result
    # Seed with SMA
    result[period - 1] = sum(data[:period]) / period
    mult = 2.0 / (period + 1)
    for i in range(period, n):
        result[i] = (data[i] - result[i - 1]) * mult + result[i - 1]
    return result


def _rsi(closes: list, period: int = 14) -> list:
    """Wilder's RSI."""
    n = len(closes)
    result = [float("nan")] * n
    if n < period + 1:
        return result
    gains, losses = [], []
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    def _rs_to_rsi(ag, al):
        if al == 0:
            return 100.0
        return 100 - (100 / (1 + ag / al))
    result[period] = _rs_to_rsi(avg_gain, avg_loss)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result[i + 1] = _rs_to_rsi(avg_gain, avg_loss)
    return result


def _macd(closes: list, fast: int = 12, slow: int = 26, sig: int = 9):
    """MACD line, signal line, histogram."""
    n = len(closes)
    ema_f = _ema(closes, fast)
    ema_s = _ema(closes, slow)
    macd_line = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(ema_f[i]) or math.isnan(ema_s[i])):
            macd_line[i] = ema_f[i] - ema_s[i]
    # Signal = EMA of MACD (fill NaN with 0 for EMA seed)
    ml_filled = [v if not math.isnan(v) else 0.0 for v in macd_line]
    signal = _ema(ml_filled, sig)
    histogram = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(macd_line[i]) or math.isnan(signal[i])):
            histogram[i] = macd_line[i] - signal[i]
    return macd_line, signal, histogram


def _bollinger(closes: list, period: int = 20, std_mult: float = 2.0):
    """Bollinger Bands: upper, mid, lower, bandwidth%."""
    n = len(closes)
    mid = _sma(closes, period)
    upper  = [float("nan")] * n
    lower  = [float("nan")] * n
    bw     = [float("nan")] * n
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        std = (sum((x - mid[i]) ** 2 for x in window) / period) ** 0.5
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
        if mid[i] > 0:
            bw[i] = (upper[i] - lower[i]) / mid[i] * 100
    return upper, mid, lower, bw


def _atr(candles: list, period: int = 14) -> list:
    """Average True Range (Wilder smoothed)."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period:
        return result
    tr = [0.0] * n
    tr[0] = candles[0].high - candles[0].low
    for i in range(1, n):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        tr[i] = max(h - l, abs(h - pc), abs(l - pc))
    result[period - 1] = sum(tr[:period]) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def _supertrend(candles: list, period: int = 10, mult: float = 3.0):
    """Supertrend indicator. Returns (values, directions) where direction: 1=bull, -1=bear."""
    n = len(candles)
    atr = _atr(candles, period)
    direction = [1] * n
    st = [0.0] * n
    upper_basic = [0.0] * n
    lower_basic = [0.0] * n

    for i in range(n):
        hl2 = (candles[i].high + candles[i].low) / 2
        if not math.isnan(atr[i]):
            upper_basic[i] = hl2 + mult * atr[i]
            lower_basic[i] = hl2 - mult * atr[i]

    final_upper = [0.0] * n
    final_lower = [0.0] * n
    for i in range(n):
        if i == 0:
            final_upper[i] = upper_basic[i]
            final_lower[i] = lower_basic[i]
        else:
            final_upper[i] = (upper_basic[i]
                              if upper_basic[i] < final_upper[i - 1] or candles[i - 1].close > final_upper[i - 1]
                              else final_upper[i - 1])
            final_lower[i] = (lower_basic[i]
                              if lower_basic[i] > final_lower[i - 1] or candles[i - 1].close < final_lower[i - 1]
                              else final_lower[i - 1])

    for i in range(1, n):
        if st[i - 1] == final_upper[i - 1]:
            st[i] = final_lower[i] if candles[i].close > final_upper[i] else final_upper[i]
        else:
            st[i] = final_upper[i] if candles[i].close < final_lower[i] else final_lower[i]
        direction[i] = 1 if candles[i].close > st[i] else -1

    return st, direction


def _adx(candles: list, period: int = 14):
    """ADX with +DI and -DI."""
    n = len(candles)
    plus_di  = [float("nan")] * n
    minus_di = [float("nan")] * n
    adx_arr  = [float("nan")] * n
    if n < period + 1:
        return plus_di, minus_di, adx_arr

    tr   = [0.0] * n
    pdm  = [0.0] * n
    mdm  = [0.0] * n
    for i in range(1, n):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        tr[i]  = max(h - l, abs(h - pc), abs(l - pc))
        up     = h - candles[i - 1].high
        dn     = candles[i - 1].low - l
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    # Wilder smoothing seed
    str_v = sum(tr[1 : period + 1])
    spdm  = sum(pdm[1 : period + 1])
    smdm  = sum(mdm[1 : period + 1])
    dx_vals = []

    for i in range(period, n):
        if i > period:
            str_v = str_v - str_v / period + tr[i]
            spdm  = spdm  - spdm  / period + pdm[i]
            smdm  = smdm  - smdm  / period + mdm[i]

        pdi = (spdm / str_v * 100) if str_v > 0 else 0.0
        mdi = (smdm / str_v * 100) if str_v > 0 else 0.0
        plus_di[i]  = pdi
        minus_di[i] = mdi

        di_sum = pdi + mdi
        dx = abs(pdi - mdi) / di_sum * 100 if di_sum > 0 else 0.0
        dx_vals.append(dx)

        if len(dx_vals) == period:
            adx_arr[i] = sum(dx_vals) / period
        elif len(dx_vals) > period:
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx) / period

    return plus_di, minus_di, adx_arr


def _stoch_rsi(closes: list, rsi_period: int = 14, stoch_period: int = 14,
               k_smooth: int = 3, d_smooth: int = 3):
    """Stochastic RSI."""
    rsi = _rsi(closes, rsi_period)
    n = len(rsi)
    k_raw = [float("nan")] * n
    for i in range(stoch_period - 1, n):
        window = [v for v in rsi[i - stoch_period + 1 : i + 1] if not math.isnan(v)]
        if not window:
            continue
        lo, hi = min(window), max(window)
        if hi > lo and not math.isnan(rsi[i]):
            k_raw[i] = (rsi[i] - lo) / (hi - lo) * 100
        else:
            k_raw[i] = 50.0
    k_filled = [v if not math.isnan(v) else 50.0 for v in k_raw]
    k_smooth_arr = _sma(k_filled, k_smooth)
    d_filled = [v if not math.isnan(v) else 50.0 for v in k_smooth_arr]
    d_smooth_arr = _sma(d_filled, d_smooth)
    return k_smooth_arr, d_smooth_arr


def _keltner(candles: list, period: int = 20, mult: float = 1.5):
    """Keltner Channel: upper, mid (EMA), lower."""
    closes = [c.close for c in candles]
    atr    = _atr(candles, period)
    mid    = _ema(closes, period)
    n      = len(candles)
    upper  = [float("nan")] * n
    lower  = [float("nan")] * n
    for i in range(n):
        if not (math.isnan(mid[i]) or math.isnan(atr[i])):
            upper[i] = mid[i] + mult * atr[i]
            lower[i] = mid[i] - mult * atr[i]
    return upper, mid, lower


def _ichimoku(candles: list, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52):
    """Ichimoku Cloud: tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou."""
    n = len(candles)
    highs  = [c.high  for c in candles]
    lows   = [c.low   for c in candles]
    closes = [c.close for c in candles]

    def _midpoint(h_arr, l_arr, period, i):
        if i < period - 1:
            return float("nan")
        window_h = h_arr[i - period + 1 : i + 1]
        window_l = l_arr[i - period + 1 : i + 1]
        return (max(window_h) + min(window_l)) / 2

    tk  = [_midpoint(highs, lows, tenkan,   i) for i in range(n)]
    kj  = [_midpoint(highs, lows, kijun,    i) for i in range(n)]
    spA = [float("nan")] * n
    spB = [float("nan")] * n
    for i in range(kijun, n):
        if not (math.isnan(tk[i - kijun]) or math.isnan(kj[i - kijun])):
            spA[i] = (tk[i - kijun] + kj[i - kijun]) / 2
        mb = _midpoint(highs, lows, senkou_b, i - kijun)
        if not math.isnan(mb):
            spB[i] = mb
    chikou = [float("nan")] * n
    for i in range(n - kijun):
        chikou[i] = closes[i + kijun]
    return tk, kj, spA, spB, chikou


def _psar(candles: list, af_start: float = 0.02, af_max: float = 0.2):
    """Parabolic SAR. Returns (sar_values, directions) direction: 1=bull, -1=bear."""
    n = len(candles)
    if n < 2:
        return [float("nan")] * n, [1] * n
    sar    = [0.0] * n
    bull   = [True] * n
    ep     = [0.0] * n
    af_arr = [af_start] * n
    sar[0] = candles[0].low
    ep[0]  = candles[0].high
    for i in range(1, n):
        prev_bull = bull[i - 1]
        prev_sar  = sar[i - 1]
        prev_ep   = ep[i - 1]
        prev_af   = af_arr[i - 1]

        new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
        if prev_bull:
            new_sar = min(new_sar, candles[i - 1].low, candles[max(0, i - 2)].low)
            if candles[i].low < new_sar:
                bull[i]   = False
                sar[i]    = prev_ep
                ep[i]     = candles[i].low
                af_arr[i] = af_start
            else:
                bull[i] = True
                sar[i]  = new_sar
                if candles[i].high > prev_ep:
                    ep[i]     = candles[i].high
                    af_arr[i] = min(prev_af + af_start, af_max)
                else:
                    ep[i]     = prev_ep
                    af_arr[i] = prev_af
        else:
            new_sar = max(new_sar, candles[i - 1].high, candles[max(0, i - 2)].high)
            if candles[i].high > new_sar:
                bull[i]   = True
                sar[i]    = prev_ep
                ep[i]     = candles[i].high
                af_arr[i] = af_start
            else:
                bull[i] = False
                sar[i]  = new_sar
                if candles[i].low < prev_ep:
                    ep[i]     = candles[i].low
                    af_arr[i] = min(prev_af + af_start, af_max)
                else:
                    ep[i]     = prev_ep
                    af_arr[i] = prev_af

    directions = [1 if b else -1 for b in bull]
    return sar, directions


def _obv(candles: list):
    """On Balance Volume."""
    obv = [0.0] * len(candles)
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv[i] = obv[i - 1] + candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            obv[i] = obv[i - 1] - candles[i].volume
        else:
            obv[i] = obv[i - 1]
    return obv


def _mfi(candles: list, period: int = 14) -> list:
    """Money Flow Index."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period + 1:
        return result
    tp    = [(c.high + c.low + c.close) / 3 for c in candles]
    rmf   = [tp[i] * candles[i].volume for i in range(n)]
    for i in range(period, n):
        pos_mf = sum(rmf[j] for j in range(i - period + 1, i + 1) if tp[j] > tp[j - 1])
        neg_mf = sum(rmf[j] for j in range(i - period + 1, i + 1) if tp[j] < tp[j - 1])
        if neg_mf == 0:
            result[i] = 100.0
        else:
            result[i] = 100 - (100 / (1 + pos_mf / neg_mf))
    return result


def _cmf(candles: list, period: int = 20) -> list:
    """Chaikin Money Flow."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period:
        return result
    mfv = []
    for c in candles:
        rng = c.high - c.low
        if rng == 0:
            mfv.append(0.0)
        else:
            mfv.append(((c.close - c.low) - (c.high - c.close)) / rng * c.volume)
    for i in range(period - 1, n):
        vol_sum = sum(candles[j].volume for j in range(i - period + 1, i + 1))
        mfv_sum = sum(mfv[j] for j in range(i - period + 1, i + 1))
        result[i] = mfv_sum / vol_sum if vol_sum > 0 else 0.0
    return result


def _cci(candles: list, period: int = 20) -> list:
    """Commodity Channel Index."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period:
        return result
    tp = [(c.high + c.low + c.close) / 3 for c in candles]
    for i in range(period - 1, n):
        window = tp[i - period + 1 : i + 1]
        mean   = sum(window) / period
        mad    = sum(abs(x - mean) for x in window) / period
        if mad > 0:
            result[i] = (tp[i] - mean) / (0.015 * mad)
        else:
            result[i] = 0.0
    return result


def _williams_r(candles: list, period: int = 14) -> list:
    """Williams %R."""
    n = len(candles)
    result = [float("nan")] * n
    if n < period:
        return result
    for i in range(period - 1, n):
        h = max(c.high  for c in candles[i - period + 1 : i + 1])
        l = min(c.low   for c in candles[i - period + 1 : i + 1])
        if h > l:
            result[i] = (h - candles[i].close) / (h - l) * -100
        else:
            result[i] = -50.0
    return result


def _roc(closes: list, period: int = 12) -> list:
    """Rate of Change %."""
    n = len(closes)
    result = [float("nan")] * n
    for i in range(period, n):
        if closes[i - period] != 0:
            result[i] = (closes[i] - closes[i - period]) / closes[i - period] * 100
    return result


def _vwap(candles: list) -> float:
    """Session VWAP (over the provided candle window)."""
    cum_pv  = sum((c.high + c.low + c.close) / 3 * c.volume for c in candles)
    cum_vol = sum(c.volume for c in candles)
    return cum_pv / cum_vol if cum_vol > 0 else 0.0


# ============================================================
# MASTER INDICATOR ANALYSIS — one call per timeframe
# ============================================================

def analyze_all_indicators(candles: List[Candle], prefix: str = "5m") -> Dict:
    """
    Calculate ALL indicators for a candle list.
    Returns a flat dict of named indicator values and boolean flags.

    prefix: "5m", "15m", "1h", "4h"

    IMPORTANT: candles should include WARMUP_CANDLES extra bars.
    Only the last candle's values are used for boolean signals.
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

    r   = {}
    n   = len(candles)
    if n < 5:
        return r

    closes  = [c.close  for c in candles]
    highs   = [c.high   for c in candles]
    lows    = [c.low    for c in candles]
    volumes = [c.volume for c in candles]
    cl      = closes[-1]

    # ----------------------------------------------------------
    # RSI
    # ----------------------------------------------------------
    rsi_arr = _rsi(closes, RSI_PERIOD)
    rsi     = rsi_arr[-1] if not math.isnan(rsi_arr[-1]) else 50.0
    r[f"{prefix}_rsi"]          = round(rsi, 2)
    r[f"{prefix}_rsi_ob"]       = rsi > 70
    r[f"{prefix}_rsi_os"]       = rsi < 30
    r[f"{prefix}_rsi_bull"]     = rsi > 50
    r[f"{prefix}_rsi_bear"]     = rsi < 50
    # RSI divergence over last 10 bars
    if n >= 10 and not math.isnan(rsi_arr[-10]):
        price_chg = cl - closes[-10]
        rsi_chg   = rsi - rsi_arr[-10]
        r[f"{prefix}_rsi_bull_div"] = price_chg < 0 and rsi_chg > 0
        r[f"{prefix}_rsi_bear_div"] = price_chg > 0 and rsi_chg < 0
    else:
        r[f"{prefix}_rsi_bull_div"] = False
        r[f"{prefix}_rsi_bear_div"] = False

    # RSI zones (useful for binning in analysis)
    r[f"{prefix}_rsi_zone"] = (
        "extreme_os"  if rsi < 20 else
        "oversold"    if rsi < 35 else
        "neutral_low" if rsi < 50 else
        "neutral_hi"  if rsi < 65 else
        "overbought"  if rsi < 80 else
        "extreme_ob"
    )

    # ----------------------------------------------------------
    # EMA
    # ----------------------------------------------------------
    ef   = _ema(closes, EMA_FAST)
    em   = _ema(closes, EMA_MID)
    es   = _ema(closes, EMA_SLOW)
    e200 = _ema(closes, EMA_200) if n >= EMA_200 else [float("nan")] * n

    def _ev(arr):
        v = arr[-1] if arr else float("nan")
        return v if not math.isnan(v) else 0.0

    ef_v, em_v, es_v, e200_v = _ev(ef), _ev(em), _ev(es), _ev(e200)

    r[f"{prefix}_ema_fast"]     = round(ef_v, 6)
    r[f"{prefix}_ema_mid"]      = round(em_v, 6)
    r[f"{prefix}_ema_slow"]     = round(es_v, 6)
    r[f"{prefix}_ema200"]       = round(e200_v, 6)
    r[f"{prefix}_ema_bull"]     = ef_v > es_v
    r[f"{prefix}_ema_bear"]     = ef_v < es_v
    r[f"{prefix}_above_ema200"] = cl > e200_v > 0
    r[f"{prefix}_below_ema200"] = cl < e200_v and e200_v > 0
    # Crossover (last bar)
    if len(ef) >= 2 and len(es) >= 2:
        ef2, es2 = ef[-2], es[-2]
        r[f"{prefix}_ema_bull_x"] = (not math.isnan(ef2)) and (not math.isnan(es2)) and ef2 <= es2 and ef_v > es_v
        r[f"{prefix}_ema_bear_x"] = (not math.isnan(ef2)) and (not math.isnan(es2)) and ef2 >= es2 and ef_v < es_v
    else:
        r[f"{prefix}_ema_bull_x"] = False
        r[f"{prefix}_ema_bear_x"] = False

    # EMA ribbon (all 4 EMAs ordered)
    if all(x > 0 for x in [ef_v, em_v, es_v]):
        r[f"{prefix}_ribbon_bull"] = ef_v > em_v > es_v
        r[f"{prefix}_ribbon_bear"] = ef_v < em_v < es_v
    else:
        r[f"{prefix}_ribbon_bull"] = False
        r[f"{prefix}_ribbon_bear"] = False

    # ----------------------------------------------------------
    # MACD
    # ----------------------------------------------------------
    macd_line, macd_sig, macd_hist = _macd(closes, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    ml   = macd_line[-1] if not math.isnan(macd_line[-1]) else 0.0
    ms   = macd_sig[-1]  if not math.isnan(macd_sig[-1])  else 0.0
    mh   = macd_hist[-1] if not math.isnan(macd_hist[-1]) else 0.0
    mh2  = macd_hist[-2] if len(macd_hist) >= 2 and not math.isnan(macd_hist[-2]) else mh

    r[f"{prefix}_macd_line"]    = round(ml, 8)
    r[f"{prefix}_macd_sig"]     = round(ms, 8)
    r[f"{prefix}_macd_hist"]    = round(mh, 8)
    r[f"{prefix}_macd_above"]   = ml > ms
    r[f"{prefix}_macd_below"]   = ml < ms
    r[f"{prefix}_macd_bull_x"]  = mh > 0 and mh2 <= 0
    r[f"{prefix}_macd_bear_x"]  = mh < 0 and mh2 >= 0
    r[f"{prefix}_macd_hist_up"] = mh > mh2
    r[f"{prefix}_macd_hist_dn"] = mh < mh2

    # ----------------------------------------------------------
    # Bollinger Bands
    # ----------------------------------------------------------
    bb_up, bb_mid, bb_lo, bb_bw = _bollinger(closes, BB_PERIOD, BB_STD)
    bbu  = bb_up[-1]  if not math.isnan(bb_up[-1])  else 0.0
    bbm  = bb_mid[-1] if not math.isnan(bb_mid[-1]) else 0.0
    bbl  = bb_lo[-1]  if not math.isnan(bb_lo[-1])  else 0.0
    bwv  = bb_bw[-1]  if not math.isnan(bb_bw[-1])  else 0.0

    r[f"{prefix}_bb_upper"]     = round(bbu, 6)
    r[f"{prefix}_bb_mid"]       = round(bbm, 6)
    r[f"{prefix}_bb_lower"]     = round(bbl, 6)
    r[f"{prefix}_bb_bw"]        = round(bwv, 2)
    r[f"{prefix}_bb_squeeze"]   = 0 < bwv < 3.0
    r[f"{prefix}_bb_above_up"]  = cl > bbu > 0
    r[f"{prefix}_bb_below_lo"]  = cl < bbl and bbl > 0
    # Percent-B  (position within bands)
    if bbu > bbl:
        pb = (cl - bbl) / (bbu - bbl) * 100
        r[f"{prefix}_bb_pct_b"] = round(pb, 2)
    else:
        r[f"{prefix}_bb_pct_b"] = 50.0

    # ----------------------------------------------------------
    # ATR (absolute and % of price)
    # ----------------------------------------------------------
    atr_arr = _atr(candles, ATR_PERIOD)
    atr_v   = atr_arr[-1] if not math.isnan(atr_arr[-1]) else 0.0
    r[f"{prefix}_atr"]      = round(atr_v, 6)
    r[f"{prefix}_atr_pct"]  = round(atr_v / cl * 100, 3) if cl > 0 else 0.0

    # ----------------------------------------------------------
    # Supertrend
    # ----------------------------------------------------------
    st_vals, st_dirs = _supertrend(candles, SUPERTREND_PERIOD, SUPERTREND_MULT)
    st_d = st_dirs[-1]
    r[f"{prefix}_st_bull"]   = st_d == 1
    r[f"{prefix}_st_bear"]   = st_d == -1
    r[f"{prefix}_st_val"]    = round(st_vals[-1], 6)
    # Flip (direction change on last bar)
    if len(st_dirs) >= 2:
        r[f"{prefix}_st_bull_flip"] = st_dirs[-2] == -1 and st_d == 1
        r[f"{prefix}_st_bear_flip"] = st_dirs[-2] == 1  and st_d == -1
    else:
        r[f"{prefix}_st_bull_flip"] = False
        r[f"{prefix}_st_bear_flip"] = False

    # ----------------------------------------------------------
    # ADX
    # ----------------------------------------------------------
    plus_di, minus_di, adx_arr = _adx(candles, ADX_PERIOD)
    adx_v = adx_arr[-1]  if not math.isnan(adx_arr[-1])  else 0.0
    pdi   = plus_di[-1]  if not math.isnan(plus_di[-1])   else 0.0
    mdi   = minus_di[-1] if not math.isnan(minus_di[-1])  else 0.0
    r[f"{prefix}_adx"]        = round(adx_v, 2)
    r[f"{prefix}_plus_di"]    = round(pdi, 2)
    r[f"{prefix}_minus_di"]   = round(mdi, 2)
    r[f"{prefix}_adx_strong"] = adx_v > 25
    r[f"{prefix}_di_bull"]    = pdi > mdi
    r[f"{prefix}_di_bear"]    = pdi < mdi

    # ----------------------------------------------------------
    # Stochastic RSI
    # ----------------------------------------------------------
    sk, sd = _stoch_rsi(closes, STOCH_RSI_PERIOD, STOCH_RSI_PERIOD, STOCH_K, STOCH_D)
    sk_v = sk[-1] if sk and not math.isnan(sk[-1]) else 50.0
    sd_v = sd[-1] if sd and not math.isnan(sd[-1]) else 50.0
    r[f"{prefix}_stoch_k"]      = round(sk_v, 2)
    r[f"{prefix}_stoch_d"]      = round(sd_v, 2)
    r[f"{prefix}_stoch_ob"]     = sk_v > 80
    r[f"{prefix}_stoch_os"]     = sk_v < 20
    r[f"{prefix}_stoch_bull_x"] = sk_v > sd_v
    r[f"{prefix}_stoch_bear_x"] = sk_v < sd_v

    # ----------------------------------------------------------
    # CCI
    # ----------------------------------------------------------
    cci_arr = _cci(candles, CCI_PERIOD)
    cci_v   = cci_arr[-1] if not math.isnan(cci_arr[-1]) else 0.0
    r[f"{prefix}_cci"]      = round(cci_v, 2)
    r[f"{prefix}_cci_ob"]   = cci_v > 100
    r[f"{prefix}_cci_os"]   = cci_v < -100
    r[f"{prefix}_cci_bull"] = cci_v > 0
    r[f"{prefix}_cci_bear"] = cci_v < 0

    # ----------------------------------------------------------
    # Williams %R
    # ----------------------------------------------------------
    wr_arr = _williams_r(candles, WILLIAMS_R_PERIOD)
    wr_v   = wr_arr[-1] if not math.isnan(wr_arr[-1]) else -50.0
    r[f"{prefix}_wr"]       = round(wr_v, 2)
    r[f"{prefix}_wr_ob"]    = wr_v > -20
    r[f"{prefix}_wr_os"]    = wr_v < -80
    r[f"{prefix}_wr_bull"]  = wr_v > -50
    r[f"{prefix}_wr_bear"]  = wr_v < -50

    # ----------------------------------------------------------
    # Keltner Channel
    # ----------------------------------------------------------
    kc_up, kc_mid, kc_lo = _keltner(candles, KC_PERIOD, KC_MULT)
    kcu = kc_up[-1]  if not math.isnan(kc_up[-1])  else 0.0
    kcl = kc_lo[-1]  if not math.isnan(kc_lo[-1])  else 0.0
    r[f"{prefix}_kc_above"]  = cl > kcu > 0
    r[f"{prefix}_kc_below"]  = cl < kcl and kcl > 0
    # BB squeeze vs Keltner (BB inside KC = squeeze)
    if bbu > 0 and kcu > 0:
        r[f"{prefix}_sqz_on"]  = bbu < kcu and bbl > kcl
        r[f"{prefix}_sqz_off"] = bbu > kcu and bbl < kcl
    else:
        r[f"{prefix}_sqz_on"]  = False
        r[f"{prefix}_sqz_off"] = False

    # ----------------------------------------------------------
    # Ichimoku
    # ----------------------------------------------------------
    tk_arr, kj_arr, spA, spB, chikou = _ichimoku(candles, ICHIMOKU_TENKAN, ICHIMOKU_KIJUN, ICHIMOKU_SENKOU_B)
    tk_v  = tk_arr[-1]  if not math.isnan(tk_arr[-1])  else 0.0
    kj_v  = kj_arr[-1]  if not math.isnan(kj_arr[-1])  else 0.0
    spA_v = spA[-1]     if not math.isnan(spA[-1])      else 0.0
    spB_v = spB[-1]     if not math.isnan(spB[-1])      else 0.0
    cloud_top = max(spA_v, spB_v)
    cloud_bot = min(spA_v, spB_v)
    r[f"{prefix}_ich_above_cloud"] = cl > cloud_top > 0
    r[f"{prefix}_ich_below_cloud"] = cl < cloud_bot and cloud_bot > 0
    r[f"{prefix}_ich_in_cloud"]    = cloud_bot <= cl <= cloud_top if cloud_top > 0 else False
    r[f"{prefix}_ich_bull_cloud"]  = spA_v > spB_v               # green cloud
    r[f"{prefix}_ich_bear_cloud"]  = spA_v < spB_v               # red cloud
    r[f"{prefix}_ich_tk_kj_bull"]  = tk_v > kj_v
    r[f"{prefix}_ich_tk_kj_bear"]  = tk_v < kj_v

    # ----------------------------------------------------------
    # Parabolic SAR
    # ----------------------------------------------------------
    psar_v, psar_d = _psar(candles, PSAR_AF_START, PSAR_AF_MAX)
    ps_dir = psar_d[-1]
    r[f"{prefix}_psar_bull"]      = ps_dir == 1
    r[f"{prefix}_psar_bear"]      = ps_dir == -1
    if len(psar_d) >= 2:
        r[f"{prefix}_psar_bull_flip"] = psar_d[-2] == -1 and ps_dir == 1
        r[f"{prefix}_psar_bear_flip"] = psar_d[-2] == 1  and ps_dir == -1
    else:
        r[f"{prefix}_psar_bull_flip"] = False
        r[f"{prefix}_psar_bear_flip"] = False

    # ----------------------------------------------------------
    # OBV
    # ----------------------------------------------------------
    obv_arr   = _obv(candles)
    obv_ma    = _sma(obv_arr, OBV_MA_PERIOD)
    obv_v     = obv_arr[-1]
    obv_ma_v  = obv_ma[-1]  if not math.isnan(obv_ma[-1])  else 0.0
    r[f"{prefix}_obv"]        = round(obv_v, 0)
    r[f"{prefix}_obv_rising"] = obv_v > obv_ma_v

    # ----------------------------------------------------------
    # MFI
    # ----------------------------------------------------------
    mfi_arr = _mfi(candles, MFI_PERIOD)
    mfi_v   = mfi_arr[-1] if not math.isnan(mfi_arr[-1]) else 50.0
    r[f"{prefix}_mfi"]      = round(mfi_v, 2)
    r[f"{prefix}_mfi_ob"]   = mfi_v > 80
    r[f"{prefix}_mfi_os"]   = mfi_v < 20
    r[f"{prefix}_mfi_bull"] = mfi_v > 50

    # ----------------------------------------------------------
    # CMF
    # ----------------------------------------------------------
    cmf_arr = _cmf(candles, CMF_PERIOD)
    cmf_v   = cmf_arr[-1] if not math.isnan(cmf_arr[-1]) else 0.0
    r[f"{prefix}_cmf"]      = round(cmf_v, 4)
    r[f"{prefix}_cmf_bull"] = cmf_v > 0
    r[f"{prefix}_cmf_bear"] = cmf_v < 0

    # ----------------------------------------------------------
    # Volume
    # ----------------------------------------------------------
    vol_ma   = _sma(volumes, VOL_MA_PERIOD)
    vol_ma_v = vol_ma[-1] if not math.isnan(vol_ma[-1]) else 0.0
    vol_cur  = volumes[-1]
    vol_ratio = vol_cur / vol_ma_v if vol_ma_v > 0 else 1.0
    r[f"{prefix}_vol_ma"]      = round(vol_ma_v, 2)
    r[f"{prefix}_vol_ratio"]   = round(vol_ratio, 3)
    r[f"{prefix}_vol_spike"]   = vol_ratio > 2.0
    r[f"{prefix}_vol_high"]    = vol_ratio > 1.5
    r[f"{prefix}_vol_low"]     = vol_ratio < 0.5
    r[f"{prefix}_vol_rising"]  = vol_cur > vol_ma_v

    # ----------------------------------------------------------
    # ROC
    # ----------------------------------------------------------
    roc_arr = _roc(closes, ROC_PERIOD)
    roc_v   = roc_arr[-1] if not math.isnan(roc_arr[-1]) else 0.0
    r[f"{prefix}_roc"]      = round(roc_v, 3)
    r[f"{prefix}_roc_bull"] = roc_v > 0

    # ----------------------------------------------------------
    # VWAP (over current candle window)
    # ----------------------------------------------------------
    vwap_v = _vwap(candles)
    r[f"{prefix}_vwap"]         = round(vwap_v, 6)
    r[f"{prefix}_above_vwap"]   = cl > vwap_v > 0
    r[f"{prefix}_below_vwap"]   = cl < vwap_v and vwap_v > 0

    # ----------------------------------------------------------
    # Price Structure
    # ----------------------------------------------------------
    if n >= 3:
        # Higher High / Lower Low (last 3 candles)
        r[f"{prefix}_hh"] = highs[-1] > highs[-2] > highs[-3]
        r[f"{prefix}_ll"] = lows[-1]  < lows[-2]  < lows[-3]
        r[f"{prefix}_hl"] = lows[-1]  > lows[-2]                # higher low
        r[f"{prefix}_lh"] = highs[-1] < highs[-2]               # lower high
    # Candle patterns
    body    = abs(cl - candles[-1].open)
    candle_range = candles[-1].high - candles[-1].low
    if n >= 3:
        avg_body = sum(abs(candles[-i].close - candles[-i].open) for i in range(1, 4)) / 3
        # Three white soldiers / three black crows
        r[f"{prefix}_three_green"] = all(
            candles[-i].close > candles[-i].open for i in range(1, 4)
        )
        r[f"{prefix}_three_red"] = all(
            candles[-i].close < candles[-i].open for i in range(1, 4)
        )
        # Big green / big red candle
        r[f"{prefix}_big_green"] = (candles[-1].close > candles[-1].open and
                                    body > avg_body * 1.5)
        r[f"{prefix}_big_red"]   = (candles[-1].close < candles[-1].open and
                                    body > avg_body * 1.5)

    return r
