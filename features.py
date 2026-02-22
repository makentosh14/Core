#!/usr/bin/env python3
"""
features.py
===========
Computes all technical features from OHLCV candle data.
Each function takes a list of candle dicts:
    [{"open":f, "high":f, "low":f, "close":f, "volume":f, "ts":int}, ...]
and returns a flat dict of feature_name -> value.

NO lookahead: features are computed from the window BEFORE the event bar.
"""

import numpy as np
from typing import List, Dict, Any, Optional

# ────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────

def _arr(candles: List[Dict], key: str) -> np.ndarray:
    return np.array([float(c[key]) for c in candles], dtype=float)

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (pandas-like, not wilder)."""
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    alpha = 2.0 / (period + 1)
    out = np.full(len(arr), np.nan)
    # seed with simple average of first `period` values
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.mean(arr[i - period + 1 : i + 1])
    return out

def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    out = np.full(len(closes), np.nan)
    # Wilder smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(closes) - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100 - 100 / (1 + rs)
    return out

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    if len(closes) < 2:
        return np.full(len(closes), np.nan)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])))
    atr_arr = np.full(len(closes), np.nan)
    if len(tr) >= period:
        atr_arr[period] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_arr[i + 1] = (atr_arr[i] * (period - 1) + tr[i]) / period
    return atr_arr

def _macd(closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9):
    """Returns (macd_line, signal_line, histogram) arrays."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal = _ema(np.where(np.isnan(macd_line), 0, macd_line), sig)
    hist = macd_line - signal
    return macd_line, signal, hist

def _bollinger(closes: np.ndarray, period: int = 20, std_mult: float = 2.0):
    """Returns (upper, mid, lower, width_pct) arrays."""
    mid = _sma(closes, period)
    std = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        std[i] = np.std(closes[i - period + 1 : i + 1], ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width_pct = np.where(mid > 0, (upper - lower) / mid * 100, np.nan)
    return upper, mid, lower, width_pct

def _supertrend(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                period: int = 10, multiplier: float = 3.0):
    """Returns (supertrend_arr, direction_arr) where direction 1=bull, -1=bear."""
    atr = _atr(highs, lows, closes, period)
    hl2 = (highs + lows) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st = np.full(len(closes), np.nan)
    direction = np.zeros(len(closes))

    for i in range(1, len(closes)):
        if np.isnan(atr[i]):
            continue
        # Final upper band
        if np.isnan(st[i - 1]) or closes[i - 1] > (hl2[i-1] + multiplier * atr[i-1]):
            final_upper = upper_band[i]
        else:
            final_upper = min(upper_band[i], hl2[i-1] + multiplier * atr[i-1])

        if np.isnan(st[i - 1]) or closes[i - 1] < (hl2[i-1] - multiplier * atr[i-1]):
            final_lower = lower_band[i]
        else:
            final_lower = max(lower_band[i], hl2[i-1] - multiplier * atr[i-1])

        if closes[i] <= final_upper:
            st[i] = final_upper
            direction[i] = -1
        else:
            st[i] = final_lower
            direction[i] = 1

        if direction[i - 1] == -1 and closes[i] > final_upper:
            direction[i] = 1
            st[i] = final_lower
        elif direction[i - 1] == 1 and closes[i] < final_lower:
            direction[i] = -1
            st[i] = final_upper

    return st, direction

def _obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    obv = np.zeros(len(closes))
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    return obv

def _swing_highs(highs: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Returns array of swing high values (nan where not a swing high)."""
    out = np.full(len(highs), np.nan)
    for i in range(lookback, len(highs) - lookback):
        if highs[i] == max(highs[i - lookback : i + lookback + 1]):
            out[i] = highs[i]
    return out

def _swing_lows(lows: np.ndarray, lookback: int = 5) -> np.ndarray:
    out = np.full(len(lows), np.nan)
    for i in range(lookback, len(lows) - lookback):
        if lows[i] == min(lows[i - lookback : i + lookback + 1]):
            out[i] = lows[i]
    return out


# ────────────────────────────────────────────
# MAIN FEATURE BUILDER
# ────────────────────────────────────────────

def compute_features(candles: List[Dict[str, Any]],
                     swing_lookback: int = 20,
                     vol_ma_period: int = 20,
                     rsi_period: int = 14,
                     bb_period: int = 20,
                     bb_std: float = 2.0,
                     atr_period: int = 14,
                     ema_fast: int = 50,
                     ema_slow: int = 200,
                     macd_fast: int = 12,
                     macd_slow: int = 26,
                     macd_sig: int = 9,
                     st_period: int = 10,
                     st_mult: float = 3.0,
                     range_lookback: int = 10,
                     breakout_lookback: int = 20) -> Dict[str, Any]:
    """
    Compute all features from a candle window.
    Returns a flat dict. Uses only the provided candles (no lookahead).
    Minimum required candles: max(ema_slow, bb_period, ...) + a few.
    If insufficient data for an indicator, that feature is None.
    """

    if len(candles) < 5:
        return {}

    opens   = _arr(candles, "open")
    highs   = _arr(candles, "high")
    lows    = _arr(candles, "low")
    closes  = _arr(candles, "close")
    volumes = _arr(candles, "volume")
    n = len(closes)

    feats: Dict[str, Any] = {}

    # ── RSI ──────────────────────────────────────────────
    rsi_arr = _rsi(closes, rsi_period)
    rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
    rsi_prev = float(rsi_arr[-2]) if n >= 2 and not np.isnan(rsi_arr[-2]) else None
    feats["rsi"] = rsi_val
    feats["rsi_slope"] = (rsi_val - rsi_prev) if (rsi_val is not None and rsi_prev is not None) else None
    feats["rsi_overbought"] = (rsi_val >= 70) if rsi_val is not None else None
    feats["rsi_oversold"]   = (rsi_val <= 30) if rsi_val is not None else None

    # ── MACD ─────────────────────────────────────────────
    macd_line, macd_sig_arr, macd_hist = _macd(closes, macd_fast, macd_slow, macd_sig)
    mh = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
    mh_prev = float(macd_hist[-2]) if n >= 2 and not np.isnan(macd_hist[-2]) else None
    ml = float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
    ms = float(macd_sig_arr[-1]) if not np.isnan(macd_sig_arr[-1]) else None
    feats["macd_hist"] = mh
    feats["macd_hist_slope"] = (mh - mh_prev) if (mh is not None and mh_prev is not None) else None
    feats["macd_cross_up"]   = (ml is not None and ms is not None and
                                 mh_prev is not None and mh is not None and
                                 mh_prev < 0 <= mh)
    feats["macd_cross_down"] = (ml is not None and ms is not None and
                                 mh_prev is not None and mh is not None and
                                 mh_prev > 0 >= mh)

    # ── EMA 50 / 200 ─────────────────────────────────────
    ema50_arr  = _ema(closes, ema_fast)
    ema200_arr = _ema(closes, ema_slow)
    e50  = float(ema50_arr[-1])  if not np.isnan(ema50_arr[-1])  else None
    e200 = float(ema200_arr[-1]) if not np.isnan(ema200_arr[-1]) else None
    e50p  = float(ema50_arr[-2])  if n >= 2 and not np.isnan(ema50_arr[-2])  else None
    e200p = float(ema200_arr[-2]) if n >= 2 and not np.isnan(ema200_arr[-2]) else None
    cl = float(closes[-1])
    feats["ema50"]  = e50
    feats["ema200"] = e200
    feats["ema50_gt_ema200"] = (e50 > e200) if (e50 and e200) else None
    feats["close_gt_ema50"]  = (cl > e50)   if e50  else None
    feats["close_gt_ema200"] = (cl > e200)  if e200 else None
    feats["ema_dist_pct"] = ((e50 - e200) / e200 * 100) if (e50 and e200 and e200 != 0) else None
    # Golden cross / death cross on last bar
    feats["ema_golden_cross"] = (e50 is not None and e200 is not None and
                                  e50p is not None and e200p is not None and
                                  e50p <= e200p and e50 > e200)
    feats["ema_death_cross"]  = (e50 is not None and e200 is not None and
                                  e50p is not None and e200p is not None and
                                  e50p >= e200p and e50 < e200)

    # ── SUPERTREND ────────────────────────────────────────
    st_arr, st_dir = _supertrend(highs, lows, closes, st_period, st_mult)
    feats["supertrend_bull"] = (bool(st_dir[-1] == 1))  if not np.isnan(st_arr[-1]) else None
    feats["supertrend_bear"] = (bool(st_dir[-1] == -1)) if not np.isnan(st_arr[-1]) else None
    # Flip in last bar
    feats["supertrend_flipped_bull"] = (st_dir[-2] == -1 and st_dir[-1] == 1) if n >= 2 else None
    feats["supertrend_flipped_bear"] = (st_dir[-2] == 1  and st_dir[-1] == -1) if n >= 2 else None

    # ── BOLLINGER BANDS ───────────────────────────────────
    bb_upper, bb_mid, bb_lower, bb_width_pct = _bollinger(closes, bb_period, bb_std)
    bwp = float(bb_width_pct[-1]) if not np.isnan(bb_width_pct[-1]) else None
    bwp_prev = float(bb_width_pct[-5]) if n >= 5 and not np.isnan(bb_width_pct[-5]) else None
    bu = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
    bl = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None
    feats["bb_width_pct"] = bwp
    # Squeeze: current width in bottom 20% of recent 50-bar range
    if n >= 50:
        recent_widths = bb_width_pct[-50:]
        valid = recent_widths[~np.isnan(recent_widths)]
        if len(valid) > 5 and bwp is not None:
            pct20 = np.percentile(valid, 20)
            feats["bb_squeeze"] = (bwp <= pct20)
        else:
            feats["bb_squeeze"] = None
    else:
        feats["bb_squeeze"] = (bwp < 4.0) if bwp is not None else None
    feats["bb_width_contracting"] = (bwp < bwp_prev) if (bwp and bwp_prev) else None
    feats["close_above_upper_bb"] = (cl > bu) if bu else None
    feats["close_below_lower_bb"] = (cl < bl) if bl else None

    # ── ATR ───────────────────────────────────────────────
    atr_arr = _atr(highs, lows, closes, atr_period)
    atr_val  = float(atr_arr[-1])  if not np.isnan(atr_arr[-1])  else None
    atr_prev = float(atr_arr[-5])  if n >= 5 and not np.isnan(atr_arr[-5]) else None
    cl_sma   = float(_sma(closes, 20)[-1]) if n >= 20 else float(np.mean(closes))
    feats["atr"] = atr_val
    feats["atr_ratio"] = (atr_val / cl_sma) if (atr_val and cl_sma and cl_sma != 0) else None
    feats["atr_contracting"] = (atr_val < atr_prev) if (atr_val and atr_prev) else None
    feats["atr_ratio_lt_09"] = (feats["atr_ratio"] < 0.9) if feats["atr_ratio"] is not None else None

    # ── OBV ───────────────────────────────────────────────
    obv_arr = _obv(closes, volumes)
    obv_ma  = _sma(obv_arr, 20)
    feats["obv_rising"] = bool(obv_arr[-1] > obv_ma[-1]) if not np.isnan(obv_ma[-1]) else None
    feats["obv_falling"]= bool(obv_arr[-1] < obv_ma[-1]) if not np.isnan(obv_ma[-1]) else None

    # ── VOLUME ────────────────────────────────────────────
    vol_sma = _sma(volumes, vol_ma_period)
    vs = float(vol_sma[-1]) if not np.isnan(vol_sma[-1]) else None
    vn = float(volumes[-1])
    feats["volume_spike_ratio"] = (vn / vs) if (vs and vs > 0) else None
    feats["volume_spike_18"] = (feats["volume_spike_ratio"] > 1.8) if feats["volume_spike_ratio"] is not None else None
    feats["volume_spike_25"] = (feats["volume_spike_ratio"] > 2.5) if feats["volume_spike_ratio"] is not None else None
    # Contraction: current vol vs 5-bar avg
    if n >= 5:
        avg5 = float(np.mean(volumes[-5:]))
        feats["volume_contraction_ratio"] = (vn / avg5) if avg5 > 0 else None
        feats["volume_contracting"] = (feats["volume_contraction_ratio"] < 0.7) if feats["volume_contraction_ratio"] is not None else None
    else:
        feats["volume_contraction_ratio"] = None
        feats["volume_contracting"] = None

    # ── RANGE COMPRESSION ────────────────────────────────
    if n >= range_lookback:
        rc_highs = highs[-range_lookback:]
        rc_lows  = lows[-range_lookback:]
        rc = (max(rc_highs) - min(rc_lows)) / cl if cl > 0 else None
        feats["range_compression"] = rc
        # Compare to prior window
        if n >= range_lookback * 2:
            prior_h = highs[-range_lookback*2:-range_lookback]
            prior_l = lows[-range_lookback*2:-range_lookback]
            prior_rc = (max(prior_h) - min(prior_l)) / float(closes[-range_lookback])
            feats["range_compressed_vs_prior"] = (rc < prior_rc * 0.7) if rc else None
        else:
            feats["range_compressed_vs_prior"] = None
    else:
        feats["range_compression"] = None
        feats["range_compressed_vs_prior"] = None

    # ── BREAKOUT ─────────────────────────────────────────
    if n >= breakout_lookback + 1:
        prior_highs = highs[-(breakout_lookback + 1):-1]
        prior_lows  = lows[-(breakout_lookback + 1):-1]
        feats["breakout_up"]   = bool(cl > max(prior_highs))
        feats["breakout_down"] = bool(cl < min(prior_lows))
    else:
        feats["breakout_up"]   = None
        feats["breakout_down"] = None

    # ── LIQUIDITY SWEEP ──────────────────────────────────
    # Sweep high: current bar wicked above prior swing highs then closed back below
    if n >= swing_lookback + 2:
        prior_swings_h = _swing_highs(highs[:-1], lookback=5)
        valid_sh = prior_swings_h[~np.isnan(prior_swings_h)]
        if len(valid_sh) > 0:
            nearest_sh = float(np.max(valid_sh[-5:]) if len(valid_sh) >= 5 else np.max(valid_sh))
            wick_above = float(highs[-1]) > nearest_sh
            closed_below = float(closes[-1]) < nearest_sh
            feats["liquidity_sweep_high"] = bool(wick_above and closed_below)
        else:
            feats["liquidity_sweep_high"] = False

        prior_swings_l = _swing_lows(lows[:-1], lookback=5)
        valid_sl = prior_swings_l[~np.isnan(prior_swings_l)]
        if len(valid_sl) > 0:
            nearest_sl = float(np.min(valid_sl[-5:]) if len(valid_sl) >= 5 else np.min(valid_sl))
            wick_below = float(lows[-1]) < nearest_sl
            closed_above = float(closes[-1]) > nearest_sl
            feats["liquidity_sweep_low"] = bool(wick_below and closed_above)
        else:
            feats["liquidity_sweep_low"] = False
    else:
        feats["liquidity_sweep_high"] = None
        feats["liquidity_sweep_low"]  = None

    # ── CANDLE PATTERNS ───────────────────────────────────
    if n >= 2:
        o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
        o2, h2, l2, c2 = opens[-1], highs[-1], lows[-1], closes[-1]
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        range1 = h1 - l1 if h1 > l1 else 1e-9

        # Bullish engulfing
        feats["candle_bull_engulf"] = bool(
            c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1
        )
        # Bearish engulfing
        feats["candle_bear_engulf"] = bool(
            c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1
        )
        # Pinbar (hammer/shooting star)
        upper_wick2 = h2 - max(o2, c2)
        lower_wick2 = min(o2, c2) - l2
        feats["candle_bull_pinbar"] = bool(
            lower_wick2 > body2 * 2 and upper_wick2 < body2 * 0.5
        )
        feats["candle_bear_pinbar"] = bool(
            upper_wick2 > body2 * 2 and lower_wick2 < body2 * 0.5
        )
        # Inside bar
        feats["candle_inside_bar"] = bool(h2 <= h1 and l2 >= l1)
        # Inside bar breakout: close beyond prior range
        if n >= 3:
            h_prev2 = float(highs[-3])
            l_prev2 = float(lows[-3])
            feats["candle_ib_breakout_up"]   = bool(c2 > h_prev2)
            feats["candle_ib_breakout_down"]  = bool(c2 < l_prev2)
        else:
            feats["candle_ib_breakout_up"]   = None
            feats["candle_ib_breakout_down"]  = None
    else:
        for k in ["candle_bull_engulf","candle_bear_engulf","candle_bull_pinbar",
                  "candle_bear_pinbar","candle_inside_bar","candle_ib_breakout_up",
                  "candle_ib_breakout_down"]:
            feats[k] = None

    return feats


def compute_features_for_window(candles: List[Dict], window_sizes: List[int], **kwargs) -> Dict[str, Dict]:
    """
    For each window size W, take the last W candles and compute features.
    Returns dict keyed by window size.
    """
    results = {}
    for w in window_sizes:
        if len(candles) >= w:
            results[w] = compute_features(candles[-w:], **kwargs)
        else:
            results[w] = {}
    return results
