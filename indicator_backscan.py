#!/usr/bin/env python3
"""
indicator_backscan_v2.py - MTF Indicator Backscan v2 (Comprehensive)
=====================================================================
Detects pumps/dumps on 15m, then checks ALL indicators on 15m/1h/4h BEFORE the move.

INDICATORS PER TIMEFRAME (30+ indicators, 150+ signals):
  Trend:      Supertrend, EMA (9/21/55/200), EMA Ribbon, ADX/DI+/DI-, Ichimoku Cloud, Parabolic SAR
  Momentum:   RSI, Stochastic RSI, MACD, Williams %R, CCI, ROC, Momentum
  Volatility: Bollinger Bands, Keltner Channels, ATR, BB Squeeze vs Keltner
  Volume:     OBV, MFI, CMF, VWAP, Volume Ratio, Volume Trend
  Structure:  Support/Resistance proximity, Higher Highs/Lower Lows, Candle Patterns
  MTF:        Cross-timeframe confluence signals

OUTPUT:
  backscan_mtf_events.csv     - Every event with ALL indicators from ALL 3 timeframes
  backscan_mtf_summary.json   - Statistical analysis optimized for AI analysis
  backscan_mtf_indicator.pine - Auto-generated Pine Script v6

Usage:
    python indicator_backscan_v2.py
    python indicator_backscan_v2.py --symbols BTCUSDT,ETHUSDT --days 90
    python indicator_backscan_v2.py --top 50 --days 180
"""

import asyncio, aiohttp, json, time, csv, os, argparse
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# ============================================================
# CONFIGURATION
# ============================================================

BYBIT_API = "https://api.bybit.com"
RATE_DELAY = 0.12

# Entry TF = 15m, Confirm = 1h, Trend = 4h
TF_ENTRY = "15"
TF_CONFIRM = "60"
TF_TREND = "240"
ALL_TFS = [TF_ENTRY, TF_CONFIRM, TF_TREND]
TF_LABELS = {TF_ENTRY: "15m", TF_CONFIRM: "1h", TF_TREND: "4h"}

# Pump/Dump detection on 15m
PUMP_PCT = 3.0
DUMP_PCT = -3.0
MOVE_WINDOW = 4       # candles
COOLDOWN = 8          # candles between events
STRONG_PUMP_PCT = 5.0
STRONG_DUMP_PCT = -5.0
STRONG_WINDOW = 6
STRONG_COOLDOWN = 12

# Indicator parameters (matching your bot)
RSI_P = 14
EMA_F = 9; EMA_M = 21; EMA_S = 55; EMA_200 = 200
MACD_F = 12; MACD_SL = 26; MACD_SIG = 9
BB_P = 20; BB_STD = 2.0
ST_P = 10; ST_MULT = 3.0
STOCH_P = 14; STOCH_K = 3; STOCH_D = 3
ATR_P = 14; VOL_MA = 20
ADX_P = 14
CCI_P = 20
WR_P = 14       # Williams %R
KC_P = 20; KC_MULT = 1.5  # Keltner Channel
ICH_T = 9; ICH_K = 26; ICH_B = 52  # Ichimoku
PSAR_AF = 0.02; PSAR_MAX = 0.2  # Parabolic SAR
MFI_P = 14
CMF_P = 20
OBV_MA = 20
ROC_P = 12
MOM_P = 10

# ============================================================
# DATA
# ============================================================

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class MTFEvent:
    symbol: str
    move_type: str       # pump / dump
    strength: str        # normal / strong
    move_pct: float
    start_time: str
    end_time: str
    start_price: float
    end_price: float
    volume_ratio: float
    # Indicator states per timeframe (prefixed: 15m_, 1h_, 4h_)
    indicators: Dict = field(default_factory=dict)
    # MTF confluence signals (prefixed: mtf_)
    mtf_signals: Dict = field(default_factory=dict)


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def _ema(prices, period):
    if len(prices) < period: return []
    vals = [np.mean(prices[:period])]
    m = 2.0 / (period + 1)
    for i in range(period, len(prices)):
        vals.append((prices[i] - vals[-1]) * m + vals[-1])
    return vals

def _sma(data, period):
    if len(data) < period: return []
    return [np.mean(data[i-period+1:i+1]) for i in range(period-1, len(data))]

def _rsi(candles, period=RSI_P):
    if len(candles) < period + 2: return []
    cl = [c.close for c in candles]
    d = [cl[i] - cl[i-1] for i in range(1, len(cl))]
    g = [max(0, x) for x in d]; l = [max(0, -x) for x in d]
    ag = np.mean(g[:period]); al = np.mean(l[:period])
    out = []
    for i in range(period, len(d)):
        ag = (ag * (period-1) + g[i]) / period
        al = (al * (period-1) + l[i]) / period
        out.append(100.0 if al == 0 else 100.0 - 100.0/(1.0 + ag/al))
    return out

def _macd(candles, fast=MACD_F, slow=MACD_SL, sig=MACD_SIG):
    cl = [c.close for c in candles]
    if len(cl) < slow + sig: return [], [], []
    ef = _ema(cl, fast); es = _ema(cl, slow)
    o = len(ef) - len(es)
    ml = [f - s for f, s in zip(ef[o:], es)]
    if len(ml) < sig: return [], [], []
    sl = _ema(ml, sig); o2 = len(ml) - len(sl)
    mt = ml[o2:]
    return mt, sl, [m - s for m, s in zip(mt, sl)]

def _bb(candles, period=BB_P, mult=BB_STD):
    cl = [c.close for c in candles]
    if len(cl) < period: return [], [], []
    u, m, l = [], [], []
    for i in range(period-1, len(cl)):
        w = cl[i-period+1:i+1]; s = np.mean(w); d = np.std(w)
        m.append(s); u.append(s + mult*d); l.append(s - mult*d)
    return u, m, l

def _supertrend(candles, period=ST_P, mult=ST_MULT):
    if len(candles) < period + 2: return []
    cl = [c.close for c in candles]; hi = [c.high for c in candles]; lo = [c.low for c in candles]
    tr = [max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])) for i in range(1, len(candles))]
    if len(tr) < period: return []
    atr = [np.mean(tr[:period])]
    for i in range(period, len(tr)):
        atr.append((atr[-1]*(period-1) + tr[i]) / period)
    sigs = []; pu = pl = 0; pt = 1; si = period
    for i in range(len(atr)):
        ci = si + i
        if ci >= len(candles): break
        h2 = (hi[ci]+lo[ci])/2
        ub = h2 + mult*atr[i]; lb = h2 - mult*atr[i]
        if pu > 0:
            ub = min(ub, pu) if cl[ci-1] <= pu else ub
            lb = max(lb, pl) if cl[ci-1] >= pl else lb
        t = 1 if cl[ci] > ub else (-1 if cl[ci] < lb else pt)
        sigs.append(t)
        pu = ub; pl = lb; pt = t
    return sigs  # 1=bull, -1=bear

def _stoch_rsi(candles, rp=STOCH_P, sp=STOCH_P, kp=STOCH_K, dp=STOCH_D):
    rv = _rsi(candles, rp)
    if len(rv) < sp: return [], []
    sv = []
    for i in range(sp-1, len(rv)):
        w = rv[i-sp+1:i+1]; mn = min(w); mx = max(w)
        sv.append(((rv[i]-mn)/(mx-mn))*100 if mx-mn > 0 else 50.0)
    kv = _sma(sv, kp); dv = _sma(kv, dp)
    return kv, dv

def _atr(candles, period=ATR_P):
    if len(candles) < period + 2: return []
    tr = [max(candles[i].high-candles[i].low, abs(candles[i].high-candles[i-1].close),
              abs(candles[i].low-candles[i-1].close)) for i in range(1, len(candles))]
    a = [np.mean(tr[:period])]
    for i in range(period, len(tr)):
        a.append((a[-1]*(period-1) + tr[i]) / period)
    return a

def _adx(candles, period=ADX_P):
    """ADX with +DI and -DI"""
    if len(candles) < period*2 + 2: return {}, {}, {}
    hi = [c.high for c in candles]; lo = [c.low for c in candles]; cl = [c.close for c in candles]
    plus_dm = []; minus_dm = []; tr_list = []
    for i in range(1, len(candles)):
        up = hi[i] - hi[i-1]; down = lo[i-1] - lo[i]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr_list.append(max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1])))
    if len(tr_list) < period: return {}, {}, {}
    # Smoothed
    s_tr = [sum(tr_list[:period])]; s_pdm = [sum(plus_dm[:period])]; s_mdm = [sum(minus_dm[:period])]
    for i in range(period, len(tr_list)):
        s_tr.append(s_tr[-1] - s_tr[-1]/period + tr_list[i])
        s_pdm.append(s_pdm[-1] - s_pdm[-1]/period + plus_dm[i])
        s_mdm.append(s_mdm[-1] - s_mdm[-1]/period + minus_dm[i])
    pdi = [(100*s_pdm[i]/s_tr[i] if s_tr[i] > 0 else 0) for i in range(len(s_tr))]
    mdi = [(100*s_mdm[i]/s_tr[i] if s_tr[i] > 0 else 0) for i in range(len(s_tr))]
    dx = [(100*abs(pdi[i]-mdi[i])/(pdi[i]+mdi[i]) if pdi[i]+mdi[i] > 0 else 0) for i in range(len(pdi))]
    if len(dx) < period: return pdi, mdi, []
    adx = [np.mean(dx[:period])]
    for i in range(period, len(dx)):
        adx.append((adx[-1]*(period-1) + dx[i]) / period)
    return pdi, mdi, adx

def _williams_r(candles, period=WR_P):
    if len(candles) < period: return []
    out = []
    for i in range(period-1, len(candles)):
        hh = max(c.high for c in candles[i-period+1:i+1])
        ll = min(c.low for c in candles[i-period+1:i+1])
        out.append(((hh - candles[i].close) / (hh - ll)) * -100 if hh-ll > 0 else -50)
    return out

def _cci(candles, period=CCI_P):
    if len(candles) < period: return []
    tp = [(c.high + c.low + c.close)/3 for c in candles]
    out = []
    for i in range(period-1, len(tp)):
        w = tp[i-period+1:i+1]; m = np.mean(w)
        md = np.mean([abs(x - m) for x in w])
        out.append((tp[i] - m) / (0.015 * md) if md > 0 else 0)
    return out

def _roc(candles, period=ROC_P):
    if len(candles) < period + 1: return []
    cl = [c.close for c in candles]
    return [((cl[i] - cl[i-period]) / cl[i-period]) * 100 if cl[i-period] > 0 else 0
            for i in range(period, len(cl))]

def _momentum(candles, period=MOM_P):
    if len(candles) < period + 1: return []
    cl = [c.close for c in candles]
    return [cl[i] - cl[i-period] for i in range(period, len(cl))]

def _obv(candles):
    if len(candles) < 2: return []
    obv = [0]
    for i in range(1, len(candles)):
        if candles[i].close > candles[i-1].close:
            obv.append(obv[-1] + candles[i].volume)
        elif candles[i].close < candles[i-1].close:
            obv.append(obv[-1] - candles[i].volume)
        else:
            obv.append(obv[-1])
    return obv

def _mfi(candles, period=MFI_P):
    if len(candles) < period + 1: return []
    tp = [(c.high+c.low+c.close)/3 for c in candles]
    mf = [tp[i] * candles[i].volume for i in range(len(candles))]
    out = []
    for i in range(period, len(candles)):
        pmf = sum(mf[j] for j in range(i-period+1, i+1) if tp[j] > tp[j-1])
        nmf = sum(mf[j] for j in range(i-period+1, i+1) if tp[j] < tp[j-1])
        out.append(100.0 - 100.0/(1.0 + pmf/nmf) if nmf > 0 else 100.0)
    return out

def _cmf(candles, period=CMF_P):
    if len(candles) < period: return []
    out = []
    for i in range(period-1, len(candles)):
        w = candles[i-period+1:i+1]
        sv = sum(((c.close-c.low)-(c.high-c.close))/(c.high-c.low)*c.volume if c.high-c.low > 0 else 0 for c in w)
        tv = sum(c.volume for c in w)
        out.append(sv/tv if tv > 0 else 0)
    return out

def _vwap(candles):
    """Session VWAP (cumulative)"""
    if not candles: return []
    cum_tv = 0; cum_v = 0; out = []
    for c in candles:
        tp = (c.high + c.low + c.close) / 3
        cum_tv += tp * c.volume; cum_v += c.volume
        out.append(cum_tv / cum_v if cum_v > 0 else tp)
    return out

def _keltner(candles, period=KC_P, mult=KC_MULT, atr_period=ATR_P):
    cl = [c.close for c in candles]
    ema_vals = _ema(cl, period)
    atr_vals = _atr(candles, atr_period)
    if not ema_vals or not atr_vals: return [], [], []
    ln = min(len(ema_vals), len(atr_vals))
    ema_vals = ema_vals[-ln:]; atr_vals = atr_vals[-ln:]
    upper = [ema_vals[i] + mult * atr_vals[i] for i in range(ln)]
    lower = [ema_vals[i] - mult * atr_vals[i] for i in range(ln)]
    return upper, ema_vals, lower

def _ichimoku(candles, tenkan=ICH_T, kijun=ICH_K, senkou_b=ICH_B):
    if len(candles) < senkou_b + 1: return {}
    def _donchian(data, period, idx):
        s = max(0, idx - period + 1)
        return (max(c.high for c in data[s:idx+1]) + min(c.low for c in data[s:idx+1])) / 2
    i = len(candles) - 1
    tenkan_val = _donchian(candles, tenkan, i)
    kijun_val = _donchian(candles, kijun, i)
    span_a = (tenkan_val + kijun_val) / 2
    span_b = _donchian(candles, senkou_b, i)
    # Current cloud (shifted back 26 periods, so current cloud = calculated 26 bars ago)
    if i >= kijun:
        span_a_cur = (_donchian(candles, tenkan, i-kijun) + _donchian(candles, kijun, i-kijun)) / 2
        span_b_cur = _donchian(candles, senkou_b, i-kijun) if i >= senkou_b else span_b
    else:
        span_a_cur = span_a; span_b_cur = span_b
    cloud_top = max(span_a_cur, span_b_cur)
    cloud_bot = min(span_a_cur, span_b_cur)
    return {
        "tenkan": tenkan_val, "kijun": kijun_val,
        "span_a": span_a_cur, "span_b": span_b_cur,
        "cloud_top": cloud_top, "cloud_bot": cloud_bot,
        "price_vs_cloud": "above" if candles[-1].close > cloud_top else ("below" if candles[-1].close < cloud_bot else "inside"),
        "tk_cross_bull": tenkan_val > kijun_val,
    }

def _parabolic_sar(candles, af_start=PSAR_AF, af_max=PSAR_MAX):
    if len(candles) < 3: return []
    sigs = []
    trend = 1  # 1 = up, -1 = down
    sar = candles[0].low; ep = candles[0].high; af = af_start
    for i in range(1, len(candles)):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            sar = min(sar, candles[i-1].low, candles[max(0,i-2)].low if i >= 2 else candles[i-1].low)
            if candles[i].low < sar:
                trend = -1; sar = ep; ep = candles[i].low; af = af_start
            else:
                if candles[i].high > ep: ep = candles[i].high; af = min(af + af_start, af_max)
        else:
            sar = max(sar, candles[i-1].high, candles[max(0,i-2)].high if i >= 2 else candles[i-1].high)
            if candles[i].high > sar:
                trend = 1; sar = ep; ep = candles[i].high; af = af_start
            else:
                if candles[i].low < ep: ep = candles[i].low; af = min(af + af_start, af_max)
        sigs.append(trend)  # 1 = bullish (SAR below), -1 = bearish (SAR above)
    return sigs


# ============================================================
# PER-TIMEFRAME ANALYZER (150+ signals per TF)
# ============================================================

def analyze_tf(candles, prefix):
    """Analyze all indicators for one timeframe. Returns dict with prefixed keys."""
    if len(candles) < 60: return {}
    r = {}
    cl = candles[-1].close
    pre = candles  # all candles up to before the move

    # ===== RSI =====
    rsi = _rsi(pre)
    if rsi:
        v = rsi[-1]; r[f"{prefix}_rsi"] = round(v, 1)
        r[f"{prefix}_rsi_os"] = v < 30; r[f"{prefix}_rsi_ob"] = v > 70
        r[f"{prefix}_rsi_extreme_os"] = v < 20; r[f"{prefix}_rsi_extreme_ob"] = v > 80
        r[f"{prefix}_rsi_bull_zone"] = 50 < v < 70; r[f"{prefix}_rsi_bear_zone"] = 30 < v < 50
        r[f"{prefix}_rsi_neutral"] = 45 <= v <= 55
        if len(rsi) >= 3: r[f"{prefix}_rsi_rising"] = rsi[-1] > rsi[-3]; r[f"{prefix}_rsi_falling"] = rsi[-1] < rsi[-3]
        if len(rsi) >= 6:
            r[f"{prefix}_rsi_accel_up"] = rsi[-1] > rsi[-3] > rsi[-6]
            r[f"{prefix}_rsi_accel_dn"] = rsi[-1] < rsi[-3] < rsi[-6]
        if len(rsi) >= 10 and len(pre) >= 10:
            ps = pre[-1].close - pre[-5].close; rs = rsi[-1] - rsi[-5]
            r[f"{prefix}_rsi_bull_div"] = ps < 0 and rs > 0
            r[f"{prefix}_rsi_bear_div"] = ps > 0 and rs < 0

    # ===== Stoch RSI =====
    kv, dv = _stoch_rsi(pre)
    if kv and dv:
        r[f"{prefix}_stoch_k"] = round(kv[-1], 1); r[f"{prefix}_stoch_d"] = round(dv[-1], 1)
        r[f"{prefix}_stoch_os"] = kv[-1] < 20 and dv[-1] < 20
        r[f"{prefix}_stoch_ob"] = kv[-1] > 80 and dv[-1] > 80
        if len(kv) >= 2 and len(dv) >= 2:
            r[f"{prefix}_stoch_bull_x"] = kv[-2] < dv[-2] and kv[-1] > dv[-1]
            r[f"{prefix}_stoch_bear_x"] = kv[-2] > dv[-2] and kv[-1] < dv[-1]

    # ===== MACD =====
    ml, sl, hl = _macd(pre)
    if ml and sl and hl:
        r[f"{prefix}_macd_above"] = ml[-1] > sl[-1]; r[f"{prefix}_macd_below"] = ml[-1] < sl[-1]
        r[f"{prefix}_macd_hist_pos"] = hl[-1] > 0; r[f"{prefix}_macd_hist_neg"] = hl[-1] < 0
        if len(hl) >= 2:
            r[f"{prefix}_macd_hist_rising"] = hl[-1] > hl[-2]; r[f"{prefix}_macd_hist_falling"] = hl[-1] < hl[-2]
        if len(ml) >= 2 and len(sl) >= 2:
            r[f"{prefix}_macd_bull_x"] = ml[-2] < sl[-2] and ml[-1] > sl[-1]
            r[f"{prefix}_macd_bear_x"] = ml[-2] > sl[-2] and ml[-1] < sl[-1]
        r[f"{prefix}_macd_above_zero"] = ml[-1] > 0; r[f"{prefix}_macd_below_zero"] = ml[-1] < 0
        if len(ml) >= 10 and len(pre) >= 10:
            ps = pre[-1].close - pre[-5].close; ms = ml[-1] - ml[-5]
            r[f"{prefix}_macd_bull_div"] = ps < 0 and ms > 0
            r[f"{prefix}_macd_bear_div"] = ps > 0 and ms < 0

    # ===== EMA =====
    closes = [c.close for c in pre]
    ef = _ema(closes, EMA_F); em = _ema(closes, EMA_M); es = _ema(closes, EMA_S)
    e200 = _ema(closes, EMA_200) if len(closes) >= EMA_200 else []
    if ef and es:
        r[f"{prefix}_ema_bull"] = ef[-1] > es[-1]; r[f"{prefix}_ema_bear"] = ef[-1] < es[-1]
        if len(ef) >= 2 and len(es) >= 2:
            r[f"{prefix}_ema_bull_x"] = ef[-2] <= es[-2] and ef[-1] > es[-1]
            r[f"{prefix}_ema_bear_x"] = ef[-2] >= es[-2] and ef[-1] < es[-1]
    if ef and em and es:
        r[f"{prefix}_ribbon_bull"] = ef[-1] > em[-1] > es[-1]
        r[f"{prefix}_ribbon_bear"] = ef[-1] < em[-1] < es[-1]
        r[f"{prefix}_ribbon_mixed"] = not (ef[-1] > em[-1] > es[-1]) and not (ef[-1] < em[-1] < es[-1])
        if es[-1] > 0:
            sp = abs((ef[-1] - es[-1]) / es[-1] * 100)
            r[f"{prefix}_ema_squeeze"] = sp < 0.5; r[f"{prefix}_ema_spread"] = round(sp, 2)
    if e200:
        r[f"{prefix}_above_200ema"] = cl > e200[-1]; r[f"{prefix}_below_200ema"] = cl < e200[-1]
    r[f"{prefix}_price_vs_ema9"] = round((cl / ef[-1] - 1) * 100, 2) if ef else 0
    r[f"{prefix}_price_vs_ema21"] = round((cl / em[-1] - 1) * 100, 2) if em else 0

    # ===== Bollinger Bands =====
    bu, bm, bl = _bb(pre)
    if bu and bl and bm:
        bw = (bu[-1] - bl[-1]) / bm[-1] * 100 if bm[-1] > 0 else 0
        r[f"{prefix}_bb_bw"] = round(bw, 2)
        r[f"{prefix}_bb_above_up"] = cl > bu[-1]; r[f"{prefix}_bb_below_lo"] = cl < bl[-1]
        r[f"{prefix}_bb_near_up"] = cl > bu[-1] * 0.995; r[f"{prefix}_bb_near_lo"] = cl < bl[-1] * 1.005
        r[f"{prefix}_bb_squeeze"] = bw < 3.0; r[f"{prefix}_bb_wide"] = bw > 10
        pct_b = (cl - bl[-1]) / (bu[-1] - bl[-1]) if bu[-1] - bl[-1] > 0 else 0.5
        r[f"{prefix}_bb_pctb"] = round(pct_b, 2)
        if len(bu) >= 5:
            pbw = (bu[-5] - bl[-5]) / bm[-5] * 100 if bm[-5] > 0 else 0
            r[f"{prefix}_bb_expanding"] = bw > pbw; r[f"{prefix}_bb_contracting"] = bw < pbw

    # ===== Supertrend =====
    st = _supertrend(pre)
    if st:
        r[f"{prefix}_st_bull"] = st[-1] == 1; r[f"{prefix}_st_bear"] = st[-1] == -1
        if len(st) >= 2:
            r[f"{prefix}_st_flip_bull"] = st[-2] == -1 and st[-1] == 1
            r[f"{prefix}_st_flip_bear"] = st[-2] == 1 and st[-1] == -1
        cons = 1
        for i in range(len(st)-2, -1, -1):
            if st[i] == st[-1]: cons += 1
            else: break
        r[f"{prefix}_st_strong"] = cons >= 5; r[f"{prefix}_st_streak"] = cons

    # ===== ADX / DI =====
    pdi, mdi, adx = _adx(pre)
    if adx:
        r[f"{prefix}_adx"] = round(adx[-1], 1)
        r[f"{prefix}_adx_strong_trend"] = adx[-1] > 25; r[f"{prefix}_adx_very_strong"] = adx[-1] > 40
        r[f"{prefix}_adx_weak"] = adx[-1] < 20; r[f"{prefix}_adx_no_trend"] = adx[-1] < 15
    if pdi and mdi:
        r[f"{prefix}_di_bull"] = pdi[-1] > mdi[-1]; r[f"{prefix}_di_bear"] = pdi[-1] < mdi[-1]
        if len(pdi) >= 2 and len(mdi) >= 2:
            r[f"{prefix}_di_bull_x"] = pdi[-2] < mdi[-2] and pdi[-1] > mdi[-1]
            r[f"{prefix}_di_bear_x"] = pdi[-2] > mdi[-2] and pdi[-1] < mdi[-1]

    # ===== Williams %R =====
    wr = _williams_r(pre)
    if wr:
        r[f"{prefix}_wr"] = round(wr[-1], 1)
        r[f"{prefix}_wr_os"] = wr[-1] < -80; r[f"{prefix}_wr_ob"] = wr[-1] > -20
        r[f"{prefix}_wr_extreme_os"] = wr[-1] < -95; r[f"{prefix}_wr_extreme_ob"] = wr[-1] > -5

    # ===== CCI =====
    cci = _cci(pre)
    if cci:
        r[f"{prefix}_cci"] = round(cci[-1], 1)
        r[f"{prefix}_cci_ob"] = cci[-1] > 100; r[f"{prefix}_cci_os"] = cci[-1] < -100
        r[f"{prefix}_cci_extreme_ob"] = cci[-1] > 200; r[f"{prefix}_cci_extreme_os"] = cci[-1] < -200

    # ===== ROC =====
    roc = _roc(pre)
    if roc:
        r[f"{prefix}_roc"] = round(roc[-1], 2)
        r[f"{prefix}_roc_pos"] = roc[-1] > 0; r[f"{prefix}_roc_neg"] = roc[-1] < 0
        r[f"{prefix}_roc_strong_up"] = roc[-1] > 2; r[f"{prefix}_roc_strong_dn"] = roc[-1] < -2

    # ===== Momentum =====
    mom = _momentum(pre)
    if mom:
        r[f"{prefix}_mom_pos"] = mom[-1] > 0; r[f"{prefix}_mom_neg"] = mom[-1] < 0
        if len(mom) >= 3:
            r[f"{prefix}_mom_accel"] = mom[-1] > mom[-2] > mom[-3]
            r[f"{prefix}_mom_decel"] = mom[-1] < mom[-2] < mom[-3]

    # ===== OBV =====
    obv = _obv(pre)
    if obv and len(obv) >= OBV_MA + 1:
        obv_ma = _sma(obv, OBV_MA)
        if obv_ma:
            r[f"{prefix}_obv_above_ma"] = obv[-1] > obv_ma[-1]
            r[f"{prefix}_obv_below_ma"] = obv[-1] < obv_ma[-1]
        if len(obv) >= 5:
            r[f"{prefix}_obv_rising"] = obv[-1] > obv[-5]; r[f"{prefix}_obv_falling"] = obv[-1] < obv[-5]
        # OBV divergence
        if len(obv) >= 10 and len(pre) >= 10:
            ps = pre[-1].close - pre[-5].close; os_ = obv[-1] - obv[-5]
            r[f"{prefix}_obv_bull_div"] = ps < 0 and os_ > 0
            r[f"{prefix}_obv_bear_div"] = ps > 0 and os_ < 0

    # ===== MFI =====
    mfi = _mfi(pre)
    if mfi:
        r[f"{prefix}_mfi"] = round(mfi[-1], 1)
        r[f"{prefix}_mfi_os"] = mfi[-1] < 20; r[f"{prefix}_mfi_ob"] = mfi[-1] > 80

    # ===== CMF =====
    cmf = _cmf(pre)
    if cmf:
        r[f"{prefix}_cmf"] = round(cmf[-1], 3)
        r[f"{prefix}_cmf_pos"] = cmf[-1] > 0.05; r[f"{prefix}_cmf_neg"] = cmf[-1] < -0.05
        r[f"{prefix}_cmf_strong_buy"] = cmf[-1] > 0.15; r[f"{prefix}_cmf_strong_sell"] = cmf[-1] < -0.15

    # ===== VWAP =====
    vwap = _vwap(pre[-200:])  # last 200 candles for VWAP
    if vwap:
        r[f"{prefix}_above_vwap"] = cl > vwap[-1]; r[f"{prefix}_below_vwap"] = cl < vwap[-1]
        if vwap[-1] > 0:
            r[f"{prefix}_vwap_dist"] = round((cl / vwap[-1] - 1) * 100, 2)

    # ===== Keltner Channels =====
    ku, km, kl = _keltner(pre)
    if ku and kl:
        r[f"{prefix}_kc_above_up"] = cl > ku[-1]; r[f"{prefix}_kc_below_lo"] = cl < kl[-1]
        r[f"{prefix}_kc_inside"] = kl[-1] <= cl <= ku[-1]
        # BB Squeeze vs Keltner (TTM Squeeze concept)
        if bu and bl:
            r[f"{prefix}_ttm_squeeze"] = bl[-1] > kl[-1] and bu[-1] < ku[-1]  # BB inside KC = squeeze
            r[f"{prefix}_ttm_no_squeeze"] = not (bl[-1] > kl[-1] and bu[-1] < ku[-1])

    # ===== Ichimoku Cloud =====
    ich = _ichimoku(pre)
    if ich:
        r[f"{prefix}_ich_above_cloud"] = ich["price_vs_cloud"] == "above"
        r[f"{prefix}_ich_below_cloud"] = ich["price_vs_cloud"] == "below"
        r[f"{prefix}_ich_in_cloud"] = ich["price_vs_cloud"] == "inside"
        r[f"{prefix}_ich_tk_bull"] = ich["tk_cross_bull"]
        r[f"{prefix}_ich_tk_bear"] = not ich["tk_cross_bull"]
        if cl > 0:
            cloud_size = abs(ich["cloud_top"] - ich["cloud_bot"]) / cl * 100
            r[f"{prefix}_ich_cloud_thick"] = cloud_size > 2
            r[f"{prefix}_ich_cloud_thin"] = cloud_size < 0.5

    # ===== Parabolic SAR =====
    psar = _parabolic_sar(pre)
    if psar:
        r[f"{prefix}_psar_bull"] = psar[-1] == 1; r[f"{prefix}_psar_bear"] = psar[-1] == -1
        if len(psar) >= 2:
            r[f"{prefix}_psar_flip_bull"] = psar[-2] == -1 and psar[-1] == 1
            r[f"{prefix}_psar_flip_bear"] = psar[-2] == 1 and psar[-1] == -1

    # ===== Volume =====
    vm = _sma([c.volume for c in pre], VOL_MA)
    if vm and pre[-1].volume > 0 and vm[-1] > 0:
        vr = pre[-1].volume / vm[-1]; r[f"{prefix}_vol_ratio"] = round(vr, 2)
        r[f"{prefix}_vol_spike"] = vr > 2.0; r[f"{prefix}_vol_high"] = vr > 1.5
        r[f"{prefix}_vol_low"] = vr < 0.5; r[f"{prefix}_vol_dry"] = vr < 0.3
    if len(pre) >= 5:
        rv = [c.volume for c in pre[-5:]]
        r[f"{prefix}_vol_increasing"] = all(rv[i] >= rv[i-1] * 0.95 for i in range(1, len(rv)))
        r[f"{prefix}_vol_decreasing"] = all(rv[i] <= rv[i-1] * 1.05 for i in range(1, len(rv)))

    # ===== ATR / Volatility =====
    at = _atr(pre)
    if at and cl > 0:
        ap = at[-1] / cl * 100; r[f"{prefix}_atr_pct"] = round(ap, 3)
        r[f"{prefix}_high_vol"] = ap > 2.0; r[f"{prefix}_low_vol"] = ap < 0.5
        if len(at) >= 5:
            r[f"{prefix}_atr_expanding"] = at[-1] > at[-5]
            r[f"{prefix}_atr_contracting"] = at[-1] < at[-5]

    # ===== Price Structure =====
    if len(pre) >= 10:
        highs = [c.high for c in pre[-10:]]; lows = [c.low for c in pre[-10:]]
        r[f"{prefix}_hh"] = highs[-1] > max(highs[:-1])  # higher high
        r[f"{prefix}_ll"] = lows[-1] < min(lows[:-1])    # lower low
        r[f"{prefix}_hl"] = lows[-1] > min(lows[:-3])    # higher low
        r[f"{prefix}_lh"] = highs[-1] < max(highs[:-3])  # lower high
    if len(pre) >= 20:
        h20 = max(c.high for c in pre[-20:]); l20 = min(c.low for c in pre[-20:])
        if h20 - l20 > 0:
            pos = (cl - l20) / (h20 - l20); r[f"{prefix}_range_pos"] = round(pos, 2)
            r[f"{prefix}_near_range_high"] = pos > 0.9; r[f"{prefix}_near_range_low"] = pos < 0.1

    # ===== Candlestick Patterns =====
    if len(pre) >= 3:
        c1, c2, c3 = pre[-3], pre[-2], pre[-1]
        r[f"{prefix}_bull_engulf"] = (c2.close < c2.open and c3.close > c3.open and c3.close > c2.open and c3.open < c2.close)
        r[f"{prefix}_bear_engulf"] = (c2.close > c2.open and c3.close < c3.open and c3.close < c2.open and c3.open > c2.close)
        bd = abs(c3.close - c3.open); lw = min(c3.open, c3.close) - c3.low; uw = c3.high - max(c3.open, c3.close)
        if bd > 0:
            r[f"{prefix}_hammer"] = lw > bd * 2 and uw < bd * 0.5
            r[f"{prefix}_shooting_star"] = uw > bd * 2 and lw < bd * 0.5
            r[f"{prefix}_doji"] = bd < (c3.high - c3.low) * 0.1 if c3.high - c3.low > 0 else False
        r[f"{prefix}_3green"] = all(pre[-i].close > pre[-i].open for i in range(1, 4))
        r[f"{prefix}_3red"] = all(pre[-i].close < pre[-i].open for i in range(1, 4))
        # Big candle (momentum)
        if len(pre) >= 20:
            avg_body = np.mean([abs(c.close - c.open) for c in pre[-20:]])
            if avg_body > 0:
                r[f"{prefix}_big_bull_candle"] = (c3.close - c3.open) > avg_body * 2
                r[f"{prefix}_big_bear_candle"] = (c3.open - c3.close) > avg_body * 2

    return r


# ============================================================
# MTF CONFLUENCE ANALYZER
# ============================================================

def analyze_mtf(e15, c1h, t4h):
    """Cross-timeframe alignment analysis"""
    m = {}

    # Supertrend alignment
    sb = [e15.get("15m_st_bull", False), c1h.get("1h_st_bull", False), t4h.get("4h_st_bull", False)]
    se = [e15.get("15m_st_bear", False), c1h.get("1h_st_bear", False), t4h.get("4h_st_bear", False)]
    m["mtf_st_all_bull"] = all(sb); m["mtf_st_all_bear"] = all(se)
    m["mtf_st_2of3_bull"] = sum(sb) >= 2; m["mtf_st_2of3_bear"] = sum(se) >= 2

    # EMA ribbon alignment
    rb = [e15.get("15m_ribbon_bull", False), c1h.get("1h_ribbon_bull", False), t4h.get("4h_ribbon_bull", False)]
    re = [e15.get("15m_ribbon_bear", False), c1h.get("1h_ribbon_bear", False), t4h.get("4h_ribbon_bear", False)]
    m["mtf_ribbon_all_bull"] = all(rb); m["mtf_ribbon_all_bear"] = all(re)

    # MACD alignment
    mb = [e15.get("15m_macd_above", False), c1h.get("1h_macd_above", False), t4h.get("4h_macd_above", False)]
    me = [e15.get("15m_macd_below", False), c1h.get("1h_macd_below", False), t4h.get("4h_macd_below", False)]
    m["mtf_macd_all_bull"] = all(mb); m["mtf_macd_all_bear"] = all(me)

    # RSI confluence
    r15 = e15.get("15m_rsi", 50); r1h = c1h.get("1h_rsi", 50); r4h = t4h.get("4h_rsi", 50)
    m["mtf_rsi_all_bull"] = r15 > 50 and r1h > 50 and r4h > 50
    m["mtf_rsi_all_bear"] = r15 < 50 and r1h < 50 and r4h < 50
    m["mtf_rsi_dip_uptrend"] = r15 < 35 and r1h > 45 and r4h > 50
    m["mtf_rsi_rally_downtrend"] = r15 > 65 and r1h < 55 and r4h < 50
    m["mtf_rsi_all_os"] = r15 < 30 and r1h < 35 and r4h < 40
    m["mtf_rsi_all_ob"] = r15 > 70 and r1h > 65 and r4h > 60

    # ADX trend strength
    adx_4h = t4h.get("4h_adx", 0); adx_1h = c1h.get("1h_adx", 0)
    m["mtf_strong_trend"] = adx_4h > 25 and adx_1h > 25
    m["mtf_no_trend"] = adx_4h < 20 and adx_1h < 20

    # Ichimoku alignment
    ib = [e15.get("15m_ich_above_cloud", False), c1h.get("1h_ich_above_cloud", False), t4h.get("4h_ich_above_cloud", False)]
    ie = [e15.get("15m_ich_below_cloud", False), c1h.get("1h_ich_below_cloud", False), t4h.get("4h_ich_below_cloud", False)]
    m["mtf_ich_all_bull"] = all(ib); m["mtf_ich_all_bear"] = all(ie)

    # Parabolic SAR alignment
    pb = [e15.get("15m_psar_bull", False), c1h.get("1h_psar_bull", False), t4h.get("4h_psar_bull", False)]
    pe = [e15.get("15m_psar_bear", False), c1h.get("1h_psar_bear", False), t4h.get("4h_psar_bear", False)]
    m["mtf_psar_all_bull"] = all(pb); m["mtf_psar_all_bear"] = all(pe)

    # Volume money flow
    cmf_pos = [e15.get("15m_cmf_pos", False), c1h.get("1h_cmf_pos", False), t4h.get("4h_cmf_pos", False)]
    cmf_neg = [e15.get("15m_cmf_neg", False), c1h.get("1h_cmf_neg", False), t4h.get("4h_cmf_neg", False)]
    m["mtf_cmf_all_pos"] = all(cmf_pos); m["mtf_cmf_all_neg"] = all(cmf_neg)

    # OBV alignment
    ob_r = [e15.get("15m_obv_rising", False), c1h.get("1h_obv_rising", False), t4h.get("4h_obv_rising", False)]
    ob_f = [e15.get("15m_obv_falling", False), c1h.get("1h_obv_falling", False), t4h.get("4h_obv_falling", False)]
    m["mtf_obv_all_rising"] = all(ob_r); m["mtf_obv_all_falling"] = all(ob_f)

    # TTM Squeeze multi-TF
    sq = [e15.get("15m_ttm_squeeze", False), c1h.get("1h_ttm_squeeze", False)]
    m["mtf_ttm_squeeze_multi"] = all(sq)

    # Volume confirmation
    m["mtf_vol_spike_confirmed"] = e15.get("15m_vol_spike", False) and (c1h.get("1h_vol_high", False) or t4h.get("4h_vol_high", False))

    # Divergence on any TF
    m["mtf_rsi_bull_div_any"] = any(d.get(f"{p}_rsi_bull_div", False) for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")])
    m["mtf_rsi_bear_div_any"] = any(d.get(f"{p}_rsi_bear_div", False) for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")])
    m["mtf_obv_bull_div_any"] = any(d.get(f"{p}_obv_bull_div", False) for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")])
    m["mtf_obv_bear_div_any"] = any(d.get(f"{p}_obv_bear_div", False) for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")])

    # Combined bull/bear signal counts
    bull_keys = ["st_bull", "ribbon_bull", "ema_bull", "macd_above", "psar_bull", "ich_above_cloud",
                 "di_bull", "obv_rising", "cmf_pos", "roc_pos", "mom_pos", "above_vwap"]
    bear_keys = ["st_bear", "ribbon_bear", "ema_bear", "macd_below", "psar_bear", "ich_below_cloud",
                 "di_bear", "obv_falling", "cmf_neg", "roc_neg", "mom_neg", "below_vwap"]
    bc = 0; bec = 0
    for k in bull_keys:
        for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")]:
            if d.get(f"{p}_{k}", False): bc += 1
    for k in bear_keys:
        for d, p in [(e15,"15m"),(c1h,"1h"),(t4h,"4h")]:
            if d.get(f"{p}_{k}", False): bec += 1
    m["mtf_bull_count"] = bc; m["mtf_bear_count"] = bec
    m["mtf_strong_bull"] = bc >= 20; m["mtf_strong_bear"] = bec >= 20
    m["mtf_moderate_bull"] = bc >= 14; m["mtf_moderate_bear"] = bec >= 14

    return m


# ============================================================
# BYBIT API
# ============================================================

async def fetch_symbols(session):
    syms = []
    url = f"{BYBIT_API}/v5/market/instruments-info"
    cursor = None
    while True:
        params = {"category": "linear", "limit": "1000"}
        if cursor: params["cursor"] = cursor
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0: break
                for inst in data.get("result", {}).get("list", []):
                    if inst.get("symbol", "").endswith("USDT") and inst.get("status") == "Trading":
                        syms.append(inst["symbol"])
                nc = data.get("result", {}).get("nextPageCursor")
                if not nc or nc == cursor: break
                cursor = nc; await asyncio.sleep(RATE_DELAY)
        except: break
    return sorted(syms)

async def fetch_klines(session, symbol, interval, days):
    all_c = []; end = int(time.time()*1000); start = int((time.time()-days*86400)*1000)
    url = f"{BYBIT_API}/v5/market/kline"; cur_end = end; mx = 300
    while cur_end > start and mx > 0:
        mx -= 1
        params = {"category":"linear","symbol":symbol,"interval":interval,"end":str(cur_end),"limit":"200"}
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                klines = data.get("result",{}).get("list",[])
                if not klines: break
                for k in klines:
                    ts,o,h,l,c,v = int(k[0]),float(k[1]),float(k[2]),float(k[3]),float(k[4]),float(k[5])
                    if ts >= start: all_c.append(Candle(ts,o,h,l,c,v))
                oldest = int(klines[-1][0])
                if oldest >= cur_end: break
                cur_end = oldest - 1
                await asyncio.sleep(RATE_DELAY)
        except: break
    seen = set(); unique = []
    for c in all_c:
        if c.timestamp not in seen: seen.add(c.timestamp); unique.append(c)
    return sorted(unique, key=lambda x: x.timestamp)


# ============================================================
# MOVE DETECTION + SCANNING
# ============================================================

def detect_moves(candles_15m):
    """Detect pumps and dumps on 15m candles"""
    moves = []
    cooldown_until = -1
    for i in range(len(candles_15m)):
        if i < cooldown_until: continue
        # Normal moves
        for w, pp, dp, cd, st in [(MOVE_WINDOW, PUMP_PCT, DUMP_PCT, COOLDOWN, "normal"),
                                    (STRONG_WINDOW, STRONG_PUMP_PCT, STRONG_DUMP_PCT, STRONG_COOLDOWN, "strong")]:
            if i + w >= len(candles_15m): continue
            start_p = candles_15m[i].close
            end_p = candles_15m[i+w].close
            if start_p == 0: continue
            pct = ((end_p - start_p) / start_p) * 100
            if pct >= pp:
                vols = [candles_15m[i+j].volume for j in range(w+1)]
                avg_v = np.mean(vols) if vols else 1
                peak_v = max(vols) if vols else 1
                moves.append({
                    "type": "pump", "strength": st, "pct": round(pct, 2),
                    "start_idx": i, "end_idx": i+w,
                    "start_time": datetime.fromtimestamp(candles_15m[i].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "end_time": datetime.fromtimestamp(candles_15m[i+w].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "start_price": start_p, "end_price": end_p,
                    "vol_ratio": round(peak_v / avg_v, 2) if avg_v > 0 else 0,
                })
                cooldown_until = i + cd; break
            elif pct <= dp:
                vols = [candles_15m[i+j].volume for j in range(w+1)]
                avg_v = np.mean(vols) if vols else 1
                peak_v = max(vols) if vols else 1
                moves.append({
                    "type": "dump", "strength": st, "pct": round(pct, 2),
                    "start_idx": i, "end_idx": i+w,
                    "start_time": datetime.fromtimestamp(candles_15m[i].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "end_time": datetime.fromtimestamp(candles_15m[i+w].timestamp/1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
                    "start_price": start_p, "end_price": end_p,
                    "vol_ratio": round(peak_v / avg_v, 2) if avg_v > 0 else 0,
                })
                cooldown_until = i + cd; break
    return moves

def find_1h_idx(ts_15m, candles_1h):
    """Find the 1h candle that contains the 15m timestamp"""
    for i in range(len(candles_1h)-1, -1, -1):
        if candles_1h[i].timestamp <= ts_15m: return i
    return -1

def find_4h_idx(ts_15m, candles_4h):
    for i in range(len(candles_4h)-1, -1, -1):
        if candles_4h[i].timestamp <= ts_15m: return i
    return -1


async def scan_symbol(session, symbol, days):
    """Scan one symbol: fetch 3 TFs, detect moves on 15m, analyze MTF indicators"""
    events = []

    # Fetch all 3 timeframes
    c15m = await fetch_klines(session, symbol, TF_ENTRY, days)
    c1h = await fetch_klines(session, symbol, TF_CONFIRM, days)
    c4h = await fetch_klines(session, symbol, TF_TREND, days)

    if len(c15m) < 200 or len(c1h) < 60 or len(c4h) < 30:
        return events

    # Detect moves on 15m
    moves = detect_moves(c15m)

    for move in moves:
        idx = move["start_idx"]
        if idx < 60: continue  # need enough history

        # Get candles BEFORE the move for each TF
        pre_15m = c15m[:idx]
        ts_15m = c15m[idx].timestamp
        idx_1h = find_1h_idx(ts_15m, c1h)
        idx_4h = find_4h_idx(ts_15m, c4h)
        if idx_1h < 30 or idx_4h < 15: continue
        pre_1h = c1h[:idx_1h]
        pre_4h = c4h[:idx_4h]

        # Analyze indicators on all 3 timeframes
        ind_15m = analyze_tf(pre_15m, "15m")
        ind_1h = analyze_tf(pre_1h, "1h")
        ind_4h = analyze_tf(pre_4h, "4h")

        # MTF confluence
        mtf = analyze_mtf(ind_15m, ind_1h, ind_4h)

        # Merge all indicators
        all_ind = {}
        all_ind.update(ind_15m)
        all_ind.update(ind_1h)
        all_ind.update(ind_4h)
        all_ind.update(mtf)

        events.append(MTFEvent(
            symbol=symbol,
            move_type=move["type"],
            strength=move["strength"],
            move_pct=move["pct"],
            start_time=move["start_time"],
            end_time=move["end_time"],
            start_price=move["start_price"],
            end_price=move["end_price"],
            volume_ratio=move["vol_ratio"],
            indicators=all_ind,
            mtf_signals=mtf,
        ))

    return events


# ============================================================
# EXPORT & REPORTING
# ============================================================

def save_progress(events, scanned, current, total):
    os.makedirs("backscan_progress", exist_ok=True)
    if not events: return
    all_keys = sorted(set(k for e in events for k in e.indicators.keys()))
    with open("backscan_progress/events_progress.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["symbol","type","strength","move_pct","start_time","end_time","start_price","end_price","vol_ratio"] + all_keys
        w.writerow(hdr)
        for e in events:
            row = [e.symbol, e.move_type, e.strength, e.move_pct, e.start_time, e.end_time, e.start_price, e.end_price, e.volume_ratio]
            for k in all_keys: row.append(e.indicators.get(k, ""))
            w.writerow(row)
    with open("backscan_progress/progress.json","w") as f:
        json.dump({"saved_at": datetime.now(timezone.utc).isoformat(), "scanned": current, "total": total,
                    "events": len(events), "scanned_symbols": scanned}, f, indent=2, default=str)
    print(f"  💾 [{current}/{total}] Saved {len(events)} events → backscan_progress/")


def export_csv(events, filename="backscan_mtf_events.csv"):
    if not events: print("  No events to export."); return
    all_keys = sorted(set(k for e in events for k in e.indicators.keys()))
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["symbol","type","strength","move_pct","start_time","end_time","start_price","end_price","vol_ratio"] + all_keys
        w.writerow(hdr)
        for e in events:
            row = [e.symbol, e.move_type, e.strength, e.move_pct, e.start_time, e.end_time, e.start_price, e.end_price, e.volume_ratio]
            for k in all_keys: row.append(e.indicators.get(k, ""))
            w.writerow(row)
    print(f"\n  ✅ Exported {len(events)} events → {filename} ({len(all_keys)} indicator columns)")


def export_summary(events, filename="backscan_mtf_summary.json"):
    """Create comprehensive summary optimized for AI analysis"""
    if not events: return

    summary = {
        "meta": {
            "total_events": len(events),
            "unique_symbols": len(set(e.symbol for e in events)),
            "pumps": sum(1 for e in events if e.move_type == "pump"),
            "dumps": sum(1 for e in events if e.move_type == "dump"),
            "strong_pumps": sum(1 for e in events if e.move_type == "pump" and e.strength == "strong"),
            "strong_dumps": sum(1 for e in events if e.move_type == "dump" and e.strength == "strong"),
            "avg_pump_pct": round(np.mean([e.move_pct for e in events if e.move_type == "pump"]), 2),
            "avg_dump_pct": round(np.mean([e.move_pct for e in events if e.move_type == "dump"]), 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "indicator_hit_rates": {},
        "mtf_confluence_rates": {},
        "numeric_distributions": {},
        "top_combos": {},
    }

    pumps = [e for e in events if e.move_type == "pump"]
    dumps = [e for e in events if e.move_type == "dump"]

    # Boolean indicator hit rates
    all_bool_keys = set()
    for e in events:
        for k, v in e.indicators.items():
            if isinstance(v, bool): all_bool_keys.add(k)

    hit_rates = {}
    for k in sorted(all_bool_keys):
        pc = sum(1 for e in pumps if e.indicators.get(k, False)) / len(pumps) * 100 if pumps else 0
        dc = sum(1 for e in dumps if e.indicators.get(k, False)) / len(dumps) * 100 if dumps else 0
        diff = pc - dc
        hit_rates[k] = {"pump_rate": round(pc, 1), "dump_rate": round(dc, 1), "diff": round(diff, 1),
                         "predicts": "pump" if diff > 0 else "dump", "abs_edge": round(abs(diff), 1)}

    # Sort by absolute edge
    sorted_rates = dict(sorted(hit_rates.items(), key=lambda x: x[1]["abs_edge"], reverse=True))
    summary["indicator_hit_rates"] = sorted_rates

    # Separate by timeframe
    for tf_prefix in ["15m", "1h", "4h", "mtf"]:
        tf_rates = {k: v for k, v in sorted_rates.items() if k.startswith(tf_prefix)}
        top_pump = sorted([x for x in tf_rates.items() if x[1]["diff"] > 0], key=lambda x: x[1]["diff"], reverse=True)[:15]
        top_dump = sorted([x for x in tf_rates.items() if x[1]["diff"] < 0], key=lambda x: x[1]["diff"])[:15]
        summary[f"top_{tf_prefix}_pump_signals"] = {k: v for k, v in top_pump}
        summary[f"top_{tf_prefix}_dump_signals"] = {k: v for k, v in top_dump}

    # Numeric distributions
    numeric_keys = set()
    for e in events:
        for k, v in e.indicators.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool): numeric_keys.add(k)

    for k in sorted(numeric_keys):
        pv = [e.indicators.get(k) for e in pumps if isinstance(e.indicators.get(k), (int, float))]
        dv = [e.indicators.get(k) for e in dumps if isinstance(e.indicators.get(k), (int, float))]
        if pv and dv:
            summary["numeric_distributions"][k] = {
                "pump_mean": round(np.mean(pv), 2), "pump_median": round(np.median(pv), 2),
                "dump_mean": round(np.mean(dv), 2), "dump_median": round(np.median(dv), 2),
                "diff_mean": round(np.mean(pv) - np.mean(dv), 2),
            }

    with open(filename, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✅ Summary → {filename}")


def print_report(events):
    """Print console report"""
    if not events: return
    pumps = [e for e in events if e.move_type == "pump"]
    dumps = [e for e in events if e.move_type == "dump"]

    print(f"\n{'='*70}")
    print(f"  MTF BACKSCAN v2 RESULTS")
    print(f"{'='*70}")
    print(f"  Total events: {len(events)} ({len(pumps)} pumps, {len(dumps)} dumps)")
    print(f"  Symbols: {len(set(e.symbol for e in events))}")
    print(f"  Indicator columns per event: {len(set(k for e in events for k in e.indicators.keys()))}")

    # Top discriminating signals
    all_bool_keys = set()
    for e in events:
        for k, v in e.indicators.items():
            if isinstance(v, bool): all_bool_keys.add(k)

    results = []
    for k in all_bool_keys:
        pc = sum(1 for e in pumps if e.indicators.get(k, False)) / len(pumps) * 100 if pumps else 0
        dc = sum(1 for e in dumps if e.indicators.get(k, False)) / len(dumps) * 100 if dumps else 0
        results.append((k, pc, dc, pc - dc))

    for tf_prefix, tf_name in [("15m", "15m ENTRY"), ("1h", "1h CONFIRM"), ("4h", "4h TREND"), ("mtf", "MTF CONFLUENCE")]:
        tf_r = [x for x in results if x[0].startswith(tf_prefix)]
        if not tf_r: continue
        print(f"\n  {'─'*60}")
        print(f"  {tf_name} — Top Discriminating Signals")
        print(f"  {'─'*60}")
        top = sorted(tf_r, key=lambda x: abs(x[3]), reverse=True)[:12]
        for name, pc, dc, diff in top:
            direction = "PUMP" if diff > 0 else "DUMP"
            bar = ("🟢" if diff > 0 else "🔴") * min(int(abs(diff) / 3), 10)
            print(f"    {name:<40} p:{pc:5.1f}% d:{dc:5.1f}% diff:{diff:+6.1f}% → {direction} {bar}")


# ============================================================
# MAIN
# ============================================================

async def run(symbols=None, days=180, top_n=0, min_move=0.0):
    print("=" * 70)
    print("  MTF INDICATOR BACKSCAN v2 — Comprehensive Analysis")
    print("  15m Entry | 1h Confirmation | 4h Trend")
    print("=" * 70)
    print(f"  Lookback: {days} days")
    print(f"  Thresholds: pump≥{PUMP_PCT}% / dump≤{DUMP_PCT}% in {MOVE_WINDOW} candles (15m)")

    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if not symbols:
            print("\n  📡 Fetching symbols...")
            symbols = await fetch_symbols(session)
            if top_n > 0:
                print(f"  Filtering to top {top_n} by volume...")
                try:
                    async with session.get(f"{BYBIT_API}/v5/market/tickers", params={"category":"linear"}) as resp:
                        data = await resp.json()
                        vol_map = {t["symbol"]: float(t.get("turnover24h",0)) for t in data.get("result",{}).get("list",[])}
                        symbols = sorted(symbols, key=lambda s: vol_map.get(s,0), reverse=True)[:top_n]
                except: pass

        print(f"\n  ✅ Scanning {len(symbols)} symbols")
        print(f"  💾 Auto-saves every 10 symbols to backscan_progress/")
        print(f"  ⚡ Safe to Ctrl+C — progress saved!\n")

        all_events = []; scanned = []; total = len(symbols); stopped = False

        for idx, sym in enumerate(symbols):
            pct = ((idx+1)/total)*100
            print(f"  [{idx+1}/{total}] ({pct:.0f}%) {sym}...", end="", flush=True)
            try:
                evts = await scan_symbol(session, sym, days)
                if min_move > 0: evts = [e for e in evts if abs(e.move_pct) >= min_move]
                all_events.extend(evts); scanned.append(sym)
                pc = sum(1 for e in evts if e.move_type == "pump")
                dc = sum(1 for e in evts if e.move_type == "dump")
                print(f" {pc} pumps, {dc} dumps")
            except KeyboardInterrupt:
                print(f"\n\n  ⚠️ Ctrl+C — saving..."); stopped = True; break
            except Exception as e:
                print(f" ❌ {e}")

            if (idx+1) % 10 == 0:
                try: save_progress(all_events, scanned, idx+1, total)
                except: pass

    status = "STOPPED EARLY" if stopped else "COMPLETE"
    print(f"\n{'='*70}")
    print(f"  {status}: {len(all_events)} events from {len(scanned)}/{total} symbols")
    print(f"{'='*70}")

    export_csv(all_events)
    export_summary(all_events)
    print_report(all_events)
    save_progress(all_events, scanned, len(scanned), total)

    return all_events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTF Indicator Backscan v2")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--days", type=int, default=180, help="Days lookback (default 180)")
    parser.add_argument("--top", type=int, default=0, help="Top N by volume (0=all)")
    parser.add_argument("--min-move", type=float, default=0.0, help="Min move % to include")
    parser.add_argument("--pump-pct", type=float, default=None, help="Override pump threshold")
    parser.add_argument("--dump-pct", type=float, default=None, help="Override dump threshold")
    args = parser.parse_args()

    if args.pump_pct is not None: PUMP_PCT = args.pump_pct
    if args.dump_pct is not None: DUMP_PCT = args.dump_pct

    syms = args.symbols.split(",") if args.symbols else None
    asyncio.run(run(symbols=syms, days=args.days, top_n=args.top, min_move=args.min_move))
