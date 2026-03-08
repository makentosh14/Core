#!/usr/bin/env python3
"""
mtf_analyzer.py - Multi-Timeframe Confluence Analyzer
=======================================================
Combines indicator data from all 4 timeframes (5m/15m/1h/4h) into
a synchronized snapshot evaluated every 5 minutes.

RESEARCH ADDITIONS vs original:
  - market_regime field  (trending/ranging/high_vol/low_vol/compression/expansion etc.)
  - setup_type field     (early/confirmation/late/fake_breakout/squeeze_release etc.)
  - vol_state field      (spike/high/normal/low/dry)
  - trend_state field    (strong_bull/mild_bull/neutral/mild_bear/strong_bear)
  - structure_state      (breakout/breakdown/range/squeeze)
  These extra fields feed directly into pattern analysis bins.
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

from config import (
    REGIME_TRENDING_ADX, REGIME_STRONG_ADX,
    REGIME_HIGH_VOL_ATR, REGIME_LOW_VOL_ATR,
    REGIME_SQUEEZE_BW,
)


# ============================================================
# SNAPSHOT DATACLASS
# ============================================================

@dataclass
class MTFSnapshot:
    """One synchronized 5-minute research snapshot across all timeframes."""
    timestamp:  int
    dt_str:     str
    symbol:     str
    price:      float

    # Per-TF indicator dicts (flat, prefixed)
    ind_5m:  Dict = field(default_factory=dict)
    ind_15m: Dict = field(default_factory=dict)
    ind_1h:  Dict = field(default_factory=dict)
    ind_4h:  Dict = field(default_factory=dict)

    # MTF confluence signals
    mtf_signals: Dict = field(default_factory=dict)

    # v1 scoring
    pump_score: float = 0.0
    dump_score: float = 0.0
    net_score:  float = 0.0

    # v3 tier signals
    v3_signals: Dict = field(default_factory=dict)

    # Final signal assessment
    direction:   str   = ""    # "pump" | "dump" | ""
    signal_type: str   = ""    # e.g. "v3_pump_sniper"
    tier:        int   = 0     # 0=none 1=std 2=elite 3=sniper
    confidence:  float = 0.0   # 0–100

    # === RESEARCH FIELDS ===
    market_regime:  str = ""   # trending/ranging/high_vol/low_vol/compression/expansion/mixed
    setup_type:     str = ""   # early/confirmation/late/squeeze_release/reversal/fake_breakout/continuation/failed
    vol_state:      str = ""   # spike/high/normal/low/dry
    trend_state:    str = ""   # strong_bull/mild_bull/neutral/mild_bear/strong_bear
    structure_state:str = ""   # breakout/breakdown/range/squeeze/expansion


# ============================================================
# SAFE GETTERS
# ============================================================

def b(d: Dict, key: str) -> bool:
    """Safe bool getter."""
    if not d:
        return False
    return bool(d.get(key, False))


def g(d: Dict, key: str, default: float = 0.0) -> float:
    """Safe float getter."""
    if not d:
        return default
    v = d.get(key, default)
    if isinstance(v, (int, float)) and not math.isnan(float(v)):
        return float(v)
    return default


# ============================================================
# MTF CONFLUENCE SIGNALS
# ============================================================

def compute_mtf_confluence(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Dict:
    """Compute cross-timeframe confluence signals."""

    # --- Bull counts ---
    bull_signals = [
        b(d5m,  "5m_st_bull"),   b(d15m, "15m_st_bull"),
        b(d1h,  "1h_st_bull"),   b(d4h,  "4h_st_bull"),
        b(d5m,  "5m_ema_bull"),  b(d15m, "15m_ema_bull"),
        b(d1h,  "1h_ema_bull"),  b(d4h,  "4h_ema_bull"),
        b(d5m,  "5m_rsi_bull"),  b(d15m, "15m_rsi_bull"),
        b(d1h,  "1h_rsi_bull"),  b(d4h,  "4h_rsi_bull"),
        b(d5m,  "5m_macd_above"),b(d15m, "15m_macd_above"),
        b(d1h,  "1h_macd_above"),b(d4h,  "4h_macd_above"),
        b(d5m,  "5m_psar_bull"), b(d15m, "15m_psar_bull"),
        b(d1h,  "1h_psar_bull"), b(d4h,  "4h_psar_bull"),
        b(d5m,  "5m_ich_above_cloud"), b(d15m, "15m_ich_above_cloud"),
        b(d1h,  "1h_ich_above_cloud"),  b(d4h,  "4h_ich_above_cloud"),
    ]
    bear_signals = [
        b(d5m,  "5m_st_bear"),   b(d15m, "15m_st_bear"),
        b(d1h,  "1h_st_bear"),   b(d4h,  "4h_st_bear"),
        b(d5m,  "5m_ema_bear"),  b(d15m, "15m_ema_bear"),
        b(d1h,  "1h_ema_bear"),  b(d4h,  "4h_ema_bear"),
        b(d5m,  "5m_rsi_bear"),  b(d15m, "15m_rsi_bear"),
        b(d1h,  "1h_rsi_bear"),  b(d4h,  "4h_rsi_bear"),
        b(d5m,  "5m_macd_below"),b(d15m, "15m_macd_below"),
        b(d1h,  "1h_macd_below"),b(d4h,  "4h_macd_below"),
        b(d5m,  "5m_psar_bear"), b(d15m, "15m_psar_bear"),
        b(d1h,  "1h_psar_bear"), b(d4h,  "4h_psar_bear"),
        b(d5m,  "5m_ich_below_cloud"), b(d15m, "15m_ich_below_cloud"),
        b(d1h,  "1h_ich_below_cloud"),  b(d4h,  "4h_ich_below_cloud"),
    ]

    bull_count = sum(bull_signals)
    bear_count = sum(bear_signals)
    total      = len(bull_signals)

    # All-timeframe agreement flags
    mtf_st_all_bull    = all([b(d5m,"5m_st_bull"), b(d15m,"15m_st_bull"), b(d1h,"1h_st_bull"), b(d4h,"4h_st_bull")])
    mtf_st_all_bear    = all([b(d5m,"5m_st_bear"), b(d15m,"15m_st_bear"), b(d1h,"1h_st_bear"), b(d4h,"4h_st_bear")])
    mtf_ema_all_bull   = all([b(d5m,"5m_ema_bull"),b(d15m,"15m_ema_bull"),b(d1h,"1h_ema_bull"),b(d4h,"4h_ema_bull")])
    mtf_ema_all_bear   = all([b(d5m,"5m_ema_bear"),b(d15m,"15m_ema_bear"),b(d1h,"1h_ema_bear"),b(d4h,"4h_ema_bear")])
    mtf_ribbon_all_bull= all([b(d5m,"5m_ribbon_bull"),b(d15m,"15m_ribbon_bull"),b(d1h,"1h_ribbon_bull"),b(d4h,"4h_ribbon_bull")])
    mtf_ribbon_all_bear= all([b(d5m,"5m_ribbon_bear"),b(d15m,"15m_ribbon_bear"),b(d1h,"1h_ribbon_bear"),b(d4h,"4h_ribbon_bear")])
    mtf_macd_all_bull  = all([b(d5m,"5m_macd_above"), b(d15m,"15m_macd_above"), b(d1h,"1h_macd_above"), b(d4h,"4h_macd_above")])
    mtf_macd_all_bear  = all([b(d5m,"5m_macd_below"), b(d15m,"15m_macd_below"), b(d1h,"1h_macd_below"), b(d4h,"4h_macd_below")])
    mtf_psar_all_bull  = all([b(d5m,"5m_psar_bull"), b(d15m,"15m_psar_bull"), b(d1h,"1h_psar_bull"), b(d4h,"4h_psar_bull")])
    mtf_psar_all_bear  = all([b(d5m,"5m_psar_bear"), b(d15m,"15m_psar_bear"), b(d1h,"1h_psar_bear"), b(d4h,"4h_psar_bear")])
    mtf_ich_all_bull   = all([b(d5m,"5m_ich_above_cloud"),b(d15m,"15m_ich_above_cloud"),b(d1h,"1h_ich_above_cloud"),b(d4h,"4h_ich_above_cloud")])
    mtf_ich_all_bear   = all([b(d5m,"5m_ich_below_cloud"),b(d15m,"15m_ich_below_cloud"),b(d1h,"1h_ich_below_cloud"),b(d4h,"4h_ich_below_cloud")])
    mtf_obv_all_rising = all([b(d5m,"5m_obv_rising"),b(d15m,"15m_obv_rising"),b(d1h,"1h_obv_rising"),b(d4h,"4h_obv_rising")])

    # Moderate alignment (3 of 4 TFs)
    mtf_moderate_bull = bull_count >= (total * 0.65)
    mtf_moderate_bear = bear_count >= (total * 0.65)

    return {
        "mtf_bull_count":       bull_count,
        "mtf_bear_count":       bear_count,
        "mtf_total_signals":    total,
        "mtf_bull_pct":         round(bull_count / total * 100, 1) if total else 0,
        "mtf_bear_pct":         round(bear_count / total * 100, 1) if total else 0,
        "mtf_st_all_bull":      mtf_st_all_bull,
        "mtf_st_all_bear":      mtf_st_all_bear,
        "mtf_ema_all_bull":     mtf_ema_all_bull,
        "mtf_ema_all_bear":     mtf_ema_all_bear,
        "mtf_ribbon_all_bull":  mtf_ribbon_all_bull,
        "mtf_ribbon_all_bear":  mtf_ribbon_all_bear,
        "mtf_macd_all_bull":    mtf_macd_all_bull,
        "mtf_macd_all_bear":    mtf_macd_all_bear,
        "mtf_psar_all_bull":    mtf_psar_all_bull,
        "mtf_psar_all_bear":    mtf_psar_all_bear,
        "mtf_ich_all_bull":     mtf_ich_all_bull,
        "mtf_ich_all_bear":     mtf_ich_all_bear,
        "mtf_obv_all_rising":   mtf_obv_all_rising,
        "mtf_moderate_bull":    mtf_moderate_bull,
        "mtf_moderate_bear":    mtf_moderate_bear,
    }


# ============================================================
# V1 SCORING
# ============================================================

def compute_v1_scores(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Tuple[float, float, float]:
    """
    Weighted pump/dump scoring.
    Returns (pump_score, dump_score, net_score).
    """
    pump = 0.0
    dump = 0.0

    # Supertrend (high weight)
    if b(d4h, "4h_st_bull"):   pump += 2.0
    if b(d1h, "1h_st_bull"):   pump += 1.5
    if b(d15m,"15m_st_bull"):  pump += 1.0
    if b(d5m, "5m_st_bull"):   pump += 0.5
    if b(d4h, "4h_st_bear"):   dump += 2.0
    if b(d1h, "1h_st_bear"):   dump += 1.5
    if b(d15m,"15m_st_bear"):  dump += 1.0
    if b(d5m, "5m_st_bear"):   dump += 0.5

    # EMA alignment
    if b(d4h, "4h_ribbon_bull"):  pump += 1.5
    if b(d1h, "1h_ribbon_bull"):  pump += 1.0
    if b(d4h, "4h_ribbon_bear"):  dump += 1.5
    if b(d1h, "1h_ribbon_bear"):  dump += 1.0

    # MACD
    if b(d4h, "4h_macd_above"):   pump += 1.0
    if b(d4h, "4h_macd_hist_up"): pump += 0.5
    if b(d4h, "4h_macd_below"):   dump += 1.0
    if b(d4h, "4h_macd_hist_dn"): dump += 0.5

    # RSI zones
    rsi_4h = g(d4h, "4h_rsi", 50.0)
    if 45 < rsi_4h < 65:   pump += 0.5
    if 30 < rsi_4h < 55:   dump += 0.5
    if b(d4h, "4h_rsi_ob"): dump += 1.0
    if b(d4h, "4h_rsi_os"): pump += 1.0

    # Volume
    if b(d5m, "5m_vol_spike"):  pump += 1.0
    if b(d15m,"15m_vol_spike"): pump += 0.5

    # Ichimoku
    if b(d4h, "4h_ich_above_cloud"): pump += 1.0
    if b(d4h, "4h_ich_below_cloud"): dump += 1.0
    if b(d1h, "1h_ich_above_cloud"): pump += 0.5
    if b(d1h, "1h_ich_below_cloud"): dump += 0.5

    # PSAR
    if b(d4h, "4h_psar_bull"): pump += 0.5
    if b(d4h, "4h_psar_bear"): dump += 0.5

    # OBV
    if b(d4h, "4h_obv_rising") and b(d1h, "1h_obv_rising"):  pump += 0.5

    net = round(pump - dump, 2)
    return round(pump, 2), round(dump, 2), net


# ============================================================
# V3 COMBOS
# ============================================================

def detect_v3_combos(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Dict:
    """
    Detect tiered signal combos (sniper / elite / standard).
    Logic mirrors the original bot.
    """
    # Shared conditions
    four_h_3green  = b(d4h, "4h_three_green")
    vol_spike_5m   = b(d5m, "5m_vol_spike")
    high_vol_15m   = b(d15m,"15m_vol_high")
    low_vol_15m    = b(d15m,"15m_vol_low")
    pump_trend_gate = b(d4h,"4h_st_bull") or b(d4h,"4h_ema_bull")

    mtf_ribbon_all_bull = (b(d5m,"5m_ribbon_bull") and b(d15m,"15m_ribbon_bull") and
                           b(d1h,"1h_ribbon_bull")  and b(d4h,"4h_ribbon_bull"))
    mtf_ribbon_all_bear = (b(d5m,"5m_ribbon_bear") and b(d15m,"15m_ribbon_bear") and
                           b(d1h,"1h_ribbon_bear")  and b(d4h,"4h_ribbon_bear"))
    mtf_ich_all_bull    = (b(d5m,"5m_ich_above_cloud") and b(d15m,"15m_ich_above_cloud") and
                           b(d1h,"1h_ich_above_cloud")  and b(d4h,"4h_ich_above_cloud"))
    mtf_ich_all_bear    = (b(d5m,"5m_ich_below_cloud") and b(d15m,"15m_ich_below_cloud") and
                           b(d1h,"1h_ich_below_cloud")  and b(d4h,"4h_ich_below_cloud"))
    mtf_psar_all_bull   = (b(d5m,"5m_psar_bull") and b(d15m,"15m_psar_bull") and
                           b(d1h,"1h_psar_bull")  and b(d4h,"4h_psar_bull"))
    mtf_psar_all_bear   = (b(d5m,"5m_psar_bear") and b(d15m,"15m_psar_bear") and
                           b(d1h,"1h_psar_bear")  and b(d4h,"4h_psar_bear"))
    mtf_macd_all_bear   = (b(d5m,"5m_macd_below") and b(d15m,"15m_macd_below") and
                           b(d1h,"1h_macd_below")  and b(d4h,"4h_macd_below"))

    # ── PUMP SNIPER ──
    pump_sniper_1 = (four_h_3green and vol_spike_5m and
                     b(d5m,"5m_st_bull") and b(d15m,"15m_st_bull") and
                     b(d1h,"1h_st_bull"))
    pump_sniper_2 = (b(d4h,"4h_bb_squeeze") and b(d5m,"5m_st_bull_flip") and
                     b(d4h,"4h_st_bull")    and b(d15m,"15m_vol_high"))
    pump_sniper_3 = (mtf_ribbon_all_bull and mtf_ich_all_bull and b(d5m,"5m_rsi_os"))
    pump_sniper   = pump_sniper_1 or pump_sniper_2 or pump_sniper_3

    # ── PUMP ELITE ──
    pump_elite_1 = (b(d4h,"4h_st_bull") and b(d1h,"1h_st_bull") and
                    b(d5m,"5m_st_bull") and high_vol_15m and b(d4h,"4h_macd_above"))
    pump_elite_2 = (mtf_ribbon_all_bull and vol_spike_5m and b(d4h,"4h_rsi_os"))
    pump_elite_3 = (b(d4h,"4h_bb_squeeze") and b(d1h,"1h_bb_squeeze") and
                    b(d5m,"5m_st_bull") and pump_trend_gate)
    pump_elite_4 = (b(d4h,"4h_adx_strong") and b(d4h,"4h_di_bull") and
                    b(d15m,"15m_st_bull") and b(d5m,"5m_vol_high"))
    pump_elite_5 = (mtf_psar_all_bull and b(d1h,"1h_rsi_os") and b(d4h,"4h_st_bull"))
    pump_elite_6 = (b(d4h,"4h_ich_above_cloud") and b(d1h,"1h_ich_above_cloud") and
                    b(d5m,"5m_st_bull_flip"))
    pump_elite   = (pump_elite_1 or pump_elite_2 or pump_elite_3 or
                    pump_elite_4 or pump_elite_5 or pump_elite_6) and not pump_sniper

    # ── PUMP STD ──
    pump_std_1   = (b(d4h,"4h_st_bull") and b(d1h,"1h_st_bull") and b(d5m,"5m_rsi_os"))
    pump_std_2   = (b(d4h,"4h_macd_bull_x") and b(d4h,"4h_st_bull"))
    pump_std_3   = (b(d4h,"4h_three_green") and b(d4h,"4h_hh"))
    pump_pullback = (b(d4h,"4h_st_bull") and b(d1h,"1h_st_bull") and
                     b(d5m,"5m_st_bear") and b(d5m,"5m_rsi_os"))
    pump_std     = (pump_std_1 or pump_std_2 or pump_std_3) and not pump_sniper and not pump_elite
    raw_v3_pump  = pump_sniper or pump_elite or pump_std or pump_pullback

    # ── DUMP SNIPER ──
    dump_sniper_1 = (b(d4h,"4h_three_red") and vol_spike_5m and
                     b(d5m,"5m_st_bear") and b(d15m,"15m_st_bear") and b(d1h,"1h_st_bear"))
    dump_sniper_2 = (b(d4h,"4h_bb_squeeze") and b(d5m,"5m_st_bear_flip") and
                     b(d4h,"4h_st_bear")    and b(d15m,"15m_vol_high"))
    dump_sniper_3 = (mtf_ribbon_all_bear and mtf_ich_all_bear and b(d5m,"5m_rsi_ob"))
    dump_sniper   = dump_sniper_1 or dump_sniper_2 or dump_sniper_3

    # ── DUMP ELITE ──
    dump_elite_1 = (b(d4h,"4h_st_bear") and b(d1h,"1h_st_bear") and
                    b(d5m,"5m_st_bear") and high_vol_15m and b(d4h,"4h_macd_below"))
    dump_elite_2 = (mtf_ribbon_all_bear and vol_spike_5m and b(d4h,"4h_rsi_ob"))
    dump_elite_3 = (b(d4h,"4h_bb_squeeze") and b(d1h,"1h_bb_squeeze") and
                    b(d5m,"5m_st_bear") and b(d4h,"4h_st_bear"))
    dump_elite_4 = (b(d4h,"4h_adx_strong") and b(d4h,"4h_di_bear") and
                    b(d15m,"15m_st_bear") and b(d5m,"5m_vol_high"))
    dump_elite_5 = (mtf_psar_all_bear and b(d1h,"1h_rsi_ob") and b(d4h,"4h_st_bear"))
    dump_elite_6 = (b(d4h,"4h_ich_below_cloud") and b(d1h,"1h_ich_below_cloud") and
                    b(d5m,"5m_st_bear_flip"))
    dump_elite   = (dump_elite_1 or dump_elite_2 or dump_elite_3 or
                    dump_elite_4 or dump_elite_5 or dump_elite_6) and not dump_sniper

    # ── DUMP STD ──
    dump_std_1  = (low_vol_15m and mtf_ich_all_bear)
    dump_std_2  = (b(d1h,"1h_bb_below_lo") and b(d4h,"4h_rsi_os"))
    dump_std_3  = (b(d4h,"4h_three_red") and b(d4h,"4h_ll"))
    dump_std    = (dump_std_1 or dump_std_2 or dump_std_3) and not dump_sniper and not dump_elite
    raw_v3_dump = dump_sniper or dump_elite or dump_std

    pump_tier = 3 if pump_sniper else (2 if pump_elite else (1 if (pump_std or pump_pullback) else 0))
    dump_tier = 3 if dump_sniper else (2 if dump_elite else (1 if dump_std else 0))

    return {
        "raw_v3_pump":     raw_v3_pump,
        "raw_v3_dump":     raw_v3_dump,
        "pump_sniper":     pump_sniper,
        "pump_elite":      pump_elite,
        "pump_std":        pump_std,
        "pump_pullback":   pump_pullback,
        "dump_sniper":     dump_sniper,
        "dump_elite":      dump_elite,
        "dump_std":        dump_std,
        "pump_tier":       pump_tier,
        "dump_tier":       dump_tier,
        # context flags
        "four_h_3green":        four_h_3green,
        "pump_trend_gate":      pump_trend_gate,
        "mtf_obv_all_rising":   (b(d5m,"5m_obv_rising") and b(d15m,"15m_obv_rising") and
                                 b(d1h,"1h_obv_rising") and b(d4h,"4h_obv_rising")),
        "mtf_ribbon_all_bear":  mtf_ribbon_all_bear,
        "mtf_ich_all_bear":     mtf_ich_all_bear,
        "mtf_psar_all_bear":    mtf_psar_all_bear,
        "mtf_macd_all_bear":    mtf_macd_all_bear,
    }


# ============================================================
# MARKET REGIME CLASSIFICATION
# ============================================================

def classify_market_regime(d4h: Dict, d1h: Dict) -> str:
    """
    Classify the current market regime using 4h as the reference.
    Returns one of:
      trending_bull / trending_bear / ranging / high_vol /
      low_vol / compression / expansion / mixed
    """
    adx   = g(d4h, "4h_adx", 0.0)
    bb_bw = g(d4h, "4h_bb_bw", 5.0)
    atr_p = g(d4h, "4h_atr_pct", 0.015)
    st_bull = b(d4h, "4h_st_bull")
    st_bear = b(d4h, "4h_st_bear")

    # Check compression first (BB squeeze on 4h)
    if bb_bw < REGIME_SQUEEZE_BW:
        return "compression"

    # High volatility (ATR% above threshold)
    if atr_p > REGIME_HIGH_VOL_ATR:
        if adx > REGIME_TRENDING_ADX:
            return "trending_bull" if st_bull else "trending_bear"
        return "high_vol"

    # Low volatility
    if atr_p < REGIME_LOW_VOL_ATR:
        return "low_vol"

    # Trending
    if adx > REGIME_STRONG_ADX:
        return "trending_bull" if st_bull else "trending_bear"

    if adx > REGIME_TRENDING_ADX:
        # 1h confirms direction?
        if b(d1h, "1h_st_bull") and st_bull:
            return "trending_bull"
        if b(d1h, "1h_st_bear") and st_bear:
            return "trending_bear"
        return "mixed"

    # Expansion (BB recently fired)
    if bb_bw > 8.0:
        return "expansion"

    # Default = ranging
    return "ranging"


# ============================================================
# SETUP TYPE CLASSIFICATION
# ============================================================

def classify_setup_type(
    d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict,
    direction: str, v3: Dict,
) -> str:
    """
    Classify the setup type for research labeling.

    early           = signal on compressed/quiet market, move has not started
    confirmation    = lower TF aligns with higher TF, classic entry
    late            = price already extended, RSI overbought/oversold on fast TF
    squeeze_release = BB squeeze + ST flip
    reversal        = counter-trend signal after extended move
    fake_breakout   = price broke level but key TF indicator still opposing
    continuation    = pullback to support/EMA in existing trend
    failed          = (set later by outcome labeler)
    """
    if not direction:
        return ""

    is_pump = direction == "pump"

    # Squeeze release: BB was compressed and now ST flipped
    sq_4h  = b(d4h, "4h_bb_squeeze")
    sq_1h  = b(d1h, "1h_bb_squeeze")
    flip_5m = b(d5m, "5m_st_bull_flip" if is_pump else "5m_st_bear_flip")
    if (sq_4h or sq_1h) and flip_5m:
        return "squeeze_release"

    # Late setup: fast TF RSI overbought/oversold + slow TF already strongly aligned
    fast_rsi = g(d5m, "5m_rsi", 50.0)
    if is_pump and fast_rsi > 72 and b(d4h, "4h_st_bull"):
        return "late"
    if not is_pump and fast_rsi < 28 and b(d4h, "4h_st_bear"):
        return "late"

    # Reversal: fast TF signal AGAINST the higher TF trend
    if is_pump and b(d4h, "4h_st_bear") and b(d4h, "4h_ema_bear"):
        return "reversal"
    if not is_pump and b(d4h, "4h_st_bull") and b(d4h, "4h_ema_bull"):
        return "reversal"

    # Continuation / pullback: higher TF trending, lower TF pulled back
    if is_pump and b(d4h, "4h_st_bull") and b(d1h, "1h_st_bull") and b(d5m, "5m_st_bear"):
        return "continuation"
    if not is_pump and b(d4h, "4h_st_bear") and b(d1h, "1h_st_bear") and b(d5m, "5m_st_bull"):
        return "continuation"

    # Sniper = early signal (all aligned but not yet extended)
    if v3.get("pump_sniper") or v3.get("dump_sniper"):
        return "early"

    # Elite = confirmation
    if v3.get("pump_elite") or v3.get("dump_elite"):
        return "confirmation"

    # Std = generic confirmation
    return "confirmation"


# ============================================================
# VOLUME STATE
# ============================================================

def classify_vol_state(d5m: Dict, d15m: Dict) -> str:
    vol_ratio_5m  = g(d5m,  "5m_vol_ratio",  1.0)
    vol_ratio_15m = g(d15m, "15m_vol_ratio", 1.0)
    avg           = (vol_ratio_5m + vol_ratio_15m) / 2
    if avg > 3.0:   return "spike"
    if avg > 1.5:   return "high"
    if avg > 0.7:   return "normal"
    if avg > 0.4:   return "low"
    return "dry"


# ============================================================
# TREND STATE
# ============================================================

def classify_trend_state(d4h: Dict, d1h: Dict) -> str:
    adx = g(d4h, "4h_adx", 0.0)
    st_bull_4h = b(d4h, "4h_st_bull")
    st_bull_1h = b(d1h, "1h_st_bull")
    if adx > 35 and st_bull_4h and st_bull_1h:     return "strong_bull"
    if adx > 20 and st_bull_4h:                     return "mild_bull"
    if adx > 35 and not st_bull_4h and not st_bull_1h: return "strong_bear"
    if adx > 20 and not st_bull_4h:                 return "mild_bear"
    return "neutral"


# ============================================================
# STRUCTURE STATE
# ============================================================

def classify_structure_state(d4h: Dict, d1h: Dict, d5m: Dict) -> str:
    if b(d4h, "4h_bb_squeeze") and b(d1h, "1h_bb_squeeze"):
        return "squeeze"
    bw_4h = g(d4h, "4h_bb_bw", 5.0)
    if bw_4h > 10.0:
        return "expansion"
    if b(d5m, "5m_bb_above_up") and b(d5m, "5m_hh"):
        return "breakout"
    if b(d5m, "5m_bb_below_lo") and b(d5m, "5m_ll"):
        return "breakdown"
    return "range"


# ============================================================
# BUILD SYNCHRONIZED SNAPSHOT
# ============================================================

def build_snapshot(
    symbol:    str,
    timestamp: int,
    price:     float,
    ind_5m:    Dict,
    ind_15m:   Dict,
    ind_1h:    Dict,
    ind_4h:    Dict,
) -> MTFSnapshot:
    """
    Build a complete MTF research snapshot for one point in time.
    Combines all indicator data, MTF confluence, v1 scores, v3 combos,
    and the new research classification fields.
    """
    dt_str = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    mtf          = compute_mtf_confluence(ind_5m, ind_15m, ind_1h, ind_4h)
    pump_sc, dump_sc, net_sc = compute_v1_scores(ind_5m, ind_15m, ind_1h, ind_4h)
    v3           = detect_v3_combos(ind_5m, ind_15m, ind_1h, ind_4h)

    # Direction + tier
    direction    = ""
    signal_type  = ""
    tier         = 0

    if v3["raw_v3_pump"]:
        direction   = "pump"
        tier        = v3["pump_tier"]
        signal_type = ("v3_pump_sniper"   if v3["pump_sniper"]   else
                       "v3_pump_elite"    if v3["pump_elite"]    else
                       "v3_pump_pullback" if v3["pump_pullback"] else
                       "v3_pump_std")
    elif v3["raw_v3_dump"]:
        direction   = "dump"
        tier        = v3["dump_tier"]
        signal_type = ("v3_dump_sniper" if v3["dump_sniper"] else
                       "v3_dump_elite"  if v3["dump_elite"]  else
                       "v3_dump_std")
    elif net_sc >= 4.0:
        direction   = "pump"
        signal_type = "v1_strong_pump" if net_sc >= 7.0 else "v1_pump"
        tier        = 2 if net_sc >= 7.0 else 1
    elif net_sc <= -4.0:
        direction   = "dump"
        signal_type = "v1_strong_dump" if net_sc <= -7.0 else "v1_dump"
        tier        = 2 if net_sc <= -7.0 else 1

    # Confidence
    total_possible = mtf.get("mtf_total_signals", 24)
    if direction == "pump":
        confidence = min(100.0, (mtf["mtf_bull_count"] / total_possible) * 100 + tier * 10)
    elif direction == "dump":
        confidence = min(100.0, (mtf["mtf_bear_count"] / total_possible) * 100 + tier * 10)
    else:
        confidence = 0.0

    # Research classification fields
    market_regime   = classify_market_regime(ind_4h, ind_1h)
    setup_type      = classify_setup_type(ind_5m, ind_15m, ind_1h, ind_4h, direction, v3)
    vol_state       = classify_vol_state(ind_5m, ind_15m)
    trend_state     = classify_trend_state(ind_4h, ind_1h)
    structure_state = classify_structure_state(ind_4h, ind_1h, ind_5m)

    return MTFSnapshot(
        timestamp       = timestamp,
        dt_str          = dt_str,
        symbol          = symbol,
        price           = price,
        ind_5m          = ind_5m,
        ind_15m         = ind_15m,
        ind_1h          = ind_1h,
        ind_4h          = ind_4h,
        mtf_signals     = mtf,
        pump_score      = pump_sc,
        dump_score      = dump_sc,
        net_score       = net_sc,
        v3_signals      = v3,
        direction       = direction,
        signal_type     = signal_type,
        tier            = tier,
        confidence      = round(confidence, 1),
        market_regime   = market_regime,
        setup_type      = setup_type,
        vol_state       = vol_state,
        trend_state     = trend_state,
        structure_state = structure_state,
    )
