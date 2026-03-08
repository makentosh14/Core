#!/usr/bin/env python3
"""
mtf_analyzer.py - Multi-Timeframe Confluence Analyzer
=======================================================
Combines indicator data from all 4 timeframes (5m, 15m, 1h, 4h)
into a synchronized view evaluated every 5 minutes.

Implements:
  - MTF alignment detection (all TFs agree)
  - v1 scoring system (pump/dump scores)
  - v3 combo system (sniper/elite/std tiers)
  - Synchronized 5-minute snapshot generation
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class MTFSnapshot:
    """One synchronized 5-minute snapshot across all timeframes."""
    timestamp: int
    dt_str: str
    symbol: str
    price: float
    # Per-TF indicators (flat dicts)
    ind_5m: Dict = field(default_factory=dict)
    ind_15m: Dict = field(default_factory=dict)
    ind_1h: Dict = field(default_factory=dict)
    ind_4h: Dict = field(default_factory=dict)
    # MTF confluence signals
    mtf_signals: Dict = field(default_factory=dict)
    # v1 scores
    pump_score: float = 0.0
    dump_score: float = 0.0
    net_score: float = 0.0
    # v3 tier signals
    v3_signals: Dict = field(default_factory=dict)
    # Final assessment
    direction: str = ""           # "pump", "dump", ""
    signal_type: str = ""         # e.g., "v3_pump_sniper"
    tier: int = 0                 # 0=none, 1=std, 2=elite, 3=sniper
    confidence: float = 0.0       # 0-100


def b(d: Dict, key: str) -> bool:
    """Safe bool getter."""
    return bool(d.get(key, False)) if d else False


def g(d: Dict, key: str, default: float = 0.0) -> float:
    """Safe float getter."""
    v = d.get(key, default) if d else default
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    return default


# ============================================================
# MTF CONFLUENCE SIGNALS
# ============================================================

def compute_mtf_confluence(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Dict:
    """
    Compute cross-timeframe confluence signals.
    Exactly mirrors your bot's indicator_backscan mtf analysis.
    """
    m = {}

    # ── Supertrend alignment ──
    sb = [b(d5m, "5m_st_bull"), b(d15m, "15m_st_bull"), b(d1h, "1h_st_bull"), b(d4h, "4h_st_bull")]
    se = [b(d5m, "5m_st_bear"), b(d15m, "15m_st_bear"), b(d1h, "1h_st_bear"), b(d4h, "4h_st_bear")]
    m["mtf_st_all_bull"] = all(sb)
    m["mtf_st_all_bear"] = all(se)
    m["mtf_st_3of4_bull"] = sum(sb) >= 3
    m["mtf_st_3of4_bear"] = sum(se) >= 3

    # ── EMA Ribbon alignment ──
    rb = [b(d5m, "5m_ribbon_bull"), b(d15m, "15m_ribbon_bull"), b(d1h, "1h_ribbon_bull"), b(d4h, "4h_ribbon_bull")]
    re = [b(d5m, "5m_ribbon_bear"), b(d15m, "15m_ribbon_bear"), b(d1h, "1h_ribbon_bear"), b(d4h, "4h_ribbon_bear")]
    m["mtf_ribbon_all_bull"] = all(rb)
    m["mtf_ribbon_all_bear"] = all(re)

    # ── MACD alignment ──
    mb = [b(d5m, "5m_macd_above"), b(d15m, "15m_macd_above"), b(d1h, "1h_macd_above"), b(d4h, "4h_macd_above")]
    me = [b(d5m, "5m_macd_below"), b(d15m, "15m_macd_below"), b(d1h, "1h_macd_below"), b(d4h, "4h_macd_below")]
    m["mtf_macd_all_bull"] = all(mb)
    m["mtf_macd_all_bear"] = all(me)

    # ── RSI confluence ──
    r5 = g(d5m, "5m_rsi", 50)
    r15 = g(d15m, "15m_rsi", 50)
    r1h = g(d1h, "1h_rsi", 50)
    r4h = g(d4h, "4h_rsi", 50)
    m["mtf_rsi_all_bull"] = r5 > 50 and r15 > 50 and r1h > 50 and r4h > 50
    m["mtf_rsi_all_bear"] = r5 < 50 and r15 < 50 and r1h < 50 and r4h < 50
    m["mtf_rsi_dip_uptrend"] = r5 < 35 and r15 < 40 and r1h > 45 and r4h > 50
    m["mtf_rsi_rally_downtrend"] = r5 > 65 and r15 > 60 and r1h < 55 and r4h < 50
    m["mtf_rsi_all_os"] = r5 < 30 and r15 < 35 and r1h < 35 and r4h < 40
    m["mtf_rsi_all_ob"] = r5 > 70 and r15 > 65 and r1h > 65 and r4h > 60

    # ── ADX trend strength ──
    adx_4h = g(d4h, "4h_adx", 0)
    adx_1h = g(d1h, "1h_adx", 0)
    m["mtf_strong_trend"] = adx_4h > 25 and adx_1h > 25
    m["mtf_no_trend"] = adx_4h < 20 and adx_1h < 20

    # ── Ichimoku alignment ──
    ib = [b(d5m, "5m_ich_above"), b(d15m, "15m_ich_above"), b(d1h, "1h_ich_above"), b(d4h, "4h_ich_above")]
    ie = [b(d5m, "5m_ich_below"), b(d15m, "15m_ich_below"), b(d1h, "1h_ich_below"), b(d4h, "4h_ich_below")]
    m["mtf_ich_all_bull"] = all(ib)
    m["mtf_ich_all_bear"] = all(ie)

    # ── Parabolic SAR alignment ──
    pb = [b(d5m, "5m_psar_bull"), b(d15m, "15m_psar_bull"), b(d1h, "1h_psar_bull"), b(d4h, "4h_psar_bull")]
    pe = [b(d5m, "5m_psar_bear"), b(d15m, "15m_psar_bear"), b(d1h, "1h_psar_bear"), b(d4h, "4h_psar_bear")]
    m["mtf_psar_all_bull"] = all(pb)
    m["mtf_psar_all_bear"] = all(pe)

    # ── OBV alignment ──
    ob_r = [b(d5m, "5m_obv_rising"), b(d15m, "15m_obv_rising"), b(d1h, "1h_obv_rising"), b(d4h, "4h_obv_rising")]
    ob_f = [b(d5m, "5m_obv_falling"), b(d15m, "15m_obv_falling"), b(d1h, "1h_obv_falling"), b(d4h, "4h_obv_falling")]
    m["mtf_obv_all_rising"] = all(ob_r)
    m["mtf_obv_all_falling"] = all(ob_f)

    # ── CMF alignment ──
    cmf_pos = [b(d5m, "5m_cmf_pos"), b(d15m, "15m_cmf_pos"), b(d1h, "1h_cmf_pos"), b(d4h, "4h_cmf_pos")]
    cmf_neg = [b(d5m, "5m_cmf_neg"), b(d15m, "15m_cmf_neg"), b(d1h, "1h_cmf_neg"), b(d4h, "4h_cmf_neg")]
    m["mtf_cmf_all_pos"] = all(cmf_pos)
    m["mtf_cmf_all_neg"] = all(cmf_neg)

    # ── TTM Squeeze multi-TF ──
    sq = [b(d5m, "5m_ttm_squeeze"), b(d15m, "15m_ttm_squeeze"), b(d1h, "1h_ttm_squeeze")]
    m["mtf_ttm_squeeze_multi"] = sum(sq) >= 2

    # ── Volume spike confirmed ──
    m["mtf_vol_spike_confirmed"] = (
        b(d5m, "5m_vol_spike") and (b(d1h, "1h_vol_high") or b(d4h, "4h_vol_high"))
    )

    # ── Divergence on any TF ──
    m["mtf_rsi_bull_div_any"] = any([
        b(d5m, "5m_rsi_bull_div"), b(d15m, "15m_rsi_bull_div"),
        b(d1h, "1h_rsi_bull_div"), b(d4h, "4h_rsi_bull_div")
    ])
    m["mtf_rsi_bear_div_any"] = any([
        b(d5m, "5m_rsi_bear_div"), b(d15m, "15m_rsi_bear_div"),
        b(d1h, "1h_rsi_bear_div"), b(d4h, "4h_rsi_bear_div")
    ])
    m["mtf_obv_bull_div_any"] = any([
        b(d5m, "5m_obv_bull_div"), b(d15m, "15m_obv_bull_div"),
        b(d1h, "1h_obv_bull_div"), b(d4h, "4h_obv_bull_div")
    ])
    m["mtf_obv_bear_div_any"] = any([
        b(d5m, "5m_obv_bear_div"), b(d15m, "15m_obv_bear_div"),
        b(d1h, "1h_obv_bear_div"), b(d4h, "4h_obv_bear_div")
    ])

    # ── Aggregate bull/bear signal count ──
    bull_keys = [
        "st_bull", "ribbon_bull", "ema_bull", "macd_above", "psar_bull",
        "ich_above", "di_bull", "obv_rising", "cmf_pos", "roc_strong",
        "mom_pos", "above_vwap"
    ]
    bear_keys = [
        "st_bear", "ribbon_bear", "ema_bear", "macd_below", "psar_bear",
        "ich_below", "di_bear", "obv_falling", "cmf_neg", "roc_weak",
        "mom_neg", "below_vwap"
    ]

    bc = 0
    bec = 0
    for k in bull_keys:
        for d, p in [(d5m, "5m"), (d15m, "15m"), (d1h, "1h"), (d4h, "4h")]:
            if b(d, f"{p}_{k}"):
                bc += 1
    for k in bear_keys:
        for d, p in [(d5m, "5m"), (d15m, "15m"), (d1h, "1h"), (d4h, "4h")]:
            if b(d, f"{p}_{k}"):
                bec += 1

    m["mtf_bull_count"] = bc
    m["mtf_bear_count"] = bec
    m["mtf_strong_bull"] = bc >= 28  # ~60% of 48 possible
    m["mtf_strong_bear"] = bec >= 28
    m["mtf_moderate_bull"] = bc >= 18
    m["mtf_moderate_bear"] = bec >= 18

    return m


# ============================================================
# V1 SCORING SYSTEM
# ============================================================

def compute_v1_scores(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Tuple[float, float, float]:
    """
    Compute pump_score and dump_score from indicator confluence.
    Returns (pump_score, dump_score, net_score).
    """
    pump = 0.0
    dump = 0.0

    # Weight map: higher TFs carry more weight
    weights = {"5m": 0.5, "15m": 0.8, "1h": 1.2, "4h": 1.5}

    for d, prefix, w in [
        (d5m, "5m", weights["5m"]),
        (d15m, "15m", weights["15m"]),
        (d1h, "1h", weights["1h"]),
        (d4h, "4h", weights["4h"]),
    ]:
        if not d:
            continue
        # Trend signals
        if b(d, f"{prefix}_st_bull"):
            pump += 0.5 * w
        if b(d, f"{prefix}_st_bear"):
            dump += 0.5 * w
        if b(d, f"{prefix}_ribbon_bull"):
            pump += 0.4 * w
        if b(d, f"{prefix}_ribbon_bear"):
            dump += 0.4 * w
        if b(d, f"{prefix}_macd_above"):
            pump += 0.3 * w
        if b(d, f"{prefix}_macd_below"):
            dump += 0.3 * w

        # Momentum
        rsi = g(d, f"{prefix}_rsi", 50)
        if rsi > 60:
            pump += 0.3 * w
        elif rsi < 40:
            dump += 0.3 * w
        if b(d, f"{prefix}_roc_strong"):
            pump += 0.2 * w
        if b(d, f"{prefix}_roc_weak"):
            dump += 0.2 * w

        # Volume
        if b(d, f"{prefix}_obv_rising"):
            pump += 0.2 * w
        if b(d, f"{prefix}_obv_falling"):
            dump += 0.2 * w
        if b(d, f"{prefix}_cmf_pos"):
            pump += 0.15 * w
        if b(d, f"{prefix}_cmf_neg"):
            dump += 0.15 * w
        if b(d, f"{prefix}_vol_spike"):
            pump += 0.3 * w  # volume spike is directionally neutral but adds energy

        # Structure
        if b(d, f"{prefix}_hh"):
            pump += 0.3 * w
        if b(d, f"{prefix}_ll"):
            dump += 0.3 * w

        # Ichimoku
        if b(d, f"{prefix}_ich_above"):
            pump += 0.2 * w
        if b(d, f"{prefix}_ich_below"):
            dump += 0.2 * w

        # PSAR
        if b(d, f"{prefix}_psar_bull"):
            pump += 0.15 * w
        if b(d, f"{prefix}_psar_bear"):
            dump += 0.15 * w

    net = pump - dump
    return round(pump, 2), round(dump, 2), round(net, 2)


# ============================================================
# V3 COMBO SYSTEM (Sniper / Elite / Standard)
# ============================================================

def detect_v3_combos(d5m: Dict, d15m: Dict, d1h: Dict, d4h: Dict) -> Dict:
    """
    Detect v3 pump/dump combos matching your bot's tiered system exactly.
    Uses 15m as entry TF, 1h as confirm, 4h as trend (same as pine_backscan).
    Also uses 5m for additional precision.
    """
    # Gates and common conditions
    four_h_3green = b(d4h, "4h_macd_above") and b(d4h, "4h_roc_strong") and b(d4h, "4h_obv_rising")
    pump_trend_gate = b(d4h, "4h_macd_above") and b(d4h, "4h_rsi_bull") and b(d4h, "4h_hh")
    low_vol_5m = b(d5m, "5m_vol_low")
    low_vol_15m = b(d15m, "15m_vol_low") if d15m else low_vol_5m
    ich_above_1h = b(d1h, "1h_ich_above") or b(d1h, "1h_ich_inside")

    # ─── PUMP PULLBACK ───
    rsi_low_5m = g(d5m, "5m_rsi", 99) < 40
    bb_touch_lo = b(d5m, "5m_bb_near_lo") or b(d5m, "5m_bb_below_lo")
    pullback_setup = rsi_low_5m or bb_touch_lo
    pullback_trigger = b(d5m, "5m_st_flip_bull") and not b(d5m, "5m_vol_low")
    pump_pullback = pump_trend_gate and pullback_setup and pullback_trigger

    # ─── PUMP SNIPERS ───
    pump_sniper_1 = four_h_3green and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_cmf_pos")
    pump_sniper_2 = b(d4h, "4h_hh") and b(d4h, "4h_bb_squeeze") and b(d1h, "1h_wr_ob")
    pump_sniper_3 = four_h_3green and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_roc_strong")
    pump_sniper_4 = b(d1h, "1h_hh") and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_roc_strong")
    pump_sniper = pump_sniper_1 or pump_sniper_2 or pump_sniper_3 or pump_sniper_4

    # ─── PUMP ELITE ───
    mtf_obv_all = b(d5m, "5m_obv_rising") and b(d15m, "15m_obv_rising") and b(d1h, "1h_obv_rising") and b(d4h, "4h_obv_rising")
    pump_elite_1 = four_h_3green and b(d4h, "4h_bb_squeeze") and ich_above_1h
    pump_elite_2 = b(d1h, "1h_hh") and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_obv_rising")
    pump_elite_3 = four_h_3green and b(d4h, "4h_bb_squeeze") and b(d1h, "1h_roc_strong")
    pump_elite_4 = b(d1h, "1h_hh") and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_cmf_pos")
    pump_elite_5 = b(d4h, "4h_cmf_pos") and mtf_obv_all
    pump_elite_6 = b(d1h, "1h_hh") and b(d4h, "4h_bb_squeeze") and ich_above_1h
    pump_elite = (pump_elite_1 or pump_elite_2 or pump_elite_3 or
                  pump_elite_4 or pump_elite_5 or pump_elite_6) and not pump_sniper

    # ─── PUMP STANDARD ───
    mtf_rsi_all = g(d5m, "5m_rsi", 0) > 50 and g(d15m, "15m_rsi", 0) > 50 and g(d1h, "1h_rsi", 0) > 50 and b(d4h, "4h_rsi_bull")
    pump_std_1 = b(d4h, "4h_hh") and b(d4h, "4h_bb_squeeze") and b(d1h, "1h_roc_strong")
    pump_std_2 = b(d4h, "4h_hh") and b(d4h, "4h_bb_squeeze") and b(d4h, "4h_cmf_pos")
    pump_std_3 = b(d1h, "1h_hh") and b(d4h, "4h_bb_squeeze") and mtf_rsi_all
    pump_std = (pump_std_1 or pump_std_2 or pump_std_3) and not pump_sniper and not pump_elite

    pump_breakout = (pump_sniper or pump_elite or pump_std) and b(d4h, "4h_macd_above")
    raw_v3_pump = pump_breakout or pump_pullback

    # ─── DUMP COMBOS ───
    mtf_ich_all_bear = b(d5m, "5m_ich_below") and b(d15m, "15m_ich_below") and b(d1h, "1h_ich_below") and b(d4h, "4h_ich_below")
    mtf_ribbon_all_bear = b(d5m, "5m_ribbon_bear") and b(d15m, "15m_ribbon_bear") and b(d1h, "1h_ribbon_bear") and b(d4h, "4h_ribbon_bear")
    mtf_psar_all_bear = b(d5m, "5m_psar_bear") and b(d15m, "15m_psar_bear") and b(d4h, "4h_psar_bear")
    mtf_macd_all_bear = (
        not b(d5m, "5m_macd_above") and not b(d15m, "15m_macd_above") and
        not b(d1h, "1h_macd_above") and not b(d4h, "4h_macd_above")
    )

    # Dump Snipers
    dump_sniper_1 = low_vol_15m and b(d1h, "1h_mfi_os") and mtf_ribbon_all_bear
    dump_sniper_2 = low_vol_15m and b(d1h, "1h_mfi_os") and mtf_ich_all_bear
    dump_sniper_3 = low_vol_15m and b(d1h, "1h_mfi_os") and b(d4h, "4h_wr_extreme_os")
    dump_sniper = dump_sniper_1 or dump_sniper_2 or dump_sniper_3

    # Dump Elite
    dump_elite_1 = low_vol_15m and mtf_ich_all_bear and b(d4h, "4h_stoch_os")
    dump_elite_2 = mtf_macd_all_bear and b(d4h, "4h_three_red")
    dump_elite_3 = low_vol_15m and b(d1h, "1h_kc_below")
    dump_elite_4 = b(d1h, "1h_rsi_os") and b(d4h, "4h_three_red") and b(d4h, "4h_mfi_os")
    dump_elite_5 = b(d4h, "4h_three_red") and mtf_psar_all_bear
    dump_elite_6 = b(d4h, "4h_three_red") and b(d1h, "1h_bb_below_lo")
    dump_elite = (dump_elite_1 or dump_elite_2 or dump_elite_3 or
                  dump_elite_4 or dump_elite_5 or dump_elite_6) and not dump_sniper

    # Dump Standard
    dump_std_1 = low_vol_15m and mtf_ich_all_bear
    dump_std_2 = b(d1h, "1h_bb_below_lo") and b(d4h, "4h_rsi_os")
    dump_std_3 = b(d4h, "4h_three_red") and b(d4h, "4h_ll")
    dump_std = (dump_std_1 or dump_std_2 or dump_std_3) and not dump_sniper and not dump_elite
    raw_v3_dump = dump_sniper or dump_elite or dump_std

    pump_tier = 3 if pump_sniper else (2 if pump_elite else (1 if (pump_std or pump_pullback) else 0))
    dump_tier = 3 if dump_sniper else (2 if dump_elite else (1 if dump_std else 0))

    return {
        "raw_v3_pump": raw_v3_pump,
        "raw_v3_dump": raw_v3_dump,
        "pump_sniper": pump_sniper,
        "pump_elite": pump_elite,
        "pump_std": pump_std,
        "pump_pullback": pump_pullback,
        "dump_sniper": dump_sniper,
        "dump_elite": dump_elite,
        "dump_std": dump_std,
        "pump_tier": pump_tier,
        "dump_tier": dump_tier,
        # Context flags
        "four_h_3green": four_h_3green,
        "pump_trend_gate": pump_trend_gate,
        "mtf_obv_all_rising": b(d5m, "5m_obv_rising") and b(d15m, "15m_obv_rising") and b(d1h, "1h_obv_rising") and b(d4h, "4h_obv_rising"),
        "mtf_ribbon_all_bear": mtf_ribbon_all_bear,
        "mtf_ich_all_bear": mtf_ich_all_bear,
        "mtf_psar_all_bear": mtf_psar_all_bear,
        "mtf_macd_all_bear": mtf_macd_all_bear,
    }


# ============================================================
# BUILD SYNCHRONIZED SNAPSHOT
# ============================================================

def build_snapshot(
    symbol: str,
    timestamp: int,
    price: float,
    ind_5m: Dict,
    ind_15m: Dict,
    ind_1h: Dict,
    ind_4h: Dict,
) -> MTFSnapshot:
    """
    Build a complete MTF snapshot for a single 5-minute point in time.
    Combines all indicator data, MTF confluence, v1 scores, and v3 combos.
    """
    dt_str = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    # MTF confluence
    mtf = compute_mtf_confluence(ind_5m, ind_15m, ind_1h, ind_4h)

    # v1 scoring
    pump_sc, dump_sc, net_sc = compute_v1_scores(ind_5m, ind_15m, ind_1h, ind_4h)

    # v3 combos
    v3 = detect_v3_combos(ind_5m, ind_15m, ind_1h, ind_4h)

    # Determine direction and tier
    direction = ""
    signal_type = ""
    tier = 0

    if v3["raw_v3_pump"]:
        direction = "pump"
        tier = v3["pump_tier"]
        if v3["pump_sniper"]:
            signal_type = "v3_pump_sniper"
        elif v3["pump_elite"]:
            signal_type = "v3_pump_elite"
        elif v3["pump_pullback"]:
            signal_type = "v3_pump_pullback"
        else:
            signal_type = "v3_pump_std"
    elif v3["raw_v3_dump"]:
        direction = "dump"
        tier = v3["dump_tier"]
        if v3["dump_sniper"]:
            signal_type = "v3_dump_sniper"
        elif v3["dump_elite"]:
            signal_type = "v3_dump_elite"
        else:
            signal_type = "v3_dump_std"
    elif net_sc >= 4.0:
        direction = "pump"
        signal_type = "v1_strong_pump" if net_sc >= 7.0 else "v1_pump"
        tier = 2 if net_sc >= 7.0 else 1
    elif net_sc <= -4.0:
        direction = "dump"
        signal_type = "v1_strong_dump" if net_sc <= -7.0 else "v1_dump"
        tier = 2 if net_sc <= -7.0 else 1

    # Confidence = weighted average of agreement signals
    bull_c = mtf.get("mtf_bull_count", 0)
    bear_c = mtf.get("mtf_bear_count", 0)
    total_possible = 48  # 12 signals × 4 TFs
    if direction == "pump":
        confidence = min(100, (bull_c / total_possible) * 100 + tier * 10)
    elif direction == "dump":
        confidence = min(100, (bear_c / total_possible) * 100 + tier * 10)
    else:
        confidence = 0

    return MTFSnapshot(
        timestamp=timestamp,
        dt_str=dt_str,
        symbol=symbol,
        price=price,
        ind_5m=ind_5m,
        ind_15m=ind_15m,
        ind_1h=ind_1h,
        ind_4h=ind_4h,
        mtf_signals=mtf,
        pump_score=pump_sc,
        dump_score=dump_sc,
        net_score=net_sc,
        v3_signals=v3,
        direction=direction,
        signal_type=signal_type,
        tier=tier,
        confidence=round(confidence, 1),
    )
