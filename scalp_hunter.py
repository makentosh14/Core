#!/usr/bin/env python3
"""
scalp_hunter.py - 1% Scalp Hunter Strategy Module
===================================================
Detects volatility compression → expansion → volume spike → structure break.
Designed for 20x leverage scalps targeting 0.8%–1.2% moves.

Integrates with: main.py, monitor.py, telegram_bot.py, trade_executor.py
Author: Senior Quant HFT Developer
"""

import asyncio
import time
import statistics
from collections import defaultdict, deque
from typing import Dict, Any, Optional, Tuple, List
from logger import log

# ─────────────────────────────────────────────
# CONFIGURATION — tune these for more/less signals
# ─────────────────────────────────────────────
SCALP_CONFIG = {
    # Compression detection
    "bb_width_ratio_threshold": 0.35,      # BB width / BB width MA — below = squeeze
    "atr_ratio_threshold": 0.70,           # ATR / ATR MA — below = squeeze
    "range_ratio_threshold": 0.75,         # Avg range / range MA — below = contraction
    "compression_min_candles": 7,         # Minimum candles in squeeze         # Minimum candles in squeeze
    "compression_max_candles": 40,         # If longer, move is less likely explosive

    # Volume spike
    "volume_spike_multiplier": 1.8,        # current_vol > vol_MA * this
    "volume_lookback": 20,                 # Candles for volume MA

    # EMA stack
    "ema_fast": 9,
    "ema_mid": 20,
    "ema_slow": 50,

    # Micro range break
    "range_lookback": 10,                  # Candles to define micro range

    # Watchlist / watch mode
    "watch_mode_ttl": 3,                   # Candles before watchlist entry expires
    "watch_score_threshold": 3,            # Partial score to enter watchlist
    "trigger_score_threshold": 5,          # Full score needed to trigger trade

    # BTC filter
    "btc_ema_fast": 9,
    "btc_ema_slow": 21,
    "btc_atr_choppy_ratio": 0.60,          # BTC ATR/ATR_MA below this = choppy, block trades
    "btc_slope_min": 0.0001,               # Minimum EMA slope (% per candle) to confirm trend

    # Liquidity / spread
    "min_quote_volume": 300_000,           # Minimum 24h quote volume (in USDT)
    "min_candle_count_1m": 30,             # Minimum 1m candles needed

    # Risk management (20x leverage)
    "sl_atr_multiplier": 1.0,             # SL = entry ± ATR * this
    "sl_min_pct": 0.0035,                  # Minimum SL distance (0.35%)
    "sl_max_pct": 0.0060,                  # Maximum SL distance (0.60%)
    "tp1_pct": 0.0090,                     # TP1 default (0.9%)
    "tp1_partial_close": 0.50,             # Close 50% at TP1
    "leverage": 20,

    # Exit thresholds (tighter than intraday/swing)
    "score_exit_threshold": 3,             # Exit if score drops below this
    "score_exit_cycles": 2,                # For this many consecutive checks
    "volume_fade_ratio": 0.70,             # Exit if current_vol < vol_MA * this
    "momentum_exit_cycles": 3,             # Candles of fading momentum before exit

    # Cooldown / loss protection
    "symbol_loss_cooldown": 1800,          # 30 min cooldown after loss
    "loss_streak_block": 3,                # Block symbol after N consecutive losses
    "chop_regime_block": True,             # Block all scalps when BTC is choppy
}

# ─────────────────────────────────────────────
# MODULE STATE
# ─────────────────────────────────────────────
# Watchlist: symbol -> {"score": int, "direction": str, "candles_watched": int, "ts": float}
_watchlist: Dict[str, Dict] = {}

# Per-symbol loss tracking
_symbol_loss_streak: Dict[str, int] = defaultdict(int)
_symbol_last_loss_time: Dict[str, float] = {}

# BTC state cache
_btc_state: Dict[str, Any] = {
    "trend": "unknown",   # "long", "short", "choppy"
    "slope": 0.0,
    "atr_ratio": 1.0,
    "updated_at": 0.0,
    "impulse": False,
}

# Scalp-specific exit tracking per symbol
# symbol -> {"score_history": deque, "volume_history": deque}
_scalp_trade_state: Dict[str, Dict] = {}


# ─────────────────────────────────────────────
# HELPER MATH FUNCTIONS (no external deps needed)
# ─────────────────────────────────────────────

def _closes(candles: List[Dict]) -> List[float]:
    return [float(c.get("close", c.get("c", 0))) for c in candles]

def _highs(candles: List[Dict]) -> List[float]:
    return [float(c.get("high", c.get("h", 0))) for c in candles]

def _lows(candles: List[Dict]) -> List[float]:
    return [float(c.get("low", c.get("l", 0))) for c in candles]

def _volumes(candles: List[Dict]) -> List[float]:
    return [float(c.get("volume", c.get("v", 0))) for c in candles]

def _ema(values: List[float], period: int) -> List[float]:
    """Simple EMA calculation."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result

def _sma(values: List[float], period: int) -> List[float]:
    """Simple moving average."""
    if len(values) < period:
        return []
    return [sum(values[i:i+period]) / period for i in range(len(values) - period + 1)]

def _atr(candles: List[Dict], period: int = 14) -> List[float]:
    """Average True Range."""
    if len(candles) < period + 1:
        return []
    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i].get("high", 0))
        l = float(candles[i].get("low", 0))
        pc = float(candles[i-1].get("close", 0))
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return []
    atrs = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs

def _bollinger_width(candles: List[Dict], period: int = 20) -> List[float]:
    """BB width (upper - lower) / middle."""
    closes = _closes(candles)
    if len(closes) < period:
        return []
    widths = []
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = sum(window) / period
        std = (sum((x - mean) ** 2 for x in window) / period) ** 0.5
        upper = mean + 2 * std
        lower = mean - 2 * std
        widths.append((upper - lower) / mean if mean != 0 else 0)
    return widths

def _candle_ranges(candles: List[Dict]) -> List[float]:
    """True range per candle (simplified H-L)."""
    return [float(c.get("high", 0)) - float(c.get("low", 0)) for c in candles]


# ─────────────────────────────────────────────
# BTC MICRO-TREND FILTER
# ─────────────────────────────────────────────

def update_btc_state(btc_candles_1m: List[Dict], btc_candles_5m: List[Dict]) -> Dict:
    """
    Analyse BTC 1m+5m candles and update global BTC state.
    Called from main scan loop before processing alts.
    Returns the state dict.
    """
    cfg = SCALP_CONFIG
    state = _btc_state

    # Need at least 55 candles for EMA50
    if len(btc_candles_1m) < 55:
        state["trend"] = "unknown"
        state["updated_at"] = time.time()
        return state

    closes_1m = _closes(btc_candles_1m)

    # EMA alignment on 1m
    ema_fast = _ema(closes_1m, cfg["btc_ema_fast"])
    ema_slow = _ema(closes_1m, cfg["btc_ema_slow"])

    if not ema_fast or not ema_slow:
        state["trend"] = "unknown"
        return state

    ef = ema_fast[-1]
    es = ema_slow[-1]

    # EMA slope (rate of change over last 3 candles)
    if len(ema_fast) >= 4:
        slope = (ema_fast[-1] - ema_fast[-4]) / (ema_fast[-4] if ema_fast[-4] != 0 else 1)
    else:
        slope = 0.0

    state["slope"] = slope

    # ATR ratio for chop detection
    atrs_1m = _atr(btc_candles_1m, 14)
    if len(atrs_1m) >= 15:
        atr_now = atrs_1m[-1]
        atr_ma = sum(atrs_1m[-15:]) / 15
        state["atr_ratio"] = atr_now / atr_ma if atr_ma > 0 else 1.0
    else:
        state["atr_ratio"] = 1.0

    # Determine BTC regime
    if state["atr_ratio"] < cfg["btc_atr_choppy_ratio"] and abs(slope) < cfg["btc_slope_min"]:
        state["trend"] = "choppy"
    elif ef > es and slope > cfg["btc_slope_min"]:
        state["trend"] = "long"
    elif ef < es and slope < -cfg["btc_slope_min"]:
        state["trend"] = "short"
    else:
        state["trend"] = "choppy"

    # Impulse: last candle body dominance + volume spike on 1m
    last = btc_candles_1m[-1]
    body = abs(float(last.get("close", 0)) - float(last.get("open", 0)))
    full_range = float(last.get("high", 0)) - float(last.get("low", 0))
    body_ratio = body / full_range if full_range > 0 else 0
    vols = _volumes(btc_candles_1m)
    vol_ma = sum(vols[-20:-1]) / 19 if len(vols) >= 20 else sum(vols[:-1]) / max(len(vols) - 1, 1)
    state["impulse"] = body_ratio > 0.6 and vols[-1] > vol_ma * 1.5

    state["updated_at"] = time.time()
    log(f"🔵 BTC state: trend={state['trend']} slope={slope:.5f} atr_ratio={state['atr_ratio']:.2f} impulse={state['impulse']}")
    return state


def btc_allows_direction(direction: str) -> Tuple[bool, str]:
    """
    Check if BTC trend allows trading in the given direction.
    Returns (allowed, reason_tag).
    """
    state = _btc_state

    # Stale BTC data (>5 min)
    if time.time() - state.get("updated_at", 0) > 300:
        return False, "BTC data stale"

    trend = state.get("trend", "unknown")

    if trend == "choppy" and SCALP_CONFIG["chop_regime_block"]:
        return False, "BTC choppy — blocked"

    if trend == "unknown":
        return False, "BTC trend unknown"

    if direction.lower() == "long" and trend == "long":
        return True, "BTC aligned ✅"
    if direction.lower() == "short" and trend == "short":
        return True, "BTC aligned ✅"

    # Counter-trend — allow only if BTC is at least not strongly opposing
    if trend == "choppy":
        return False, "BTC choppy — blocked"

    return False, f"BTC opposing ({trend}) — blocked"


# ─────────────────────────────────────────────
# COMPRESSION DETECTOR
# ─────────────────────────────────────────────

def detect_compression(candles: List[Dict]) -> Tuple[bool, int, float]:
    """
    Detect volatility squeeze.
    Returns (in_compression, compression_candle_count, compression_score_0_to_1).
    Requires at least 2 of 3 signals to confirm compression.
    Uses closed candles only (excludes last candle if it's open/current).
    """
    cfg = SCALP_CONFIG
    closed = candles[:-1]  # Exclude last (potentially open) candle

    if len(closed) < 30:
        return False, 0, 0.0

    signals = 0
    compression_counts = []

    # --- Signal 1: Bollinger Band width below its own MA ---
    bb_widths = _bollinger_width(closed, 20)
    if len(bb_widths) >= 20:
        bb_ma = sum(bb_widths[-20:]) / 20
        bb_now = bb_widths[-1]
        ratio = bb_now / bb_ma if bb_ma > 0 else 1.0
        if ratio < cfg["bb_width_ratio_threshold"] + 0.15:  # slightly relaxed check
            signals += 1
            # Count consecutive candles in squeeze
            count = 0
            for w in reversed(bb_widths):
                if w / bb_ma < cfg["bb_width_ratio_threshold"] + 0.20:
                    count += 1
                else:
                    break
            compression_counts.append(count)

    # --- Signal 2: ATR% below its MA ---
    atrs = _atr(closed, 14)
    if len(atrs) >= 20:
        atr_ma = sum(atrs[-20:]) / 20
        atr_now = atrs[-1]
        ratio = atr_now / atr_ma if atr_ma > 0 else 1.0
        if ratio < cfg["atr_ratio_threshold"]:
            signals += 1
            count = 0
            for a in reversed(atrs):
                if a / atr_ma < cfg["atr_ratio_threshold"] + 0.10:
                    count += 1
                else:
                    break
            compression_counts.append(count)

    # --- Signal 3: Candle range contraction ---
    ranges = _candle_ranges(closed)
    if len(ranges) >= 20:
        range_ma = sum(ranges[-20:]) / 20
        range_now = sum(ranges[-3:]) / 3  # Average last 3 candles
        ratio = range_now / range_ma if range_ma > 0 else 1.0
        if ratio < cfg["range_ratio_threshold"]:
            signals += 1
            count = 0
            for r in reversed(ranges):
                if r / range_ma < cfg["range_ratio_threshold"] + 0.10:
                    count += 1
                else:
                    break
            compression_counts.append(count)

    if signals < 2:
        return False, 0, 0.0

    # Duration: use minimum count (most conservative)
    duration = min(compression_counts) if compression_counts else 0
    if duration < cfg["compression_min_candles"]:
        return False, duration, 0.0
    if duration > cfg["compression_max_candles"]:
        return False, duration, 0.0  # Too long = stale setup

    compression_score = min(1.0, signals / 3.0 + duration / 30.0)
    return True, duration, compression_score


# ─────────────────────────────────────────────
# EXPANSION + VOLUME SPIKE TRIGGER
# ─────────────────────────────────────────────

def detect_trigger(candles: List[Dict]) -> Tuple[bool, str, float, Dict]:
    """
    Detect volume spike + structure break on current (just-closed) candle.
    Returns (triggered, direction, trigger_score, details).

    direction: "long" or "short"
    trigger_score: 0.0–1.0
    """
    cfg = SCALP_CONFIG
    if len(candles) < 55:
        return False, "", 0.0, {}

    closed = candles[:-1]  # Use closed candles for all calculations
    last_closed = closed[-1]

    closes = _closes(closed)
    highs = _highs(closed)
    lows = _lows(closed)
    vols = _volumes(closed)

    last_close = float(last_closed.get("close", 0))
    last_open = float(last_closed.get("open", 0))
    last_high = float(last_closed.get("high", 0))
    last_low = float(last_closed.get("low", 0))
    last_vol = float(last_closed.get("volume", 0))

    # --- Volume spike ---
    vol_lookback = cfg["volume_lookback"]
    if len(vols) < vol_lookback + 1:
        return False, "", 0.0, {}
    vol_ma = sum(vols[-(vol_lookback + 1):-1]) / vol_lookback
    vol_spike = last_vol / vol_ma if vol_ma > 0 else 0.0
    has_vol_spike = vol_spike >= cfg["volume_spike_multiplier"]

    if not has_vol_spike:
        return False, "", 0.0, {}

    # --- EMA stack ---
    ema9 = _ema(closes, cfg["ema_fast"])
    ema20 = _ema(closes, cfg["ema_mid"])
    ema50 = _ema(closes, cfg["ema_slow"])

    if not ema9 or not ema20 or not ema50:
        return False, "", 0.0, {}

    e9, e20, e50 = ema9[-1], ema20[-1], ema50[-1]
    bullish_ema = e9 > e20 > e50
    bearish_ema = e9 < e20 < e50

    # --- Micro range break (Option A: momentum continuation) ---
    lookback = cfg["range_lookback"]
    micro_high = max(highs[-(lookback + 1):-1])
    micro_low = min(lows[-(lookback + 1):-1])

    range_break_long = last_close > micro_high and bullish_ema
    range_break_short = last_close < micro_low and bearish_ema

    # --- Liquidity sweep reversal (Option B) ---
    # Price sweeps recent swing low/high then closes back inside range
    recent_lows = lows[-(lookback + 1):-1]
    recent_highs = highs[-(lookback + 1):-1]
    swing_low = min(recent_lows)
    swing_high = max(recent_highs)

    # Long sweep: price dipped below swing_low but close is back above micro_low
    sweep_long = (last_low < swing_low and last_close > micro_low
                  and last_close > last_open)
    # Short sweep: price spiked above swing_high but close is back below micro_high
    sweep_short = (last_high > swing_high and last_close < micro_high
                   and last_close < last_open)

    # Determine direction and trigger type
    triggered = False
    direction = ""
    trigger_type = ""
    trigger_score = 0.0

    if range_break_long or sweep_long:
        triggered = True
        direction = "long"
        trigger_type = "Range Break" if range_break_long else "Liquidity Sweep Reversal"
        trigger_score = vol_spike / cfg["volume_spike_multiplier"]  # Normalised
    elif range_break_short or sweep_short:
        triggered = True
        direction = "short"
        trigger_type = "Range Break" if range_break_short else "Liquidity Sweep Reversal"
        trigger_score = vol_spike / cfg["volume_spike_multiplier"]

    if not triggered:
        return False, "", 0.0, {}

    details = {
        "vol_spike_ratio": round(vol_spike, 2),
        "trigger_type": trigger_type,
        "micro_high": micro_high,
        "micro_low": micro_low,
        "ema_aligned": bullish_ema if direction == "long" else bearish_ema,
    }

    return True, direction, min(trigger_score, 1.0), details


# ─────────────────────────────────────────────
# LIQUIDITY / SPREAD FILTER
# ─────────────────────────────────────────────

def passes_liquidity_filter(candles_1m: List[Dict], symbol: str) -> Tuple[bool, str]:
    """
    Check minimum volume and wick quality.
    Returns (passes, reason).
    """
    cfg = SCALP_CONFIG

    if len(candles_1m) < cfg["min_candle_count_1m"]:
        return False, f"Insufficient 1m candles ({len(candles_1m)})"

    # Quote volume proxy: close * volume (USDT equivalent)
    vols = _volumes(candles_1m[-20:])
    closes = _closes(candles_1m[-20:])
    quote_vols = [v * c for v, c in zip(vols, closes)]
    avg_quote_vol = sum(quote_vols) / len(quote_vols) if quote_vols else 0

    if avg_quote_vol < cfg["min_quote_volume"]:
        return False, f"Low liquidity (avg quote vol {avg_quote_vol:.0f} < {cfg['min_quote_volume']})"

    # Long wick filter: avoid candles where wicks >> body (low follow-through)
    last_5 = candles_1m[-6:-1]  # Last 5 closed candles
    wick_rejection_count = 0
    for c in last_5:
        h = float(c.get("high", 0))
        l = float(c.get("low", 0))
        op = float(c.get("open", 0))
        cl = float(c.get("close", 0))
        body = abs(cl - op)
        full = h - l
        if full > 0 and body / full < 0.25:  # Doji / wick-heavy
            wick_rejection_count += 1

    if wick_rejection_count >= 4:
        return False, "High wick rejection (choppy price action)"

    return True, "Liquidity OK"


# ─────────────────────────────────────────────
# SL / TP CALCULATOR
# ─────────────────────────────────────────────

def calculate_scalp_sl_tp(
    candles: List[Dict],
    direction: str,
    entry_price: float
) -> Tuple[float, float, float, float]:
    """
    Calculate SL and TP1 for scalp trade.
    Returns (sl_price, tp1_price, sl_pct, tp1_pct).
    """
    cfg = SCALP_CONFIG
    closed = candles[:-1]
    atrs = _atr(closed, 14)
    current_price = entry_price

    if atrs:
        atr = atrs[-1]
        atr_pct = atr / current_price
        raw_sl_pct = atr_pct * cfg["sl_atr_multiplier"]
    else:
        raw_sl_pct = (cfg["sl_min_pct"] + cfg["sl_max_pct"]) / 2

    # Apply caps
    sl_pct = max(cfg["sl_min_pct"], min(cfg["sl_max_pct"], raw_sl_pct))
    tp1_pct = cfg["tp1_pct"]

    if direction.lower() == "long":
        sl_price = current_price * (1 - sl_pct)
        tp1_price = current_price * (1 + tp1_pct)
    else:
        sl_price = current_price * (1 + sl_pct)
        tp1_price = current_price * (1 - tp1_pct)

    return round(sl_price, 6), round(tp1_price, 6), round(sl_pct, 6), round(tp1_pct, 6)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT: evaluate_scalp_setup()
# ─────────────────────────────────────────────

def evaluate_scalp_setup(
    symbol: str,
    candles_by_tf: Dict[str, List[Dict]],
    current_price: float,
) -> Dict[str, Any]:
    """
    Main function called per symbol from the scan loop.
    
    Args:
        symbol: e.g. "BTCUSDT"
        candles_by_tf: {"1": [...], "3": [...], "5": [...]}  closed-candle lists
        current_price: latest price

    Returns dict with keys:
        should_watch: bool — put on watchlist
        should_trade: bool — fire signal now
        direction: "long" | "short" | ""
        confidence: float 0–100
        score: int 0–7 (component count)
        sl_price: float
        tp1_price: float
        sl_pct: float
        tp1_pct: float
        reasons: List[str]
        details: Dict  (for logging)
    """
    result = {
        "should_watch": False,
        "should_trade": False,
        "direction": "",
        "confidence": 0.0,
        "score": 0,
        "sl_price": 0.0,
        "tp1_price": 0.0,
        "sl_pct": SCALP_CONFIG["sl_min_pct"],
        "tp1_pct": SCALP_CONFIG["tp1_pct"],
        "reasons": [],
        "details": {},
        "watch_mode": False,
        "btc_state": _btc_state.get("trend", "unknown"),
    }

    cfg = SCALP_CONFIG
    candles_1m = candles_by_tf.get("1", [])
    candles_3m = candles_by_tf.get("3", [])
    candles_5m = candles_by_tf.get("5", [])

    # Use 3m if 1m thin, else 1m
    primary_candles = candles_1m if len(candles_1m) >= 40 else candles_3m

    if len(primary_candles) < 30:
        return result

    # ── Loss streak check ──────────────────────────────────────
    if _symbol_loss_streak[symbol] >= cfg["loss_streak_block"]:
        last_loss = _symbol_last_loss_time.get(symbol, 0)
        if time.time() - last_loss < cfg["symbol_loss_cooldown"]:
            result["reasons"].append(f"Loss streak cooldown ({_symbol_loss_streak[symbol]} losses)")
            return result
        else:
            _symbol_loss_streak[symbol] = 0  # Reset after cooldown

    # ── Liquidity filter ──────────────────────────────────────
    liq_ok, liq_reason = passes_liquidity_filter(candles_1m or primary_candles, symbol)
    if not liq_ok:
        result["details"]["liquidity_fail"] = liq_reason
        return result
    result["reasons"].append("Liquidity ✅")

    # ── Compression check ────────────────────────────────────
    # Use pre-trigger window for compression check (exclude latest closed candle = trigger candle)
    # This correctly asks: "were we compressed BEFORE the breakout?"
    pre_trigger_candles = primary_candles[:-1] if len(primary_candles) > 31 else primary_candles
    in_compression, comp_duration, comp_score = detect_compression(pre_trigger_candles)
    result["details"]["compression"] = {
        "active": in_compression,
        "duration": comp_duration,
        "score": round(comp_score, 3),
    }

    if not in_compression:
        # No compression = no scalp setup
        return result

    result["score"] += 2
    result["reasons"].append(f"Compression ({comp_duration} candles) 🗜️")

    # ── Volume + trigger check ───────────────────────────────
    triggered, direction, trig_score, trig_details = detect_trigger(primary_candles)
    result["details"]["trigger"] = trig_details

    if not triggered:
        # Compression exists but no trigger yet → watchlist candidate
        result["should_watch"] = True
        result["score"] = 2
        result["details"]["phase"] = "compression_only"
        return result

    result["score"] += 2
    result["reasons"].append(f"{trig_details.get('trigger_type', 'Trigger')} 💥")
    result["reasons"].append(f"Vol Spike {trig_details.get('vol_spike_ratio', 0)}x 📊")
    result["direction"] = direction

    # ── BTC filter ───────────────────────────────────────────
    if symbol != "BTCUSDT":
        btc_ok, btc_reason = btc_allows_direction(direction)
        result["details"]["btc_filter"] = btc_reason
        if not btc_ok:
            result["reasons"].append(f"BTC blocked: {btc_reason}")
            # Still watchlist-worthy if compression is good
            result["should_watch"] = comp_score > 0.5
            return result
        result["score"] += 1
        result["reasons"].append(btc_reason)
    else:
        result["score"] += 1
        result["details"]["btc_filter"] = "N/A (is BTC)"

    # ── SL / TP calculation ──────────────────────────────────
    sl_price, tp1_price, sl_pct, tp1_pct = calculate_scalp_sl_tp(
        primary_candles, direction, current_price
    )
    result["sl_price"] = sl_price
    result["tp1_price"] = tp1_price
    result["sl_pct"] = sl_pct
    result["tp1_pct"] = tp1_pct

    # ── Final scoring ────────────────────────────────────────
    # Max score = 7 (liquidity implicit, compression=2, trigger=2, btc=1, ema=1, compression quality=1)
    ema_bonus = _check_ema_alignment(primary_candles, direction)
    if ema_bonus:
        result["score"] += 1
        result["reasons"].append("EMA stack aligned 📈")

    comp_quality_bonus = comp_score > 0.6
    if comp_quality_bonus:
        result["score"] += 1
        result["reasons"].append("Strong squeeze 🔒")

    # Confidence = score / 7 * 100, then adjusted by comp_score + trig_score
    raw_confidence = (result["score"] / 7.0) * 100
    raw_confidence = raw_confidence * 0.7 + (comp_score + trig_score) / 2.0 * 30
    result["confidence"] = round(min(99.0, raw_confidence), 1)

    # ── Trade decision ───────────────────────────────────────
    threshold = cfg["trigger_score_threshold"]
    if result["score"] >= threshold:
        result["should_trade"] = True
        result["should_watch"] = False
        result["details"]["phase"] = "trigger"
    else:
        result["should_watch"] = True
        result["details"]["phase"] = "watch"

    return result


def _check_ema_alignment(candles: List[Dict], direction: str) -> bool:
    """Check EMA9 > EMA20 > EMA50 for long (reverse for short)."""
    cfg = SCALP_CONFIG
    closes = _closes(candles[:-1])  # Closed candles
    ema9 = _ema(closes, cfg["ema_fast"])
    ema20 = _ema(closes, cfg["ema_mid"])
    ema50 = _ema(closes, cfg["ema_slow"])
    if not ema9 or not ema20 or not ema50:
        return False
    if direction.lower() == "long":
        return ema9[-1] > ema20[-1] > ema50[-1]
    else:
        return ema9[-1] < ema20[-1] < ema50[-1]


# ─────────────────────────────────────────────
# WATCHLIST MANAGER
# ─────────────────────────────────────────────

def update_watchlist(symbol: str, scalp_result: Dict) -> Dict:
    """
    Manage the watchlist for "watch mode" pre-entry confirmation.
    
    Call this every scan cycle for any symbol with should_watch=True.
    Returns updated scalp_result with should_trade=True if watchlist confirms.
    """
    cfg = SCALP_CONFIG
    now = time.time()

    if scalp_result.get("should_trade"):
        # Already confirmed — remove from watchlist
        _watchlist.pop(symbol, None)
        return scalp_result

    if scalp_result.get("should_watch"):
        entry = _watchlist.get(symbol)
        if entry is None:
            # Add to watchlist
            _watchlist[symbol] = {
                "score": scalp_result["score"],
                "direction": scalp_result["direction"],
                "candles_watched": 1,
                "ts": now,
            }
            log(f"👁️ WATCH: {symbol} added to scalp watchlist (score={scalp_result['score']})")
        else:
            entry["candles_watched"] += 1
            new_score = scalp_result["score"]

            if entry["candles_watched"] > cfg["watch_mode_ttl"]:
                # Expired
                log(f"⌛ WATCH: {symbol} watchlist expired")
                _watchlist.pop(symbol, None)
            elif new_score >= cfg["trigger_score_threshold"]:
                # Confirmed!
                log(f"✅ WATCH CONFIRMED: {symbol} score rose to {new_score} — triggering!")
                scalp_result["should_trade"] = True
                scalp_result["should_watch"] = False
                scalp_result["reasons"].append("Watch confirmed 🎯")
                _watchlist.pop(symbol, None)
            elif new_score < entry["score"]:
                # Score dropping — abort
                log(f"❌ WATCH: {symbol} score dropped from {entry['score']} to {new_score} — removing")
                _watchlist.pop(symbol, None)
            else:
                entry["score"] = new_score
    else:
        # No longer watching
        if symbol in _watchlist:
            log(f"🗑️ WATCH: {symbol} removed (no longer qualifying)")
        _watchlist.pop(symbol, None)

    return scalp_result


# ─────────────────────────────────────────────
# SCALP EXIT MONITOR (called from monitor.py)
# ─────────────────────────────────────────────

def register_scalp_trade(symbol: str):
    """Call when a scalp trade is opened. Sets up exit tracking state."""
    _scalp_trade_state[symbol] = {
        "score_history": deque(maxlen=10),
        "volume_history": deque(maxlen=20),
        "momentum_fade_count": 0,
        "entry_ts": time.time(),
    }
    log(f"📌 SCALP EXIT: Registered exit monitoring for {symbol}")


def check_scalp_early_exit(
    symbol: str,
    current_score: float,
    current_candles: List[Dict],
    direction: str,
) -> Tuple[bool, str]:
    """
    Check if a scalp trade should be exited early.
    Called from monitor loop every cycle.
    
    Returns (should_exit_early, reason).
    """
    cfg = SCALP_CONFIG
    state = _scalp_trade_state.get(symbol)
    if not state:
        return False, ""

    state["score_history"].append(current_score)

    # Check volume fade
    if current_candles:
        vols = _volumes(current_candles)
        if len(vols) >= 5:
            current_vol = vols[-1]
            vol_ma = sum(vols[-21:-1]) / 20 if len(vols) >= 21 else sum(vols[:-1]) / max(len(vols) - 1, 1)
            state["volume_history"].append(current_vol / vol_ma if vol_ma > 0 else 1.0)

            if len(state["volume_history"]) >= 3:
                recent_ratios = list(state["volume_history"])[-3:]
                if all(r < cfg["volume_fade_ratio"] for r in recent_ratios):
                    return True, "Volume fading 📉"

    # Score drop check
    if len(state["score_history"]) >= cfg["score_exit_cycles"]:
        recent_scores = list(state["score_history"])[-cfg["score_exit_cycles"]:]
        if all(s < cfg["score_exit_threshold"] for s in recent_scores):
            return True, f"Score dropped below {cfg['score_exit_threshold']} for {cfg['score_exit_cycles']} cycles"

    # Engulfing / rejection pattern on current candle
    if current_candles and len(current_candles) >= 3:
        c1 = current_candles[-2]  # Previous closed
        c2 = current_candles[-3]  # One before
        if _is_engulfing_reversal(c1, c2, direction):
            return True, "Engulfing reversal candle 🕯️"

    # BTC reversal impulse
    btc_state = _btc_state
    if btc_state.get("impulse") and btc_state.get("trend") not in [direction.lower(), "unknown"]:
        return True, "BTC reversed impulse 🔄"

    return False, ""


def _is_engulfing_reversal(c1: Dict, c2: Dict, direction: str) -> bool:
    """Detect bearish engulfing (for long) or bullish engulfing (for short)."""
    o1 = float(c1.get("open", 0))
    c1v = float(c1.get("close", 0))
    o2 = float(c2.get("open", 0))
    c2v = float(c2.get("close", 0))

    if direction.lower() == "long":
        # Previous candle was bullish, current bearish and larger
        prev_bullish = c2v > o2
        curr_bearish = c1v < o1
        curr_larger = abs(c1v - o1) > abs(c2v - o2)
        return prev_bullish and curr_bearish and curr_larger
    else:
        # Previous candle was bearish, current bullish and larger
        prev_bearish = c2v < o2
        curr_bullish = c1v > o1
        curr_larger = abs(c1v - o1) > abs(c2v - o2)
        return prev_bearish and curr_bullish and curr_larger


# ─────────────────────────────────────────────
# LOSS TRACKING API
# ─────────────────────────────────────────────

def record_scalp_result(symbol: str, win: bool):
    """
    Record win/loss for a completed scalp trade.
    Updates loss streak and cooldown state.
    """
    if win:
        _symbol_loss_streak[symbol] = 0
        log(f"✅ SCALP WIN: {symbol} — streak reset")
    else:
        _symbol_loss_streak[symbol] += 1
        _symbol_last_loss_time[symbol] = time.time()
        log(f"❌ SCALP LOSS: {symbol} — streak now {_symbol_loss_streak[symbol]}")

    # Cleanup trade state
    _scalp_trade_state.pop(symbol, None)


# ─────────────────────────────────────────────
# TELEGRAM FORMAT
# ─────────────────────────────────────────────

def format_scalp_signal(
    symbol: str,
    direction: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    sl_pct: float,
    tp1_pct: float,
    score: int,
    confidence: float,
    reasons: List[str],
    details: Dict,
    btc_state: str,
) -> str:
    """Format the Telegram scalp signal message."""
    emoji = "🟢" if direction.lower() == "long" else "🔴"
    dir_label = "LONG" if direction.lower() == "long" else "SHORT"
    quality_emoji = "⭐⭐⭐" if score >= 6 else "⭐⭐" if score >= 4 else "⭐"

    # Build reason tags
    reason_tag = " → ".join(reasons[:5]) if reasons else "N/A"
    vol_ratio = details.get("trigger", {}).get("vol_spike_ratio", "N/A")
    comp_dur = details.get("compression", {}).get("duration", "N/A")
    btc_filter = details.get("btc_filter", btc_state)

    # Risk calculation
    risk_pct_per_trade = sl_pct * 20 * 100  # With 20x leverage

    msg = (
        f"{emoji} <b>🎯 1% SCALP HUNTER — {dir_label}</b>\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>TF:</b> 1m/3m | <b>Leverage:</b> 20x\n\n"
        f"<b>Entry:</b> {entry_price:.6f}\n"
        f"<b>SL:</b> {sl_price:.6f} ({sl_pct*100:.2f}%) ⛔\n"
        f"<b>TP1:</b> {tp1_price:.6f} ({tp1_pct*100:.2f}%) 🎯 [50% close]\n\n"
        f"<b>Score:</b> {score}/7 {quality_emoji}\n"
        f"<b>Confidence:</b> {confidence:.1f}%\n"
        f"<b>Account Risk:</b> ~{risk_pct_per_trade:.1f}% (with 20x)\n\n"
        f"<b>Setup:</b> {reason_tag}\n"
        f"<b>Vol Spike:</b> {vol_ratio}x | <b>Compression:</b> {comp_dur} candles\n"
        f"<b>BTC Filter:</b> {btc_filter}\n\n"
        f"<i>After TP1: Move SL to breakeven + enable Smart Trailing</i>"
    )
    return msg


def format_scalp_exit_message(
    symbol: str,
    direction: str,
    exit_reason: str,
    pnl_pct: float,
    entry_price: float,
    exit_price: float,
) -> str:
    """Format early exit Telegram message."""
    pnl_emoji = "✅" if pnl_pct > 0 else "❌"
    return (
        f"⚡ <b>Scalp Early Exit</b> — {symbol}\n"
        f"Direction: {'🟢 LONG' if direction.lower() == 'long' else '🔴 SHORT'}\n"
        f"Entry: {entry_price:.6f} → Exit: {exit_price:.6f}\n"
        f"P&L: {pnl_emoji} {pnl_pct:+.2f}%\n"
        f"Reason: {exit_reason}"
    )


# Export surface
__all__ = [
    "SCALP_CONFIG",
    "update_btc_state",
    "btc_allows_direction",
    "evaluate_scalp_setup",
    "update_watchlist",
    "register_scalp_trade",
    "check_scalp_early_exit",
    "record_scalp_result",
    "format_scalp_signal",
    "format_scalp_exit_message",
    "_btc_state",
    "_watchlist",
]
