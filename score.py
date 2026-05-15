# score.py - Enhanced with Advanced Pattern Detection Integration
# QUALITY PATCH (May 2026) — Win-rate optimization
# =============================================================================
# Changes vs previous version:
#   - WEIGHTS rebalanced to favor proven backscan-edge indicators
#   - determine_direction(): stricter consensus check (was total > 1.0)
#   - check_4h_alignment_veto(): hard veto for trades fighting 4h trend
#   - check_not_chasing(): blocks already-extended entries
#   - calculate_confidence(): tighter baseline + counter-trend penalties
#   - _apply_strong_indicator_gate(): require ≥ 2 *aligned* strong indicators
# =============================================================================

from logger import log
from rsi import (
    calculate_rsi, calculate_rsi_with_bands, calculate_stoch_rsi,
    analyze_multi_timeframe_rsi, detect_rsi_divergence, calculate_rsi_with_scoring
)
from macd import detect_macd_cross, get_macd_divergence, get_macd_momentum
from supertrend import (
    calculate_supertrend_signal, get_supertrend_state,
    detect_supertrend_squeeze, calculate_multi_timeframe_supertrend
)
from ema import (
    detect_ema_crossover, calculate_ema_ribbon,
    analyze_ema_ribbon, detect_ema_squeeze
)
from bollinger import (
    calculate_bollinger_bands, detect_band_walk,
    get_bollinger_signal, detect_bollinger_squeeze
)
from pattern_detector import (
    detect_pattern, analyze_pattern_strength, detect_pattern_cluster,
    get_pattern_direction, pattern_success_probability, get_all_patterns,
    PATTERN_WEIGHTS, REVERSAL_PATTERNS, CONTINUATION_PATTERNS
)
from volume import (
    is_volume_spike, get_average_volume, detect_volume_climax,
    get_volume_profile, get_volume_weighted_average_price, analyze_volume_trend
)
from stealth_detector import (
    detect_volume_divergence, detect_slow_breakout,
    detect_stealth_accumulation_advanced
)
from whale_detector import (
    detect_whale_activity, detect_whale_activity_advanced,
    analyze_whale_impact
)
from error_handler import send_error_to_telegram
from config import ALWAYS_ALLOW_SWING
from indicator_fixes import (
    rebalance_indicator_scores, get_balanced_rsi_signal,
    analyze_volume_direction
)
from enhanced_entry_validator import entry_validator
from pattern_context_analyzer import pattern_context_analyzer
from divergence_detector import divergence_detector
import numpy as np


# ── WEIGHTS (quality-tuned, May 2026) ────────────────────────────────────────
# Punishes single-TF noise (single MACD/EMA cross), rewards confluence
# (MTF Supertrend, MTF RSI), divergences, and whale activity.
WEIGHTS = {
    # Momentum / trend
    "macd":               0.5,   # ↓ 0.6 — single cross alone is weak
    "macd_divergence":    1.1,   # ↑ 1.0 — divergences higher quality
    "macd_momentum":      1.3,   # ↑ 1.2 — best MACD signal
    "ema":                0.7,   # ↓ 0.9 — single-TF EMA cross overfires
    "ema_ribbon":         1.1,   # ↑ 1.0 — ribbon trend more reliable
    "ema_squeeze":        0.5,   # ↓ 0.6 — mostly noise on alts
    # Volume
    "volume_spike":       0.8,   # ↑ 0.7 — combined with quality check
    "volume_climax":      1.2,   # ↑ 1.1
    "volume_profile":     0.5,
    # VWAP / supertrend
    "vwap":               0.9,   # ↑ 0.8 — strong predictor
    "supertrend":         1.2,   # ↑ 1.1
    "supertrend_squeeze": 0.6,   # ↓ 0.7
    "supertrend_mtf":     1.5,   # ↑ 1.3 — MTF alignment is THE edge
    # RSI family
    "rsi":                0.6,   # ↓ 0.7 — single RSI is weak
    "rsi_divergence":     1.2,   # ↑ 1.0 — divergences strong
    "stoch_rsi":          0.7,   # ↓ 0.8
    "rsi_mtf":            1.3,   # ↑ 1.1 — MTF confluence strong
    # Bollinger / band walk
    "bollinger":          0.5,   # ↓ 0.6
    "bollinger_squeeze":  0.0,   # REMOVED — anti-correlated with longs
    "band_walk":          1.0,   # ↑ 0.9 — band walks confirm trend
    # Patterns
    "pattern":            0.7,   # ↓ 0.8 — patterns alone misleading
    "pattern_cluster":    0.5,   # ↑ 0.4
    "pattern_quality":    0.7,   # ↑ 0.6
    "pattern_confluence": 0.6,   # ↑ 0.5
    # Other
    "divergence":         0.7,   # ↑ 0.6
    "slow_breakout":      0.9,   # ↑ 0.8
    "whale":              0.8,
    "whale_advanced":     1.1,   # ↑ 1.0 — high-edge
    "momentum":           1.3,   # ↑ 1.2
    "stealth":            0.8,
    "strong_stealth":     1.1,   # ↑ 1.0
}

# Trade type to timeframe mapping
TRADE_TYPE_TF = {
    "Scalp":    ["1", "3"],
    "Intraday": ["5", "15"],
    "Swing":    ["30", "60", "240"],
}

MIN_TF_REQUIRED = {
    "Scalp":    1,
    "Intraday": 1,
    "Swing":    2,
}

MAX_PATTERN_CONTRIBUTION = 2.0

# Volume spike threshold — backscan: avg vol_ratio at pump = 2.09x
VOLUME_SPIKE_THRESHOLD = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_detect_momentum_strength(candles):
    """Safe wrapper for detect_momentum_strength"""
    try:
        if not candles or len(candles) < 10:
            return False, None, 0
        if isinstance(candles, (list, tuple)):
            return detect_momentum_strength(candles)
        else:
            return False, None, 0
    except Exception:
        return False, None, 0


def detect_momentum_strength(candles, lookback=5):
    """Detect if price is showing strong momentum"""
    if len(candles) < lookback + 5:
        return False, None, 0

    recent = candles[-lookback:]
    prior = candles[-(lookback + 5):-lookback]

    recent_vol_avg = sum(float(c['volume']) for c in recent) / len(recent)
    prior_vol_avg = sum(float(c['volume']) for c in prior) / len(prior)
    vol_increase = recent_vol_avg / prior_vol_avg if prior_vol_avg > 0 else 1

    consecutive_up = 0
    consecutive_down = 0

    for i in range(len(recent)):
        candle_close = float(recent[i]['close'])
        candle_open = float(recent[i]['open'])
        if candle_close > candle_open:
            consecutive_up += 1
            consecutive_down = 0
        elif candle_close < candle_open:
            consecutive_down += 1
            consecutive_up = 0

    first_candle_open = float(recent[0]['open'])
    last_candle_close = float(recent[-1]['close'])
    price_change_pct = ((last_candle_close - first_candle_open) / first_candle_open) * 100

    direction = "bullish" if price_change_pct > 0 else "bearish"

    strength = 0
    if consecutive_up >= 3 or consecutive_down >= 3:
        strength += 0.4
    if vol_increase >= 1.5:
        strength += 0.3
    if abs(price_change_pct) >= 1.0:
        strength += 0.3

    has_momentum = strength >= 0.6
    return has_momentum, direction, strength


# ─────────────────────────────────────────────────────────────────────────────
# 4H TREND GATE — backscan: 4h is most predictive (+7-8% diff)
# ─────────────────────────────────────────────────────────────────────────────

def get_4h_trend_gate(candles_by_timeframe):
    """
    Returns a score adjustment based on 4h momentum indicators.
    Called in score_symbol() before the final return.
    Returns: (adjustment: float, reason: str)
    """
    candles_4h = (
        candles_by_timeframe.get("240", []) or
        candles_by_timeframe.get("4h", []) or
        []
    )
    if not candles_4h or len(candles_4h) < 20:
        return 0.0, "no_4h_data"

    adjustment = 0.0
    reasons = []

    # 1. 4H Supertrend direction (+6.4% diff)
    try:
        trend_4h = calculate_supertrend_signal(candles_4h)
        if trend_4h == "bullish":
            adjustment += 0.5
            reasons.append("4h_st_bull")
        elif trend_4h == "bearish":
            adjustment -= 0.5
            reasons.append("4h_st_bear")
    except Exception:
        pass

    # 2. 4H MACD histogram direction (+7.4% diff)
    try:
        macd_mom_4h = get_macd_momentum(candles_4h)
        if macd_mom_4h > 0.3:
            adjustment += 0.6
            reasons.append("4h_macd_hist_pos")
        elif macd_mom_4h < -0.3:
            adjustment -= 0.6
            reasons.append("4h_macd_hist_neg")
    except Exception:
        pass

    # 3. 4H Momentum direction (+8.2% diff — highest single indicator)
    try:
        closes_4h = [float(c["close"]) for c in candles_4h]
        if len(closes_4h) >= 12:
            mom_10 = closes_4h[-1] - closes_4h[-11]
            if mom_10 > 0:
                adjustment += 0.6
                reasons.append("4h_mom_pos")
            else:
                adjustment -= 0.6
                reasons.append("4h_mom_neg")
    except Exception:
        pass

    # Cap at ±1.5
    adjustment = max(-1.5, min(1.5, adjustment))
    return round(adjustment, 2), ", ".join(reasons)


def check_4h_alignment_veto(candles_by_timeframe, direction):
    """
    QUALITY PATCH: Hard-veto trades fighting the 4h trend.
    Long against 2+ bearish 4h signals → reject.
    Returns (allow: bool, reason: str).
    """
    candles_4h = (
        candles_by_timeframe.get("240", []) or
        candles_by_timeframe.get("4h", []) or
        []
    )
    if not candles_4h or len(candles_4h) < 20:
        return True, "no_4h_data"  # Don't punish data gaps

    try:
        st_4h = calculate_supertrend_signal(candles_4h)
        macd_mom_4h = get_macd_momentum(candles_4h)
        closes_4h = [float(c["close"]) for c in candles_4h]
        mom_10 = closes_4h[-1] - closes_4h[-11] if len(closes_4h) >= 12 else 0

        bearish_signals_4h = sum([
            st_4h == "bearish",
            macd_mom_4h < -0.3,
            mom_10 < 0,
        ])
        bullish_signals_4h = sum([
            st_4h == "bullish",
            macd_mom_4h > 0.3,
            mom_10 > 0,
        ])

        if direction == "Long" and bearish_signals_4h >= 2:
            return False, f"4h_bearish_veto({bearish_signals_4h}/3)"
        if direction == "Short" and bullish_signals_4h >= 2:
            return False, f"4h_bullish_veto({bullish_signals_4h}/3)"

        # Diagnosis fix (no shorts): the previous "bullish_signals_4h >= 1
        # → veto short" line made shorts mathematically near-impossible to
        # generate. ANY single mildly-bullish 4h signal (Supertrend OR MACD
        # OR momentum) would reject every short outright. Combined with the
        # 2+ rule above we already have a real bullish-veto. Removed the
        # 1-signal veto so symmetric long/short logic actually works.

        return True, "4h_aligned"
    except Exception as e:
        log(f"⚠️ 4h veto check failed: {e}", level="WARN")
        return True, "4h_check_error"  # Fail open


# ─────────────────────────────────────────────────────────────────────────────
# ANTI-CHASE FILTER — block already-extended entries
# ─────────────────────────────────────────────────────────────────────────────

def check_not_chasing(candles_by_timeframe, direction, trade_type):
    """
    Block already-extended entries — but not so tightly that you miss
    actual breakouts.

    Diagnosis fix (big movers missed): previous thresholds (1.0% / 1.8% /
    3.0%) were tighter than the typical pre-signal lag of the scoring
    chain. By the time MACD/Supertrend confirmed a move, price had
    already moved that much and anti-chase blocked the entry. Loosened
    to thresholds that match observed signal-to-entry lag:
      Scalp:    2.0% over 3 × 1m bars
      Intraday: 3.0% over 3 × 5m bars
      Swing:    5.0% over 4 × 15m bars

    Returns (allow: bool, reason: str).
    """
    if trade_type == "Scalp":
        tf, max_recent_move_pct, lookback = "1", 2.0, 3
    elif trade_type == "Intraday":
        tf, max_recent_move_pct, lookback = "5", 3.0, 3
    else:  # Swing
        tf, max_recent_move_pct, lookback = "15", 5.0, 4

    candles = candles_by_timeframe.get(tf, [])
    if len(candles) < lookback + 1:
        return True, "insufficient_data"

    try:
        ref_close = float(candles[-(lookback + 1)]["close"])
        cur_close = float(candles[-1]["close"])
        if ref_close <= 0:
            return True, "bad_price"

        move_pct = ((cur_close - ref_close) / ref_close) * 100

        if direction == "Long" and move_pct > max_recent_move_pct:
            return False, f"chasing_long(+{move_pct:.2f}%)"
        if direction == "Short" and move_pct < -max_recent_move_pct:
            return False, f"chasing_short({move_pct:.2f}%)"

        return True, f"entry_clean({move_pct:+.2f}%)"
    except Exception:
        return True, "chase_check_error"


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_minimum_volume_threshold(trade_type):
    """Get minimum volume threshold based on trade type"""
    thresholds = {
        "Scalp":    1.5,
        "Intraday": 1.8,
        "Swing":    2.0,
    }
    return thresholds.get(trade_type, 1.8)


def check_volume_quality(candles, trade_type):
    """Check if volume quality meets requirements"""
    if not candles or len(candles) < 10:
        return False, "insufficient_data"

    threshold = get_minimum_volume_threshold(trade_type)
    avg_vol = get_average_volume(candles)
    current_vol = float(candles[-1].get('volume', 0))

    if avg_vol <= 0:
        return False, "zero_volume"

    ratio = current_vol / avg_vol
    if ratio >= threshold:
        return True, f"volume_ok_{ratio:.1f}x"
    return False, f"volume_low_{ratio:.1f}x"


def calculate_volume_penalty(candles, trade_type):
    """Calculate penalty for poor volume"""
    if not candles or len(candles) < 10:
        return 0

    threshold = get_minimum_volume_threshold(trade_type)
    avg_vol = get_average_volume(candles)
    current_vol = float(candles[-1].get('volume', 0))

    if avg_vol <= 0:
        return 0

    ratio = current_vol / avg_vol
    if ratio < threshold:
        shortfall = threshold - ratio
        return min(shortfall * 0.5, 1.5)
    return 0


def has_pump_potential(candles_by_timeframe, direction):
    """Check if symbol shows pump potential"""
    if direction != "Long":
        return False

    for tf in ['1', '5', '15']:
        if tf in candles_by_timeframe and candles_by_timeframe[tf]:
            candles = candles_by_timeframe[tf]
            if len(candles) >= 10:
                has_momentum, mom_dir, strength = safe_detect_momentum_strength(candles)
                if has_momentum and mom_dir == "bullish" and strength > 0.7:
                    return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTION (QUALITY PATCH — stricter consensus)
# ─────────────────────────────────────────────────────────────────────────────

def determine_direction(tf_scores):
    """
    QUALITY PATCH: Stricter consensus check.
    Requires:
      - total bias > 2.0 (was > 1.0)
      - majority of TFs agreeing
      - no strongly opposing TF (>2.0)
    Returns "Long", "Short", or None (no clean direction = no trade).
    """
    if not tf_scores:
        return None

    values = list(tf_scores.values())
    total_score = sum(values)
    n = len(values)

    if n == 0:
        return None

    bullish_tfs = sum(1 for v in values if v > 0.5)
    bearish_tfs = sum(1 for v in values if v < -0.5)

    has_strong_bearish_tf = any(v < -2.0 for v in values)
    has_strong_bullish_tf = any(v > 2.0 for v in values)

    # Long requires total > 2.0, more bullish than bearish TFs,
    # no strong bearish TF, and ≥ 50% TFs aligned
    if total_score > 2.0 and bullish_tfs > bearish_tfs and not has_strong_bearish_tf:
        if bullish_tfs / max(n, 1) >= 0.5:
            return "Long"

    # Short — same logic in reverse
    if total_score < -2.0 and bearish_tfs > bullish_tfs and not has_strong_bullish_tf:
        if bearish_tfs / max(n, 1) >= 0.5:
            return "Short"

    return None  # No clean direction = no trade


def determine_direction_with_pattern_priority(tf_scores, candles_by_timeframe):
    """Determine direction with strong pattern taking priority"""
    basic_direction = determine_direction(tf_scores)

    for tf in ['15', '5', '1']:
        if tf in candles_by_timeframe and candles_by_timeframe[tf]:
            candles = candles_by_timeframe[tf]
            pattern = detect_pattern(candles)
            if pattern:
                pattern_dir = get_pattern_direction(pattern)
                if pattern_dir in ["bullish", "bearish"]:
                    strength = analyze_pattern_strength(pattern, candles)
                    if strength > 0.7:
                        return "Long" if pattern_dir == "bullish" else "Short"

    return basic_direction


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE (QUALITY PATCH — tighter baseline + counter-trend penalty)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_confidence(score, tf_scores, market_context, trade_type):
    """
    QUALITY PATCH: Tighter confidence — must EARN the 65% gate.
    Old baseline: score * 7.5 (score 9 → 67.5% trivially)
    New baseline: score * 6.0 (score 9 → 54%, must earn bonuses)
    """
    if not tf_scores:
        return 0

    # Tighter baseline
    base_confidence = min(score * 6.0, 78)

    # TF alignment — reward strong consensus, not just majority
    if len(tf_scores) > 0:
        positive_tfs = sum(1 for v in tf_scores.values() if v > 0.5)
        negative_tfs = sum(1 for v in tf_scores.values() if v < -0.5)
        consensus = max(positive_tfs, negative_tfs) / len(tf_scores)
        if consensus >= 0.75:
            alignment_bonus = 12
        elif consensus >= 0.5:
            alignment_bonus = 6
        else:
            alignment_bonus = -3   # Mixed signals = penalty
    else:
        alignment_bonus = 0

    # Trade type bonus — slightly reduced
    type_bonus = {"Scalp": 3, "Intraday": 2, "Swing": 1}.get(trade_type, 0)

    # Market context bonus — counter-trend gets harsh penalty
    market_bonus = 0
    if market_context:
        btc_trend = market_context.get("btc_trend", "neutral")
        regime = market_context.get("regime", "").lower()
        if btc_trend == "bullish" and trade_type != "Short":
            market_bonus = 4
        elif btc_trend == "bearish" and trade_type == "Short":
            market_bonus = 4
        elif btc_trend in ("bearish", "downtrend"):
            market_bonus = -8   # Counter-trend long = harsh penalty
        elif regime == "ranging":
            market_bonus = -3

    confidence = base_confidence + alignment_bonus + type_bonus + market_bonus
    return max(0, min(int(confidence), 100))


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN SCORING
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_pattern_scoring(candles, tf_label, score, indicator_scores, used_indicators):
    """Enhanced pattern scoring with advanced pattern detection"""

    pattern_score_total = 0

    # Detect primary pattern
    pattern = detect_pattern(candles)

    if pattern:
        pattern_strength = analyze_pattern_strength(pattern, candles)
        pattern_direction = get_pattern_direction(pattern)
        base_pattern_score = WEIGHTS.get("pattern", 0.7)

        if pattern in ["spinning_top", "doji", "harami"]:
            adjusted_score = base_pattern_score * pattern_strength * 0.5
            pattern_score_total += adjusted_score
            indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score
        else:
            adjusted_score = base_pattern_score * pattern_strength
            if pattern_direction == "bullish":
                pattern_score_total += adjusted_score
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score
            elif pattern_direction == "bearish":
                pattern_score_total -= adjusted_score
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = -adjusted_score
            else:
                pattern_score_total += adjusted_score * 0.3
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score * 0.3

        used_indicators.add(f"pattern_{pattern}")

        # Direction-aware cluster scoring
        pattern_cluster = detect_pattern_cluster(candles, lookback=10)
        if len(pattern_cluster) >= 2:
            cluster_patterns = [p['pattern'] for p in pattern_cluster]

            bullish_count = sum(
                1 for p in cluster_patterns
                if p in REVERSAL_PATTERNS["bullish"] or p in CONTINUATION_PATTERNS["bullish"]
            )
            bearish_count = sum(
                1 for p in cluster_patterns
                if p in REVERSAL_PATTERNS["bearish"] or p in CONTINUATION_PATTERNS["bearish"]
            )

            if bullish_count >= 2:
                cluster_bonus = WEIGHTS["pattern_cluster"] * bullish_count
                score += cluster_bonus
                indicator_scores[f"{tf_label}_pattern_cluster_bullish"] = cluster_bonus
                used_indicators.add("pattern_cluster")
                log(f"📊 Bullish pattern cluster on {tf_label}: {cluster_patterns} (+{cluster_bonus:.2f})")
            elif bearish_count >= 2:
                cluster_bonus = WEIGHTS["pattern_cluster"] * bearish_count
                score -= cluster_bonus
                indicator_scores[f"{tf_label}_pattern_cluster_bearish"] = -cluster_bonus
                used_indicators.add("pattern_cluster")
                log(f"📊 Bearish pattern cluster on {tf_label}: {cluster_patterns} (-{cluster_bonus:.2f})")
            else:
                log(f"📊 Mixed/neutral pattern cluster on {tf_label}: {cluster_patterns} (no bonus)")

    # All-pattern confluence
    all_patterns = get_all_patterns(candles)
    pattern_count = sum(1 for detected in all_patterns.values() if detected)

    if pattern_count >= 3:
        confluence_bonus = WEIGHTS["pattern_confluence"]
        score += confluence_bonus
        indicator_scores[f"{tf_label}_pattern_confluence"] = confluence_bonus
        used_indicators.add("pattern_confluence")
        detected_patterns = [name for name, detected in all_patterns.items() if detected]
        log(f"📊 Pattern confluence on {tf_label}: {detected_patterns}")

    capped_pattern_score = min(pattern_score_total, MAX_PATTERN_CONTRIBUTION)
    score += capped_pattern_score

    return score, indicator_scores, used_indicators


# ─────────────────────────────────────────────────────────────────────────────
# STRONG INDICATOR GATE (QUALITY PATCH)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_strong_indicator_gate(symbol, original_score, indicator_scores, tf_scores):
    """
    QUALITY PATCH: Require ≥ 2 strong indicators ALIGNED with direction.
    - Any strong indicator pointing wrong way alone vetoes.
    - 1 aligned strong = 50% penalty.
    - 0 aligned strong = reject.
    Returns (passed: bool, adjusted_score: float).
    """
    direction = determine_direction(tf_scores)
    if direction is None:
        return False, 0.0

    expected_sign = 1 if direction == "Long" else -1

    strong_aligned = 0
    strong_against = 0
    for k, v in indicator_scores.items():
        if abs(v) >= 0.8:
            if (v > 0 and expected_sign > 0) or (v < 0 and expected_sign < 0):
                strong_aligned += 1
            else:
                strong_against += 1

    # Hard veto: more strong-against than aligned (or close to it)
    if strong_against >= 1 and strong_aligned < strong_against + 1:
        log(f"❌ {symbol}: Rejected — {strong_against} strong indicator(s) "
            f"against {direction} ({strong_aligned} aligned)")
        return False, 0.0

    if strong_aligned == 0:
        log(f"❌ {symbol}: Rejected — zero aligned strong indicators")
        return False, 0.0

    if strong_aligned == 1:
        adj = round(original_score * 0.5, 2)
        log(f"⚠️ {symbol}: Only 1 aligned strong indicator — 50% penalty "
            f"(score {original_score:.2f} → {adj:.2f})")
        return True, adj

    return True, original_score


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_symbol(symbol, candles_by_timeframe, market_context=None):
    """
    Score a symbol using multi-timeframe indicator analysis.
    Weights are optimized from backscan of 9,343 pump/dump events.
    """
    if market_context is None:
        market_context = {}

    tf_scores = {}
    type_scores = {"Scalp": 0, "Intraday": 0, "Swing": 0}
    tf_count = {"Scalp": 0, "Intraday": 0, "Swing": 0}
    indicator_scores = {}
    used_indicators = set()

    # ── Initialize MTF variables ─────────────────────────────────────────────
    mtf_supertrend = {'alignment': 0, 'overall_trend': None}
    mtf_rsi = {'overall_signal': None, 'confluence_strength': 0}
    vwap_values = {}

    try:
        mtf_supertrend = calculate_multi_timeframe_supertrend(candles_by_timeframe)
    except Exception as e:
        log(f"⚠️ MTF Supertrend calculation failed: {e}", level="WARN")

    try:
        mtf_rsi = analyze_multi_timeframe_rsi(symbol, candles_by_timeframe)
    except Exception as e:
        log(f"⚠️ MTF RSI calculation failed: {e}", level="WARN")

    for tf, candles in candles_by_timeframe.items():
        if candles and len(candles) >= 10:
            try:
                vwap = get_volume_weighted_average_price(candles)
                if vwap:
                    vwap_values[tf] = vwap
            except Exception:
                pass

    # Get current price
    current_price = 0
    for tf in ['1', '5', '15', '30', '60']:
        if tf in candles_by_timeframe and candles_by_timeframe[tf]:
            try:
                current_price = float(candles_by_timeframe[tf][-1]['close'])
                break
            except Exception:
                continue

    # Momentum data
    momentum_data = {
        "1m":  safe_detect_momentum_strength(candles_by_timeframe.get("1", [])),
        "5m":  safe_detect_momentum_strength(candles_by_timeframe.get("5", [])),
        "15m": safe_detect_momentum_strength(candles_by_timeframe.get("15", [])),
    }
    has_momentum = any(data[0] for data in momentum_data.values())
    momentum_direction = None
    for data in momentum_data.values():
        if data[0]:
            momentum_direction = data[1]
            break

    # ── Main scoring loop ────────────────────────────────────────────────────
    for tf, candles in candles_by_timeframe.items():
        if not candles or len(candles) < 10:
            continue

        score = 0
        tf_label = f"{tf}m"

        try:
            # ── Common indicators for ALL timeframes ─────────────────────────

            # 1. VWAP Analysis
            if tf in vwap_values and current_price > 0:
                vwap = vwap_values[tf]
                if current_price > vwap * 1.005:
                    score += WEIGHTS["vwap"]
                    indicator_scores[f"{tf_label}_vwap"] = WEIGHTS["vwap"]
                elif current_price < vwap * 0.995:
                    score -= WEIGHTS["vwap"]
                    indicator_scores[f"{tf_label}_vwap"] = -WEIGHTS["vwap"]
                used_indicators.add("vwap")

            # 2. Advanced Whale Detection
            whale_advanced = detect_whale_activity_advanced(candles, symbol)
            if whale_advanced.get('detected'):
                strength = whale_advanced.get('strength', 0)
                rec = whale_advanced.get('recommendation', '')
                if rec == 'potential_long':
                    score += WEIGHTS["whale_advanced"] * strength
                    indicator_scores[f"{tf_label}_whale_advanced"] = WEIGHTS["whale_advanced"] * strength
                elif rec == 'potential_short':
                    score -= WEIGHTS["whale_advanced"] * strength
                    indicator_scores[f"{tf_label}_whale_advanced"] = -WEIGHTS["whale_advanced"] * strength
                used_indicators.add("whale_advanced")

            # ── Timeframe-specific indicators ────────────────────────────────

            if tf in TRADE_TYPE_TF["Scalp"]:
                # SCALP INDICATORS (1m, 3m)
                macd = detect_macd_cross(candles)
                ema = detect_ema_crossover(candles)

                if macd == "bullish":
                    score += WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = WEIGHTS["macd"]
                elif macd == "bearish":
                    score -= WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = -WEIGHTS["macd"]

                # MACD Divergence
                macd_div = get_macd_divergence(candles)
                if macd_div:
                    if macd_div['type'] == 'bullish_divergence':
                        score += WEIGHTS["macd_divergence"]
                        indicator_scores[f"{tf_label}_macd_divergence"] = WEIGHTS["macd_divergence"]
                    else:
                        score -= WEIGHTS["macd_divergence"]
                        indicator_scores[f"{tf_label}_macd_divergence"] = -WEIGHTS["macd_divergence"]
                    used_indicators.add("macd_divergence")

                # MACD Momentum
                macd_momentum = get_macd_momentum(candles)
                if abs(macd_momentum) > 0.5:
                    score += WEIGHTS["macd_momentum"] * macd_momentum
                    indicator_scores[f"{tf_label}_macd_momentum"] = WEIGHTS["macd_momentum"] * macd_momentum
                    used_indicators.add("macd_momentum")

                if ema == "bullish":
                    score += WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = WEIGHTS["ema"]
                elif ema == "bearish":
                    score -= WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = -WEIGHTS["ema"]

                # Enhanced RSI Analysis
                rsi_data = calculate_rsi_with_bands(candles)
                if rsi_data:
                    market_trend = market_context.get("btc_trend", "neutral")
                    rsi_signal, rsi_strength = get_balanced_rsi_signal(rsi_data, market_trend)
                    
                    # Extra veto if strong opposite 4h
                    if market_trend == "downtrend" and rsi_signal == "buy":
                        rsi_strength *= 0.4
                    if rsi_signal == "buy":
                        score += WEIGHTS["rsi"] * rsi_strength
                        indicator_scores[f"{tf_label}_rsi"] = WEIGHTS["rsi"] * rsi_strength
                    elif rsi_signal == "sell":
                        score -= WEIGHTS["rsi"] * rsi_strength
                        indicator_scores[f"{tf_label}_rsi"] = -WEIGHTS["rsi"] * rsi_strength

                    # RSI momentum
                    if rsi_data.get('momentum'):
                        momentum_score = WEIGHTS["rsi"] * 0.3 * (rsi_data['momentum'] / 10)
                        score += momentum_score
                        indicator_scores[f"{tf_label}_rsi_momentum"] = momentum_score
                        used_indicators.add("rsi_momentum")

                # Volume spike (1.8x)
                volume_spike = bool(is_volume_spike(candles, VOLUME_SPIKE_THRESHOLD))
                if volume_spike:
                    score += WEIGHTS["volume_spike"]
                    indicator_scores[f"{tf_label}_volume"] = WEIGHTS["volume_spike"]

                # Volume Climax.
                # Bug fix (volume.py audit): previous code added +volume_climax
                # score for ANY climax. But detect_volume_climax returns
                # ("buying", "selling") and a "buying climax" in classical TA
                # is an EXHAUSTION TOP — bearish for longs. Now type-aware:
                #   buying climax  → bearish for long signals (subtract)
                #   selling climax → bullish for long signals (add)
                vol_climax_result = detect_volume_climax(candles)
                if isinstance(vol_climax_result, tuple) and vol_climax_result[0]:
                    climax_type = vol_climax_result[1]  # "buying" / "selling"
                    if climax_type == "selling":
                        # Selling climax = bearish exhaustion = potential bullish reversal
                        score += WEIGHTS["volume_climax"]
                        indicator_scores[f"{tf_label}_volume_climax_sell"] = WEIGHTS["volume_climax"]
                    elif climax_type == "buying":
                        # Buying climax = bullish exhaustion = potential bearish reversal
                        score -= WEIGHTS["volume_climax"]
                        indicator_scores[f"{tf_label}_volume_climax_buy"] = -WEIGHTS["volume_climax"]
                    used_indicators.add("volume_climax")

                # Stealth accumulation
                try:
                    stealth_result = detect_stealth_accumulation_advanced(candles, symbol)
                    if stealth_result.get('detected'):
                        stealth_score = WEIGHTS["stealth"] * stealth_result.get('strength', 0)
                        score += stealth_score
                        indicator_scores[f"{tf_label}_stealth"] = stealth_score
                        used_indicators.add("stealth")
                        if stealth_result.get('recommendation') == 'strong_accumulation':
                            score += WEIGHTS["strong_stealth"] * 0.5
                            indicator_scores[f"{tf_label}_strong_stealth"] = WEIGHTS["strong_stealth"] * 0.5
                except Exception:
                    pass

                if detect_volume_divergence(candles):
                    score += WEIGHTS["divergence"]
                    indicator_scores[f"{tf_label}_divergence"] = WEIGHTS["divergence"]

                if detect_slow_breakout(candles):
                    score += WEIGHTS["slow_breakout"]
                    indicator_scores[f"{tf_label}_slow_breakout"] = WEIGHTS["slow_breakout"]

                if detect_whale_activity(candles):
                    score += WEIGHTS["whale"]
                    indicator_scores[f"{tf_label}_whale"] = WEIGHTS["whale"]

                # Enhanced pattern detection
                score, indicator_scores, used_indicators = enhanced_pattern_scoring(
                    candles, tf_label, score, indicator_scores, used_indicators
                )

                type_scores["Scalp"] += score
                tf_count["Scalp"] += 1
                used_indicators.update(["macd", "ema", "volume", "divergence", "slow_breakout", "whale"])

            elif tf in TRADE_TYPE_TF["Intraday"]:
                # INTRADAY INDICATORS (5m, 15m)
                macd = detect_macd_cross(candles)
                ema = detect_ema_crossover(candles)
                trend = calculate_supertrend_signal(candles)

                # Volume spike
                volume_spike = bool(is_volume_spike(candles, VOLUME_SPIKE_THRESHOLD))
                if volume_spike:
                    score += WEIGHTS["volume_spike"]
                    indicator_scores[f"{tf_label}_volume"] = WEIGHTS["volume_spike"]

                if macd == "bullish":
                    score += WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = WEIGHTS["macd"]
                elif macd == "bearish":
                    score -= WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = -WEIGHTS["macd"]

                # MACD Divergence
                macd_div = get_macd_divergence(candles)
                if macd_div:
                    if macd_div['type'] == 'bullish_divergence':
                        score += WEIGHTS["macd_divergence"]
                        indicator_scores[f"{tf_label}_macd_divergence"] = WEIGHTS["macd_divergence"]
                    else:
                        score -= WEIGHTS["macd_divergence"]
                        indicator_scores[f"{tf_label}_macd_divergence"] = -WEIGHTS["macd_divergence"]
                    used_indicators.add("macd_divergence")

                # MACD Momentum
                macd_momentum = get_macd_momentum(candles)
                if abs(macd_momentum) > 0.5:
                    score += WEIGHTS["macd_momentum"] * macd_momentum
                    indicator_scores[f"{tf_label}_macd_momentum"] = WEIGHTS["macd_momentum"] * macd_momentum
                    used_indicators.add("macd_momentum")

                if ema == "bullish":
                    score += WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = WEIGHTS["ema"]
                elif ema == "bearish":
                    score -= WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = -WEIGHTS["ema"]

                # EMA Ribbon
                ribbon = calculate_ema_ribbon(candles)
                ribbon_analysis = analyze_ema_ribbon(ribbon)
                if ribbon_analysis['trend'] == 'bullish':
                    score += WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                elif ribbon_analysis['trend'] == 'bearish':
                    score -= WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = -WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                used_indicators.add("ema_ribbon")

                if trend == "bullish":
                    score += WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = WEIGHTS["supertrend"]
                elif trend == "bearish":
                    score -= WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = -WEIGHTS["supertrend"]

                if detect_volume_divergence(candles):
                    score += WEIGHTS["divergence"]
                    indicator_scores[f"{tf_label}_divergence"] = WEIGHTS["divergence"]

                if detect_slow_breakout(candles):
                    score += WEIGHTS["slow_breakout"]
                    indicator_scores[f"{tf_label}_slow_breakout"] = WEIGHTS["slow_breakout"]

                if detect_whale_activity(candles):
                    score += WEIGHTS["whale"]
                    indicator_scores[f"{tf_label}_whale"] = WEIGHTS["whale"]

                # Enhanced pattern detection
                score, indicator_scores, used_indicators = enhanced_pattern_scoring(
                    candles, tf_label, score, indicator_scores, used_indicators
                )

                type_scores["Intraday"] += score
                tf_count["Intraday"] += 1
                used_indicators.update(["macd", "ema", "supertrend", "volume", "divergence", "slow_breakout", "whale"])

            elif tf in TRADE_TYPE_TF["Swing"]:
                # SWING INDICATORS (30m, 60m, 240m)

                # RSI
                rsi_data = calculate_rsi_with_bands(candles, symbol=f"{symbol}_{tf_label}")
                if rsi_data:
                    rsi_signal, rsi_strength = get_balanced_rsi_signal(
                        rsi_data, market_trend=market_context.get("btc_trend", "neutral")
                    )
                    if rsi_signal == "buy":
                        score += WEIGHTS["rsi"] * rsi_strength
                        indicator_scores[f"{tf_label}_rsi"] = WEIGHTS["rsi"] * rsi_strength
                    elif rsi_signal == "sell":
                        score -= WEIGHTS["rsi"] * rsi_strength
                        indicator_scores[f"{tf_label}_rsi"] = -WEIGHTS["rsi"] * rsi_strength

                    # RSI Divergence
                    if rsi_data.get('divergence'):
                        if rsi_data['divergence'] == 'bullish_divergence':
                            score += WEIGHTS["rsi_divergence"]
                            indicator_scores[f"{tf_label}_rsi_divergence"] = WEIGHTS["rsi_divergence"]
                        else:
                            score -= WEIGHTS["rsi_divergence"]
                            indicator_scores[f"{tf_label}_rsi_divergence"] = -WEIGHTS["rsi_divergence"]
                        used_indicators.add("rsi_divergence")

                    # RSI momentum
                    if rsi_data.get('momentum'):
                        momentum_score = WEIGHTS["rsi"] * 0.3 * (rsi_data['momentum'] / 10)
                        score += momentum_score
                        indicator_scores[f"{tf_label}_rsi_momentum"] = momentum_score
                        used_indicators.add("rsi_momentum")

                # Stochastic RSI
                stoch_rsi = calculate_stoch_rsi(candles)
                if stoch_rsi:
                    if stoch_rsi['oversold']:
                        score += WEIGHTS["stoch_rsi"]
                        indicator_scores[f"{tf_label}_stoch_rsi"] = WEIGHTS["stoch_rsi"]
                    elif stoch_rsi['overbought']:
                        score -= WEIGHTS["stoch_rsi"]
                        indicator_scores[f"{tf_label}_stoch_rsi"] = -WEIGHTS["stoch_rsi"]

                    if stoch_rsi.get('cross') == 'bullish_cross':
                        score += WEIGHTS["stoch_rsi"] * 0.5
                        indicator_scores[f"{tf_label}_stoch_rsi_cross"] = WEIGHTS["stoch_rsi"] * 0.5
                    elif stoch_rsi.get('cross') == 'bearish_cross':
                        score -= WEIGHTS["stoch_rsi"] * 0.5
                        indicator_scores[f"{tf_label}_stoch_rsi_cross"] = -WEIGHTS["stoch_rsi"] * 0.5
                    used_indicators.add("stoch_rsi")

                ema = detect_ema_crossover(candles)
                trend = calculate_supertrend_signal(candles)
                bb = calculate_bollinger_bands(candles)

                if ema == "bullish":
                    score += WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = WEIGHTS["ema"]
                elif ema == "bearish":
                    score -= WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = -WEIGHTS["ema"]

                # EMA Ribbon
                ribbon = calculate_ema_ribbon(candles)
                ribbon_analysis = analyze_ema_ribbon(ribbon)
                if ribbon_analysis['trend'] == 'bullish':
                    score += WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                elif ribbon_analysis['trend'] == 'bearish':
                    score -= WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = -WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                used_indicators.add("ema_ribbon")

                # EMA Squeeze
                ema_squeeze = detect_ema_squeeze(ribbon)
                if ema_squeeze['squeezing']:
                    score += WEIGHTS["ema_squeeze"] * ema_squeeze['intensity']
                    indicator_scores[f"{tf_label}_ema_squeeze"] = WEIGHTS["ema_squeeze"] * ema_squeeze['intensity']
                    used_indicators.add("ema_squeeze")

                if trend == "bullish":
                    score += WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = WEIGHTS["supertrend"]
                elif trend == "bearish":
                    score -= WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = -WEIGHTS["supertrend"]

                # Supertrend State
                st_state = get_supertrend_state(candles)
                if st_state.get('trend'):
                    strength_bonus = WEIGHTS["supertrend"] * st_state.get('strength', 0) * 0.5
                    if st_state['trend'] == 'up':
                        score += strength_bonus
                        indicator_scores[f"{tf_label}_supertrend_strength"] = strength_bonus
                    else:
                        score -= strength_bonus
                        indicator_scores[f"{tf_label}_supertrend_strength"] = -strength_bonus
                    used_indicators.add("supertrend_strength")

                # Bollinger
                if bb and bb[-1]:
                    close = float(candles[-1]["close"])
                    if close < bb[-1]["lower"]:
                        score += WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = WEIGHTS["bollinger"]
                    elif close > bb[-1]["upper"]:
                        score -= WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = -WEIGHTS["bollinger"]
                    # bollinger_squeeze weight is 0 — kept for structural clarity
                    if bb[-1].get('squeeze'):
                        score += WEIGHTS["bollinger_squeeze"]
                        indicator_scores[f"{tf_label}_bollinger_squeeze"] = WEIGHTS["bollinger_squeeze"]
                        used_indicators.add("bollinger_squeeze")

                # Band walk
                band_walk = detect_band_walk(candles, bb)
                if band_walk:
                    if band_walk.get('walking_upper'):
                        score += WEIGHTS["band_walk"] * band_walk.get('strength', 0)
                        indicator_scores[f"{tf_label}_band_walk"] = WEIGHTS["band_walk"] * band_walk.get('strength', 0)
                    elif band_walk.get('walking_lower'):
                        score -= WEIGHTS["band_walk"] * band_walk.get('strength', 0)
                        indicator_scores[f"{tf_label}_band_walk"] = -WEIGHTS["band_walk"] * band_walk.get('strength', 0)
                    used_indicators.add("band_walk")

                # Bollinger Signal
                bb_signal = get_bollinger_signal(candles)
                if bb_signal['signal'] in ['squeeze_breakout_up', 'strong_bullish']:
                    score += WEIGHTS["bollinger"] * bb_signal['strength']
                    indicator_scores[f"{tf_label}_bollinger_signal"] = WEIGHTS["bollinger"] * bb_signal['strength']
                elif bb_signal['signal'] in ['squeeze_breakout_down', 'strong_bearish']:
                    score -= WEIGHTS["bollinger"] * bb_signal['strength']
                    indicator_scores[f"{tf_label}_bollinger_signal"] = -WEIGHTS["bollinger"] * bb_signal['strength']
                used_indicators.add("bollinger_signal")

                # Enhanced pattern detection
                score, indicator_scores, used_indicators = enhanced_pattern_scoring(
                    candles, tf_label, score, indicator_scores, used_indicators
                )

                if detect_whale_activity(candles):
                    score += WEIGHTS["whale"]
                    indicator_scores[f"{tf_label}_whale"] = WEIGHTS["whale"]

                type_scores["Swing"] += score
                tf_count["Swing"] += 1
                used_indicators.update(["rsi", "ema", "supertrend", "bollinger", "whale"])

        except Exception as e:
            log(f"❌ Scoring error for {symbol} [{tf}m]: {str(e)}", level="ERROR")

        # Apply rebalancing
        indicator_scores = rebalance_indicator_scores(indicator_scores, market_context)
        tf_scores[tf] = round(score, 2)

    # ── Apply timeframe bonuses ──────────────────────────────────────────────
    # REVERTED: prior rebalance (Scalp 1.25x/1.10x, Intraday 1.08x) made
    # Scalp dominant — 244 of 265 trades over 30 days were Scalp at PF 0.71,
    # losing $94 of $1000. The Scalp tier loses money on these weights;
    # restoring the original tier-bonus values so Intraday wins the max()
    # selection by default (and Scalp tier mostly stays dormant).
    if type_scores["Scalp"] > 0 and tf_count["Scalp"] >= 2:
        type_scores["Scalp"] *= 1.2
    if type_scores["Intraday"] > 0 and tf_count["Intraday"] >= 2:
        type_scores["Intraday"] *= 1.15

    # ── Find best trade type ─────────────────────────────────────────────────
    valid_types = [t for t in type_scores if tf_count[t] >= MIN_TF_REQUIRED[t]]

    if not valid_types:
        if '1' in candles_by_timeframe and '3' in candles_by_timeframe:
            best_type = "Scalp"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Scalp")
        elif '5' in candles_by_timeframe and '15' in candles_by_timeframe:
            best_type = "Intraday"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Intraday")
        else:
            best_type = "Intraday"
            log(f"ℹ️ {symbol}: No valid trade types, using Intraday as default")
        best_score = type_scores[best_type]
    else:
        best_type = max(valid_types, key=lambda t: type_scores[t])
        best_score = type_scores[best_type]

    # Swing momentum check
    if best_type == "Swing":
        has_swing_momentum, _, strength = safe_detect_momentum_strength(
            candles_by_timeframe.get("60", [])
        )
        if not has_swing_momentum or strength < 0.3:
            log(f"⚠️ {symbol} Swing trade allowed with moderate momentum (strength={strength:.2f})")
            best_score *= 0.8

    # ── MTF Bonuses ──────────────────────────────────────────────────────────
    if mtf_supertrend['alignment'] > 0.7:
        mtf_bonus = WEIGHTS["supertrend_mtf"] * mtf_supertrend['alignment']
        if mtf_supertrend['overall_trend'] == 'up':
            best_score += mtf_bonus
            indicator_scores["mtf_supertrend"] = mtf_bonus
        else:
            best_score -= mtf_bonus
            indicator_scores["mtf_supertrend"] = -mtf_bonus
        used_indicators.add("mtf_supertrend")

    if mtf_rsi.get('overall_signal') == 'bullish' and mtf_rsi.get('confluence_strength', 0) > 0.6:
        best_score += WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")
    elif mtf_rsi.get('overall_signal') == 'bearish' and mtf_rsi.get('confluence_strength', 0) > 0.6:
        best_score -= WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = -WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")

    # ── 4H Trend Gate (additive bonus) ──────────────────────────────────────
    gate_adj, gate_reason = get_4h_trend_gate(candles_by_timeframe)
    if gate_adj != 0:
        best_score += gate_adj
        indicator_scores["4h_gate"] = gate_adj
        log(f"📊 4H gate for {symbol}: {gate_adj:+.2f} ({gate_reason})")

    # Momentum bonus — only fire when momentum is real AND aligned with TF consensus.
    # Previous version had broken indentation that nested this inside the 4h block
    # and dereferenced `expected_direction` before it was bound (NameError when
    # has_momentum was False). Fixed: bind & check in one guarded block.
    if has_momentum and best_score > 6.0 and momentum_direction:
        direction = determine_direction(tf_scores)
        if direction is not None:
            expected_direction = "bullish" if direction == "Long" else "bearish"
            if momentum_direction == expected_direction:
                bonus = 0.8
                best_score += bonus
                log(f"🚀 Momentum bonus applied to {symbol}: +{bonus} (aligned {momentum_direction})")

    return round(best_score, 2), tf_scores, best_type, indicator_scores, list(used_indicators)


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED SCORE — adds quality gates on top of score_symbol
# ─────────────────────────────────────────────────────────────────────────────

def enhanced_score_symbol(symbol, candles_by_timeframe, market_context=None):
    """Enhanced scoring with all quality validations"""

    # 1. Get base score
    original_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
        symbol, candles_by_timeframe, market_context
    )

    # Skip enhanced checks if score is too low
    if original_score < 5:
        return original_score, tf_scores, trade_type, indicator_scores, used_indicators

    # 2. Determine direction (now stricter)
    direction = determine_direction(tf_scores)

        # CRITICAL FIX: Strong downtrend veto
    gate_adj, gate_reason = get_4h_trend_gate(candles_by_timeframe)
    if direction == "Long" and "4h_mom_neg" in gate_reason and gate_adj < -0.5:
        log(f"❌ {symbol}: Strong 4h downtrend → forcing Short/0 (was Long)")
        return -5.0, tf_scores, trade_type, indicator_scores, list(used_indicators)

    # QUALITY GATE 1: No clean direction → reject
    if direction is None:
        log(f"⏭️ {symbol}: No clean direction (TFs disagree) — skipped")
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)

    # QUALITY GATE 2: 4h direction veto
    allow_4h, reason_4h = check_4h_alignment_veto(candles_by_timeframe, direction)
    if not allow_4h:
        log(f"❌ {symbol}: {reason_4h}")
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)

    # QUALITY GATE 3: Anti-chase
    allow_chase, reason_chase = check_not_chasing(candles_by_timeframe, direction, trade_type)
    if not allow_chase:
        log(f"❌ {symbol}: {reason_chase}")
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)

    # 3. Get current price
    try:
        current_price = float(candles_by_timeframe['1'][-1]['close'])
    except (KeyError, IndexError):
        current_price = 0
        for tf in ['5', '15', '30']:
            if tf in candles_by_timeframe and candles_by_timeframe[tf]:
                current_price = float(candles_by_timeframe[tf][-1]['close'])
                break
        if current_price == 0:
            return original_score, tf_scores, trade_type, indicator_scores, used_indicators

    # 4. Validate entry timing
    entry_valid, entry_reason = entry_validator.validate_entry(
        symbol, candles_by_timeframe, direction, current_price, trade_type, original_score
    )

    if not entry_valid:
        log(f"❌ Entry validation failed for {symbol}: {entry_reason}")
        original_score *= 0.7
        indicator_scores["entry_validation_failed"] = -3.0
        if isinstance(used_indicators, list):
            used_indicators.append("entry_validation_failed")
        else:
            used_indicators = list(used_indicators)
            used_indicators.append("entry_validation_failed")
        return original_score, tf_scores, trade_type, indicator_scores, used_indicators

    # 5. Pattern context
    pattern_context = None
    try:
        pattern_context = pattern_context_analyzer.analyze_pattern_context(
            candles_by_timeframe, direction, trade_type
        )
        if pattern_context and pattern_context.get('score_adjustment'):
            original_score += pattern_context['score_adjustment']
            indicator_scores["pattern_context"] = pattern_context['score_adjustment']
    except Exception as e:
        log(f"⚠️ Pattern context analysis failed: {e}", level="WARN")

    # 6. Divergences
    divergences_found = []
    try:
        divergences = divergence_detector.detect_all_divergences(candles_by_timeframe, direction)
        if divergences:
            divergences_found = divergences
            div_score = sum(d.get('strength', 0) * 0.5 for d in divergences)
            original_score += div_score
            indicator_scores["divergences"] = div_score
    except Exception as e:
        log(f"⚠️ Divergence detection failed: {e}", level="WARN")

    # 7. Entry quality micro-bonuses
    entry_quality_score = 0
    try:
        if entry_validator.check_momentum_alignment(candles_by_timeframe, direction, trade_type)[0]:
            entry_quality_score += 0.3
    except Exception:
        pass
    try:
        if entry_validator.check_timeframe_alignment(candles_by_timeframe, direction, trade_type)[0]:
            entry_quality_score += 0.3
    except Exception:
        pass
    try:
        if entry_validator.check_market_structure(candles_by_timeframe, trade_type)[0]:
            entry_quality_score += 0.2
    except Exception:
        pass

    original_score += entry_quality_score
    indicator_scores["entry_quality"] = entry_quality_score

    # QUALITY GATE 4: Strong-indicator gate (tightened — must have ≥ 2 aligned)
    passed, original_score = _apply_strong_indicator_gate(
        symbol, original_score, indicator_scores, tf_scores
    )
    if not passed:
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)

    # Log analysis
    log(f"📊 Enhanced scoring for {symbol}:")
    log(f"   Final score: {original_score:.2f} | Direction: {direction}")
    log(f"   Entry validation: {entry_reason}")
    if pattern_context:
        log(f"   Pattern context: {pattern_context.get('location', 'N/A')} - "
            f"{pattern_context.get('trend_before', 'N/A')}")
    if divergences_found:
        log(f"   Divergences: {[d['type'] + ' ' + d['indicator'] for d in divergences_found]}")

    return original_score, tf_scores, trade_type, indicator_scores, list(used_indicators)


# ─────────────────────────────────────────────────────────────────────────────
# ML-BASED SCORER (Phase 7 Turn 3)
# ─────────────────────────────────────────────────────────────────────────────
#
# Drop-in replacement for enhanced_score_symbol that uses a trained
# logistic-regression model instead of the hand-tuned WEIGHTS dict.
#
# The model was trained on labeled_dataset_30d.csv with:
#   * label = 1 iff a >= 2% move happens within next 20 × 5m bars (100 min)
#   * features = ~436 indicator booleans/floats from analyze_tf on 5m/15m/60m
#   * temporal train/test split (no shuffling)
#
# Phase 7 Turn 2 result on this dataset:
#   AUC = 0.612  (genuine predictive signal, above the 0.6 audit threshold)
#
# How to enable:
#   Set the module-level _ML_MODEL via load_ml_model() once at startup,
#   then call score_symbol_ml() instead of score_symbol/enhanced_score_symbol.
#   main.py can swap based on a config flag.
#
# Returns the SAME 5-tuple shape as score_symbol so it's drop-in compatible:
#   (score: float, tf_scores: dict, trade_type: str, indicator_scores: dict, used_indicators: list)
#
# Score scaling: model.predict_proba returns 0-1 → multiplied by 100 so the
# downstream gates (MIN_INTRADAY_SCORE = 12.0) need to be re-tuned for the
# ML model. Suggested gate for AUC 0.612 model: ~85 (top ~15% by probability).

_ML_MODEL = None
_ML_FEATURE_COLUMNS = None
_ML_THRESHOLD = 0.5


def load_ml_model(
    model_path: str = "score_model.pkl",
    columns_path: str = "score_model_columns.json",
):
    """Load the trained model + feature list once. Idempotent."""
    global _ML_MODEL, _ML_FEATURE_COLUMNS, _ML_THRESHOLD
    if _ML_MODEL is not None:
        return _ML_MODEL  # already loaded
    try:
        import joblib
        import json as _json
        _ML_MODEL = joblib.load(model_path)
        with open(columns_path, "r") as f:
            data = _json.load(f)
        _ML_FEATURE_COLUMNS = data.get("feature_columns", [])
        _ML_THRESHOLD = float(data.get("chosen_threshold", 0.5))
        log(f"✅ ML scorer loaded: {len(_ML_FEATURE_COLUMNS)} features, "
            f"threshold={_ML_THRESHOLD:.4f}")
    except Exception as e:
        log(f"❌ Failed to load ML model: {e}", level="ERROR")
        _ML_MODEL = None
        _ML_FEATURE_COLUMNS = None
    return _ML_MODEL


def score_symbol_ml(symbol, candles_by_timeframe, market_context=None):
    """Drop-in replacement for enhanced_score_symbol using the trained model.

    Returns the same 5-tuple shape:
      (score, tf_scores, trade_type, indicator_scores, used_indicators)

    Score is the model's predicted probability × 100 (range 0-100). The
    downstream gate in main.py / BacktestEngine should be set around 50-85
    depending on desired precision/recall trade-off.

    If the model isn't loaded, falls back to enhanced_score_symbol.
    """
    if _ML_MODEL is None:
        # Attempt to load on first call; if still fails, fall back.
        load_ml_model()
        if _ML_MODEL is None:
            return enhanced_score_symbol(symbol, candles_by_timeframe, market_context)

    try:
        from labeled_dataset_builder import extract_features_for_inference
        features = extract_features_for_inference(candles_by_timeframe)
    except Exception as e:
        log(f"⚠️ {symbol}: ML feature extraction failed ({e}), falling back", level="WARN")
        return enhanced_score_symbol(symbol, candles_by_timeframe, market_context)

    if not features:
        return 0.0, {}, "Intraday", {}, []

    # Build a feature vector in the exact column order the model expects.
    # Missing features → 0. Bool → int.
    import numpy as np
    try:
        row = []
        for col in _ML_FEATURE_COLUMNS:
            v = features.get(col, 0)
            if isinstance(v, bool):
                v = int(v)
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(0.0)
        X = np.array([row], dtype=float)
        proba = float(_ML_MODEL.predict_proba(X)[0, 1])
    except Exception as e:
        log(f"⚠️ {symbol}: ML inference failed ({e}), falling back", level="WARN")
        return enhanced_score_symbol(symbol, candles_by_timeframe, market_context)

    # Score = proba × 100 for compatibility with existing gate scale.
    score = round(proba * 100.0, 2)

    # For trade_type / direction, fall back to score.py's existing logic by
    # examining the TF closes. The model itself only predicts "move imminent
    # in either direction" — we use the recent EMA slope to pick a side.
    tf_scores: dict = {}
    direction_hint = "Long"
    try:
        for tf in ("1", "3", "5", "15"):
            bars = candles_by_timeframe.get(tf, [])
            if len(bars) < 21:
                continue
            closes = [float(c["close"]) for c in bars[-21:]]
            ema9 = sum(closes[-9:]) / 9
            ema21 = sum(closes) / 21
            tf_scores[tf] = round((ema9 - ema21) / ema21 * 100, 3) if ema21 > 0 else 0.0
        # Direction from EMA slope on primary 5m bar
        if "5" in tf_scores and tf_scores["5"] < 0:
            direction_hint = "Short"
    except Exception:
        pass

    # We expose the probability as the only "indicator score" — it's a
    # single composite. trade_type defaults to Intraday (what the model
    # was trained for).
    indicator_scores = {"ml_proba": proba, "ml_direction_hint": direction_hint}
    used_indicators = ["ml_model"]

    return score, tf_scores, "Intraday", indicator_scores, used_indicators


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    'score_symbol',
    'enhanced_score_symbol',
    'score_symbol_ml',
    'load_ml_model',
    'determine_direction',
    'determine_direction_with_pattern_priority',
    'calculate_confidence',
    'has_pump_potential',
    'detect_momentum_strength',
    'safe_detect_momentum_strength',
    'enhanced_pattern_scoring',
    'get_minimum_volume_threshold',
    'check_volume_quality',
    'calculate_volume_penalty',
    'get_4h_trend_gate',
    'check_4h_alignment_veto',
    'check_not_chasing',
    'WEIGHTS',
    'TRADE_TYPE_TF',
    'MIN_TF_REQUIRED',
    'VOLUME_SPIKE_THRESHOLD',
]
