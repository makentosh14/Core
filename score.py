# score.py - Enhanced with Advanced Pattern Detection Integration
# FIXED VERSION - Backscan-optimized weights (9,343 events analyzed Feb 2026)
# Changes vs previous version:
#   - macd weight: 1.2 → 0.6 (MACD cross fires only 3-4% before pumps)
#   - macd_momentum: 0.8 → 1.2 (MACD hist direction has +7.4% pump-diff)
#   - ema_ribbon: 0.8 → 1.0 (ribbon trend more stable signal)
#   - volume_spike: 1.0 → 0.7 (10% hit rate, threshold too strict)
#   - supertrend: 1.0 → 1.1 (ST +6.4% diff, slightly underweighted)
#   - supertrend_mtf: 1.0 → 1.3 (4h MTF alignment is THE most predictive factor)
#   - rsi: 0.8 → 0.7 (RSI near 50 at pump start)
#   - rsi_mtf: 0.9 → 1.1 (MTF RSI confluence +5-6% diff)
#   - bollinger_squeeze: 0.8 → 0.0 (REMOVED - anti-correlated with longs!)
#   - vwap: 0.6 → 0.8 (above_vwap +6.3% diff)
#   - whale: 1.0 → 0.8 (normalize)
#   - whale_advanced: 1.1 → 1.0 (normalize)
#   - Volume spike threshold: 2.5x → 1.8x (avg vol_ratio at pump = 2.09x)
#   - Added get_4h_trend_gate() — 4h momentum gate applied before final return
#   - Replaced hard strong_count < 2 rejection with soft 30% penalty

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

# ── WEIGHTS (backscan-optimized) ─────────────────────────────────────────────
WEIGHTS = {
    "macd":               0.6,   # was 1.2 — cross fires only 3-4% before pumps
    "macd_divergence":    1.0,
    "macd_momentum":      1.2,   # was 0.8 — hist direction +7.4% diff, best MACD signal
    "ema":                0.9,   # was 1.0
    "ema_ribbon":         1.0,   # was 0.8
    "ema_squeeze":        0.6,
    "volume_spike":       0.7,   # was 1.0 — 10% hit rate before pumps
    "volume_climax":      1.1,
    "volume_profile":     0.5,
    "vwap":               0.8,   # was 0.6 — above_vwap +6.3% diff
    "supertrend":         1.1,   # was 1.0 — ST +6.4% diff
    "supertrend_squeeze": 0.7,
    "supertrend_mtf":     1.3,   # was 1.0 — 4h MTF alignment is most predictive
    "rsi":                0.7,   # was 0.8 — RSI near 50 at pump start
    "rsi_divergence":     1.0,
    "stoch_rsi":          0.8,
    "rsi_mtf":            1.1,   # was 0.9 — MTF RSI confluence +5-6% diff
    "bollinger":          0.6,
    "bollinger_squeeze":  0.0,   # was 0.8 — REMOVED: anti-correlated with longs (-2.1%)
    "band_walk":          0.9,
    "pattern":            0.8,
    "pattern_cluster":    0.4,
    "pattern_quality":    0.6,
    "pattern_confluence": 0.5,
    "divergence":         0.6,
    "slow_breakout":      0.8,
    "whale":              0.8,   # was 1.0
    "whale_advanced":     1.0,   # was 1.1
    "momentum":           1.2,
    "stealth":            0.8,
    "strong_stealth":     1.0,
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

# Volume spike threshold — lowered from 2.5x because avg vol_ratio at pump events = 2.09x
VOLUME_SPIKE_THRESHOLD = 1.8


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


def get_4h_trend_gate(candles_by_timeframe):
    """
    Returns a score adjustment based on 4h momentum indicators.
    4h is the most predictive timeframe per backscan (9,343 events):
      4h_mom_pos   → +8.2% pump-diff (highest single indicator)
      4h_macd_hist → +7.4% pump-diff
      4h_st_bull   → +6.4% pump-diff

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


def enhanced_pattern_scoring(candles, tf_label, score, indicator_scores, used_indicators):
    """Enhanced pattern scoring with advanced pattern detection"""

    pattern_score_total = 0

    # Detect primary pattern
    pattern = detect_pattern(candles)

    if pattern:
        pattern_strength = analyze_pattern_strength(pattern, candles)
        pattern_direction = get_pattern_direction(pattern)
        base_pattern_score = WEIGHTS.get("pattern", 0.8)

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

        pattern_cluster = detect_pattern_cluster(candles, lookback=10)
        if len(pattern_cluster) >= 2:
            cluster_bonus = WEIGHTS["pattern_cluster"] * len(pattern_cluster)
            score += cluster_bonus
            indicator_scores[f"{tf_label}_pattern_cluster"] = cluster_bonus
            used_indicators.add("pattern_cluster")
            cluster_patterns = [p['pattern'] for p in pattern_cluster]
            log(f"📊 Pattern cluster detected on {tf_label}: {cluster_patterns}")

    # Scan for all patterns for comprehensive analysis
    all_patterns = get_all_patterns(candles)
    pattern_count = sum(1 for detected in all_patterns.values() if detected)

    if pattern_count >= 3:
        confluence_bonus = WEIGHTS["pattern_confluence"]
        score += confluence_bonus
        indicator_scores[f"{tf_label}_pattern_confluence"] = confluence_bonus
        used_indicators.add("pattern_confluence")
        detected_patterns = [name for name, detected in all_patterns.items() if detected]
        log(f"📊 Pattern confluence on {tf_label}: {detected_patterns}")

    # Apply cap to total pattern contribution
    capped_pattern_score = min(pattern_score_total, MAX_PATTERN_CONTRIBUTION)
    score += capped_pattern_score

    return score, indicator_scores, used_indicators


def determine_direction(tf_scores):
    """Determine trading direction from timeframe scores"""
    if not tf_scores:
        return None

    total_score = sum(tf_scores.values())
    if total_score > 1.0:
        return "Long"
    elif total_score < -1.0:
        return "Short"
    return None


def determine_direction_with_pattern_priority(tf_scores, candles_by_timeframe):
    """Determine direction with pattern taking priority"""
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


def calculate_confidence(score, tf_scores, market_context, trade_type):
    """Calculate confidence percentage for a trade signal"""
    if not tf_scores:
        return 0

    base_confidence = min(score * 5, 70)
    tf_alignment = sum(1 for v in tf_scores.values() if v > 0) / max(len(tf_scores), 1)
    alignment_bonus = tf_alignment * 20

    market_bonus = 0
    if market_context:
        btc_trend = market_context.get("btc_trend", "neutral")
        if btc_trend == "bullish":
            market_bonus = 5
        elif btc_trend == "bearish":
            market_bonus = -5

    confidence = base_confidence + alignment_bonus + market_bonus
    return min(int(confidence), 100)


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
            if whale_advanced['detected']:
                strength = whale_advanced['strength']
                if whale_advanced['recommendation'] == 'potential_long':
                    score += WEIGHTS["whale_advanced"] * strength
                    indicator_scores[f"{tf_label}_whale_advanced"] = WEIGHTS["whale_advanced"] * strength
                elif whale_advanced['recommendation'] == 'potential_short':
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

                if ema == "bullish":
                    score += WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = WEIGHTS["ema"]
                elif ema == "bearish":
                    score -= WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = -WEIGHTS["ema"]

                # Enhanced RSI Analysis
                rsi_data = calculate_rsi_with_bands(candles)
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
                    used_indicators.add("rsi")

                # Volume spike — threshold lowered to 1.8x (was 2.5x)
                if is_volume_spike(candles, VOLUME_SPIKE_THRESHOLD):
                    score += WEIGHTS["volume_spike"]
                    indicator_scores[f"{tf_label}_volume"] = WEIGHTS["volume_spike"]
                    used_indicators.add("volume_spike")

                # Stealth Accumulation
                stealth_result = detect_stealth_accumulation_advanced(candles, symbol)
                if stealth_result['detected']:
                    stealth_score = WEIGHTS["stealth"] * stealth_result['strength']
                    score += stealth_score
                    indicator_scores[f"{tf_label}_stealth"] = stealth_score
                    if stealth_result['patterns']:
                        log(f"🕵️ Stealth patterns on {symbol}: {', '.join(stealth_result['patterns'])}")
                    if stealth_result['recommendation'] == 'strong_accumulation':
                        score += 0.5
                        indicator_scores[f"{tf_label}_strong_stealth"] = 0.5

                if detect_volume_divergence(candles):
                    score += WEIGHTS["divergence"]
                    indicator_scores[f"{tf_label}_divergence"] = WEIGHTS["divergence"]

                if detect_slow_breakout(candles):
                    score += WEIGHTS["slow_breakout"]
                    indicator_scores[f"{tf_label}_slow_breakout"] = WEIGHTS["slow_breakout"]

                if detect_whale_activity(candles):
                    score += WEIGHTS["whale"]
                    indicator_scores[f"{tf_label}_whale"] = WEIGHTS["whale"]

                type_scores["Scalp"] += score
                tf_count["Scalp"] += 1
                used_indicators.update(["macd", "ema", "volume", "divergence", "slow_breakout", "whale"])

            elif tf in TRADE_TYPE_TF["Intraday"]:
                # INTRADAY INDICATORS (5m, 15m)
                macd = detect_macd_cross(candles)
                ema = detect_ema_crossover(candles)
                trend = calculate_supertrend_signal(candles)

                # Volume spike — threshold lowered to 1.8x
                if is_volume_spike(candles, VOLUME_SPIKE_THRESHOLD):
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

                # MACD Momentum (histogram direction — most predictive MACD signal)
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

                # Supertrend Squeeze
                st_squeeze = detect_supertrend_squeeze(candles)
                if st_squeeze['squeeze']:
                    score += WEIGHTS["supertrend_squeeze"] * st_squeeze['intensity']
                    indicator_scores[f"{tf_label}_supertrend_squeeze"] = WEIGHTS["supertrend_squeeze"] * st_squeeze['intensity']
                    used_indicators.add("supertrend_squeeze")

                # Enhanced Pattern Detection
                score, indicator_scores, used_indicators = enhanced_pattern_scoring(
                    candles, tf_label, score, indicator_scores, used_indicators
                )

                if detect_volume_divergence(candles):
                    score += WEIGHTS["divergence"]
                    indicator_scores[f"{tf_label}_divergence"] = WEIGHTS["divergence"]

                if detect_slow_breakout(candles):
                    score += WEIGHTS["slow_breakout"]
                    indicator_scores[f"{tf_label}_slow_breakout"] = WEIGHTS["slow_breakout"]

                if detect_whale_activity(candles):
                    score += WEIGHTS["whale"]
                    indicator_scores[f"{tf_label}_whale"] = WEIGHTS["whale"]

                type_scores["Intraday"] += score
                tf_count["Intraday"] += 1
                used_indicators.update(["macd", "ema", "supertrend", "volume", "divergence", "slow_breakout", "whale"])

            elif tf in TRADE_TYPE_TF["Swing"]:
                # SWING INDICATORS (30m, 60m, 240m)

                # Enhanced RSI Analysis
                rsi_data = calculate_rsi_with_bands(candles)
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

                    # RSI Momentum
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

                # Supertrend State Analysis
                st_state = get_supertrend_state(candles)
                if st_state['trend']:
                    strength_bonus = WEIGHTS["supertrend"] * st_state['strength'] * 0.5
                    if st_state['trend'] == 'up':
                        score += strength_bonus
                        indicator_scores[f"{tf_label}_supertrend_strength"] = strength_bonus
                    else:
                        score -= strength_bonus
                        indicator_scores[f"{tf_label}_supertrend_strength"] = -strength_bonus
                    used_indicators.add("supertrend_strength")

                # Bollinger Bands (price vs bands — NOT squeeze for longs)
                if bb and bb[-1]:
                    close = float(candles[-1]["close"])
                    if close < bb[-1]["lower"]:
                        score += WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = WEIGHTS["bollinger"]
                    elif close > bb[-1]["upper"]:
                        score -= WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = -WEIGHTS["bollinger"]

                    # NOTE: bollinger_squeeze weight is 0.0 (removed) — backscan shows
                    # BB squeeze is anti-correlated with longs (-2.1% diff).
                    # The if-block below will add 0.0 so it has no effect, but kept
                    # for structural clarity if weight is ever re-evaluated.
                    if bb[-1].get('squeeze'):
                        score += WEIGHTS["bollinger_squeeze"]   # = 0.0
                        indicator_scores[f"{tf_label}_bollinger_squeeze"] = WEIGHTS["bollinger_squeeze"]
                        used_indicators.add("bollinger_squeeze")

                # Band Walk
                band_walk = detect_band_walk(candles, bb)
                if band_walk:
                    if band_walk['walking_upper']:
                        score += WEIGHTS["band_walk"] * band_walk['strength']
                        indicator_scores[f"{tf_label}_band_walk"] = WEIGHTS["band_walk"] * band_walk['strength']
                    elif band_walk['walking_lower']:
                        score -= WEIGHTS["band_walk"] * band_walk['strength']
                        indicator_scores[f"{tf_label}_band_walk"] = -WEIGHTS["band_walk"] * band_walk['strength']
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

                # Enhanced Pattern Detection
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

        # Apply indicator rebalancing
        indicator_scores = rebalance_indicator_scores(indicator_scores, market_context)
        tf_scores[tf] = round(score, 2)

    # ── Timeframe bonuses ────────────────────────────────────────────────────
    if type_scores["Scalp"] > 0 and tf_count["Scalp"] >= 2:
        type_scores["Scalp"] *= 1.2

    if type_scores["Intraday"] > 0 and tf_count["Intraday"] >= 2:
        type_scores["Intraday"] *= 1.15

    # Find best trade type
    valid_types = [t for t in type_scores if tf_count[t] >= MIN_TF_REQUIRED[t]]

    if not valid_types:
        if '1' in candles_by_timeframe and '3' in candles_by_timeframe:
            best_type = "Scalp"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Scalp based on available TFs")
        elif '5' in candles_by_timeframe and '15' in candles_by_timeframe:
            best_type = "Intraday"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Intraday based on available TFs")
        else:
            best_type = "Intraday"
            log(f"ℹ️ {symbol}: No valid trade types, using Intraday as default")
        best_score = type_scores[best_type]
    else:
        best_type = max(valid_types, key=lambda t: type_scores[t])
        best_score = type_scores[best_type]

    # Swing momentum check
    if best_type == "Swing":
        has_swing_momentum, direction, strength = safe_detect_momentum_strength(
            candles_by_timeframe.get("60", [])
        )
        if not has_swing_momentum or strength < 0.3:
            log(f"⚠️ {symbol} Swing trade allowed with moderate momentum (strength={strength:.2f})")
            best_score *= 0.8

    # ── MTF Bonuses ──────────────────────────────────────────────────────────

    # Supertrend MTF Alignment
    if mtf_supertrend['alignment'] > 0.7:
        mtf_bonus = WEIGHTS["supertrend_mtf"] * mtf_supertrend['alignment']
        if mtf_supertrend['overall_trend'] == 'up':
            best_score += mtf_bonus
            indicator_scores["mtf_supertrend"] = mtf_bonus
        else:
            best_score -= mtf_bonus
            indicator_scores["mtf_supertrend"] = -mtf_bonus
        used_indicators.add("mtf_supertrend")

    # RSI MTF Confluence
    if mtf_rsi.get('overall_signal') == 'bullish' and mtf_rsi.get('confluence_strength', 0) > 0.6:
        best_score += WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")
    elif mtf_rsi.get('overall_signal') == 'bearish' and mtf_rsi.get('confluence_strength', 0) > 0.6:
        best_score -= WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = -WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")

    # ── 4H Trend Gate (backscan: 4h is THE most predictive TF, +7-8% diff) ──
    gate_adj, gate_reason = get_4h_trend_gate(candles_by_timeframe)
    if gate_adj != 0:
        best_score += gate_adj
        indicator_scores["4h_gate"] = gate_adj
        log(f"📊 4H gate for {symbol}: {gate_adj:+.2f} ({gate_reason})")

    # Momentum bonus
    if has_momentum and best_score > 6.0 and momentum_direction:
        expected_direction = "bullish" if determine_direction(tf_scores) == "Long" else "bearish"
        if momentum_direction == expected_direction:
            bonus = 0.8
            best_score += bonus
            log(f"🚀 Momentum bonus applied to {symbol}: +{bonus} (aligned {momentum_direction})")

    return round(best_score, 2), tf_scores, best_type, indicator_scores, list(used_indicators)


def enhanced_score_symbol(symbol, candles_by_timeframe, market_context=None):
    """Enhanced scoring with all the new validations"""

    # First get the original score
    original_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
        symbol, candles_by_timeframe, market_context
    )

    # Skip enhanced checks if score is too low
    if original_score < 5:
        return original_score, tf_scores, trade_type, indicator_scores, used_indicators

    # Get current price and direction
    try:
        current_price = float(candles_by_timeframe['1'][-1]['close'])
    except (KeyError, IndexError):
        for tf in ['5', '15', '30']:
            if tf in candles_by_timeframe and candles_by_timeframe[tf]:
                current_price = float(candles_by_timeframe[tf][-1]['close'])
                break
        else:
            return original_score, tf_scores, trade_type, indicator_scores, used_indicators

    direction = determine_direction(tf_scores)

    # 1. Validate entry timing
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

    # 2. Analyze pattern context
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

    # 3. Check for divergences
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

    # 4. Apply final score adjustments based on entry quality
    entry_quality_score = 0

    try:
        momentum_check = entry_validator.check_momentum_alignment(
            candles_by_timeframe, direction, trade_type
        )
        if momentum_check[0]:
            entry_quality_score += 0.3
    except Exception:
        pass

    try:
        tf_check = entry_validator.check_timeframe_alignment(
            candles_by_timeframe, direction, trade_type
        )
        if tf_check[0]:
            entry_quality_score += 0.3
    except Exception:
        pass

    try:
        structure_check = entry_validator.check_market_structure(
            candles_by_timeframe, trade_type
        )
        if structure_check[0]:
            entry_quality_score += 0.2
    except Exception:
        pass

    original_score += entry_quality_score
    indicator_scores["entry_quality"] = entry_quality_score

    # ── Strong indicator count check (soft penalty instead of hard rejection) ──
    # Previously: hard reject if strong_count < 2
    # Now: soft 30% penalty for 1 strong indicator, hard reject only if 0
    # Reason: MACD cross fires 3.4%, vol spike 10% — hard to get 2 strong indicators
    strong_count = sum(1 for k, v in indicator_scores.items() if abs(v) >= 0.8)
    if strong_count == 0:
        log(f"⚠️ {symbol}: Rejected — zero strong indicators")
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)
    elif strong_count == 1:
        log(f"⚠️ {symbol}: Only 1 strong indicator — applying 30% penalty (score {original_score:.2f} → {original_score * 0.7:.2f})")
        original_score = round(original_score * 0.7, 2)
    # 2+ strong indicators: no change

    # Log enhanced analysis
    log(f"📊 Enhanced scoring for {symbol}:")
    log(f"   Original score: {original_score:.2f}")
    log(f"   Entry validation: {entry_reason}")
    if pattern_context:
        log(f"   Pattern context: {pattern_context.get('location', 'N/A')} - {pattern_context.get('trend_before', 'N/A')}")
    if divergences_found:
        log(f"   Divergences: {[d['type'] + ' ' + d['indicator'] for d in divergences_found]}")

    return original_score, tf_scores, trade_type, indicator_scores, list(used_indicators)


# Export all functions
__all__ = [
    'score_symbol',
    'enhanced_score_symbol',
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
    'WEIGHTS',
    'TRADE_TYPE_TF',
    'MIN_TF_REQUIRED',
    'VOLUME_SPIKE_THRESHOLD',
]
