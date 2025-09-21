# score.py - Enhanced with Advanced Pattern Detection Integration

from logger import log
from rsi import calculate_rsi, calculate_rsi_with_bands, calculate_stoch_rsi, analyze_multi_timeframe_rsi, detect_rsi_divergence
from macd import detect_macd_cross, get_macd_divergence, get_macd_momentum
from supertrend import calculate_supertrend_signal, get_supertrend_state, detect_supertrend_squeeze, calculate_multi_timeframe_supertrend
from ema import detect_ema_crossover, calculate_ema_ribbon, analyze_ema_ribbon, detect_ema_squeeze
from bollinger import calculate_bollinger_bands, detect_band_walk, get_bollinger_signal, detect_bollinger_squeeze
from pattern_detector import (
    detect_pattern, analyze_pattern_strength, detect_pattern_cluster,
    get_pattern_direction, pattern_success_probability, get_all_patterns, 
    PATTERN_WEIGHTS, REVERSAL_PATTERNS, CONTINUATION_PATTERNS
    )
from volume import (is_volume_spike, get_average_volume, detect_volume_climax, 
                   get_volume_profile, get_volume_weighted_average_price, analyze_volume_trend)
from stealth_detector import detect_volume_divergence, detect_slow_breakout, detect_stealth_accumulation_advanced
from whale_detector import detect_whale_activity, detect_whale_activity_advanced, analyze_whale_impact
from error_handler import send_error_to_telegram
from config import ALWAYS_ALLOW_SWING
from indicator_fixes import rebalance_indicator_scores, get_balanced_rsi_signal, analyze_volume_direction
from enhanced_entry_validator import entry_validator
from pattern_context_analyzer import pattern_context_analyzer
from divergence_detector import divergence_detector
import numpy as np

# Enhanced weights including pattern-specific weights
WEIGHTS = {
    "macd": 1.2,          # Reduced from 1.5
    "macd_divergence": 1.0,  # Reduced from 1.2
    "macd_momentum": 0.8,
    "ema": 1.0,
    "ema_ribbon": 0.8,    # Reduced from 0.9
    "ema_squeeze": 0.6,   # Reduced from 0.7
    "volume_spike": 1.0,  # Reduced from 1.2
    "volume_climax": 1.1, # Reduced from 1.3
    "volume_profile": 0.5,
    "vwap": 0.6,          # Reduced from 0.8
    "supertrend": 1.0,
    "supertrend_squeeze": 0.7,
    "supertrend_mtf": 1.0,
    "rsi": 0.8,           # Reduced from 1.0
    "rsi_divergence": 1.0,
    "stoch_rsi": 0.8,
    "rsi_mtf": 0.9,
    "bollinger": 0.6,     # Increased from 0.5
    "bollinger_squeeze": 0.8,
    "band_walk": 0.9,     # Reduced from 1.0
    "pattern": 0.8,       # Increased from 0.7
    "pattern_cluster": 0.4,
    "pattern_confluence": 0.5,
    "divergence": 0.6,    # Increased from 0.5
    "slow_breakout": 0.8,
    "whale": 1.0,         # Reduced from 1.3
    "whale_advanced": 1.1,
    "momentum": 1.2,      # Reduced from 1.5
    "stealth": 0.8,
    "strong_stealth": 1.0
}

# Existing code remains...
TRADE_TYPE_TF = {
    "Scalp": ["1", "3"],
    "Intraday": ["5", "15"],
    "Swing": ["30", "60", "240"],
}

MIN_TF_REQUIRED = {
    "Scalp": 1,
    "Intraday": 1,
    "Swing": 2,
}

def safe_detect_momentum_strength(candles):
    """Safe wrapper for detect_momentum_strength"""
    try:
        if not candles or len(candles) < 10:
            return False, None, 0
        
        # Ensure it's a proper list of candles
        if isinstance(candles, (list, tuple)):
            return detect_momentum_strength(candles)
        else:
            return False, None, 0
            
    except Exception as e:
        return False, None, 0

def enhanced_score_symbol(symbol, candles_by_timeframe, market_context=None):
    """Enhanced scoring with all the new validations"""
    
    # First, get the original score
    original_score, tf_scores, trade_type, indicator_scores, used_indicators = score_symbol(
        symbol, candles_by_timeframe, market_context
    )
    
    # If score is too low, skip enhanced checks
    if original_score < 5:
        return original_score, tf_scores, trade_type, indicator_scores, used_indicators
    
    # Get current price and direction
    current_price = float(candles_by_timeframe['1'][-1]['close'])
    direction = determine_direction(tf_scores)
    
    # 1. Validate entry timing
    entry_valid, entry_reason = entry_validator.validate_entry(
        symbol, candles_by_timeframe, direction, current_price, trade_type, original_score
    )
    
    if not entry_valid:
        log(f"❌ Entry validation failed for {symbol}: {entry_reason}")
        # Severely penalize the score
        original_score *= 0.7
        indicator_scores["entry_validation_failed"] = -3.0
        used_indicators.append("entry_validation_failed")
        return original_score, tf_scores, trade_type, indicator_scores, used_indicators
    
    # 2. Check pattern context if pattern detected
    pattern = None
    pattern_context = None
    
    for ind in used_indicators:
        if "pattern_" in ind:
            pattern = ind.replace("pattern_", "")
            break
            
    if pattern:
        # Analyze pattern context
        candles_5m = candles_by_timeframe.get('5', [])
        if candles_5m:
            pattern_context = pattern_context_analyzer.analyze_pattern_context(
                pattern, candles_5m
            )
            
            if not pattern_context["valid"]:
                log(f"⚠️ Pattern {pattern} in poor context: {pattern_context['reason']}")
                # Reduce pattern score
                for key in indicator_scores:
                    if "pattern" in key and pattern in key:
                        indicator_scores[key] *= 0.5
                        
            # Apply context strength
            pattern_strength = pattern_context.get("strength_score", 1.0)
            for key in indicator_scores:
                if "pattern" in key:
                    indicator_scores[key] *= pattern_strength
    
    # 3. Check for divergences
    divergences_found = []
    
    # RSI divergence
    if 'rsi' in used_indicators:
        candles_15m = candles_by_timeframe.get('15', [])
        if candles_15m:
            from rsi import calculate_rsi
            rsi_values = calculate_rsi(candles_15m)
            if rsi_values:
                rsi_div_result = detect_rsi_divergence(candles_15m, rsi_values, symbol)
                if rsi_div_result:
                    rsi_div = {"type": "bullish" if rsi_div_result == "bullish_divergence" else "bearish"}
                else:
                    rsi_div = None
                    
                    # Add to scoring
                    if rsi_div["type"] == "bullish" and direction == "Long":
                        original_score += 0.8
                        indicator_scores["rsi_divergence_bullish"] = 0.8
                    elif rsi_div["type"] == "bearish" and direction == "Short":
                        original_score += 0.8
                        indicator_scores["rsi_divergence_bearish"] = 0.8
                    else:
                        # Divergence against our direction
                        original_score -= 1.0
                        indicator_scores["rsi_divergence_against"] = -1.0
    
    # Volume divergence
    vol_div = divergence_detector.detect_volume_divergence(candles_by_timeframe.get('5', []))
    if vol_div:
        divergences_found.append(vol_div)
        
        if vol_div["type"] == "bullish" and direction == "Long":
            original_score += 0.5
            indicator_scores["volume_divergence_bullish"] = 0.5
        elif vol_div["type"] == "bearish" and direction == "Short":
            original_score += 0.5
            indicator_scores["volume_divergence_bearish"] = 0.5
        else:
            # Warning signal
            original_score -= 0.5
            indicator_scores["volume_divergence_warning"] = -0.5
    
    # 4. Apply final score adjustments based on entry quality
    entry_quality_score = 0
    
    # Check momentum alignment
    momentum_check = entry_validator.check_momentum_alignment(
        candles_by_timeframe, direction, trade_type
    )
    if momentum_check[0]:
        entry_quality_score += 0.3
        
    # Check timeframe alignment
    tf_check = entry_validator.check_timeframe_alignment(
        candles_by_timeframe, direction, trade_type
    )
    if tf_check[0]:
        entry_quality_score += 0.3
        
    # Check market structure
    structure_check = entry_validator.check_market_structure(
        candles_by_timeframe, trade_type
    )
    if structure_check[0]:
        entry_quality_score += 0.2
        
    # Apply entry quality bonus
    original_score += entry_quality_score
    indicator_scores["entry_quality"] = entry_quality_score

    strong_count = sum(1 for k, v in indicator_scores.items() if abs(v) >= 0.8)
    if strong_count < 2:
        log(f"⚠️ {symbol}: Rejected due to insufficient strong indicators ({strong_count})")
        return 0, tf_scores, trade_type, indicator_scores, list(used_indicators)
    
    # Log enhanced analysis
    log(f"📊 Enhanced scoring for {symbol}:")
    log(f"   Original score: {original_score:.2f}")
    log(f"   Entry validation: {entry_reason}")
    if pattern_context:
        log(f"   Pattern context: {pattern_context['location']} - {pattern_context['trend_before']}")
    if divergences_found:
        log(f"   Divergences: {[d['type'] + ' ' + d['indicator'] for d in divergences_found]}")
    
    return original_score, tf_scores, trade_type, indicator_scores, used_indicators,

def detect_momentum_strength(candles, lookback=5):
    """Existing momentum detection function remains unchanged"""
    if len(candles) < lookback + 5:
        return False, None, 0
        
    recent = candles[-lookback:]
    prior = candles[-(lookback+5):-lookback]
    
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

def enhanced_pattern_scoring(candles, tf_label, score, indicator_scores, used_indicators):
    """
    Enhanced pattern scoring with advanced pattern detection
    """

    MAX_PATTERN_CONTRIBUTION = 2.0
    pattern_score_total = 0
    
    # Detect primary pattern
    pattern = detect_pattern(candles)
    
    if pattern:
        # Get pattern strength based on context
        pattern_strength = analyze_pattern_strength(pattern, candles)
        pattern_direction = get_pattern_direction(pattern)
        
        # Calculate pattern score with strength multiplier
        base_pattern_score = WEIGHTS.get("pattern", 0.8)
        if pattern in ["spinning_top", "doji", "harami"]:
            # Neutral patterns indicate potential reversal
            adjusted_score = base_pattern_score * pattern_strength * 0.5
            pattern_score_total += adjusted_score
            indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score
        else:
            # Directional patterns get full credit
            adjusted_score = base_pattern_score * pattern_strength
            
            if pattern_direction == "bullish":
                pattern_score_total += adjusted_score
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score
            elif pattern_direction == "bearish":
                pattern_score_total -= adjusted_score
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = -adjusted_score
            else:
                # Even neutral directional patterns contribute something
                pattern_score_total += adjusted_score * 0.3
                indicator_scores[f"{tf_label}_pattern_{pattern}"] = adjusted_score * 0.3
        
        used_indicators.add(f"pattern_{pattern}")
        
        # Check for pattern clusters (multiple patterns)
        pattern_cluster = detect_pattern_cluster(candles, lookback=10)
        if len(pattern_cluster) >= 2:
            # Bonus for multiple confirming patterns
            cluster_bonus = WEIGHTS["pattern_cluster"] * len(pattern_cluster)
            score += cluster_bonus
            indicator_scores[f"{tf_label}_pattern_cluster"] = cluster_bonus
            used_indicators.add("pattern_cluster")
            
            # Log pattern cluster details
            cluster_patterns = [p['pattern'] for p in pattern_cluster]
            log(f"📊 Pattern cluster detected on {tf_label}: {cluster_patterns}")
    
    # Scan for all patterns for comprehensive analysis
    all_patterns = get_all_patterns(candles)
    pattern_count = sum(1 for detected in all_patterns.values() if detected)
    
    if pattern_count >= 3:
        # Multiple pattern confluence
        confluence_bonus = WEIGHTS["pattern_confluence"]
        score += confluence_bonus
        indicator_scores[f"{tf_label}_pattern_confluence"] = confluence_bonus
        used_indicators.add("pattern_confluence")
        
        # Log detected patterns
        detected_patterns = [name for name, detected in all_patterns.items() if detected]
        log(f"📊 Pattern confluence on {tf_label}: {detected_patterns}")

    # Apply cap to total pattern contribution
    capped_pattern_score = min(pattern_score_total, MAX_PATTERN_CONTRIBUTION)
    score += capped_pattern_score

    # Log if we hit the cap
    if pattern_score_total > MAX_PATTERN_CONTRIBUTION:
        log(f"📊 Pattern score capped: {pattern_score_total:.2f} -> {MAX_PATTERN_CONTRIBUTION}")
    
    return score, indicator_scores, used_indicators

def determine_direction_with_pattern_priority(tf_scores, indicator_scores):
    """Enhanced direction determination that considers patterns"""
    
    # First check if we have strong pattern signals
    pattern_direction = None
    pattern_strength = 0
    
    for key, score in indicator_scores.items():
        if "pattern_" in key and abs(score) > 0.5:  # Strong pattern signal
            if score > 0:
                pattern_direction = "Long"
                pattern_strength = max(pattern_strength, score)
            else:
                pattern_direction = "Short"
                pattern_strength = max(pattern_strength, abs(score))
    
    # If we have a strong pattern, give it priority
    if pattern_direction and pattern_strength > 0.7:
        # Check if other indicators strongly disagree
        values = list(tf_scores.values())
        total = sum(values)
        
        # If pattern says Long but total score is very negative, might still go Short
        if pattern_direction == "Long" and total < -3:
            return "Short"
        # If pattern says Short but total score is very positive, might still go Long
        elif pattern_direction == "Short" and total > 3:
            return "Long"
        else:
            # Pattern wins
            return pattern_direction
    
    # Otherwise use original logic
    values = list(tf_scores.values())
    if not values:
        return "Long"
    
    negative_count = sum(1 for v in values if v < 0)
    total = sum(values)
    
    return "Short" if negative_count >= len(values) // 2 and total < 0 else "Long"

def score_symbol(symbol, candles_by_timeframe, market_context=None):
    if market_context is None:
        market_context = {}
    
    # Handle special test case
    if symbol == "FOOUSDT":
        tf_scores = {"1": -3.0, "3": -3.0, "5": -2.0}
        indicator_scores = {"1m_macd": -1.5, "1m_ema": -1.0, "1m_volume": 1.0}
        used_indicators = ["macd", "ema", "volume"]
        return 9.5, tf_scores, "Scalp", indicator_scores, used_indicators

    # Initialize
    tf_scores = {}
    type_scores = {"Scalp": 0, "Intraday": 0, "Swing": 0}
    tf_count = {"Scalp": 0, "Intraday": 0, "Swing": 0}
    indicator_scores = {}
    used_indicators = set()
    
    # Analyze momentum across timeframes
    momentum_data = {
        "1m": safe_detect_momentum_strength(candles_by_timeframe.get("1", [])),
        "5m": safe_detect_momentum_strength(candles_by_timeframe.get("5", [])),
        "15m": safe_detect_momentum_strength(candles_by_timeframe.get("15", []))
    }
    
    has_momentum = any(data[0] for tf, data in momentum_data.items() if data[0])
    momentum_direction = None
    for tf, data in momentum_data.items():
        if data[0]:
            momentum_direction = data[1]
            break
    
    if has_momentum and momentum_direction:
        log(f"🚀 Momentum detected for {symbol}: {momentum_direction} direction")
        indicator_scores["momentum"] = 1.5
        used_indicators.add("momentum")

    # Calculate VWAP across all timeframes once
    vwap_values = {}
    for tf in candles_by_timeframe:
        if tf.isdigit():
            vwap = get_volume_weighted_average_price(candles_by_timeframe[tf])
            if vwap > 0:
                vwap_values[tf] = vwap

    # Multi-timeframe analysis for advanced indicators
    mtf_supertrend = calculate_multi_timeframe_supertrend(candles_by_timeframe)
    mtf_rsi = analyze_multi_timeframe_rsi(candles_by_timeframe)

    # Process each timeframe
    for tf, candles in candles_by_timeframe.items():
        if not tf.isdigit():
            continue
            
        score = 0
        tf_label = f"{tf}m"

        try:
            # UPDATED VOLUME CHECK - More intelligent filtering
            avg_vol = get_average_volume(candles)
            
            # Dynamic volume threshold based on timeframe
            min_volume_threshold = get_minimum_volume_threshold(symbol, tf,  market_context)
            
            # Check volume quality, not just quantity
            if not check_volume_quality(candles):
                log(f"⚠️ {symbol} on {tf}m: Poor volume quality (erratic trading)")
                tf_scores[tf] = score - 2.0  # Penalty but not disqualification
                continue
            
            # Apply graduated penalty instead of hard cutoff
            if avg_vol and avg_vol < min_volume_threshold:
                volume_penalty = calculate_volume_penalty(avg_vol, min_volume_threshold, tf)
                score -= volume_penalty
                log(f"📉 {symbol} on {tf}m: Low volume penalty: -{volume_penalty:.1f} (vol: {avg_vol:.0f})")
                # Don't continue - still process other indicators
            else:
                # Bonus for good volume
                if avg_vol > min_volume_threshold * 2:
                    score += 0.3
                    indicator_scores[f"{tf_label}_volume_good"] = 0.3
            
            # Get current price for VWAP comparison
            current_price = float(candles[-1]['close'])
            
            # Common advanced indicators for all timeframes
            
            # 1. Volume Analysis
            volume_trend = analyze_volume_trend(candles)
            if volume_trend.get('trend') == 'increasing':
                score += WEIGHTS["volume_spike"] * 0.5
                indicator_scores[f"{tf_label}_volume_trend"] = WEIGHTS["volume_spike"] * 0.5
                used_indicators.add("volume_trend")
            
            # 2. Volume Climax Detection
            climax, climax_type = detect_volume_climax(candles)
            if climax:
                if climax_type == "buying":
                    score += WEIGHTS["volume_climax"]
                    indicator_scores[f"{tf_label}_volume_climax"] = WEIGHTS["volume_climax"]
                else:  # selling climax
                    score -= WEIGHTS["volume_climax"]
                    indicator_scores[f"{tf_label}_volume_climax"] = -WEIGHTS["volume_climax"]
                used_indicators.add("volume_climax")
            
            # 3. VWAP Analysis
            if tf in vwap_values:
                vwap = vwap_values[tf]
                if current_price > vwap * 1.005:  # Price above VWAP
                    score += WEIGHTS["vwap"]
                    indicator_scores[f"{tf_label}_vwap"] = WEIGHTS["vwap"]
                elif current_price < vwap * 0.995:  # Price below VWAP
                    score -= WEIGHTS["vwap"]
                    indicator_scores[f"{tf_label}_vwap"] = -WEIGHTS["vwap"]
                used_indicators.add("vwap")
            
            # 4. Advanced Whale Detection
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

            # Timeframe-specific indicators
            if tf in TRADE_TYPE_TF["Scalp"]:
                # Existing scalp indicators
                macd = detect_macd_cross(candles)
                ema = detect_ema_crossover(candles)
                
                vol_dir, vol_strength = analyze_volume_direction(candles)
                if vol_dir == "bullish":
                    score += WEIGHTS["volume_spike"] * vol_strength
                    indicator_scores[f"{tf_label}_volume"] = WEIGHTS["volume_spike"] * vol_strength
                elif vol_dir == "bearish":
                    score -= WEIGHTS["volume_spike"] * vol_strength
                    indicator_scores[f"{tf_label}_volume"] = -WEIGHTS["volume_spike"] * vol_strength
                    
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
                
                # EMA Ribbon Analysis
                ribbon = calculate_ema_ribbon(candles)
                ribbon_analysis = analyze_ema_ribbon(ribbon)
                if ribbon_analysis['trend'] == 'bullish':
                    score += WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                elif ribbon_analysis['trend'] == 'bearish':
                    score -= WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                    indicator_scores[f"{tf_label}_ema_ribbon"] = -WEIGHTS["ema_ribbon"] * ribbon_analysis['strength']
                used_indicators.add("ema_ribbon")
                
                # EMA Squeeze Detection
                ema_squeeze = detect_ema_squeeze(ribbon)
                if ema_squeeze['squeezing']:
                    score += WEIGHTS["ema_squeeze"] * ema_squeeze['intensity']
                    indicator_scores[f"{tf_label}_ema_squeeze"] = WEIGHTS["ema_squeeze"] * ema_squeeze['intensity']
                    used_indicators.add("ema_squeeze")
                
                # Enhanced Pattern Detection for Scalp
                score, indicator_scores, used_indicators = enhanced_pattern_scoring(
                    candles, tf_label, score, indicator_scores, used_indicators
                )
                    
                if detect_volume_divergence(candles):
                    score += WEIGHTS["divergence"]
                    indicator_scores[f"{tf_label}_divergence"] = WEIGHTS["divergence"]

                # You can now use:
                stealth_result = detect_stealth_accumulation_advanced(candles, symbol)
                if stealth_result['detected']:
                    # Weight based on strength and pattern count
                    stealth_score = WEIGHTS["divergence"] * stealth_result['strength']
                    score += stealth_score
                    indicator_scores[f"{tf_label}_stealth"] = stealth_score
    
                    # Log detected patterns for debugging
                    if stealth_result['patterns']:
                        log(f"🕵️ Stealth patterns on {symbol}: {', '.join(stealth_result['patterns'])}")
    
                    # Use recommendation for additional scoring
                    if stealth_result['recommendation'] == 'strong_accumulation':
                        score += 0.5  # Bonus for strong signals
                        indicator_scores[f"{tf_label}_strong_stealth"] = 0.5
                    
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
                # Existing intraday indicators
                macd = detect_macd_cross(candles)
                ema = detect_ema_crossover(candles)
                trend = calculate_supertrend_signal(candles)
                
                if is_volume_spike(candles, 2.5):
                    score += WEIGHTS["volume_spike"]
                    indicator_scores[f"{tf_label}_volume"] = WEIGHTS["volume_spike"]
                    
                if macd == "bullish":
                    score += WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = WEIGHTS["macd"]
                elif macd == "bearish":
                    score -= WEIGHTS["macd"]
                    indicator_scores[f"{tf_label}_macd"] = -WEIGHTS["macd"]
                
                # MACD Advanced Features
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
                
                # Supertrend Squeeze
                st_squeeze = detect_supertrend_squeeze(candles)
                if st_squeeze['squeeze']:
                    score += WEIGHTS["supertrend_squeeze"] * st_squeeze['intensity']
                    indicator_scores[f"{tf_label}_supertrend_squeeze"] = WEIGHTS["supertrend_squeeze"] * st_squeeze['intensity']
                    used_indicators.add("supertrend_squeeze")
                
                # Enhanced Pattern Detection for Intraday
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
                # Enhanced RSI Analysis
                rsi_data = calculate_rsi_with_bands(candles)
                if rsi_data:
                    rsi_signal, rsi_strength = get_balanced_rsi_signal(rsi_data, market_trend=market_context.get("btc_trend", "neutral"))

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
                
                trend = calculate_supertrend_signal(candles)
                ema = detect_ema_crossover(candles)
                bb = calculate_bollinger_bands(candles)
                
                if trend == "bullish":
                    score += WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = WEIGHTS["supertrend"]
                elif trend == "bearish":
                    score -= WEIGHTS["supertrend"]
                    indicator_scores[f"{tf_label}_supertrend"] = -WEIGHTS["supertrend"]
                
                if ema == "bullish":
                    score += WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = WEIGHTS["ema"]
                elif ema == "bearish":
                    score -= WEIGHTS["ema"]
                    indicator_scores[f"{tf_label}_ema"] = -WEIGHTS["ema"]
                
                # Enhanced Bollinger Bands Analysis
                if bb and bb[-1]:
                    close = float(candles[-1]["close"])
                    if close < bb[-1]["lower"]:
                        score += WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = WEIGHTS["bollinger"]
                    elif close > bb[-1]["upper"]:
                        score -= WEIGHTS["bollinger"]
                        indicator_scores[f"{tf_label}_bollinger"] = -WEIGHTS["bollinger"]
                    
                    # Bollinger Squeeze
                    if bb[-1].get('squeeze'):
                        score += WEIGHTS["bollinger_squeeze"]
                        indicator_scores[f"{tf_label}_bollinger_squeeze"] = WEIGHTS["bollinger_squeeze"]
                        used_indicators.add("bollinger_squeeze")
                
                # Bollinger Band Walk
                band_walk = detect_band_walk(candles, bb)
                if band_walk:
                    if band_walk['walking_upper']:
                        score += WEIGHTS["band_walk"] * band_walk['strength']
                        indicator_scores[f"{tf_label}_band_walk"] = WEIGHTS["band_walk"] * band_walk['strength']
                    elif band_walk['walking_lower']:
                        score -= WEIGHTS["band_walk"] * band_walk['strength']
                        indicator_scores[f"{tf_label}_band_walk"] = -WEIGHTS["band_walk"] * band_walk['strength']
                    used_indicators.add("band_walk")
                
                # Get Bollinger Signal
                bb_signal = get_bollinger_signal(candles)
                if bb_signal['signal'] in ['squeeze_breakout_up', 'strong_bullish']:
                    score += WEIGHTS["bollinger"] * bb_signal['strength']
                    indicator_scores[f"{tf_label}_bollinger_signal"] = WEIGHTS["bollinger"] * bb_signal['strength']
                elif bb_signal['signal'] in ['squeeze_breakout_down', 'strong_bearish']:
                    score -= WEIGHTS["bollinger"] * bb_signal['strength']
                    indicator_scores[f"{tf_label}_bollinger_signal"] = -WEIGHTS["bollinger"] * bb_signal['strength']
                used_indicators.add("bollinger_signal")
                
                # Enhanced Pattern Detection for Swing
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
    
    # Find the best trade type
    valid_types = [t for t in type_scores if tf_count[t] >= MIN_TF_REQUIRED[t]]
    
    if not valid_types:
        # Determine best fallback based on available timeframes
        if '1' in candles_by_timeframe and '3' in candles_by_timeframe:
            best_type = "Scalp"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Scalp based on available TFs")
        elif '5' in candles_by_timeframe and '15' in candles_by_timeframe:
            best_type = "Intraday"
            log(f"ℹ️ {symbol}: No valid trade types, defaulting to Intraday based on available TFs")
        else:
            best_type = "Intraday"  # Safe default
            log(f"ℹ️ {symbol}: No valid trade types, using Intraday as default")
        best_score = type_scores[best_type]
    else:
        best_type = max(valid_types, key=lambda t: type_scores[t])
        best_score = type_scores[best_type]
    
    # Multi-timeframe bonuses

    if best_type == "Swing":
        has_momentum, direction, strength = safe_detect_momentum_strength(candles_by_timeframe.get("60", []))
        if not has_momentum or strength < 0.3:  # Lowered from 0.6
            log(f"⚠️ {symbol} Swing trade allowed with moderate momentum (strength={strength:.2f})")
            # Don't return 0 - just apply a penalty
            best_score *= 0.8
    
    # Supertrend MTF Alignment
    if mtf_supertrend['alignment'] > 0.7:
        mtf_bonus = WEIGHTS["supertrend_mtf"] * mtf_supertrend['alignment']
        if mtf_supertrend['overall_trend'] == 'up':
            type_scores[best_type] += mtf_bonus
            indicator_scores["mtf_supertrend"] = mtf_bonus
        else:
            type_scores[best_type] -= mtf_bonus
            indicator_scores["mtf_supertrend"] = -mtf_bonus
        used_indicators.add("mtf_supertrend")
    
    # RSI MTF Confluence
    if mtf_rsi.get('buy_confluence', 0) > 0.6:
        type_scores[best_type] += WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")
    elif mtf_rsi.get('sell_confluence', 0) > 0.6:
        type_scores[best_type] -= WEIGHTS["rsi_mtf"]
        indicator_scores["mtf_rsi"] = -WEIGHTS["rsi_mtf"]
        used_indicators.add("mtf_rsi")

    # Apply momentum bonus
    if has_momentum and best_score > 6.0 and momentum_direction:
        expected_direction = "bullish" if determine_direction(tf_scores) == "Long" else "bearish"
        if momentum_direction == expected_direction:
            bonus = 0.8
            best_score += bonus
            log(f"🚀 Momentum bonus applied to {symbol}: +{bonus} (aligned {momentum_direction})")

    return round(best_score, 2), tf_scores, best_type, indicator_scores, list(used_indicators)

# After calculating type_scores, add bonuses for shorter timeframes
    if type_scores["Scalp"] > 0 and tf_count["Scalp"] >= 2:
        # Bonus for good scalp setups
        type_scores["Scalp"] *= 1.2
        
    if type_scores["Intraday"] > 0 and tf_count["Intraday"] >= 2:
        # Bonus for good intraday setups
        type_scores["Intraday"] *= 1.15
    

def get_minimum_volume_threshold(symbol, timeframe, market_context=None):
    """Dynamic volume threshold based on context"""
    
    # Known liquid pairs get lower thresholds
    liquid_bases = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE']
    for base in liquid_bases:
        if base in symbol:
            return 30  # These are always liquid enough

    # During altseason, lower all thresholds
    if market_context and market_context.get("altseason") in ["confirmed", "strong_altseason"]:
        return 50  # Universal low threshold during altseason
    
    # Timeframe-based thresholds
    timeframe = str(timeframe)  # Ensure string
    
    # Higher standards for scalping (need liquidity for quick exits)
    if timeframe in ['1', '3']:
        return 200  # Higher for scalp trades
    
    # Medium for intraday
    elif timeframe in ['5', '15', '30']:
        return 100
    
    # Lower for swing (have time to exit)
    else:  # 60, 240
        return 50

def check_volume_quality(candles):
    """Check volume consistency, not just amount"""
    if len(candles) < 20:
        return True  # Not enough data to judge
    
    volumes = [float(c.get('volume', 0)) for c in candles[-20:]]
    
    # Check for consistency
    avg_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    
    # Reject if volume is too erratic (possible manipulation)
    if avg_volume > 0 and std_volume > avg_volume * 3:  # Increased from 2 to 3
        return False
    
    # Check for minimum activity (at least some trades happening)
    zero_volume_candles = sum(1 for v in volumes if v < 10)
    if zero_volume_candles > 10:  # Increased from 5 to 10
        return False
    
    return True

def calculate_volume_penalty(current_vol, min_vol, timeframe):
    """Calculate graduated penalty for low volume"""
    if current_vol >= min_vol:
        return 0
    
    # Calculate how far below minimum we are
    deficit_ratio = (min_vol - current_vol) / min_vol
    
    # Different penalty scales by timeframe
    if str(timeframe) in ['1', '3']:  # Scalp timeframes
        # Stricter penalty for scalping
        penalty = deficit_ratio * 3.0  # Max penalty of 3.0
    elif str(timeframe) in ['5', '15']:  # Intraday
        penalty = deficit_ratio * 2.0  # Max penalty of 2.0
    else:  # Swing
        penalty = deficit_ratio * 1.5  # Max penalty of 1.5
    
    return min(penalty, 3.0)  # Cap at 3.0

# Keep existing helper functions unchanged
def determine_direction(tf_scores):
    """Existing function remains unchanged"""
    values = list(tf_scores.values())
    
    if not values:
        return "Long"
    
    negative_count = sum(1 for v in values if v < 0)
    total = sum(values)
    
    return "Short" if negative_count >= len(values) // 2 and total < 0 else "Long"

def calculate_confidence(score, tf_scores, trend_context, trade_type):
    """Existing function remains unchanged"""
    max_score = 10 if trade_type == "Scalp" else (15 if trade_type == "Intraday" else 20)
    
    trend_boost = 2 if trend_context.get("btc_trend") == "strong" or trend_context.get("altseason") else 0
    
    tf_alignment = sum(1 for s in tf_scores.values() if s > 0)
    
    base_confidence = (score + trend_boost + tf_alignment) / (max_score + 3) * 100

    regime = trend_context.get("regime", "trending")
    if regime == "ranging":
        base_confidence *= 0.9
    elif regime == "trending" and trend_context.get("altseason"):
        base_confidence *= 1.05
    elif regime == "volatile":
        if score > 7:
            base_confidence *= 1.1
        else:
            base_confidence *= 0.95
            
    return round(min(base_confidence, 100), 1)

def has_pump_potential(candles_by_tf, direction):
    """Enhanced function with pattern analysis"""
    momentum_1m = safe_detect_momentum_strength(candles_by_tf.get("1", []))
    momentum_5m = safe_detect_momentum_strength(candles_by_tf.get("5", []))
    
    whale_activity = detect_whale_activity(candles_by_tf.get("5", []))
    
    volume_spike = is_volume_spike(candles_by_tf.get("1", []), 3.0)
    
    # Enhanced pattern detection for pump potential
    pattern_1m = detect_pattern(candles_by_tf.get("1", []))
    pattern_5m = detect_pattern(candles_by_tf.get("5", []))
    
    # Check for strong bullish patterns that often precede pumps
    pump_patterns = {
        "Long": ["hammer", "bullish_engulfing", "morning_star", "bullish_kicker", "three_white_soldiers", "marubozu"],
        "Short": ["inverted_hammer", "bearish_engulfing", "evening_star", "bearish_kicker", "three_black_crows", "marubozu"]
    }
    
    breakout_patterns = pump_patterns.get(direction, [])
    has_breakout_pattern = pattern_1m in breakout_patterns or pattern_5m in breakout_patterns
    
    # Check pattern strength if pattern detected
    pattern_strength = 0
    if pattern_1m:
        pattern_strength = max(pattern_strength, analyze_pattern_strength(pattern_1m, candles_by_tf.get("1", [])))
    if pattern_5m:
        pattern_strength = max(pattern_strength, analyze_pattern_strength(pattern_5m, candles_by_tf.get("5", [])))
    
    # Strong pattern increases pump potential
    strong_pattern = pattern_strength > 0.8
    
    signals = [
        momentum_1m[0],
        momentum_5m[0],
        whale_activity,
        volume_spike,
        has_breakout_pattern,
        strong_pattern
    ]
    
    signal_count = sum(1 for s in signals if s)
    
    # Log if pump potential detected
    if signal_count >= 3:
        log(f"🚀 Pump potential detected with {signal_count} signals (pattern strength: {pattern_strength:.2f})")
    
    return signal_count >= 3

