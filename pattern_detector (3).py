# pattern_detector.py - Enhanced with Performance Optimizations and Advanced Patterns

import asyncio
import traceback
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
from datetime import datetime
import math
from logger import log
from error_handler import send_error_to_telegram

# Cache for pattern detection results
_pattern_cache = {}
_cache_ttl = 60  # 60 seconds cache TTL
_cache_timestamps = {}
_cache_max_size = 500

# Pattern strength weights
PATTERN_WEIGHTS = {
    # Strong reversal patterns
    "morning_star": 1.5,
    "evening_star": 1.5,
    "three_white_soldiers": 1.4,
    "three_black_crows": 1.4,
    "bullish_engulfing": 1.2,
    "bearish_engulfing": 1.2,
    
    # Moderate reversal patterns
    "hammer": 1.0,
    "inverted_hammer": 1.0,
    "shooting_star": 1.0,
    "hanging_man": 1.0,
    "piercing_line": 1.1,
    "dark_cloud_cover": 1.1,
    
    # Continuation patterns
    "inside_bar": 0.8,
    "three_line_strike": 1.3,
    "rising_three_methods": 1.2,
    "falling_three_methods": 1.2,
    
    # Neutral/Indecision patterns
    "doji": 0.6,
    "spinning_top": 0.7,
    "harami": 0.9,
    "harami_cross": 1.0,
    
    # Advanced patterns
    "marubozu": 1.1,
    "tweezer_top": 1.0,
    "tweezer_bottom": 1.0,
    "bullish_abandoned_baby": 1.6,
    "bearish_abandoned_baby": 1.6,
    "bullish_kicker": 1.4,
    "bearish_kicker": 1.4
}

# Pattern categories for analysis
REVERSAL_PATTERNS = {
    "bullish": ["hammer", "bullish_engulfing", "morning_star", "piercing_line", 
                "tweezer_bottom", "bullish_abandoned_baby", "bullish_kicker"],
    "bearish": ["inverted_hammer", "bearish_engulfing", "evening_star", "dark_cloud_cover",
                "shooting_star", "hanging_man", "tweezer_top", "bearish_abandoned_baby", "bearish_kicker"]
}

CONTINUATION_PATTERNS = {
    "bullish": ["three_white_soldiers", "rising_three_methods"],
    "bearish": ["three_black_crows", "falling_three_methods"]
}

def _get_cache_key(candles_hash: int, func_name: str) -> str:
    """Generate cache key for pattern detection"""
    return f"{candles_hash}_{func_name}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached result is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with new value"""
    _pattern_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Clean up old cache entries if needed
    if len(_pattern_cache) > _cache_max_size:
        # Remove oldest 20% of entries
        sorted_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])
        for key, _ in sorted_keys[:int(_cache_max_size * 0.2)]:
            _pattern_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def _calculate_candle_metrics(candle: Dict) -> Dict:
    """Pre-calculate common candle metrics for reuse"""
    high = float(candle["high"])
    low = float(candle["low"])
    open_ = float(candle["open"])
    close = float(candle["close"])
    
    body = abs(close - open_)
    range_ = high - low
    upper_shadow = high - max(open_, close)
    lower_shadow = min(open_, close) - low
    body_pct = body / range_ if range_ > 0 else 0
    is_bullish = close > open_
    
    return {
        "high": high,
        "low": low,
        "open": open_,
        "close": close,
        "body": body,
        "range": range_,
        "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
        "body_pct": body_pct,
        "is_bullish": is_bullish,
        "midpoint": (high + low) / 2
    }

# Enhanced pattern detection functions

def is_bullish_engulfing(prev: Dict, curr: Dict) -> bool:
    """Enhanced bullish engulfing with volume confirmation"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Basic engulfing criteria
    if not (not prev_m["is_bullish"] and curr_m["is_bullish"]):
        return False
    
    if not (curr_m["open"] < prev_m["close"] and curr_m["close"] > prev_m["open"]):
        return False
    
    # Enhanced: Check for volume confirmation
    if "volume" in prev and "volume" in curr:
        if float(curr["volume"]) < float(prev["volume"]) * 1.2:  # Need 20% more volume
            return False
    
    # Enhanced: Body size requirement
    if curr_m["body"] < prev_m["body"] * 1.1:  # Current body should be 10% larger
        return False
    
    return True

def is_bearish_engulfing(prev: Dict, curr: Dict) -> bool:
    """Enhanced bearish engulfing with volume confirmation"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Basic engulfing criteria
    if not (prev_m["is_bullish"] and not curr_m["is_bullish"]):
        return False
    
    if not (curr_m["open"] > prev_m["close"] and curr_m["close"] < prev_m["open"]):
        return False
    
    # Enhanced: Check for volume confirmation
    if "volume" in prev and "volume" in curr:
        if float(curr["volume"]) < float(prev["volume"]) * 1.2:
            return False
    
    # Enhanced: Body size requirement
    if curr_m["body"] < prev_m["body"] * 1.1:
        return False
    
    return True

def is_hammer(candle: Dict, trend_context: Optional[str] = None) -> bool:
    """Enhanced hammer with trend context"""
    m = _calculate_candle_metrics(candle)
    
    # Basic hammer criteria
    if m["body_pct"] >= 0.3:  # Body too large
        return False
    
    if m["lower_shadow"] < m["body"] * 2:  # Lower shadow not long enough
        return False
    
    if m["upper_shadow"] > m["body"] * 0.5:  # Upper shadow too long
        return False
    
    # Enhanced: Consider trend context (hammer is more valid at bottom of downtrend)
    if trend_context == "uptrend":
        return False  # Hammer in uptrend is less reliable
    
    return True

def is_inverted_hammer(candle: Dict, trend_context: Optional[str] = None) -> bool:
    """Enhanced inverted hammer with trend context"""
    m = _calculate_candle_metrics(candle)
    
    # Basic inverted hammer criteria
    if m["body_pct"] >= 0.3:
        return False
    
    if m["upper_shadow"] < m["body"] * 2:
        return False
    
    if m["lower_shadow"] > m["body"] * 0.5:
        return False
    
    # Enhanced: Consider trend context
    if trend_context == "downtrend":
        return False  # Inverted hammer in downtrend is less reliable
    
    return True

def is_inside_bar(prev: Dict, curr: Dict) -> bool:
    """Enhanced inside bar pattern"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Current candle completely within previous candle's range
    return (curr_m["high"] < prev_m["high"] and 
            curr_m["low"] > prev_m["low"] and
            curr_m["range"] < prev_m["range"] * 0.7)  # Enhanced: Range should be notably smaller

def is_morning_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Enhanced morning star with stricter criteria"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    m3 = _calculate_candle_metrics(c3)
    
    # First candle: bearish with decent body
    if m1["is_bullish"] or m1["body_pct"] < 0.5:
        return False
    
    # Second candle: small body (star)
    if m2["body"] >= m1["body"] * 0.3:  # Star body should be small
        return False
    
    # Gap down from first to second
    if m2["high"] >= m1["low"]:
        return False
    
    # Third candle: bullish closing above midpoint of first
    if not m3["is_bullish"] or m3["close"] <= m1["midpoint"]:
        return False
    
    # Enhanced: Volume should increase on third candle
    if "volume" in c3 and "volume" in c2:
        if float(c3["volume"]) < float(c2["volume"]) * 1.5:
            return False
    
    return True

def is_evening_star(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Enhanced evening star with stricter criteria"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    m3 = _calculate_candle_metrics(c3)
    
    # First candle: bullish with decent body
    if not m1["is_bullish"] or m1["body_pct"] < 0.5:
        return False
    
    # Second candle: small body (star)
    if m2["body"] >= m1["body"] * 0.3:
        return False
    
    # Gap up from first to second
    if m2["low"] <= m1["high"]:
        return False
    
    # Third candle: bearish closing below midpoint of first
    if m3["is_bullish"] or m3["close"] >= m1["midpoint"]:
        return False
    
    # Enhanced: Volume should increase on third candle
    if "volume" in c3 and "volume" in c2:
        if float(c3["volume"]) < float(c2["volume"]) * 1.5:
            return False
    
    return True

def is_doji(candle: Dict, precision: float = 0.1) -> bool:
    """Enhanced doji with configurable precision"""
    m = _calculate_candle_metrics(candle)
    
    # Doji has very small body relative to range
    return m["body"] <= m["range"] * precision

# New advanced pattern detection functions

def is_three_white_soldiers(candles: List[Dict]) -> bool:
    """Three consecutive bullish candles with higher closes"""
    if len(candles) < 3:
        return False
    
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    m1, m2, m3 = [_calculate_candle_metrics(c) for c in [c1, c2, c3]]
    
    # All must be bullish
    if not all([m1["is_bullish"], m2["is_bullish"], m3["is_bullish"]]):
        return False
    
    # Each close higher than previous
    if not (m1["close"] < m2["close"] < m3["close"]):
        return False
    
    # Each open within previous body
    if not (m1["open"] < m2["open"] < m1["close"]):
        return False
    if not (m2["open"] < m3["open"] < m2["close"]):
        return False
    
    # Decent body sizes
    min_body = min(m1["body"], m2["body"], m3["body"])
    avg_range = (m1["range"] + m2["range"] + m3["range"]) / 3
    if min_body < avg_range * 0.5:
        return False
    
    return True

def is_three_black_crows(candles: List[Dict]) -> bool:
    """Three consecutive bearish candles with lower closes"""
    if len(candles) < 3:
        return False
    
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    m1, m2, m3 = [_calculate_candle_metrics(c) for c in [c1, c2, c3]]
    
    # All must be bearish
    if not all([not m1["is_bullish"], not m2["is_bullish"], not m3["is_bullish"]]):
        return False
    
    # Each close lower than previous
    if not (m1["close"] > m2["close"] > m3["close"]):
        return False
    
    # Each open within previous body
    if not (m1["close"] < m2["open"] < m1["open"]):
        return False
    if not (m2["close"] < m3["open"] < m2["open"]):
        return False
    
    # Decent body sizes
    min_body = min(m1["body"], m2["body"], m3["body"])
    avg_range = (m1["range"] + m2["range"] + m3["range"]) / 3
    if min_body < avg_range * 0.5:
        return False
    
    return True

def is_piercing_line(prev: Dict, curr: Dict) -> bool:
    """Bullish reversal pattern"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Previous bearish, current bullish
    if prev_m["is_bullish"] or not curr_m["is_bullish"]:
        return False
    
    # Current opens below previous low
    if curr_m["open"] >= prev_m["low"]:
        return False
    
    # Current closes above middle of previous body
    prev_middle = (prev_m["open"] + prev_m["close"]) / 2
    if curr_m["close"] <= prev_middle:
        return False
    
    # But not above previous open (that would be engulfing)
    if curr_m["close"] >= prev_m["open"]:
        return False
    
    return True

def is_dark_cloud_cover(prev: Dict, curr: Dict) -> bool:
    """Bearish reversal pattern"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Previous bullish, current bearish
    if not prev_m["is_bullish"] or curr_m["is_bullish"]:
        return False
    
    # Current opens above previous high
    if curr_m["open"] <= prev_m["high"]:
        return False
    
    # Current closes below middle of previous body
    prev_middle = (prev_m["open"] + prev_m["close"]) / 2
    if curr_m["close"] >= prev_middle:
        return False
    
    # But not below previous open (that would be engulfing)
    if curr_m["close"] <= prev_m["open"]:
        return False
    
    return True

def is_shooting_star(candle: Dict, prev_candle: Optional[Dict] = None) -> bool:
    """Bearish reversal at top of uptrend"""
    m = _calculate_candle_metrics(candle)
    
    # Small body at lower end
    if m["body_pct"] >= 0.3:
        return False
    
    # Long upper shadow
    if m["upper_shadow"] < m["body"] * 2:
        return False
    
    # Small lower shadow
    if m["lower_shadow"] > m["body"] * 0.5:
        return False
    
    # Should be after upward movement
    if prev_candle:
        prev_m = _calculate_candle_metrics(prev_candle)
        if m["low"] <= prev_m["close"]:  # Not gapping up
            return False
    
    return True

def is_hanging_man(candle: Dict, prev_candle: Optional[Dict] = None) -> bool:
    """Bearish reversal at top of uptrend (similar to hammer but context matters)"""
    m = _calculate_candle_metrics(candle)
    
    # Small body at upper end
    if m["body_pct"] >= 0.3:
        return False
    
    # Long lower shadow
    if m["lower_shadow"] < m["body"] * 2:
        return False
    
    # Small upper shadow
    if m["upper_shadow"] > m["body"] * 0.5:
        return False
    
    # Should be after upward movement
    if prev_candle:
        prev_m = _calculate_candle_metrics(prev_candle)
        if m["close"] <= prev_m["close"]:  # Not in uptrend
            return False
    
    return True

def is_marubozu(candle: Dict) -> bool:
    """Strong trend continuation pattern with no shadows"""
    m = _calculate_candle_metrics(candle)
    
    # Very small shadows relative to body
    if m["range"] == 0:
        return False
    
    shadow_ratio = (m["upper_shadow"] + m["lower_shadow"]) / m["range"]
    return shadow_ratio < 0.05  # Less than 5% shadows

def is_spinning_top(candle: Dict) -> bool:
    """Indecision pattern with small body and long shadows"""
    m = _calculate_candle_metrics(candle)
    
    # Small body
    if m["body_pct"] >= 0.35:
        return False
    
    # Both shadows should be significant
    if m["upper_shadow"] < m["body"] or m["lower_shadow"] < m["body"]:
        return False
    
    return True

def is_harami(prev: Dict, curr: Dict) -> bool:
    """Current candle contained within previous candle's body"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Current candle's body within previous body
    prev_body_high = max(prev_m["open"], prev_m["close"])
    prev_body_low = min(prev_m["open"], prev_m["close"])
    curr_body_high = max(curr_m["open"], curr_m["close"])
    curr_body_low = min(curr_m["open"], curr_m["close"])
    
    return (curr_body_high < prev_body_high and 
            curr_body_low > prev_body_low and
            curr_m["body"] < prev_m["body"] * 0.5)

def is_harami_cross(prev: Dict, curr: Dict) -> bool:
    """Harami where second candle is a doji"""
    return is_harami(prev, curr) and is_doji(curr)

def is_tweezer_top(c1: Dict, c2: Dict) -> bool:
    """Two candles with same high (resistance)"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    
    # Highs should be very close
    high_diff = abs(m1["high"] - m2["high"]) / m1["high"]
    if high_diff > 0.001:  # Within 0.1%
        return False
    
    # First bullish, second bearish (classic)
    return m1["is_bullish"] and not m2["is_bullish"]

def is_tweezer_bottom(c1: Dict, c2: Dict) -> bool:
    """Two candles with same low (support)"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    
    # Lows should be very close
    low_diff = abs(m1["low"] - m2["low"]) / m1["low"]
    if low_diff > 0.001:  # Within 0.1%
        return False
    
    # First bearish, second bullish (classic)
    return not m1["is_bullish"] and m2["is_bullish"]

def is_bullish_abandoned_baby(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Rare but powerful bullish reversal"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    m3 = _calculate_candle_metrics(c3)
    
    # First: bearish
    if m1["is_bullish"]:
        return False
    
    # Second: doji with gap down
    if not is_doji(c2) or m2["high"] >= m1["low"]:
        return False
    
    # Third: bullish with gap up
    if not m3["is_bullish"] or m3["low"] <= m2["high"]:
        return False
    
    return True

def is_bearish_abandoned_baby(c1: Dict, c2: Dict, c3: Dict) -> bool:
    """Rare but powerful bearish reversal"""
    m1 = _calculate_candle_metrics(c1)
    m2 = _calculate_candle_metrics(c2)
    m3 = _calculate_candle_metrics(c3)
    
    # First: bullish
    if not m1["is_bullish"]:
        return False
    
    # Second: doji with gap up
    if not is_doji(c2) or m2["low"] <= m1["high"]:
        return False
    
    # Third: bearish with gap down
    if m3["is_bullish"] or m3["high"] >= m2["low"]:
        return False
    
    return True

def is_bullish_kicker(prev: Dict, curr: Dict) -> bool:
    """Strong bullish reversal with gap"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Previous bearish, current bullish
    if prev_m["is_bullish"] or not curr_m["is_bullish"]:
        return False
    
    # Gap up
    if curr_m["open"] <= prev_m["open"]:
        return False
    
    # Both should be marubozu-like (strong)
    if not (is_marubozu(prev) and is_marubozu(curr)):
        return False
    
    return True

def is_bearish_kicker(prev: Dict, curr: Dict) -> bool:
    """Strong bearish reversal with gap"""
    prev_m = _calculate_candle_metrics(prev)
    curr_m = _calculate_candle_metrics(curr)
    
    # Previous bullish, current bearish
    if not prev_m["is_bullish"] or curr_m["is_bullish"]:
        return False
    
    # Gap down
    if curr_m["open"] >= prev_m["open"]:
        return False
    
    # Both should be marubozu-like (strong)
    if not (is_marubozu(prev) and is_marubozu(curr)):
        return False
    
    return True

# Main pattern detection function

def detect_pattern(candles: List[Dict], use_cache: bool = True) -> Optional[str]:
    """
    Enhanced pattern detection with caching and all patterns
    
    Args:
        candles: List of candle dictionaries
        use_cache: Whether to use caching
        
    Returns:
        Detected pattern name or None
    """
    try:
        if not candles or len(candles) < 3:
            return None
        
        # Create cache key if caching enabled
        cache_key = None
        if use_cache:
            # Use hash of last few candles for cache key
            candles_str = str([(c.get('timestamp', ''), c.get('close', '')) for c in candles[-5:]])
            cache_key = _get_cache_key(hash(candles_str), "detect_pattern")
            
            if cache_key in _pattern_cache and _is_cache_valid(cache_key):
                return _pattern_cache[cache_key]
        
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        
        # Determine trend context for better pattern reliability
        trend_context = _determine_trend_context(candles)
        
        # Three-candle patterns (check first as they're often stronger)
        if is_morning_star(c1, c2, c3):
            pattern = "morning_star"
        elif is_evening_star(c1, c2, c3):
            pattern = "evening_star"
        elif is_three_white_soldiers(candles):
            pattern = "three_white_soldiers"
        elif is_three_black_crows(candles):
            pattern = "three_black_crows"
        elif is_bullish_abandoned_baby(c1, c2, c3):
            pattern = "bullish_abandoned_baby"
        elif is_bearish_abandoned_baby(c1, c2, c3):
            pattern = "bearish_abandoned_baby"
        
        # Two-candle patterns
        elif is_bullish_engulfing(c2, c3):
            pattern = "bullish_engulfing"
        elif is_bearish_engulfing(c2, c3):
            pattern = "bearish_engulfing"
        elif is_inside_bar(c2, c3):
            pattern = "inside_bar"
        elif is_piercing_line(c2, c3):
            pattern = "piercing_line"
        elif is_dark_cloud_cover(c2, c3):
            pattern = "dark_cloud_cover"
        elif is_harami(c2, c3):
            pattern = "harami"
        elif is_harami_cross(c2, c3):
            pattern = "harami_cross"
        elif is_tweezer_top(c2, c3):
            pattern = "tweezer_top"
        elif is_tweezer_bottom(c2, c3):
            pattern = "tweezer_bottom"
        elif is_bullish_kicker(c2, c3):
            pattern = "bullish_kicker"
        elif is_bearish_kicker(c2, c3):
            pattern = "bearish_kicker"
        
        # Single candle patterns
        elif is_hammer(c3, trend_context):
            pattern = "hammer"
        elif is_inverted_hammer(c3, trend_context):
            pattern = "inverted_hammer"
        elif is_shooting_star(c3, c2):
            pattern = "shooting_star"
        elif is_hanging_man(c3, c2):
            pattern = "hanging_man"
        elif is_marubozu(c3):
            pattern = "marubozu"
        elif is_spinning_top(c3):
            pattern = "spinning_top"
        elif is_doji(c3):
            pattern = "doji"
        else:
            pattern = None
        
        # Update cache if pattern found
        if use_cache and cache_key:
            _update_cache(cache_key, pattern)
        
        if pattern:
            log(f"🔍 Pattern Detected: {pattern}")
        
        return pattern

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Pattern Detection Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def _determine_trend_context(candles: List[Dict], lookback: int = 10) -> Optional[str]:
    """Determine if we're in uptrend, downtrend, or ranging"""
    if len(candles) < lookback:
        return None
    
    # Simple trend determination using linear regression
    closes = [float(c['close']) for c in candles[-lookback:]]
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    
    # Normalize slope by average price
    avg_price = np.mean(closes)
    normalized_slope = (slope / avg_price) * 100
    
    if normalized_slope > 0.5:  # 0.5% upward slope
        return "uptrend"
    elif normalized_slope < -0.5:  # 0.5% downward slope
        return "downtrend"
    else:
        return "ranging"

# Advanced pattern analysis functions

def analyze_pattern_strength(pattern: str, candles: List[Dict]) -> float:
    """
    Analyze the strength of a detected pattern based on context
    
    Returns:
        Strength score between 0 and 1
    """
    if not pattern or not candles:
        return 0.0
    
    base_weight = PATTERN_WEIGHTS.get(pattern, 0.5)
    strength = base_weight
    
    # Adjust based on volume
    if len(candles) >= 3:
        recent_vol = float(candles[-1].get('volume', 0))
        avg_vol = np.mean([float(c.get('volume', 0)) for c in candles[-10:-1]])
        
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            if vol_ratio > 1.5:  # High volume confirmation
                strength *= 1.2
            elif vol_ratio < 0.7:  # Low volume weakness
                strength *= 0.8
    
    # Adjust based on trend alignment
    trend = _determine_trend_context(candles)
    
    if pattern in REVERSAL_PATTERNS["bullish"] and trend == "downtrend":
        strength *= 1.3  # Bullish reversal in downtrend is stronger
    elif pattern in REVERSAL_PATTERNS["bearish"] and trend == "uptrend":
        strength *= 1.3  # Bearish reversal in uptrend is stronger
    elif pattern in CONTINUATION_PATTERNS["bullish"] and trend == "uptrend":
        strength *= 1.2  # Bullish continuation in uptrend
    elif pattern in CONTINUATION_PATTERNS["bearish"] and trend == "downtrend":
        strength *= 1.2  # Bearish continuation in downtrend
    
    return min(strength, 1.0)  # Cap at 1.0

def detect_pattern_cluster(candles: List[Dict], lookback: int = 10) -> List[Dict]:
    """
    Detect multiple patterns in recent candles
    
    Returns:
        List of pattern dictionaries with position and strength
    """
    if len(candles) < lookback:
        return []
    
    patterns = []
    
    # Check for patterns at different positions
    for i in range(len(candles) - lookback, len(candles) - 2):
        # Get candle subset
        subset = candles[:i+3]
        
        # Detect pattern
        pattern = detect_pattern(subset, use_cache=False)
        
        if pattern:
            strength = analyze_pattern_strength(pattern, subset)
            patterns.append({
                "pattern": pattern,
                "position": i,
                "strength": strength,
                "candles_ago": len(candles) - i - 3
            })
    
    return patterns

def get_pattern_direction(pattern: str) -> Optional[str]:
    """Get the directional bias of a pattern"""
    if pattern in REVERSAL_PATTERNS["bullish"] or pattern in CONTINUATION_PATTERNS["bullish"]:
        return "bullish"
    elif pattern in REVERSAL_PATTERNS["bearish"] or pattern in CONTINUATION_PATTERNS["bearish"]:
        return "bearish"
    else:
        return "neutral"

def pattern_success_probability(pattern: str, market_conditions: Dict) -> float:
    """
    Estimate pattern success probability based on market conditions
    
    Args:
        pattern: Pattern name
        market_conditions: Dict with keys like 'volatility', 'trend_strength', 'volume'
        
    Returns:
        Probability between 0 and 1
    """
    base_prob = 0.5  # Start with 50%
    
    # Pattern reliability
    if pattern in ["morning_star", "evening_star", "three_white_soldiers", "three_black_crows"]:
        base_prob = 0.65  # High reliability patterns
    elif pattern in ["bullish_engulfing", "bearish_engulfing", "hammer", "inverted_hammer"]:
        base_prob = 0.60  # Moderate reliability
    elif pattern in ["doji", "spinning_top", "inside_bar"]:
        base_prob = 0.45  # Lower reliability
    
    # Adjust for market conditions
    volatility = market_conditions.get('volatility', 'normal')
    if volatility == 'high':
        base_prob *= 0.9  # Patterns less reliable in high volatility
    elif volatility == 'low':
        base_prob *= 1.1  # Patterns more reliable in low volatility
    
    trend_strength = market_conditions.get('trend_strength', 0.5)
    if trend_strength > 0.7:
        # Strong trends favor continuation patterns
        if pattern in CONTINUATION_PATTERNS["bullish"] + CONTINUATION_PATTERNS["bearish"]:
            base_prob *= 1.2
        else:  # Reversal patterns less likely in strong trends
            base_prob *= 0.8
    
    volume_confirm = market_conditions.get('volume', 'normal')
    if volume_confirm == 'high':
        base_prob *= 1.15  # Volume confirmation increases reliability
    elif volume_confirm == 'low':
        base_prob *= 0.85  # Low volume decreases reliability
    
    return min(max(base_prob, 0.0), 1.0)  # Keep between 0 and 1

# Helper functions for performance

class PatternScanner:
    """
    High-performance pattern scanner with batch processing
    """
    def __init__(self):
        self.pattern_functions = {
            # Three-candle patterns
            "morning_star": lambda c: is_morning_star(c[-3], c[-2], c[-1]) if len(c) >= 3 else False,
            "evening_star": lambda c: is_evening_star(c[-3], c[-2], c[-1]) if len(c) >= 3 else False,
            "three_white_soldiers": is_three_white_soldiers,
            "three_black_crows": is_three_black_crows,
            
            # Two-candle patterns
            "bullish_engulfing": lambda c: is_bullish_engulfing(c[-2], c[-1]) if len(c) >= 2 else False,
            "bearish_engulfing": lambda c: is_bearish_engulfing(c[-2], c[-1]) if len(c) >= 2 else False,
            "piercing_line": lambda c: is_piercing_line(c[-2], c[-1]) if len(c) >= 2 else False,
            "dark_cloud_cover": lambda c: is_dark_cloud_cover(c[-2], c[-1]) if len(c) >= 2 else False,
            
            # Single candle patterns
            "hammer": lambda c: is_hammer(c[-1]) if len(c) >= 1 else False,
            "inverted_hammer": lambda c: is_inverted_hammer(c[-1]) if len(c) >= 1 else False,
            "doji": lambda c: is_doji(c[-1]) if len(c) >= 1 else False,
        }
    
    def scan_all_patterns(self, candles: List[Dict]) -> Dict[str, bool]:
        """Scan for all patterns at once"""
        results = {}
        
        for pattern_name, pattern_func in self.pattern_functions.items():
            try:
                results[pattern_name] = pattern_func(candles)
            except:
                results[pattern_name] = False
        
        return results

# Global scanner instance
_scanner = PatternScanner()

def get_all_patterns(candles: List[Dict]) -> Dict[str, bool]:
    """Get all patterns detected in the candles"""
    return _scanner.scan_all_patterns(candles)

# Cache cleanup
async def cleanup_pattern_cache():
    """Periodically clean up old cache entries"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in _cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > _cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            _pattern_cache.pop(key, None)
            _cache_timestamps.pop(key, None)
        
        if expired_keys:
            log(f"🧹 Cleaned {len(expired_keys)} expired pattern cache entries")

# Export main functions and constants
__all__ = [
    'detect_pattern',
    'analyze_pattern_strength',
    'detect_pattern_cluster',
    'get_pattern_direction',
    'pattern_success_probability',
    'get_all_patterns',
    'PATTERN_WEIGHTS',
    'REVERSAL_PATTERNS',
    'CONTINUATION_PATTERNS'
]
