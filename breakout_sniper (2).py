# breakout_sniper.py - Enhanced with Advanced Pattern Detection and Multi-Indicator Confirmation

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import deque

from volume import (
    is_volume_spike, get_average_volume, detect_volume_climax, 
    get_volume_momentum, analyze_volume_trend, get_volume_profile
)
from macd import detect_macd_cross, get_macd_momentum, get_macd_divergence
from rsi import calculate_rsi_with_bands, calculate_stoch_rsi, get_rsi_signal
from pattern_detector import (
    detect_pattern, analyze_pattern_strength, detect_pattern_cluster,
    get_pattern_direction, pattern_success_probability, get_all_patterns,
    PATTERN_WEIGHTS, CONTINUATION_PATTERNS
)
from ema import (
    calculate_ema, detect_ema_crossover, calculate_ema_ribbon, 
    analyze_ema_ribbon, get_ema_slope
)
from bollinger import (
    calculate_bollinger_bands_advanced, detect_band_walk, 
    get_bollinger_signal, detect_bollinger_squeeze
)
from supertrend import get_supertrend_state, calculate_supertrend_signal
from atr import calculate_atr
from whale_detector import detect_whale_activity_advanced
from logger import log, write_log

# Configuration
MIN_BREAKOUT_SCORE = 4.0
BREAKOUT_PATTERNS = {
    "bullish": ["marubozu", "bullish_kicker", "three_white_soldiers", "bullish_engulfing"],
    "bearish": ["marubozu", "bearish_kicker", "three_black_crows", "bearish_engulfing"]
}

# Cache for recent calculations
_breakout_cache = {}
_cache_ttl = 60  # 60 seconds

class BreakoutAnalyzer:
    """Advanced breakout analysis with pattern recognition"""
    
    def __init__(self):
        self.breakout_history = deque(maxlen=100)
        self.success_rate = 0.0
        self.total_trades = 0
        self.successful_trades = 0
        
    def update_performance(self, symbol: str, success: bool):
        """Update performance metrics for breakout trades"""
        self.total_trades += 1
        if success:
            self.successful_trades += 1
        self.success_rate = self.successful_trades / self.total_trades if self.total_trades > 0 else 0.0
        log(f"📊 Breakout Performance: {self.success_rate:.2%} success rate ({self.total_trades} trades)")

# Global analyzer instance
_analyzer = BreakoutAnalyzer()

def detect_resistance_break(candles: List[Dict], lookback: int = 20) -> Tuple[bool, float, float]:
    """
    Detect if price has broken through resistance
    
    Returns:
        Tuple of (is_breakout, resistance_level, breakout_strength)
    """
    if len(candles) < lookback:
        return False, 0, 0
    
    # Get highs from lookback period
    prev_highs = [float(c["high"]) for c in candles[-lookback-1:-1]]
    resistance = max(prev_highs)
    
    current_close = float(candles[-1]["close"])
    current_high = float(candles[-1]["high"])
    
    # Check if we've broken resistance
    if current_close > resistance:
        # Calculate breakout strength
        breakout_pct = ((current_close - resistance) / resistance) * 100
        
        # Verify it's a meaningful breakout (not just a wick)
        body_size = abs(float(candles[-1]["close"]) - float(candles[-1]["open"]))
        range_size = current_high - float(candles[-1]["low"])
        body_ratio = body_size / range_size if range_size > 0 else 0
        
        if body_ratio > 0.6 and breakout_pct > 0.2:  # Strong body and meaningful break
            return True, resistance, breakout_pct
    
    return False, resistance, 0

def detect_support_break(candles: List[Dict], lookback: int = 20) -> Tuple[bool, float, float]:
    """
    Detect if price has broken through support
    
    Returns:
        Tuple of (is_breakout, support_level, breakout_strength)
    """
    if len(candles) < lookback:
        return False, 0, 0
    
    # Get lows from lookback period
    prev_lows = [float(c["low"]) for c in candles[-lookback-1:-1]]
    support = min(prev_lows)
    
    current_close = float(candles[-1]["close"])
    current_low = float(candles[-1]["low"])
    
    # Check if we've broken support
    if current_close < support:
        # Calculate breakout strength
        breakout_pct = ((support - current_close) / support) * 100
        
        # Verify it's a meaningful breakout
        body_size = abs(float(candles[-1]["close"]) - float(candles[-1]["open"]))
        range_size = float(candles[-1]["high"]) - current_low
        body_ratio = body_size / range_size if range_size > 0 else 0
        
        if body_ratio > 0.6 and breakout_pct > 0.2:  # Strong body and meaningful break
            return True, support, breakout_pct
    
    return False, support, 0

def detect_volatility_expansion(candles: List[Dict], atr_multiplier: float = 1.5) -> bool:
    """
    Detect if volatility is expanding (favorable for breakouts)
    """
    if len(candles) < 30:
        return False
    
    # Calculate ATR
    atr_short = calculate_atr(candles, period=7)
    atr_long = calculate_atr(candles, period=21)
    
    if not atr_short or not atr_long or atr_long == 0:
        return False
    
    # Check if short-term volatility is expanding
    return atr_short > atr_long * atr_multiplier

def analyze_breakout_quality(candles: List[Dict], direction: str, breakout_level: float) -> float:
    """
    Analyze the quality of a breakout (0-1 score)
    """
    if len(candles) < 3:
        return 0
    
    quality_score = 0
    
    # 1. Check volume on breakout
    current_vol = float(candles[-1]["volume"])
    avg_vol = get_average_volume(candles[:-1], window=10)
    
    if avg_vol > 0:
        vol_ratio = current_vol / avg_vol
        if vol_ratio > 2.0:
            quality_score += 0.3
        elif vol_ratio > 1.5:
            quality_score += 0.2
    
    # 2. Check candle strength
    close = float(candles[-1]["close"])
    open_price = float(candles[-1]["open"])
    high = float(candles[-1]["high"])
    low = float(candles[-1]["low"])
    
    range_size = high - low
    if range_size > 0:
        if direction == "Long":
            # For bullish breakout, close should be near high
            close_position = (close - low) / range_size
            if close_position > 0.8:
                quality_score += 0.2
        else:  # Short
            # For bearish breakout, close should be near low
            close_position = (high - close) / range_size
            if close_position > 0.8:
                quality_score += 0.2
    
    # 3. Check for follow-through (multiple candles confirming)
    if len(candles) >= 3:
        confirming_candles = 0
        for i in range(-3, -1):
            candle_close = float(candles[i]["close"])
            if direction == "Long" and candle_close > breakout_level:
                confirming_candles += 1
            elif direction == "Short" and candle_close < breakout_level:
                confirming_candles += 1
        
        if confirming_candles >= 2:
            quality_score += 0.2
    
    # 4. Check for clean breakout (no immediate pullback)
    if direction == "Long":
        if low > breakout_level * 0.998:  # Didn't dip back below
            quality_score += 0.3
    else:  # Short
        if high < breakout_level * 1.002:  # Didn't rise back above
            quality_score += 0.3
    
    return min(quality_score, 1.0)

def enhanced_pattern_check_for_breakout(candles, tf, direction, score, reasons, confidence_factors):
    """Enhanced pattern confirmation specifically for breakouts"""
    
    # Get all patterns
    all_patterns = get_all_patterns(candles)
    
    # Check for breakout-favorable patterns
    if direction == "Long":
        bullish_patterns = [p for p in BREAKOUT_PATTERNS["bullish"] if all_patterns.get(p, False)]
        if bullish_patterns:
            # Find the strongest pattern
            strongest_pattern = max(bullish_patterns, key=lambda p: PATTERN_WEIGHTS.get(p, 0.5))
            pattern_strength = analyze_pattern_strength(strongest_pattern, candles)
            
            # Breakout patterns get higher weight
            pattern_score = 0.8 * pattern_strength
            score += pattern_score
            reasons[f"{tf}m_breakout_pattern_{strongest_pattern}"] = pattern_strength
            confidence_factors.append(0.7 * pattern_strength)
            
            log(f"   🎯 Bullish breakout pattern: {strongest_pattern} (strength: {pattern_strength:.2f})")
    
    elif direction == "Short":
        bearish_patterns = [p for p in BREAKOUT_PATTERNS["bearish"] if all_patterns.get(p, False)]
        if bearish_patterns:
            # Find the strongest pattern
            strongest_pattern = max(bearish_patterns, key=lambda p: PATTERN_WEIGHTS.get(p, 0.5))
            pattern_strength = analyze_pattern_strength(strongest_pattern, candles)
            
            # Breakout patterns get higher weight
            pattern_score = 0.8 * pattern_strength
            score += pattern_score
            reasons[f"{tf}m_breakout_pattern_{strongest_pattern}"] = pattern_strength
            confidence_factors.append(0.7 * pattern_strength)
            
            log(f"   🎯 Bearish breakout pattern: {strongest_pattern} (strength: {pattern_strength:.2f})")
    
    # Check for continuation patterns (favorable for breakouts)
    continuation_patterns = []
    for pattern_type in ["bullish", "bearish"]:
        for pattern in CONTINUATION_PATTERNS[pattern_type]:
            if all_patterns.get(pattern, False):
                pattern_direction = "Long" if pattern_type == "bullish" else "Short"
                if pattern_direction == direction:
                    continuation_patterns.append(pattern)
    
    if continuation_patterns:
        # Continuation patterns support breakout momentum
        for pattern in continuation_patterns:
            pattern_strength = analyze_pattern_strength(pattern, candles)
            score += 0.4 * pattern_strength
            reasons[f"{tf}m_continuation_{pattern}"] = pattern_strength
            confidence_factors.append(0.5 * pattern_strength)
        
        log(f"   📈 Continuation patterns detected: {continuation_patterns}")
    
    return score, reasons, confidence_factors

def score_breakout_sniper(symbol: str, candles_by_tf: Dict[str, List[Dict]], regime: str) -> Tuple[float, str, float, Dict]:
    """
    Enhanced breakout scoring with advanced pattern detection
    
    Args:
        symbol: Trading pair symbol
        candles_by_tf: Dictionary of candles by timeframe
        regime: Market regime
        
    Returns:
        score, direction, confidence, reasons dictionary
    """
    # Exit early if not in volatile regime
    if regime != "volatile":
        return 0, None, 0, {"not_volatile": True}
    
    # Check cache first
    cache_key = f"{symbol}_bo_{regime}"
    if cache_key in _breakout_cache:
        cached_time, cached_result = _breakout_cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_result
    
    tf_to_check = ["1", "3", "5"]  # Multiple timeframes for confirmation
    score = 0
    reasons = {}
    direction = None
    confidence_factors = []
    
    try:
        for tf in tf_to_check:
            candles = candles_by_tf.get(tf)
            if not candles or len(candles) < 30:
                continue
            
            # Basic price data
            close = float(candles[-1]["close"])
            high = float(candles[-1]["high"])
            low = float(candles[-1]["low"])
            open_price = float(candles[-1]["open"])
            
            # Check for resistance/support breaks
            resistance_break, resistance_level, resistance_strength = detect_resistance_break(candles)
            support_break, support_level, support_strength = detect_support_break(candles)
            
            # Determine primary breakout direction
            if resistance_break and not support_break:
                direction = "Long"
                breakout_level = resistance_level
                breakout_strength = resistance_strength
                score += 2.0
                reasons[f"{tf}m_resistance_break"] = resistance_strength
                confidence_factors.append(0.8)
            elif support_break and not resistance_break:
                direction = "Short"
                breakout_level = support_level
                breakout_strength = support_strength
                score += 2.0
                reasons[f"{tf}m_support_break"] = support_strength
                confidence_factors.append(0.8)
            else:
                continue  # No clear breakout on this timeframe
            
            # Analyze breakout quality
            quality = analyze_breakout_quality(candles, direction, breakout_level)
            if quality > 0.6:
                score += quality
                reasons[f"{tf}m_breakout_quality"] = quality
                confidence_factors.append(quality)
            
            # Volume analysis
            volume_analysis = analyze_volume_trend(candles)
            if volume_analysis.get('trend') == 'increasing':
                score += 0.5
                reasons[f"{tf}m_volume_trend"] = True
                confidence_factors.append(0.6)
            
            # Volume spike on breakout
            if is_volume_spike(candles, multiplier=2.0):
                score += 1.0
                reasons[f"{tf}m_volume_spike"] = True
                confidence_factors.append(0.7)
            
            # Volume climax check
            climax, climax_type = detect_volume_climax(candles)
            if climax:
                if (climax_type == "buying" and direction == "Long") or \
                   (climax_type == "selling" and direction == "Short"):
                    score += 0.8
                    reasons[f"{tf}m_volume_climax"] = climax_type
                    confidence_factors.append(0.7)
            
            # Volatility expansion check
            if detect_volatility_expansion(candles):
                score += 0.6
                reasons[f"{tf}m_volatility_expansion"] = True
                confidence_factors.append(0.6)
            
            # Enhanced pattern detection
            score, reasons, confidence_factors = enhanced_pattern_check_for_breakout(
                candles, tf, direction, score, reasons, confidence_factors
            )
            
            # Technical indicator confirmations
            
            # 1. MACD
            macd = detect_macd_cross(candles)
            if (macd == "bullish" and direction == "Long") or (macd == "bearish" and direction == "Short"):
                score += 0.8
                reasons[f"{tf}m_macd_confirmation"] = macd
                confidence_factors.append(0.7)
            
            # MACD Momentum
            macd_momentum = get_macd_momentum(candles)
            if abs(macd_momentum) > 0.6:
                if (macd_momentum > 0 and direction == "Long") or (macd_momentum < 0 and direction == "Short"):
                    score += 0.5
                    reasons[f"{tf}m_macd_momentum"] = macd_momentum
                    confidence_factors.append(0.6)
            
            # 2. RSI
            rsi_data = calculate_rsi_with_bands(candles)
            if rsi_data:
                rsi = rsi_data['rsi']
                rsi_signal, rsi_strength = get_rsi_signal(rsi_data)
                
                # RSI should not be at extremes for breakout
                if direction == "Long" and 40 < rsi < 70:
                    score += 0.5
                    reasons[f"{tf}m_rsi_favorable"] = rsi
                    confidence_factors.append(0.5)
                elif direction == "Short" and 30 < rsi < 60:
                    score += 0.5
                    reasons[f"{tf}m_rsi_favorable"] = rsi
                    confidence_factors.append(0.5)
                
                # RSI momentum
                if rsi_data.get('momentum'):
                    rsi_momentum = rsi_data['momentum']
                    if (rsi_momentum > 5 and direction == "Long") or (rsi_momentum < -5 and direction == "Short"):
                        score += 0.4
                        reasons[f"{tf}m_rsi_momentum"] = rsi_momentum
                        confidence_factors.append(0.5)
            
            # 3. EMA Analysis
            ema_cross = detect_ema_crossover(candles)
            if (ema_cross == "bullish" and direction == "Long") or (ema_cross == "bearish" and direction == "Short"):
                score += 0.7
                reasons[f"{tf}m_ema_cross"] = ema_cross
                confidence_factors.append(0.6)
            
            # EMA Ribbon
            ribbon = calculate_ema_ribbon(candles)
            ribbon_analysis = analyze_ema_ribbon(ribbon)
            if ribbon_analysis['trend'] != 'neutral':
                if (ribbon_analysis['trend'] == 'bullish' and direction == "Long") or \
                   (ribbon_analysis['trend'] == 'bearish' and direction == "Short"):
                    score += 0.6 * ribbon_analysis['strength']
                    reasons[f"{tf}m_ema_ribbon"] = ribbon_analysis['trend']
                    confidence_factors.append(ribbon_analysis['strength'])
            
            # 4. Bollinger Bands
            bb_signal = get_bollinger_signal(candles)
            if bb_signal['signal']:
                # Squeeze breakout is excellent for breakout trading
                if bb_signal['signal'] in ['squeeze_breakout_up', 'squeeze_breakout_down']:
                    if (bb_signal['signal'] == 'squeeze_breakout_up' and direction == "Long") or \
                       (bb_signal['signal'] == 'squeeze_breakout_down' and direction == "Short"):
                        score += 1.2 * bb_signal['strength']
                        reasons[f"{tf}m_bb_squeeze_breakout"] = True
                        confidence_factors.append(0.8)
                
                # Band walk confirms trend
                elif bb_signal.get('band_walk'):
                    walk_info = bb_signal['band_walk']
                    if (walk_info['walking_upper'] and direction == "Long") or \
                       (walk_info['walking_lower'] and direction == "Short"):
                        score += 0.8 * walk_info['strength']
                        reasons[f"{tf}m_band_walk"] = True
                        confidence_factors.append(0.7)
            
            # 5. Supertrend
            st_signal = calculate_supertrend_signal(candles)
            if (st_signal == "bullish" and direction == "Long") or (st_signal == "bearish" and direction == "Short"):
                score += 0.8
                reasons[f"{tf}m_supertrend"] = st_signal
                confidence_factors.append(0.7)
            
            # Supertrend State
            st_state = get_supertrend_state(candles)
            if st_state['consecutive_bars'] > 3:  # Trend persistence
                if (st_state['trend'] == 'up' and direction == "Long") or \
                   (st_state['trend'] == 'down' and direction == "Short"):
                    score += 0.5
                    reasons[f"{tf}m_trend_persistence"] = st_state['consecutive_bars']
                    confidence_factors.append(0.6)
            
            # 6. Whale Activity
            whale_activity = detect_whale_activity_advanced(candles, symbol)
            if whale_activity['detected']:
                if whale_activity['recommendation'] in ['potential_long', 'potential_short']:
                    whale_dir = "Long" if whale_activity['recommendation'] == 'potential_long' else "Short"
                    if whale_dir == direction:
                        score += 1.0 * whale_activity['strength']
                        reasons[f"{tf}m_whale_activity"] = whale_activity['recommendation']
                        confidence_factors.append(0.8)
        
        # Multi-timeframe confirmation bonus
        if len([r for r in reasons if 'resistance_break' in r or 'support_break' in r]) >= 2:
            score += 1.0
            reasons["mtf_breakout_confirmation"] = True
            confidence_factors.append(0.8)
        
        # Validate minimum requirements
        if score < MIN_BREAKOUT_SCORE:
            log(f"⚠️ Breakout score for {symbol} too low: {score:.2f} < {MIN_BREAKOUT_SCORE}")
            result = (0, None, 0, {"score_too_low": score})
            _breakout_cache[cache_key] = (time.time(), result)
            return result
        
        # Require at least 4 confirmation signals
        if len(reasons) < 4:
            log(f"⚠️ Breakout for {symbol} has insufficient indicators: {len(reasons)} < 4")
            result = (0, None, 0, {"insufficient_indicators": len(reasons)})
            _breakout_cache[cache_key] = (time.time(), result)
            return result
        
        # Ensure we have a direction
        if not direction:
            log(f"⚠️ Breakout for {symbol} has no clear direction")
            result = (0, None, 0, {"no_direction": True})
            _breakout_cache[cache_key] = (time.time(), result)
            return result
        
        # Calculate confidence
        base_confidence = (score / 8) * 100  # Base confidence from score
        
        # Adjust confidence based on confirmation factors
        if confidence_factors:
            avg_factor = np.mean(confidence_factors)
            confidence = base_confidence * avg_factor
        else:
            confidence = base_confidence
        
        # Apply analyzer's historical performance
        if _analyzer.success_rate > 0:
            confidence = confidence * (0.5 + _analyzer.success_rate * 0.5)
        
        confidence = round(min(confidence, 100), 1)
        
        log(f"✅ Valid breakout setup for {symbol}: Score {score:.2f}, Dir: {direction}, Conf: {confidence}%")
        log(f"   Reasons: {list(reasons.keys())}")
        
        # Cache the result
        result = (score, direction, confidence, reasons)
        _breakout_cache[cache_key] = (time.time(), result)
        
        return result
        
    except Exception as e:
        log(f"❌ Error in breakout scoring for {symbol}: {e}", level="ERROR")
        import traceback
        log(f"Stack trace: {traceback.format_exc()}", level="ERROR")
        return 0, None, 0, {"error": str(e)}

def clear_cache():
    """Clear the breakout cache"""
    global _breakout_cache
    _breakout_cache.clear()
    log("🧹 Cleared breakout cache")

# Export analyzer for performance tracking
def get_breakout_stats() -> Dict:
    """Get breakout strategy statistics"""
    return {
        "total_trades": _analyzer.total_trades,
        "successful_trades": _analyzer.successful_trades,
        "success_rate": _analyzer.success_rate,
        "cache_size": len(_breakout_cache)
    }

def update_breakout_performance(symbol: str, success: bool):
    """Update performance metrics for a breakout trade"""
    _analyzer.update_performance(symbol, success)
