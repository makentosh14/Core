# pattern_context_analyzer.py - Analyze where patterns occur

from typing import Dict, List, Optional, Tuple
from logger import log
import numpy as np

class PatternContextAnalyzer:
    """Analyze the context where patterns appear for better signal quality"""
    
    def analyze_pattern_context(self, pattern: str, candles: List[Dict], 
                               pattern_candle_idx: int = -1) -> Dict[str, any]:
        """
        Analyze where a pattern occurred and its significance
        
        Returns:
            Dictionary with context analysis
        """
        if not pattern or len(candles) < 20:
            return {"valid": True, "context": "unknown"}
            
        # Get pattern location
        pattern_candle = candles[pattern_candle_idx]
        pattern_price = float(pattern_candle['close'])
        
        # Analyze recent price action
        lookback_candles = candles[pattern_candle_idx-20:pattern_candle_idx]
        
        highs = [float(c['high']) for c in lookback_candles]
        lows = [float(c['low']) for c in lookback_candles]
        closes = [float(c['close']) for c in lookback_candles]
        
        recent_high = max(highs)
        recent_low = min(lows)
        
        # Determine pattern location
        price_position = (pattern_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        context = {
            "valid": True,
            "location": self._get_location_description(price_position),
            "price_position": price_position,
            "at_resistance": price_position > 0.8,
            "at_support": price_position < 0.2,
            "in_middle": 0.3 < price_position < 0.7,
            "trend_before": self._analyze_trend_before_pattern(lookback_candles),
            "volatility": self._calculate_volatility(lookback_candles),
            "strength_score": 1.0
        }
        
        # Adjust pattern validity based on context
        context["valid"], context["reason"] = self._validate_pattern_context(
            pattern, context, candles[pattern_candle_idx]
        )
        
        # Calculate context-adjusted strength
        context["strength_score"] = self._calculate_context_strength(pattern, context)
        
        return context
    
    def _get_location_description(self, price_position: float) -> str:
        """Get human-readable location description"""
        if price_position > 0.9:
            return "at_extreme_high"
        elif price_position > 0.7:
            return "near_resistance"
        elif price_position < 0.1:
            return "at_extreme_low"
        elif price_position < 0.3:
            return "near_support"
        else:
            return "mid_range"
    
    def _analyze_trend_before_pattern(self, candles: List[Dict]) -> str:
        """Analyze trend leading up to pattern"""
        if len(candles) < 5:
            return "unknown"
            
        closes = [float(c['close']) for c in candles]
        
        # Simple linear regression
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        avg_close = np.mean(closes)
        
        # Normalize slope
        normalized_slope = (slope / avg_close) * 100
        
        if normalized_slope > 0.5:
            return "strong_up"
        elif normalized_slope > 0.1:
            return "up"
        elif normalized_slope < -0.5:
            return "strong_down"
        elif normalized_slope < -0.1:
            return "down"
        else:
            return "sideways"
    
    def _calculate_volatility(self, candles: List[Dict]) -> str:
        """Calculate recent volatility"""
        if len(candles) < 5:
            return "normal"
            
        # Calculate true ranges
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i]['high'])
            low = float(candles[i]['low'])
            prev_close = float(candles[i-1]['close'])
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
            
        avg_tr = np.mean(true_ranges)
        recent_tr = np.mean(true_ranges[-3:]) if len(true_ranges) >= 3 else avg_tr
        
        vol_ratio = recent_tr / avg_tr if avg_tr > 0 else 1.0
        
        if vol_ratio > 1.5:
            return "high"
        elif vol_ratio < 0.7:
            return "low"
        else:
            return "normal"
    
    def _validate_pattern_context(self, pattern: str, context: Dict, 
                                 pattern_candle: Dict) -> Tuple[bool, str]:
        """Validate if pattern is valid in its context"""
        
        # Bullish patterns
        bullish_patterns = ["hammer", "bullish_engulfing", "morning_star", 
                           "piercing_line", "tweezer_bottom"]
        
        # Bearish patterns
        bearish_patterns = ["shooting_star", "bearish_engulfing", "evening_star",
                           "dark_cloud_cover", "hanging_man", "tweezer_top"]
        
        # Check pattern-context alignment
        if pattern in bullish_patterns:
            # Bullish patterns should appear at/near support or after downtrend
            if context["at_resistance"]:
                return False, "Bullish pattern at resistance"
            if context["trend_before"] == "strong_up":
                return False, "Bullish pattern after strong uptrend"
                
        elif pattern in bearish_patterns:
            # Bearish patterns should appear at/near resistance or after uptrend
            if context["at_support"]:
                return False, "Bearish pattern at support"
            if context["trend_before"] == "strong_down":
                return False, "Bearish pattern after strong downtrend"
        
        # Neutral patterns are valid anywhere
        return True, "Valid context"
    
    def _calculate_context_strength(self, pattern: str, context: Dict) -> float:
        """Calculate pattern strength based on context"""
        
        strength = 1.0
        
        # Location bonuses/penalties
        if context["at_support"] and "bullish" in pattern:
            strength *= 1.3
        elif context["at_resistance"] and "bearish" in pattern:
            strength *= 1.3
        elif context["in_middle"]:
            strength *= 0.8
            
        # Trend alignment
        if context["trend_before"] == "strong_down" and "bullish" in pattern:
            strength *= 1.2  # Reversal potential
        elif context["trend_before"] == "strong_up" and "bearish" in pattern:
            strength *= 1.2  # Reversal potential
        elif context["trend_before"] == "sideways":
            strength *= 0.9  # Less reliable in ranging markets
            
        # Volatility adjustment
        if context["volatility"] == "high":
            strength *= 0.9  # Patterns less reliable in high volatility
        elif context["volatility"] == "low":
            strength *= 1.1  # Patterns more reliable in low volatility
            
        return min(strength, 2.0)  # Cap at 2.0

# Global instance
pattern_context_analyzer = PatternContextAnalyzer()
