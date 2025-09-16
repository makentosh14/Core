# enhanced_entry_validator.py - FIXED VERSION: Less strict for crypto volatility

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from logger import log
from volume import get_average_volume

class EntryValidator:
    """Enhanced entry validation to prevent late entries and improve timing"""
    
    def __init__(self):
        self.key_levels_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def validate_entry(self, symbol: str, candles_by_tf: Dict, direction: str, 
                      entry_price: float, trade_type: str, score: float) -> Tuple[bool, str]:
        """
        Master validation function that checks all entry criteria
        
        Returns:
            Tuple of (is_valid, reason)
        """
        # LESS STRICT: Only check critical validations
        
        # 1. Check for extreme exhaustion (only block if very exhausted)
        exhaustion = self.detect_exhaustion(candles_by_tf, direction)
        if exhaustion:
            # Only block if score is also low
            if score < 7:
                return False, "Price exhaustion with low score"
        
        # 2. Check momentum alignment (more lenient)
        momentum_valid, momentum_reason = self.check_momentum_alignment(
            candles_by_tf, direction, trade_type
        )
        if not momentum_valid:
            # Only block if score is low
            if score < 8:
                return False, momentum_reason
            
        # 4. Check key levels (only for lower scores)
        levels_valid, levels_reason = self.check_key_levels(
            symbol, candles_by_tf, entry_price, direction
        )
        if not levels_valid and score < 7.5:
            return False, levels_reason
            
        return True, "Validation passed"
    
    def check_momentum_alignment(self, candles_by_tf: Dict, direction: str, 
                               trade_type: str) -> Tuple[bool, str]:
        """Check if current momentum aligns with trade direction - LESS STRICT"""
        
        # REDUCED thresholds for crypto
        if trade_type == "Scalp":
            check_tfs = ["1", "3"]
            momentum_threshold = 0.5  # Increased from 0.3%
        elif trade_type == "Intraday":
            check_tfs = ["5", "15"]
            momentum_threshold = 0.8  # Increased from 0.5%
        else:  # Swing
            check_tfs = ["15", "30"]
            momentum_threshold = 1.2  # Increased from 0.8%
            
        # Check recent momentum
        adverse_count = 0
        for tf in check_tfs:
            if tf not in candles_by_tf:
                continue
                
            candles = candles_by_tf[tf]
            if len(candles) < 5:
                continue
                
            # Check last 3 candles for adverse movement
            recent_candles = candles[-3:]
            first_open = float(recent_candles[0]['open'])
            last_close = float(recent_candles[-1]['close'])
            
            move_pct = ((last_close - first_open) / first_open) * 100
            
            # Check if move is against our direction
            if direction == "Long" and move_pct < -momentum_threshold:
                adverse_count += 1
            elif direction == "Short" and move_pct > momentum_threshold:
                adverse_count += 1
        
        # Only fail if BOTH timeframes show adverse momentum
        if adverse_count >= len(check_tfs):
            return False, f"Strong adverse momentum on multiple timeframes"
                
        # Reduced exhaustion check
        if self._is_momentum_exhausted(candles_by_tf, direction):
            # Don't block, just warn
            log(f"⚠️ Momentum may be exhausted but allowing trade")
            
        return True, "Momentum acceptable"
    
    def _is_momentum_exhausted(self, candles_by_tf: Dict, direction: str) -> bool:
        """Check if momentum is exhausted - LESS STRICT"""
        
        candles_1m = candles_by_tf.get("1", [])
        if len(candles_1m) < 10:  # Increased from 7
            return False
            
        # Count consecutive candles in same direction
        consecutive = 0
        for i in range(-10, -1):  # Check last 10 instead of 7
            candle = candles_1m[i]
            close = float(candle['close'])
            open_price = float(candle['open'])
            
            # Only count significant body candles
            body_pct = abs(close - open_price) / close if close > 0 else 0
            if body_pct < 0.005:  # Increased from 0.003 (0.5% vs 0.3%)
                continue
            
            if direction == "Long" and close > open_price:
                consecutive += 1
            elif direction == "Short" and close < open_price:
                consecutive += 1
            else:
                consecutive = 0
        
        # Only exhausted if 10+ consecutive (up from 7)
        return consecutive >= 10
    
    def check_key_levels(self, symbol: str, candles_by_tf: Dict, 
                        entry_price: float, direction: str) -> Tuple[bool, str]:
        """Check if entry price is near key support/resistance levels - LESS STRICT"""
        
        # Get key levels
        levels = self.calculate_key_levels(symbol, candles_by_tf)
        
        if not levels:
            return True, "No key levels detected"
            
        # INCREASED distance requirement (less strict)
        min_distance_pct = 0.4  # Reduced from 0.3% to 0.2%
        
        for level_type, level_price in levels.items():
            distance_pct = abs((entry_price - level_price) / level_price) * 100
            
            if distance_pct < min_distance_pct:
                # Only block if we're on wrong side AND very close
                if direction == "Long" and entry_price > level_price and "resistance" in level_type:
                    if distance_pct < 0.1:  # Only if VERY close
                        return False, f"Too close to {level_type} at {level_price:.6f}"
                elif direction == "Short" and entry_price < level_price and "support" in level_type:
                    if distance_pct < 0.1:  # Only if VERY close
                        return False, f"Too close to {level_type} at {level_price:.6f}"
                    
        return True, "Acceptable distance from key levels"
    
    def calculate_key_levels(self, symbol: str, candles_by_tf: Dict) -> Dict[str, float]:
        """Calculate support and resistance levels - SIMPLIFIED"""
        
        # Check cache first
        cache_key = f"{symbol}_levels"
        if cache_key in self.key_levels_cache:
            cached_time, cached_levels = self.key_levels_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_levels
                
        levels = {}
        
        # Use 15m candles for key levels
        candles = candles_by_tf.get("15", candles_by_tf.get("5", []))
        if len(candles) < 20:  # Reduced from 50
            return levels
            
        # Get highs and lows
        highs = [float(c['high']) for c in candles[-20:]]
        lows = [float(c['low']) for c in candles[-20:]]
        
        # Only track major levels
        levels["recent_high"] = max(highs)
        levels["recent_low"] = min(lows)
        
        # Cache the results
        self.key_levels_cache[cache_key] = (datetime.now(), levels)
        
        return levels
    
    def check_timeframe_alignment(self, candles_by_tf: Dict, direction: str, 
                                 trade_type: str) -> Tuple[bool, str]:
        """DISABLED - Too strict for crypto"""
        # Always pass this check
        return True, "Timeframe check disabled"
    
    def check_market_structure(self, candles_by_tf: Dict, trade_type: str) -> Tuple[bool, str]:
        """Check if market structure is suitable - MUCH LESS STRICT"""
        
        # Use appropriate timeframe
        tf = "5" if trade_type == "Scalp" else "15" if trade_type == "Intraday" else "30"
        
        candles = candles_by_tf.get(tf, [])
        if len(candles) < 20:  # Reduced from 30
            return True, "Insufficient data for structure analysis"
            
        # Check for extremely tight range only
        closes = [float(c['close']) for c in candles[-20:]]
        highest = max(closes)
        lowest = min(closes)
        range_pct = ((highest - lowest) / lowest) * 100
        
        # Only fail if EXTREMELY choppy
        if range_pct < 0.5:  # Reduced from 1.0%
            return False, "Extremely tight range"
            
        return True, "Market structure acceptable"
    
    def detect_exhaustion(self, candles_by_tf: Dict, direction: str) -> bool:
        """Detect if price movement is exhausted - LESS STRICT"""
        
        candles_5m = candles_by_tf.get("5", candles_by_tf.get("3", []))
        if len(candles_5m) < 10:
            return False
            
        # Only check for extreme exhaustion
        recent_candles = candles_5m[-5:]
        
        # Check for multiple dojis (sign of exhaustion)
        doji_count = 0
        for candle in recent_candles:
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            open_price = float(candle['open'])
            
            body_size = abs(close - open_price)
            total_range = high - low
            
            if total_range > 0:
                body_ratio = body_size / total_range
                
                # Doji = small body
                if body_ratio < 0.3:  # Reduced from 0.3
                    doji_count += 1
        
        # Only exhausted if 4+ dojis out of 5 candles
        return doji_count >= 4

# Global validator instance
entry_validator = EntryValidator()
