# supertrend.py - Enhanced Supertrend Indicator with Performance Optimizations

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
import traceback
from error_handler import send_error_to_telegram
from logger import log

# Cache for Supertrend calculations to avoid redundant computations
_supertrend_cache = {}
_cache_max_size = 100  # Maximum number of cached calculations
_cache_timestamps = {}

# Default parameters
DEFAULT_PERIOD = 10
DEFAULT_MULTIPLIER = 3.0

def calculate_atr_optimized(candles: List[Dict], period: int = 10) -> Optional[np.ndarray]:
    """
    Optimized ATR calculation using numpy for better performance
    
    Args:
        candles: List of candle dictionaries
        period: ATR period
        
    Returns:
        numpy array of ATR values or None if not enough data
    """
    try:
        if len(candles) < period + 1:
            return None
        
        # Convert to numpy arrays for vectorized operations
        highs = np.array([float(c["high"]) for c in candles])
        lows = np.array([float(c["low"]) for c in candles])
        closes = np.array([float(c["close"]) for c in candles])
        
        # Calculate True Range components
        high_low = highs - lows
        high_close = np.abs(highs[1:] - closes[:-1])
        low_close = np.abs(lows[1:] - closes[:-1])
        
        # Combine TR components
        true_ranges = np.zeros(len(candles))
        true_ranges[0] = high_low[0]  # First TR is just high-low
        
        # Vectorized TR calculation
        true_ranges[1:] = np.maximum(high_low[1:], np.maximum(high_close, low_close))
        
        # Calculate ATR using Wilder's smoothing
        atr = np.zeros(len(candles))
        atr[:period] = np.nan  # Not enough data for these indices
        
        # Initial ATR is simple average of first 'period' TRs
        atr[period] = np.mean(true_ranges[1:period+1])
        
        # Apply Wilder's smoothing for subsequent values
        smoothing_factor = 1.0 / period
        for i in range(period + 1, len(candles)):
            atr[i] = atr[i-1] * (1 - smoothing_factor) + true_ranges[i] * smoothing_factor
        
        return atr
        
    except Exception as e:
        log(f"❌ Error in optimized ATR calculation: {e}", level="ERROR")
        return None

def calculate_supertrend(candles: List[Dict], period: int = DEFAULT_PERIOD, 
                        multiplier: float = DEFAULT_MULTIPLIER) -> Optional[List[Dict]]:
    """
    Enhanced Supertrend calculation with caching and performance optimizations
    
    Args:
        candles: List of candle dictionaries
        period: ATR period for Supertrend
        multiplier: ATR multiplier for band calculation
        
    Returns:
        List of dictionaries with Supertrend values and signals
    """
    try:
        if len(candles) < period + 2:
            return None
        
        # Create cache key
        cache_key = f"{len(candles)}_{period}_{multiplier}_{candles[-1].get('timestamp', '')}"
        
        # Check cache
        if cache_key in _supertrend_cache:
            return _supertrend_cache[cache_key]
        
        # Calculate ATR
        atr_values = calculate_atr_optimized(candles, period)
        if atr_values is None:
            return None
        
        # Convert prices to numpy arrays
        highs = np.array([float(c["high"]) for c in candles])
        lows = np.array([float(c["low"]) for c in candles])
        closes = np.array([float(c["close"]) for c in candles])
        
        # Calculate HL/2 (average price)
        hl_avg = (highs + lows) / 2
        
        # Initialize arrays
        basic_upper = np.zeros(len(candles))
        basic_lower = np.zeros(len(candles))
        final_upper = np.zeros(len(candles))
        final_lower = np.zeros(len(candles))
        supertrend = np.zeros(len(candles))
        trend = np.zeros(len(candles), dtype=int)  # 1 for up, -1 for down
        
        # Calculate basic bands
        basic_upper = hl_avg + multiplier * atr_values
        basic_lower = hl_avg - multiplier * atr_values
        
        # Initialize first values
        final_upper[period] = basic_upper[period]
        final_lower[period] = basic_lower[period]
        trend[period] = 1 if closes[period] > final_upper[period] else -1
        
        # Calculate Supertrend
        for i in range(period + 1, len(candles)):
            # Final upper band
            if basic_upper[i] < final_upper[i-1] or closes[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Final lower band
            if basic_lower[i] > final_lower[i-1] or closes[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # Determine trend
            if trend[i-1] == 1:  # Previous trend was up
                if closes[i] <= final_lower[i]:
                    trend[i] = -1
                else:
                    trend[i] = 1
            else:  # Previous trend was down
                if closes[i] >= final_upper[i]:
                    trend[i] = 1
                else:
                    trend[i] = -1
            
            # Set Supertrend value
            if trend[i] == 1:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]
        
        # Build result list
        result = []
        for i in range(period, len(candles)):
            # Detect trend changes
            trend_changed = False
            signal = None
            
            if i > period:
                if trend[i] != trend[i-1]:
                    trend_changed = True
                    signal = "bullish" if trend[i] == 1 else "bearish"
            
            result.append({
                'timestamp': candles[i].get('timestamp'),
                'supertrend': round(supertrend[i], 8),
                'upper_band': round(final_upper[i], 8),
                'lower_band': round(final_lower[i], 8),
                'trend': 'up' if trend[i] == 1 else 'down',
                'trend_value': int(trend[i]),
                'signal': signal,
                'trend_changed': trend_changed,
                'atr': round(atr_values[i], 8),
                'close': closes[i],
                'strength': calculate_trend_strength(closes[i], supertrend[i], atr_values[i])
            })
        
        # Cache management
        if len(_supertrend_cache) >= _cache_max_size:
            # Remove oldest entries
            oldest_keys = list(_supertrend_cache.keys())[:20]
            for key in oldest_keys:
                _supertrend_cache.pop(key, None)
        
        # Store in cache
        _supertrend_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Supertrend Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_trend_strength(close: float, supertrend: float, atr: float) -> float:
    """
    Calculate the strength of the current trend
    
    Args:
        close: Current closing price
        supertrend: Current Supertrend value
        atr: Current ATR value
        
    Returns:
        float: Trend strength (0-1)
    """
    if atr == 0:
        return 0.5
    
    # Distance from Supertrend line as multiple of ATR
    distance = abs(close - supertrend)
    atr_multiples = distance / atr
    
    # Normalize to 0-1 range (cap at 3 ATRs)
    strength = min(atr_multiples / 3, 1.0)
    
    return round(strength, 3)

def calculate_supertrend_signal(candles: List[Dict], period: int = DEFAULT_PERIOD, 
                               multiplier: float = DEFAULT_MULTIPLIER) -> Optional[str]:
    """
    Simplified function that returns just the signal for backward compatibility
    
    Returns:
        'bullish', 'bearish', or None
    """
    try:
        result = calculate_supertrend(candles, period, multiplier)
        
        if not result or len(result) < 2:
            return None
        
        # Check the last entry for a signal
        latest = result[-1]
        return latest.get('signal')
        
    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Supertrend Signal Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def get_supertrend_state(candles: List[Dict], period: int = DEFAULT_PERIOD, 
                        multiplier: float = DEFAULT_MULTIPLIER) -> Dict:
    """
    Get comprehensive Supertrend state information
    
    Returns:
        Dictionary with current trend, strength, and recent signals
    """
    try:
        result = calculate_supertrend(candles, period, multiplier)
        
        if not result:
            return {
                'trend': None,
                'strength': 0,
                'signal': None,
                'consecutive_bars': 0,
                'distance_from_line': 0
            }
        
        latest = result[-1]
        
        # Count consecutive bars in same trend
        consecutive = 1
        current_trend = latest['trend_value']
        
        for i in range(len(result) - 2, -1, -1):
            if result[i]['trend_value'] == current_trend:
                consecutive += 1
            else:
                break
        
        # Calculate distance from Supertrend line as percentage
        distance_pct = abs(latest['close'] - latest['supertrend']) / latest['close'] * 100
        
        return {
            'trend': latest['trend'],
            'strength': latest['strength'],
            'signal': latest.get('signal'),
            'consecutive_bars': consecutive,
            'distance_from_line': round(distance_pct, 2),
            'supertrend_value': latest['supertrend'],
            'upper_band': latest['upper_band'],
            'lower_band': latest['lower_band']
        }
        
    except Exception as e:
        log(f"❌ Error getting Supertrend state: {e}", level="ERROR")
        return {
            'trend': None,
            'strength': 0,
            'signal': None,
            'consecutive_bars': 0,
            'distance_from_line': 0
        }

def detect_supertrend_squeeze(candles: List[Dict], period: int = DEFAULT_PERIOD, 
                             multiplier: float = DEFAULT_MULTIPLIER, 
                             lookback: int = 20) -> Dict:
    """
    Detect when price is consolidating near Supertrend line (potential breakout setup)
    
    Returns:
        Dictionary with squeeze information
    """
    try:
        result = calculate_supertrend(candles, period, multiplier)
        
        if not result or len(result) < lookback:
            return {'squeeze': False, 'duration': 0, 'avg_distance': 0}
        
        # Check recent bars for proximity to Supertrend
        squeeze_count = 0
        distances = []
        
        for i in range(len(result) - lookback, len(result)):
            distance_pct = abs(result[i]['close'] - result[i]['supertrend']) / result[i]['close'] * 100
            distances.append(distance_pct)
            
            # Consider it a squeeze if within 0.5% of Supertrend
            if distance_pct < 0.5:
                squeeze_count += 1
        
        avg_distance = np.mean(distances)
        is_squeeze = squeeze_count >= lookback * 0.6  # 60% of bars near Supertrend
        
        return {
            'squeeze': is_squeeze,
            'duration': squeeze_count,
            'avg_distance': round(avg_distance, 2),
            'intensity': round(1 - (avg_distance / 2), 2) if avg_distance < 2 else 0
        }
        
    except Exception as e:
        log(f"❌ Error detecting Supertrend squeeze: {e}", level="ERROR")
        return {'squeeze': False, 'duration': 0, 'avg_distance': 0}

def calculate_multi_timeframe_supertrend(candles_by_tf: Dict[str, List[Dict]], 
                                        periods: Dict[str, int] = None,
                                        multipliers: Dict[str, float] = None) -> Dict:
    """
    Calculate Supertrend across multiple timeframes for confluence
    
    Args:
        candles_by_tf: Dictionary of candles by timeframe
        periods: Custom periods by timeframe
        multipliers: Custom multipliers by timeframe
        
    Returns:
        Dictionary with multi-timeframe analysis
    """
    if periods is None:
        periods = {
            "1": 7,
            "5": 10,
            "15": 10,
            "60": 10
        }
    
    if multipliers is None:
        multipliers = {
            "1": 2.0,
            "5": 2.5,
            "15": 3.0,
            "60": 3.0
        }
    
    results = {}
    trends = []
    
    for tf, candles in candles_by_tf.items():
        if tf in periods and candles:
            period = periods.get(tf, DEFAULT_PERIOD)
            multiplier = multipliers.get(tf, DEFAULT_MULTIPLIER)
            
            state = get_supertrend_state(candles, period, multiplier)
            if state['trend']:
                results[tf] = state
                trends.append(1 if state['trend'] == 'up' else -1)
    
    # Calculate confluence
    if trends:
        avg_trend = np.mean(trends)
        trend_alignment = abs(avg_trend)  # 0 = mixed, 1 = all aligned
        
        overall_trend = 'up' if avg_trend > 0.3 else 'down' if avg_trend < -0.3 else 'mixed'
    else:
        trend_alignment = 0
        overall_trend = 'unknown'
    
    return {
        'timeframes': results,
        'overall_trend': overall_trend,
        'alignment': round(trend_alignment, 2),
        'bullish_count': sum(1 for t in trends if t > 0),
        'bearish_count': sum(1 for t in trends if t < 0)
    }

def get_supertrend_exit_signal(candles: List[Dict], position_type: str,
                              period: int = DEFAULT_PERIOD, 
                              multiplier: float = DEFAULT_MULTIPLIER) -> Tuple[bool, str]:
    """
    Determine if position should be exited based on Supertrend
    
    Args:
        candles: List of candle dictionaries
        position_type: 'long' or 'short'
        period: Supertrend period
        multiplier: Supertrend multiplier
        
    Returns:
        Tuple of (should_exit, reason)
    """
    try:
        result = calculate_supertrend(candles, period, multiplier)
        
        if not result or len(result) < 2:
            return False, "Insufficient data"
        
        latest = result[-1]
        previous = result[-2]
        
        # Check for trend reversal
        if latest['signal']:
            if position_type == 'long' and latest['signal'] == 'bearish':
                return True, "Supertrend turned bearish"
            elif position_type == 'short' and latest['signal'] == 'bullish':
                return True, "Supertrend turned bullish"
        
        # Check if price crossed Supertrend line without full reversal (early exit)
        if position_type == 'long':
            if latest['close'] < latest['supertrend'] and previous['close'] >= previous['supertrend']:
                return True, "Price crossed below Supertrend"
        else:  # short
            if latest['close'] > latest['supertrend'] and previous['close'] <= previous['supertrend']:
                return True, "Price crossed above Supertrend"
        
        return False, "No exit signal"
        
    except Exception as e:
        log(f"❌ Error checking Supertrend exit: {e}", level="ERROR")
        return False, f"Error: {str(e)}"

# Clear cache periodically to prevent memory issues
async def clear_supertrend_cache_periodically():
    """Clear the Supertrend cache every hour"""
    while True:
        await asyncio.sleep(3600)  # Wait 1 hour
        _supertrend_cache.clear()
        log("🧹 Cleared Supertrend cache")

# Incremental Supertrend calculator for real-time updates
class SupertrendCalculator:
    """
    Incremental Supertrend calculator for efficient real-time updates
    """
    def __init__(self, period: int = DEFAULT_PERIOD, multiplier: float = DEFAULT_MULTIPLIER):
        self.period = period
        self.multiplier = multiplier
        self.candles = deque(maxlen=period * 2)  # Keep enough history
        self.atr_calculator = ATRCalculator(period)
        self.final_upper = None
        self.final_lower = None
        self.trend = None
        self.supertrend = None
        
    def update(self, candle: Dict) -> Optional[Dict]:
        """
        Update Supertrend with new candle
        
        Args:
            candle: Dictionary with OHLC data
            
        Returns:
            Dictionary with current Supertrend state or None
        """
        self.candles.append(candle)
        
        if len(self.candles) < self.period + 1:
            return None
        
        # Update ATR
        atr = self.atr_calculator.update(candle)
        if atr is None:
            return None
        
        # Calculate HL/2
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        hl_avg = (high + low) / 2
        
        # Calculate basic bands
        basic_upper = hl_avg + self.multiplier * atr
        basic_lower = hl_avg - self.multiplier * atr
        
        # Initialize if first calculation
        if self.final_upper is None:
            self.final_upper = basic_upper
            self.final_lower = basic_lower
            self.trend = 1 if close > self.final_upper else -1
            self.supertrend = self.final_lower if self.trend == 1 else self.final_upper
            return None
        
        # Update final bands
        prev_close = float(self.candles[-2]['close'])
        
        # Final upper band
        if basic_upper < self.final_upper or prev_close > self.final_upper:
            self.final_upper = basic_upper
        
        # Final lower band
        if basic_lower > self.final_lower or prev_close < self.final_lower:
            self.final_lower = basic_lower
        
        # Determine trend
        prev_trend = self.trend
        
        if self.trend == 1:  # Currently up
            if close <= self.final_lower:
                self.trend = -1
        else:  # Currently down
            if close >= self.final_upper:
                self.trend = 1
        
        # Set Supertrend value
        self.supertrend = self.final_lower if self.trend == 1 else self.final_upper
        
        # Detect signal
        signal = None
        if self.trend != prev_trend:
            signal = "bullish" if self.trend == 1 else "bearish"
        
        return {
            'supertrend': round(self.supertrend, 8),
            'trend': 'up' if self.trend == 1 else 'down',
            'signal': signal,
            'upper_band': round(self.final_upper, 8),
            'lower_band': round(self.final_lower, 8),
            'atr': round(atr, 8),
            'strength': calculate_trend_strength(close, self.supertrend, atr)
        }

class ATRCalculator:
    """
    Incremental ATR calculator for Supertrend
    """
    def __init__(self, period: int):
        self.period = period
        self.true_ranges = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        
    def update(self, candle: Dict) -> Optional[float]:
        """Update ATR with new candle"""
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close)
            )
        
        self.true_ranges.append(tr)
        self.prev_close = close
        
        if len(self.true_ranges) < self.period:
            return None
        
        # Calculate or update ATR
        if self.atr is None:
            self.atr = sum(self.true_ranges) / self.period
        else:
            # Wilder's smoothing
            self.atr = (self.atr * (self.period - 1) + tr) / self.period
        
        return self.atr
