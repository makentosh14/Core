# atr.py - Enhanced ATR calculation with performance optimizations
import asyncio
import traceback
import numpy as np
from typing import List, Dict, Optional, Union
from collections import deque
from datetime import datetime
from error_handler import send_error_to_telegram
from logger import log

# Cache for ATR calculations to avoid redundant computations
_atr_cache = {}
_cache_timestamps = {}
_cache_ttl = 300  # 5 minutes cache TTL
_cache_max_size = 1000  # Maximum cache entries

# Pre-calculated ATR values for incremental updates
_incremental_atr = {}

def _get_cache_key(symbol: str, period: int, smoothed: bool = False) -> str:
    """Generate cache key for ATR calculations"""
    return f"{symbol}_{period}_{'smoothed' if smoothed else 'simple'}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached value is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with new value and manage cache size"""
    _atr_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Clean up old cache entries if cache is getting too large
    if len(_atr_cache) > _cache_max_size:
        # Remove oldest 20% of entries
        oldest_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])[:int(_cache_max_size * 0.2)]
        for key, _ in oldest_keys:
            _atr_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def calculate_true_range_vectorized(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """
    Vectorized True Range calculation for better performance
    
    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        
    Returns:
        Array of True Range values
    """
    # Shift close prices by 1 to get previous close
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]  # First element has no previous, use current
    
    # Calculate three components of True Range
    hl = highs - lows  # High - Low
    hc = np.abs(highs - prev_closes)  # |High - Previous Close|
    lc = np.abs(lows - prev_closes)  # |Low - Previous Close|
    
    # True Range is the maximum of the three
    tr = np.maximum(hl, np.maximum(hc, lc))
    
    return tr

def calculate_atr(candles: List[Dict], period: int = 14, use_cache: bool = True) -> Optional[float]:
    """
    Calculate the Average True Range (ATR) from candles with performance optimizations
    
    Args:
        candles: List of candle dictionaries with 'high', 'low', 'close' keys
        period: ATR calculation period (default 14)
        use_cache: Whether to use caching (default True)
        
    Returns:
        float: ATR value or None if not enough candles
    """
    try:
        if not candles or len(candles) < period + 1:
            return None

        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache and len(candles) > 0:
            # Use last candle timestamp for cache key
            last_timestamp = candles[-1].get('timestamp', '')
            cache_key = f"atr_{period}_{last_timestamp}_{len(candles)}"
            
            # Check cache
            if cache_key in _atr_cache and _is_cache_valid(cache_key):
                return _atr_cache[cache_key]

        # Convert to numpy arrays for vectorized operations
        try:
            highs = np.array([float(candle['high']) for candle in candles], dtype=np.float64)
            lows = np.array([float(candle['low']) for candle in candles], dtype=np.float64)
            closes = np.array([float(candle['close']) for candle in candles], dtype=np.float64)
        except (ValueError, KeyError) as e:
            log(f"⚠️ Invalid candle data for ATR calculation: {e}", level="WARN")
            return None
        
        # Basic validation
        if np.any(highs < lows) or np.any(closes < 0):
            log("⚠️ Invalid price data detected in ATR calculation", level="WARN")
            return None
            
        # Calculate True Ranges using vectorized operations
        true_ranges = calculate_true_range_vectorized(highs, lows, closes)
        
        # Skip the first element as it doesn't have a valid previous close
        true_ranges = true_ranges[1:]
        
        if len(true_ranges) < period:
            return None
            
        # Calculate ATR as Simple Moving Average of True Ranges
        # Take the last 'period' true ranges
        recent_trs = true_ranges[-period:]
        atr = np.mean(recent_trs)
        
        # Round to appropriate precision (8 decimals for crypto)
        atr = round(float(atr), 8)
        
        # Update cache if enabled
        if use_cache and cache_key:
            _update_cache(cache_key, atr)
        
        return atr
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>ATR Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_smoothed_atr(candles: List[Dict], period: int = 14, use_cache: bool = True) -> Optional[float]:
    """
    Calculate Smoothed ATR using Wilder's smoothing method (more commonly used)
    
    Args:
        candles: List of candle dictionaries
        period: ATR calculation period
        use_cache: Whether to use caching
        
    Returns:
        float: Smoothed ATR value or None if not enough candles
    """
    try:
        if not candles or len(candles) < period * 2:  # Need more data for smoothing
            return None

        # Check cache if enabled
        cache_key = None
        if use_cache and len(candles) > 0:
            last_timestamp = candles[-1].get('timestamp', '')
            cache_key = f"atr_smoothed_{period}_{last_timestamp}_{len(candles)}"
            
            if cache_key in _atr_cache and _is_cache_valid(cache_key):
                return _atr_cache[cache_key]

        # Get basic data using numpy for performance
        highs = np.array([float(c['high']) for c in candles], dtype=np.float64)
        lows = np.array([float(c['low']) for c in candles], dtype=np.float64)
        closes = np.array([float(c['close']) for c in candles], dtype=np.float64)
        
        # Calculate True Ranges
        true_ranges = calculate_true_range_vectorized(highs, lows, closes)[1:]  # Skip first
        
        if len(true_ranges) < period:
            return None
            
        # First ATR is simple average of first 'period' TRs
        first_atr = np.mean(true_ranges[:period])
        
        # Apply Wilder's smoothing for subsequent values
        smoothed_atr = first_atr
        for i in range(period, len(true_ranges)):
            # Wilder's smoothing: ATR = ((previous_ATR * (period-1)) + current_TR) / period
            smoothed_atr = ((smoothed_atr * (period - 1)) + true_ranges[i]) / period
        
        smoothed_atr = round(float(smoothed_atr), 8)
        
        # Update cache if enabled
        if use_cache and cache_key:
            _update_cache(cache_key, smoothed_atr)
        
        return smoothed_atr
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Smoothed ATR Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_atr_percentage(candles: List[Dict], period: int = 14, smoothed: bool = False) -> Optional[float]:
    """
    Calculate ATR as a percentage of current price
    Useful for comparing volatility across different price levels
    
    Args:
        candles: List of candle dictionaries
        period: ATR calculation period
        smoothed: Whether to use smoothed ATR
        
    Returns:
        float: ATR percentage or None if calculation fails
    """
    try:
        # Calculate ATR
        if smoothed:
            atr = calculate_smoothed_atr(candles, period)
        else:
            atr = calculate_atr(candles, period)
            
        if not atr or not candles:
            return None
            
        current_price = float(candles[-1]['close'])
        if current_price <= 0:
            return None
            
        atr_percentage = (atr / current_price) * 100
        return round(atr_percentage, 4)
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>ATR Percentage Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_atr_bands(candles: List[Dict], period: int = 14, multiplier: float = 2.0) -> Optional[Dict]:
    """
    Calculate ATR-based bands (similar to Keltner Channels)
    
    Args:
        candles: List of candle dictionaries
        period: ATR calculation period
        multiplier: ATR multiplier for bands
        
    Returns:
        dict: Dictionary with upper, middle, lower bands or None
    """
    try:
        if not candles or len(candles) < period + 1:
            return None
            
        atr = calculate_atr(candles, period)
        if not atr:
            return None
            
        # Use EMA or SMA for middle band
        closes = np.array([float(c['close']) for c in candles[-period:]], dtype=np.float64)
        middle = np.mean(closes)  # Simple moving average
        
        upper = middle + (atr * multiplier)
        lower = middle - (atr * multiplier)
        
        current_close = float(candles[-1]['close'])
        position = (current_close - lower) / (upper - lower) if upper != lower else 0.5
        
        return {
            'upper': round(upper, 8),
            'middle': round(middle, 8),
            'lower': round(lower, 8),
            'atr': round(atr, 8),
            'position': round(position, 4),  # 0 = at lower band, 1 = at upper band
            'width': round(upper - lower, 8),
            'width_pct': round(((upper - lower) / middle * 100), 4) if middle > 0 else 0
        }
        
    except Exception as e:
        log(f"❌ Error calculating ATR bands: {e}", level="ERROR")
        return None

def calculate_atr_trailing_stop(entry_price: float, current_price: float, atr: float, 
                               direction: str, multiplier: float = 2.0) -> float:
    """
    Calculate trailing stop based on ATR
    
    Args:
        entry_price: Entry price of the position
        current_price: Current market price
        atr: Current ATR value
        direction: 'long' or 'short'
        multiplier: ATR multiplier for stop distance
        
    Returns:
        float: Trailing stop price
    """
    try:
        stop_distance = atr * multiplier
        
        if direction.lower() == 'long':
            # For long positions, stop is below current price
            stop_price = current_price - stop_distance
            # Don't let stop go below entry (optional)
            # stop_price = max(stop_price, entry_price - stop_distance)
        else:  # short
            # For short positions, stop is above current price
            stop_price = current_price + stop_distance
            # Don't let stop go above entry (optional)
            # stop_price = min(stop_price, entry_price + stop_distance)
            
        return round(stop_price, 8)
        
    except Exception as e:
        log(f"❌ Error calculating ATR trailing stop: {e}", level="ERROR")
        return entry_price  # Return entry as fallback

def get_volatility_regime(candles: List[Dict], lookback_periods: int = 20, 
                         short_period: int = 7, long_period: int = 30) -> Dict:
    """
    Determine current volatility regime based on ATR analysis
    
    Args:
        candles: List of candle dictionaries
        lookback_periods: Number of periods to analyze
        short_period: Short-term ATR period
        long_period: Long-term ATR period
        
    Returns:
        dict: Volatility regime information
    """
    try:
        if len(candles) < long_period + lookback_periods:
            return {'regime': 'unknown', 'ratio': 1.0, 'trend': 'stable'}
            
        # Calculate short and long-term ATR
        short_atr = calculate_atr(candles, short_period)
        long_atr = calculate_atr(candles, long_period)
        
        if not short_atr or not long_atr or long_atr == 0:
            return {'regime': 'unknown', 'ratio': 1.0, 'trend': 'stable'}
            
        # Calculate volatility ratio
        vol_ratio = short_atr / long_atr
        
        # Calculate historical ATR for trend
        historical_atrs = []
        for i in range(lookback_periods):
            historical_candles = candles[:-i-1] if i > 0 else candles
            hist_atr = calculate_atr(historical_candles, short_period)
            if hist_atr:
                historical_atrs.append(hist_atr)
                
        # Determine volatility trend
        if len(historical_atrs) >= 3:
            recent_avg = np.mean(historical_atrs[:3])
            older_avg = np.mean(historical_atrs[-3:])
            
            if recent_avg > older_avg * 1.2:
                trend = 'expanding'
            elif recent_avg < older_avg * 0.8:
                trend = 'contracting'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Determine regime
        if vol_ratio < 0.8:
            regime = 'low_volatility'
        elif vol_ratio > 1.3:
            regime = 'high_volatility'
        else:
            regime = 'normal_volatility'
            
        # Calculate percentile rank of current ATR
        if historical_atrs:
            percentile = (sum(1 for x in historical_atrs if x < short_atr) / len(historical_atrs)) * 100
        else:
            percentile = 50.0
            
        return {
            'regime': regime,
            'ratio': round(vol_ratio, 3),
            'trend': trend,
            'short_atr': round(short_atr, 8),
            'long_atr': round(long_atr, 8),
            'percentile': round(percentile, 1),
            'expanding': trend == 'expanding',
            'contracting': trend == 'contracting'
        }
        
    except Exception as e:
        log(f"❌ Error determining volatility regime: {e}", level="ERROR")
        return {'regime': 'unknown', 'ratio': 1.0, 'trend': 'stable'}

class IncrementalATR:
    """
    Class for incremental ATR calculation (more efficient for real-time updates)
    """
    def __init__(self, period: int = 14):
        self.period = period
        self.true_ranges = deque(maxlen=period)
        self.atr = None
        self.prev_close = None
        
    def update(self, high: float, low: float, close: float) -> Optional[float]:
        """Update ATR with new candle data"""
        if self.prev_close is None:
            self.prev_close = close
            return None
            
        # Calculate True Range
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
            # First ATR calculation
            self.atr = sum(self.true_ranges) / self.period
        else:
            # Wilder's smoothing
            self.atr = ((self.atr * (self.period - 1)) + tr) / self.period
            
        return round(self.atr, 8)
    
    def reset(self):
        """Reset the calculator"""
        self.true_ranges.clear()
        self.atr = None
        self.prev_close = None

def get_incremental_atr(symbol: str, period: int = 14) -> IncrementalATR:
    """Get or create incremental ATR calculator for a symbol"""
    key = f"{symbol}_{period}"
    if key not in _incremental_atr:
        _incremental_atr[key] = IncrementalATR(period)
    return _incremental_atr[key]

def update_incremental_atr(symbol: str, candle: Dict, period: int = 14) -> Optional[float]:
    """Update incremental ATR for a symbol with new candle"""
    try:
        calc = get_incremental_atr(symbol, period)
        return calc.update(
            float(candle['high']),
            float(candle['low']),
            float(candle['close'])
        )
    except Exception as e:
        log(f"❌ Error updating incremental ATR: {e}", level="ERROR")
        return None

# Cleanup functions
async def cleanup_cache_periodically():
    """Periodically clean up old cache entries"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            # Remove expired entries
            current_time = datetime.now()
            expired_keys = []
            
            for key, timestamp in _cache_timestamps.items():
                if (current_time - timestamp).total_seconds() > _cache_ttl:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                _atr_cache.pop(key, None)
                _cache_timestamps.pop(key, None)
                
            # Clean up incremental ATR calculators that haven't been used
            # This would require tracking last use time
            
            log(f"🧹 ATR cache cleanup: removed {len(expired_keys)} expired entries")
            
        except Exception as e:
            log(f"❌ Error in ATR cache cleanup: {e}", level="ERROR")

def clear_atr_cache(symbol: str = None):
    """Clear ATR cache for a specific symbol or all"""
    if symbol:
        # Clear entries for specific symbol
        keys_to_remove = [k for k in _atr_cache.keys() if symbol in k]
        for key in keys_to_remove:
            _atr_cache.pop(key, None)
            _cache_timestamps.pop(key, None)
            
        # Clear incremental calculators
        inc_keys_to_remove = [k for k in _incremental_atr.keys() if symbol in k]
        for key in inc_keys_to_remove:
            del _incremental_atr[key]
    else:
        # Clear all
        _atr_cache.clear()
        _cache_timestamps.clear()
        _incremental_atr.clear()
