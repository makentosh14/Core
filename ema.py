# ema.py - Enhanced EMA with performance optimizations and advanced features

import asyncio
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import traceback
from error_handler import send_error_to_telegram
from logger import log

# Cache for EMA calculations to avoid redundant computations
_ema_cache = {}
_cache_max_size = 200  # Increased cache size
_cache_timestamps = {}
_cache_ttl = 300  # 5 minutes TTL for cache entries

# Pre-calculated EMA alphas for common periods to avoid repeated calculations
_alpha_cache = {}

class EMACalculator:
    """
    Optimized incremental EMA calculator for real-time updates
    """
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.ema = None
        self.count = 0
        
    def update(self, value: float) -> Optional[float]:
        """Update EMA with new value"""
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (value - self.ema) * self.alpha + self.ema
        
        self.count += 1
        return self.ema if self.count >= self.period else None

# Cache for incremental EMA calculators
_incremental_emas = {}

def get_alpha(period: int) -> float:
    """Get EMA alpha value with caching"""
    if period not in _alpha_cache:
        _alpha_cache[period] = 2.0 / (period + 1.0)
    return _alpha_cache[period]

def calculate_ema_vectorized(values: np.ndarray, period: int) -> np.ndarray:
    """
    Highly optimized vectorized EMA calculation using NumPy
    
    Args:
        values: NumPy array of price values
        period: EMA period
        
    Returns:
        NumPy array of EMA values
    """
    if len(values) < 1:
        return np.array([])
    
    alpha = get_alpha(period)
    ema = np.zeros_like(values)
    
    # Initialize first value
    ema[0] = values[0]
    
    # Vectorized calculation using NumPy's exponential weighted functions
    weights = np.power(1 - alpha, np.arange(len(values)))
    weights = weights[::-1]  # Reverse for proper time ordering
    
    # Use NumPy's convolve for efficient calculation
    for i in range(1, len(values)):
        if i < period:
            # Simple average for initial period
            ema[i] = np.mean(values[:i+1])
        else:
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

def _create_cache_key(values_len: int, period: int, last_values: List[float]) -> str:
    """Create a unique cache key"""
    # Use length, period, and hash of last few values for uniqueness
    last_vals_str = ','.join(f"{v:.6f}" for v in last_values[-5:])
    return f"{values_len}_{period}_{hash(last_vals_str)}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with TTL management"""
    _ema_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Clean up old cache entries if needed
    if len(_ema_cache) > _cache_max_size:
        # Remove oldest 20% of entries
        sorted_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])
        for key, _ in sorted_keys[:_cache_max_size // 5]:
            _ema_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def calculate_ema(candles: Union[List[Dict], List[float]], period: int, 
                  price_field: str = 'close', use_cache: bool = True) -> List[float]:
    """
    Enhanced EMA calculation with multiple optimizations
    
    Args:
        candles: List of candle dicts or price values
        period: EMA period
        price_field: Which price field to use (for candle dicts)
        use_cache: Whether to use caching
        
    Returns:
        List of EMA values
    """
    try:
        # Extract values based on input type
        if isinstance(candles, list) and len(candles) > 0:
            if isinstance(candles[0], dict):
                values = [float(c.get(price_field, c.get('close', 0))) for c in candles]
            else:
                values = [float(v) for v in candles]
        else:
            return []
        
        if len(values) < period:
            return []
        
        # Check cache if enabled
        cache_key = _create_cache_key(len(values), period, values) if use_cache else None
        
        if use_cache and cache_key and cache_key in _ema_cache and _is_cache_valid(cache_key):
            return _ema_cache[cache_key]
        
        # Use vectorized calculation for performance
        values_array = np.array(values, dtype=np.float64)
        ema_array = calculate_ema_vectorized(values_array, period)
        ema_list = ema_array.tolist()
        
        # Update cache if enabled
        if use_cache and cache_key:
            _update_cache(cache_key, ema_list)
        
        return ema_list

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>EMA Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return []

def calculate_dema(candles: List[Dict], period: int, price_field: str = 'close') -> List[float]:
    """
    Calculate Double EMA (DEMA) for reduced lag
    DEMA = 2 * EMA - EMA(EMA)
    """
    try:
        ema1 = calculate_ema(candles, period, price_field)
        if not ema1 or len(ema1) < period:
            return []
        
        # Create synthetic candles for second EMA calculation
        ema_candles = [{'close': v} for v in ema1]
        ema2 = calculate_ema(ema_candles, period)
        
        if not ema2:
            return []
        
        # Calculate DEMA
        dema = []
        for i in range(len(ema2)):
            if i < len(ema1):
                dema_value = 2 * ema1[i] - ema2[i]
                dema.append(round(dema_value, 8))
        
        return dema
        
    except Exception as e:
        log(f"❌ Error calculating DEMA: {e}", level="ERROR")
        return []

def calculate_tema(candles: List[Dict], period: int, price_field: str = 'close') -> List[float]:
    """
    Calculate Triple EMA (TEMA) for even less lag
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    """
    try:
        ema1 = calculate_ema(candles, period, price_field)
        if not ema1 or len(ema1) < period:
            return []
        
        # Calculate EMA of EMA
        ema2_candles = [{'close': v} for v in ema1]
        ema2 = calculate_ema(ema2_candles, period)
        
        if not ema2 or len(ema2) < period:
            return []
        
        # Calculate EMA of EMA of EMA
        ema3_candles = [{'close': v} for v in ema2]
        ema3 = calculate_ema(ema3_candles, period)
        
        if not ema3:
            return []
        
        # Calculate TEMA
        tema = []
        for i in range(len(ema3)):
            if i < len(ema1) and i < len(ema2):
                tema_value = 3 * ema1[i] - 3 * ema2[i] + ema3[i]
                tema.append(round(tema_value, 8))
        
        return tema
        
    except Exception as e:
        log(f"❌ Error calculating TEMA: {e}", level="ERROR")
        return []

def detect_ema_crossover(candles: List[Dict], fast_period: int = 9, slow_period: int = 21,
                        confirmation_candles: int = 1, min_separation: float = 0.0001) -> Optional[str]:
    """
    Enhanced EMA crossover detection with confirmation and separation filters
    
    Args:
        candles: List of candle dictionaries
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        confirmation_candles: Number of candles to confirm crossover
        min_separation: Minimum separation between EMAs to confirm signal
        
    Returns:
        "bullish", "bearish", or None
    """
    try:
        # Use vectorized calculation for better performance
        values = np.array([float(c['close']) for c in candles])
        
        if len(values) < slow_period + confirmation_candles:
            return None
        
        # Calculate EMAs efficiently
        fast_ema = calculate_ema_vectorized(values, fast_period)
        slow_ema = calculate_ema_vectorized(values, slow_period)
        
        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return None
        
        # Check for crossover with confirmation
        bullish_cross = False
        bearish_cross = False
        
        # Look at the last few candles for confirmation
        for i in range(confirmation_candles):
            idx = -(i + 1)
            prev_idx = -(i + 2)
            
            if abs(prev_idx) > len(fast_ema) or abs(prev_idx) > len(slow_ema):
                continue
            
            # Calculate separation
            separation = abs(fast_ema[idx] - slow_ema[idx]) / slow_ema[idx] if slow_ema[idx] != 0 else 0
            
            # Check for bullish crossover
            if (fast_ema[prev_idx] <= slow_ema[prev_idx] and 
                fast_ema[idx] > slow_ema[idx] and 
                separation >= min_separation):
                bullish_cross = True
                
            # Check for bearish crossover
            elif (fast_ema[prev_idx] >= slow_ema[prev_idx] and 
                  fast_ema[idx] < slow_ema[idx] and 
                  separation >= min_separation):
                bearish_cross = True
        
        # Additional trend strength confirmation
        if bullish_cross:
            # Check if fast EMA is accelerating upward
            if len(fast_ema) >= 3:
                acceleration = fast_ema[-1] - fast_ema[-3]
                if acceleration > 0:
                    return "bullish"
                    
        elif bearish_cross:
            # Check if fast EMA is accelerating downward
            if len(fast_ema) >= 3:
                acceleration = fast_ema[-1] - fast_ema[-3]
                if acceleration < 0:
                    return "bearish"
        
        return None

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>EMA Crossover Detection Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_ema_ribbon(candles: List[Dict], periods: List[int] = None, 
                        price_field: str = 'close') -> Dict[int, List[float]]:
    """
    Calculate multiple EMAs to form a ribbon for trend analysis
    
    Args:
        candles: List of candle dictionaries
        periods: List of EMA periods (default: [5, 8, 13, 21, 34, 55])
        price_field: Price field to use
        
    Returns:
        Dictionary mapping period to EMA values
    """
    if periods is None:
        periods = [5, 8, 13, 21, 34, 55]  # Fibonacci sequence
    
    ribbon = {}
    
    # Pre-extract values once for efficiency
    values = [float(c.get(price_field, c.get('close', 0))) for c in candles]
    values_array = np.array(values, dtype=np.float64)
    
    # Calculate all EMAs in parallel if possible
    for period in sorted(periods):
        if len(values) >= period:
            ema_values = calculate_ema_vectorized(values_array, period)
            ribbon[period] = ema_values.tolist()
        else:
            ribbon[period] = []
    
    return ribbon

def analyze_ema_ribbon(ribbon: Dict[int, List[float]], lookback: int = 5) -> Dict[str, any]:
    """
    Analyze EMA ribbon for trend strength and direction
    
    Args:
        ribbon: Dictionary of EMAs from calculate_ema_ribbon
        lookback: Number of periods to analyze
        
    Returns:
        Dictionary with ribbon analysis
    """
    if not ribbon or not any(ribbon.values()):
        return {'trend': 'neutral', 'strength': 0, 'convergence': 0}
    
    # Get the latest values for each EMA
    current_values = {}
    for period, values in ribbon.items():
        if values and len(values) > 0:
            current_values[period] = values[-1]
    
    if len(current_values) < 2:
        return {'trend': 'neutral', 'strength': 0, 'convergence': 0}
    
    # Sort periods
    sorted_periods = sorted(current_values.keys())
    
    # Check if EMAs are in order (bullish or bearish alignment)
    ema_values = [current_values[p] for p in sorted_periods]
    
    # Calculate trend direction
    is_bullish = all(ema_values[i] > ema_values[i+1] for i in range(len(ema_values)-1))
    is_bearish = all(ema_values[i] < ema_values[i+1] for i in range(len(ema_values)-1))
    
    # Calculate ribbon width (convergence/divergence)
    ribbon_width = max(ema_values) - min(ema_values)
    avg_price = sum(ema_values) / len(ema_values)
    ribbon_width_pct = (ribbon_width / avg_price * 100) if avg_price > 0 else 0
    
    # Calculate trend strength based on alignment and separation
    if is_bullish:
        trend = 'bullish'
        strength = min(1.0, ribbon_width_pct / 2)  # Max strength at 2% separation
    elif is_bearish:
        trend = 'bearish'
        strength = min(1.0, ribbon_width_pct / 2)
    else:
        trend = 'neutral'
        strength = 0
    
    # Check for ribbon compression (potential breakout setup)
    compression = ribbon_width_pct < 0.5  # Less than 0.5% separation
    
    return {
        'trend': trend,
        'strength': round(strength, 3),
        'ribbon_width': round(ribbon_width_pct, 3),
        'compression': compression,
        'ema_order': sorted_periods,
        'values': current_values
    }

def get_ema_slope(ema_values: List[float], periods: int = 5) -> float:
    """
    Calculate the slope of EMA over recent periods
    
    Args:
        ema_values: List of EMA values
        periods: Number of periods to calculate slope
        
    Returns:
        Slope value (positive for upward, negative for downward)
    """
    if len(ema_values) < periods:
        return 0.0
    
    recent_values = ema_values[-periods:]
    x = np.arange(len(recent_values))
    
    # Use linear regression to find slope
    slope = np.polyfit(x, recent_values, 1)[0]
    
    # Normalize slope by average value
    avg_value = np.mean(recent_values)
    if avg_value != 0:
        normalized_slope = slope / avg_value * 100  # Percentage slope
        return round(normalized_slope, 4)
    
    return 0.0

def detect_ema_squeeze(ribbon: Dict[int, List[float]], squeeze_threshold: float = 0.3,
                      lookback: int = 20) -> Dict[str, any]:
    """
    Detect when EMAs are squeezing together (low volatility, potential breakout)
    
    Args:
        ribbon: EMA ribbon data
        squeeze_threshold: Maximum ribbon width % to consider a squeeze
        lookback: Periods to analyze
        
    Returns:
        Dictionary with squeeze information
    """
    squeeze_periods = 0
    min_width = float('inf')
    
    # Check ribbon width over lookback period
    for i in range(lookback):
        idx = -(i + 1)
        
        current_values = []
        for period, values in ribbon.items():
            if len(values) > abs(idx):
                current_values.append(values[idx])
        
        if len(current_values) >= 2:
            width = max(current_values) - min(current_values)
            avg = sum(current_values) / len(current_values)
            width_pct = (width / avg * 100) if avg > 0 else 0
            
            if width_pct < squeeze_threshold:
                squeeze_periods += 1
                min_width = min(min_width, width_pct)
    
    is_squeezing = squeeze_periods >= lookback * 0.7
    
    return {
        'squeezing': is_squeezing,
        'squeeze_periods': squeeze_periods,
        'min_width': round(min_width, 3) if min_width != float('inf') else None,
        'intensity': round(1 - (min_width / squeeze_threshold), 3) if is_squeezing else 0
    }

def update_ema_incremental(symbol: str, price: float, period: int) -> Optional[float]:
    """
    Update EMA incrementally for real-time processing
    
    Args:
        symbol: Trading symbol
        price: Latest price
        period: EMA period
        
    Returns:
        Updated EMA value or None
    """
    cache_key = f"{symbol}_{period}"
    
    if cache_key not in _incremental_emas:
        _incremental_emas[cache_key] = EMACalculator(period)
    
    return _incremental_emas[cache_key].update(price)

def clear_ema_cache(symbol: str = None, period: int = None):
    """Clear EMA cache for specific symbol/period or all"""
    if symbol and period:
        cache_key = f"{symbol}_{period}"
        _incremental_emas.pop(cache_key, None)
        
        # Also clear regular cache entries for this symbol/period
        keys_to_remove = [k for k in _ema_cache.keys() if f"_{period}_" in k]
        for key in keys_to_remove:
            _ema_cache.pop(key, None)
            _cache_timestamps.pop(key, None)
    else:
        _incremental_emas.clear()
        _ema_cache.clear()
        _cache_timestamps.clear()
        log("🧹 Cleared all EMA caches")

# Periodic cache cleanup
async def cleanup_ema_cache_periodically():
    """Periodically clean up expired cache entries"""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in _cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > _cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            _ema_cache.pop(key, None)
            _cache_timestamps.pop(key, None)
        
        if expired_keys:
            log(f"🧹 Cleaned {len(expired_keys)} expired EMA cache entries")
