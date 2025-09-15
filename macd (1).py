# macd.py - Enhanced version with improved performance and accuracy

import numpy as np
import asyncio
from error_handler import send_error_to_telegram
from logger import log

# MACD configuration constants
DEFAULT_FAST_PERIOD = 12
DEFAULT_SLOW_PERIOD = 26
DEFAULT_SIGNAL_PERIOD = 9

# Cache for EMA calculations to avoid redundant computation
_ema_cache = {}
_cache_max_size = 100  # Limit cache size to prevent memory issues

def calculate_ema_optimized(values, period):
    """
    Optimized EMA calculation using numpy for better performance
    
    Args:
        values: List or array of price values
        period: EMA period
        
    Returns:
        numpy array of EMA values
    """
    if len(values) < period:
        return np.array([])
    
    # Convert to numpy array if not already
    values = np.array(values, dtype=np.float64)
    
    # Use exponential weighted moving average
    alpha = 2.0 / (period + 1.0)
    
    # Initialize EMA array
    ema = np.zeros_like(values)
    ema[0] = values[0]
    
    # Vectorized calculation for better performance
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
    
    return ema

def calculate_ema(values, period, use_cache=True):
    """
    Calculate EMA with optional caching
    
    Args:
        values: List of price values
        period: EMA period
        use_cache: Whether to use caching (default: True)
        
    Returns:
        List of EMA values
    """
    try:
        # Create cache key
        cache_key = f"{len(values)}_{period}_{hash(tuple(values[-min(10, len(values)):]))}"
        
        # Check cache if enabled
        if use_cache and cache_key in _ema_cache:
            return _ema_cache[cache_key]
        
        # Calculate EMA
        if len(values) < period:
            return []
        
        # Use optimized numpy calculation
        ema_array = calculate_ema_optimized(values, period)
        ema_list = ema_array.tolist()
        
        # Cache result if enabled
        if use_cache:
            # Clean cache if it's getting too large
            if len(_ema_cache) > _cache_max_size:
                # Remove oldest entries (simple FIFO)
                keys_to_remove = list(_ema_cache.keys())[:20]
                for key in keys_to_remove:
                    del _ema_cache[key]
            
            _ema_cache[cache_key] = ema_list
        
        return ema_list

    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>EMA Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return []

def calculate_macd(candles, fast_period=DEFAULT_FAST_PERIOD, 
                   slow_period=DEFAULT_SLOW_PERIOD, 
                   signal_period=DEFAULT_SIGNAL_PERIOD,
                   price_field='close'):
    """
    Enhanced MACD calculation with better performance and flexibility
    
    Args:
        candles: List of candle dictionaries
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
        price_field: Which price to use ('close', 'high', 'low', 'typical')
        
    Returns:
        List of MACD dictionaries with 'macd', 'signal', 'histogram' values
    """
    try:
        # Extract price values based on field
        if price_field == 'typical':
            # Typical price = (High + Low + Close) / 3
            prices = [(float(c['high']) + float(c['low']) + float(c['close'])) / 3 
                     for c in candles]
        else:
            prices = [float(c.get(price_field, c['close'])) for c in candles]
        
        # Need at least slow_period + signal_period candles for meaningful MACD
        min_required = slow_period + signal_period
        if len(prices) < min_required:
            return []
        
        # Calculate EMAs
        fast_ema = calculate_ema_optimized(prices, fast_period)
        slow_ema = calculate_ema_optimized(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        signal_line = calculate_ema_optimized(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Build result list
        macd_result = []
        for i in range(len(macd_line)):
            if i >= slow_period - 1:  # Only include valid MACD values
                macd_result.append({
                    'macd': round(macd_line[i], 8),
                    'signal': round(signal_line[i], 8),
                    'histogram': round(histogram[i], 8),
                    'timestamp': candles[i].get('timestamp')  # Include timestamp for reference
                })
        
        return macd_result

    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>MACD Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return []

def detect_macd_cross(candles, lookback=2, min_histogram_change=0.0001):
    """
    Enhanced MACD crossover detection with additional filters
    
    Args:
        candles: List of candle dictionaries
        lookback: Number of candles to look back for confirmation (default: 2)
        min_histogram_change: Minimum histogram change to confirm crossover
        
    Returns:
        "bullish", "bearish", or None
    """
    try:
        macd_data = calculate_macd(candles)
        
        if len(macd_data) < lookback + 1:
            return None
        
        # Get recent MACD values
        recent_macd = macd_data[-lookback:]
        current = recent_macd[-1]
        
        # Check for crossover with confirmation
        bullish_cross = False
        bearish_cross = False
        
        # Look for crossover in recent candles
        for i in range(len(recent_macd) - 1):
            prev = recent_macd[i]
            curr = recent_macd[i + 1]
            
            # Bullish crossover: MACD crosses above signal
            if prev["macd"] <= prev["signal"] and curr["macd"] > curr["signal"]:
                # Confirm with histogram change
                histogram_change = curr["histogram"] - prev["histogram"]
                if histogram_change >= min_histogram_change:
                    bullish_cross = True
                    
            # Bearish crossover: MACD crosses below signal
            elif prev["macd"] >= prev["signal"] and curr["macd"] < curr["signal"]:
                # Confirm with histogram change
                histogram_change = prev["histogram"] - curr["histogram"]
                if histogram_change >= min_histogram_change:
                    bearish_cross = True
        
        # Additional confirmation: Check trend strength
        if bullish_cross:
            # Verify MACD is gaining strength
            if len(macd_data) >= 3:
                macd_slope = current["macd"] - macd_data[-3]["macd"]
                if macd_slope > 0:
                    return "bullish"
                    
        elif bearish_cross:
            # Verify MACD is losing strength
            if len(macd_data) >= 3:
                macd_slope = current["macd"] - macd_data[-3]["macd"]
                if macd_slope < 0:
                    return "bearish"
        
        return None

    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>MACD Cross Detection Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def get_macd_divergence(candles, lookback_periods=14):
    """
    Detect MACD divergence (price making new highs/lows but MACD not confirming)
    
    Args:
        candles: List of candle dictionaries
        lookback_periods: Number of periods to check for divergence
        
    Returns:
        Dictionary with divergence info or None
    """
    try:
        if len(candles) < lookback_periods + 30:  # Need enough data
            return None
            
        macd_data = calculate_macd(candles)
        if len(macd_data) < lookback_periods:
            return None
            
        # Get recent data
        recent_candles = candles[-lookback_periods:]
        recent_macd = macd_data[-lookback_periods:]
        
        # Find price highs and lows
        price_highs = [float(c['high']) for c in recent_candles]
        price_lows = [float(c['low']) for c in recent_candles]
        
        # Find MACD highs and lows
        macd_values = [m['macd'] for m in recent_macd]
        
        # Check for bullish divergence (price lower low, MACD higher low)
        price_min_idx = price_lows.index(min(price_lows))
        if price_min_idx > lookback_periods // 2:  # Recent low
            # Find previous low
            prev_low_idx = price_lows[:price_min_idx-2].index(min(price_lows[:price_min_idx-2]))
            
            if price_lows[price_min_idx] < price_lows[prev_low_idx]:
                # Price made lower low
                if macd_values[price_min_idx] > macd_values[prev_low_idx]:
                    # MACD made higher low - bullish divergence
                    return {
                        'type': 'bullish_divergence',
                        'strength': abs(macd_values[price_min_idx] - macd_values[prev_low_idx]),
                        'periods_ago': lookback_periods - price_min_idx
                    }
        
        # Check for bearish divergence (price higher high, MACD lower high)
        price_max_idx = price_highs.index(max(price_highs))
        if price_max_idx > lookback_periods // 2:  # Recent high
            # Find previous high
            prev_high_idx = price_highs[:price_max_idx-2].index(max(price_highs[:price_max_idx-2]))
            
            if price_highs[price_max_idx] > price_highs[prev_high_idx]:
                # Price made higher high
                if macd_values[price_max_idx] < macd_values[prev_high_idx]:
                    # MACD made lower high - bearish divergence
                    return {
                        'type': 'bearish_divergence',
                        'strength': abs(macd_values[prev_high_idx] - macd_values[price_max_idx]),
                        'periods_ago': lookback_periods - price_max_idx
                    }
        
        return None
        
    except Exception as e:
        log(f"❌ Error detecting MACD divergence: {e}", level="ERROR")
        return None

def get_macd_momentum(candles):
    """
    Calculate MACD momentum strength (useful for identifying strong trends)
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        Float between -1 and 1 indicating momentum strength and direction
    """
    try:
        macd_data = calculate_macd(candles)
        if len(macd_data) < 10:
            return 0.0
            
        # Get recent MACD histogram values
        recent_histograms = [m['histogram'] for m in macd_data[-10:]]
        
        # Calculate momentum based on histogram trend
        increasing = 0
        decreasing = 0
        
        for i in range(1, len(recent_histograms)):
            if recent_histograms[i] > recent_histograms[i-1]:
                increasing += 1
            else:
                decreasing += 1
                
        # Calculate average histogram value
        avg_histogram = sum(recent_histograms) / len(recent_histograms)
        
        # Determine momentum strength
        trend_strength = (increasing - decreasing) / len(recent_histograms)
        
        # Combine trend and magnitude
        if avg_histogram > 0:
            momentum = min(1.0, trend_strength + abs(avg_histogram) / 100)
        else:
            momentum = max(-1.0, trend_strength - abs(avg_histogram) / 100)
            
        return round(momentum, 3)
        
    except Exception as e:
        log(f"❌ Error calculating MACD momentum: {e}", level="ERROR")
        return 0.0

# Clear cache periodically to prevent memory issues
async def clear_ema_cache_periodically():
    """Periodically clear the EMA cache to prevent memory buildup"""
    while True:
        await asyncio.sleep(3600)  # Clear every hour
        _ema_cache.clear()
        log("🧹 Cleared EMA cache")
