import asyncio
import traceback
import numpy as np
from collections import deque
from error_handler import send_error_to_telegram
from logger import log

# Cache for Bollinger Bands calculations to avoid redundant computations
_bb_cache = {}
_cache_max_size = 100  # Maximum number of cached calculations

def calculate_bollinger_bands(candles, period=20, multiplier=2):
    """
    Calculate Bollinger Bands with improved performance and accuracy
    
    Args:
        candles: list of dicts with 'close' prices as strings
        period: SMA period (default 20)
        multiplier: Standard deviation multiplier (default 2)
        
    Returns:
        list of dicts with 'middle', 'upper', 'lower', 'bandwidth', 'percent_b' or None if error
    """
    try:
        if not candles or len(candles) < period:
            return []
            
        # Create cache key based on last candle timestamp and parameters
        cache_key = f"{candles[-1].get('timestamp', '')}_{len(candles)}_{period}_{multiplier}"
        
        # Check cache first
        if cache_key in _bb_cache:
            return _bb_cache[cache_key]
        
        # Convert to numpy array for better performance
        closes = np.array([float(c['close']) for c in candles])
        
        # Pre-allocate result array
        bands = [None] * len(closes)
        
        # Use numpy's efficient rolling window calculation
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            
            # Calculate SMA (middle band)
            sma = np.mean(window)
            
            # Calculate standard deviation
            std_dev = np.std(window, ddof=0)  # Population standard deviation
            
            # Calculate bands
            upper = sma + (multiplier * std_dev)
            lower = sma - (multiplier * std_dev)
            
            # Calculate additional useful metrics
            bandwidth = (upper - lower) / sma if sma != 0 else 0
            percent_b = (closes[i] - lower) / (upper - lower) if upper != lower else 0.5
            
            bands[i] = {
                'middle': round(sma, 8),
                'upper': round(upper, 8),
                'lower': round(lower, 8),
                'std_dev': round(std_dev, 8),
                'bandwidth': round(bandwidth, 4),  # Bandwidth as percentage
                'percent_b': round(percent_b, 4),  # Position within bands (0-1)
                'squeeze': bandwidth < 0.02  # Bollinger Squeeze detection
            }
        
        # Cache management - remove oldest entries if cache is too large
        if len(_bb_cache) >= _cache_max_size:
            # Remove the oldest 20% of entries
            remove_count = _cache_max_size // 5
            for _ in range(remove_count):
                _bb_cache.pop(next(iter(_bb_cache)))
        
        # Store in cache
        _bb_cache[cache_key] = bands
        
        return bands

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Bollinger Bands Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return []

def calculate_bollinger_bands_advanced(candles, period=20, multiplier=2, use_ema=False):
    """
    Advanced Bollinger Bands calculation with EMA option and additional features
    
    Args:
        candles: list of dicts with OHLC data
        period: Moving average period
        multiplier: Standard deviation multiplier
        use_ema: Use EMA instead of SMA for middle band
        
    Returns:
        list of dicts with advanced band metrics
    """
    try:
        if not candles or len(candles) < period:
            return []
            
        closes = np.array([float(c['close']) for c in candles])
        highs = np.array([float(c['high']) for c in candles])
        lows = np.array([float(c['low']) for c in candles])
        volumes = np.array([float(c['volume']) for c in candles])
        
        bands = [None] * len(closes)
        
        # Calculate EMA if requested
        if use_ema:
            ema_multiplier = 2 / (period + 1)
            ema = np.zeros(len(closes))
            ema[0] = closes[0]
            
            for i in range(1, len(closes)):
                ema[i] = (closes[i] - ema[i-1]) * ema_multiplier + ema[i-1]
        
        for i in range(period - 1, len(closes)):
            window = closes[i - period + 1:i + 1]
            
            # Middle band calculation
            if use_ema:
                middle = ema[i]
            else:
                middle = np.mean(window)
            
            # Standard deviation using typical price for better accuracy
            typical_prices = (highs[i - period + 1:i + 1] + lows[i - period + 1:i + 1] + closes[i - period + 1:i + 1]) / 3
            std_dev = np.std(typical_prices, ddof=0)
            
            # Calculate bands
            upper = middle + (multiplier * std_dev)
            lower = middle - (multiplier * std_dev)
            
            # Advanced metrics
            bandwidth = (upper - lower) / middle if middle != 0 else 0
            percent_b = (closes[i] - lower) / (upper - lower) if upper != lower else 0.5
            
            # Volume-weighted bands (experimental)
            vol_window = volumes[i - period + 1:i + 1]
            avg_volume = np.mean(vol_window)
            vol_ratio = volumes[i] / avg_volume if avg_volume > 0 else 1
            
            # Trend strength based on band position
            if closes[i] > upper:
                trend_strength = min((closes[i] - upper) / std_dev, 3.0) if std_dev > 0 else 0
                trend_direction = "strong_up"
            elif closes[i] < lower:
                trend_strength = min((lower - closes[i]) / std_dev, 3.0) if std_dev > 0 else 0
                trend_direction = "strong_down"
            elif closes[i] > middle:
                trend_strength = (closes[i] - middle) / (upper - middle) if upper != middle else 0
                trend_direction = "up"
            else:
                trend_strength = (middle - closes[i]) / (middle - lower) if middle != lower else 0
                trend_direction = "down"
            
            bands[i] = {
                'middle': round(middle, 8),
                'upper': round(upper, 8),
                'lower': round(lower, 8),
                'std_dev': round(std_dev, 8),
                'bandwidth': round(bandwidth, 4),
                'percent_b': round(percent_b, 4),
                'squeeze': bandwidth < 0.02,
                'volume_ratio': round(vol_ratio, 2),
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 2),
                'band_width_percentile': calculate_bandwidth_percentile(bands, i, bandwidth)
            }
        
        return bands
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Advanced Bollinger Bands Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return []

def calculate_bandwidth_percentile(bands, current_index, current_bandwidth):
    """
    Calculate where current bandwidth sits relative to recent history
    Useful for detecting volatility expansion/contraction
    """
    lookback = min(100, current_index)
    if lookback < 20:
        return 50.0  # Default to middle if not enough history
        
    recent_bandwidths = []
    for i in range(current_index - lookback, current_index):
        if bands[i] and 'bandwidth' in bands[i]:
            recent_bandwidths.append(bands[i]['bandwidth'])
    
    if not recent_bandwidths:
        return 50.0
        
    # Calculate percentile
    below_count = sum(1 for bw in recent_bandwidths if bw < current_bandwidth)
    percentile = (below_count / len(recent_bandwidths)) * 100
    
    return round(percentile, 1)

def detect_bollinger_squeeze(bands, lookback=20, squeeze_threshold=0.015):
    """
    Detect Bollinger Band squeeze (low volatility) periods
    
    Args:
        bands: Bollinger bands data
        lookback: Number of periods to check
        squeeze_threshold: Bandwidth threshold for squeeze detection
        
    Returns:
        dict with squeeze information
    """
    if not bands or len(bands) < lookback:
        return {'squeeze': False, 'duration': 0}
    
    squeeze_count = 0
    min_bandwidth = float('inf')
    
    # Check recent periods for squeeze
    for i in range(len(bands) - lookback, len(bands)):
        if bands[i] and bands[i].get('bandwidth', 1) < squeeze_threshold:
            squeeze_count += 1
            min_bandwidth = min(min_bandwidth, bands[i]['bandwidth'])
    
    is_squeeze = squeeze_count >= lookback * 0.7  # 70% of recent periods in squeeze
    
    return {
        'squeeze': is_squeeze,
        'duration': squeeze_count,
        'intensity': round(1 - (min_bandwidth / squeeze_threshold), 2) if is_squeeze else 0,
        'min_bandwidth': round(min_bandwidth, 4) if min_bandwidth != float('inf') else None
    }

def detect_band_walk(candles, bands, periods=5):
    """
    Detect when price is "walking the bands" (strong trend signal)
    
    Args:
        candles: Price candles
        bands: Bollinger bands data
        periods: Number of consecutive periods to check
        
    Returns:
        dict with band walk information
    """
    if not bands or not candles or len(bands) < periods:
        return {'walking_upper': False, 'walking_lower': False, 'strength': 0}
    
    upper_touches = 0
    lower_touches = 0
    
    for i in range(len(bands) - periods, len(bands)):
        if not bands[i]:
            continue
            
        close = float(candles[i]['close'])
        upper = bands[i]['upper']
        lower = bands[i]['lower']
        
        # Check if price is near upper band (within 0.1%)
        if close >= upper * 0.999:
            upper_touches += 1
        # Check if price is near lower band
        elif close <= lower * 1.001:
            lower_touches += 1
    
    # Determine if walking the bands
    walking_upper = upper_touches >= periods * 0.8  # 80% of periods touching upper
    walking_lower = lower_touches >= periods * 0.8  # 80% of periods touching lower
    
    strength = max(upper_touches, lower_touches) / periods
    
    return {
        'walking_upper': walking_upper,
        'walking_lower': walking_lower,
        'strength': round(strength, 2),
        'consecutive_touches': max(upper_touches, lower_touches)
    }

def get_bollinger_signal(candles, period=20, multiplier=2):
    """
    Get trading signal based on Bollinger Bands analysis
    
    Returns:
        dict with signal type and strength
    """
    try:
        bands = calculate_bollinger_bands_advanced(candles, period, multiplier)
        if not bands or len(bands) < period:
            return {'signal': None, 'strength': 0, 'reason': 'Insufficient data'}
        
        # Get latest values
        latest_band = bands[-1]
        if not latest_band:
            return {'signal': None, 'strength': 0, 'reason': 'No band data'}
            
        close = float(candles[-1]['close'])
        percent_b = latest_band['percent_b']
        bandwidth = latest_band['bandwidth']
        
        # Check for squeeze
        squeeze_info = detect_bollinger_squeeze(bands)
        
        # Check for band walk
        walk_info = detect_band_walk(candles, bands)
        
        # Generate signal
        signal = None
        strength = 0
        reason = []
        
        # Strong signals
        if walk_info['walking_upper']:
            signal = 'strong_bullish'
            strength = 0.8 + (walk_info['strength'] * 0.2)
            reason.append(f"Walking upper band ({walk_info['consecutive_touches']} touches)")
            
        elif walk_info['walking_lower']:
            signal = 'strong_bearish'
            strength = 0.8 + (walk_info['strength'] * 0.2)
            reason.append(f"Walking lower band ({walk_info['consecutive_touches']} touches)")
            
        # Squeeze breakout signals
        elif squeeze_info['squeeze'] and squeeze_info['duration'] >= 10:
            # Check previous bands to see if we're breaking out
            if len(bands) >= 2 and bands[-2]:
                prev_bandwidth = bands[-2]['bandwidth']
                if bandwidth > prev_bandwidth * 1.2:  # 20% bandwidth expansion
                    if percent_b > 0.8:
                        signal = 'squeeze_breakout_up'
                        strength = 0.7 + (squeeze_info['intensity'] * 0.3)
                        reason.append(f"Squeeze breakout up (duration: {squeeze_info['duration']})")
                    elif percent_b < 0.2:
                        signal = 'squeeze_breakout_down'
                        strength = 0.7 + (squeeze_info['intensity'] * 0.3)
                        reason.append(f"Squeeze breakout down (duration: {squeeze_info['duration']})")
        
        # Mean reversion signals
        elif percent_b > 1.0:  # Above upper band
            signal = 'overbought'
            strength = min(0.6 + (percent_b - 1.0), 0.9)
            reason.append(f"Above upper band (percent_b: {percent_b:.2f})")
            
        elif percent_b < 0.0:  # Below lower band
            signal = 'oversold'
            strength = min(0.6 + abs(percent_b), 0.9)
            reason.append(f"Below lower band (percent_b: {percent_b:.2f})")
        
        # Neutral zone
        else:
            signal = 'neutral'
            strength = 0
            reason.append(f"Within bands (percent_b: {percent_b:.2f})")
        
        return {
            'signal': signal,
            'strength': round(strength, 2),
            'reason': ', '.join(reason),
            'percent_b': percent_b,
            'bandwidth': bandwidth,
            'squeeze': squeeze_info['squeeze'],
            'band_walk': walk_info
        }
        
    except Exception as e:
        log(f"❌ Error generating Bollinger signal: {e}", level="ERROR")
        return {'signal': None, 'strength': 0, 'reason': f'Error: {str(e)}'}

# Clear cache periodically to prevent memory issues
async def clear_cache_periodically():
    """Clear the Bollinger Bands cache every hour"""
    while True:
        await asyncio.sleep(3600)  # Wait 1 hour
        _bb_cache.clear()
        log("🧹 Cleared Bollinger Bands cache")
