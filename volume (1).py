# volume.py - Enhanced with better performance and more features

import asyncio
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import traceback
from error_handler import send_error_to_telegram
from logger import log

# Cache for volume calculations to improve performance
_volume_cache = {}
_cache_ttl = 300  # 5 minutes cache TTL
_cache_timestamps = {}

# Moving average cache for performance
_ma_cache = {}
_ma_cache_size = 1000  # Maximum cache entries

def _get_cache_key(symbol: str, window: int, func_name: str) -> str:
    """Generate cache key for volume calculations"""
    return f"{symbol}_{window}_{func_name}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached value is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with new value"""
    _volume_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Cleanup old cache entries if cache is getting too large
    if len(_volume_cache) > _ma_cache_size:
        oldest_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])[:100]
        for key, _ in oldest_keys:
            _volume_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def get_volumes_array(candles: List[Dict]) -> np.ndarray:
    """Convert candles to numpy array of volumes for faster computation"""
    try:
        return np.array([float(c['volume']) for c in candles])
    except (KeyError, ValueError) as e:
        log(f"Error converting volumes to array: {e}", level="ERROR")
        return np.array([])

def is_volume_spike(candles: List[Dict], multiplier: float = 2.0, 
                   use_median: bool = False, lookback: int = 10) -> bool:
    """
    Enhanced volume spike detection with multiple methods
    
    Args:
        candles: List of candle dictionaries
        multiplier: Volume spike threshold multiplier
        use_median: Use median instead of mean for comparison
        lookback: Number of previous candles to compare against
        
    Returns:
        bool: True if volume spike detected
    """
    try:
        if len(candles) < lookback + 1:
            return False

        volumes = get_volumes_array(candles[-(lookback+1):])
        if len(volumes) < lookback + 1:
            return False
            
        current_volume = volumes[-1]
        previous_volumes = volumes[:-1]
        
        # Use median for more robust comparison (less affected by outliers)
        if use_median:
            avg_volume = np.median(previous_volumes)
        else:
            avg_volume = np.mean(previous_volumes)
        
        # Avoid division by zero
        if avg_volume <= 0:
            return False
            
        return current_volume > avg_volume * multiplier
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Volume Spike Detection Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return False

def detect_volume_pattern(candles: List[Dict], pattern_type: str = "increasing") -> bool:
    """
    Detect specific volume patterns
    
    Args:
        candles: List of candle dictionaries
        pattern_type: Type of pattern to detect ("increasing", "decreasing", "divergence")
        
    Returns:
        bool: True if pattern detected
    """
    try:
        if len(candles) < 5:
            return False
            
        volumes = get_volumes_array(candles[-5:])
        prices = np.array([float(c['close']) for c in candles[-5:]])
        
        if pattern_type == "increasing":
            # Check if volumes are generally increasing
            volume_slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return volume_slope > 0
            
        elif pattern_type == "decreasing":
            # Check if volumes are generally decreasing
            volume_slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            return volume_slope < 0
            
        elif pattern_type == "divergence":
            # Check for price-volume divergence
            volume_slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
            price_slope = np.polyfit(range(len(prices)), prices, 1)[0]
            
            # Divergence: price going up but volume going down (or vice versa)
            return (price_slope > 0 and volume_slope < 0) or (price_slope < 0 and volume_slope > 0)
            
        return False
        
    except Exception as e:
        log(f"Error detecting volume pattern: {e}", level="ERROR")
        return False

def detect_slow_ramp(candles: List[Dict], lookback: int = 6, 
                    price_threshold: float = 0.02,
                    min_volume_increase: float = 1.2) -> bool:
    """
    Enhanced slow ramp detection with better performance
    
    Args:
        candles: List of candle dictionaries
        lookback: Number of candles to analyze
        price_threshold: Maximum price movement allowed (2% default)
        min_volume_increase: Minimum volume increase factor required
        
    Returns:
        bool: True if slow ramp pattern detected
    """
    try:
        if len(candles) < lookback + 2:
            return False

        # Use numpy for faster computation
        recent_candles = candles[-lookback:]
        volumes = get_volumes_array(recent_candles)
        closes = np.array([float(c['close']) for c in recent_candles])
        opens = np.array([float(c['open']) for c in recent_candles])
        
        # Check volume trend (should be increasing)
        vol_increasing = all(volumes[i] <= volumes[i + 1] * 1.1 for i in range(len(volumes) - 1))
        
        # Check for volume acceleration
        if len(volumes) >= 3:
            recent_vol_increase = volumes[-1] / volumes[-3] if volumes[-3] > 0 else 0
            vol_accelerating = recent_vol_increase >= min_volume_increase
        else:
            vol_accelerating = False
        
        # Check price stability
        price_change = abs(closes[-1] - opens[0]) / opens[0] if opens[0] > 0 else 0
        price_steady = price_change < price_threshold
        
        # Last candle should be bullish
        last_candle_bullish = closes[-1] > opens[-1]
        
        # Additional check: volume should be above average
        avg_volume = np.mean(volumes)
        current_vol_above_avg = volumes[-1] > avg_volume
        
        return (vol_increasing or vol_accelerating) and price_steady and last_candle_bullish and current_vol_above_avg
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Slow Ramp Detection Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return False

def get_average_volume(candles: List[Dict], window: int = 20) -> float:
    """
    Enhanced average volume calculation with caching
    
    Args:
        candles: List of candle dictionaries
        window: Period for average calculation
        
    Returns:
        float: Average volume over the window period
    """
    if not candles or len(candles) < window:
        return 0.0
    
    # Generate cache key (simplified without symbol for this function)
    cache_key = f"avg_vol_{len(candles)}_{window}"
    
    # Check cache
    if cache_key in _volume_cache and _is_cache_valid(cache_key):
        return _volume_cache[cache_key]
    
    try:
        volumes = get_volumes_array(candles[-window:])
        avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else 0.0
        
        # Update cache
        _update_cache(cache_key, avg_volume)
        
        return avg_volume
        
    except Exception as e:
        log(f"Error calculating average volume: {e}", level="ERROR")
        return 0.0

def get_volume_profile(candles: List[Dict], bins: int = 10) -> Dict[str, any]:
    """
    Calculate volume profile (volume at price levels)
    
    Args:
        candles: List of candle dictionaries
        bins: Number of price bins for volume distribution
        
    Returns:
        dict: Volume profile with price levels and volumes
    """
    try:
        if len(candles) < 2:
            return {}
            
        prices = []
        volumes = []
        
        for candle in candles:
            # Use average price of the candle
            avg_price = (float(candle['high']) + float(candle['low'])) / 2
            volume = float(candle['volume'])
            prices.append(avg_price)
            volumes.append(volume)
        
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        # Create price bins
        price_range = prices.max() - prices.min()
        if price_range <= 0:
            return {}
            
        bin_edges = np.linspace(prices.min(), prices.max(), bins + 1)
        
        # Calculate volume for each price bin
        volume_profile = {}
        for i in range(len(bin_edges) - 1):
            mask = (prices >= bin_edges[i]) & (prices < bin_edges[i + 1])
            bin_volume = volumes[mask].sum()
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            volume_profile[round(bin_center, 6)] = round(bin_volume, 2)
        
        # Find point of control (price level with highest volume)
        if volume_profile:
            poc = max(volume_profile.items(), key=lambda x: x[1])[0]
        else:
            poc = None
            
        return {
            'profile': volume_profile,
            'poc': poc,  # Point of Control
            'total_volume': float(volumes.sum()),
            'avg_volume': float(volumes.mean())
        }
        
    except Exception as e:
        log(f"Error calculating volume profile: {e}", level="ERROR")
        return {}

def get_volume_momentum(candles: List[Dict], short_period: int = 5, 
                       long_period: int = 20) -> float:
    """
    Calculate volume momentum (similar to price momentum but for volume)
    
    Args:
        candles: List of candle dictionaries
        short_period: Short period for momentum calculation
        long_period: Long period for momentum calculation
        
    Returns:
        float: Volume momentum ratio (>1 means increasing volume momentum)
    """
    try:
        if len(candles) < long_period:
            return 1.0
            
        volumes = get_volumes_array(candles[-long_period:])
        
        short_avg = np.mean(volumes[-short_period:])
        long_avg = np.mean(volumes)
        
        if long_avg <= 0:
            return 1.0
            
        return round(short_avg / long_avg, 3)
        
    except Exception as e:
        log(f"Error calculating volume momentum: {e}", level="ERROR")
        return 1.0

def detect_volume_climax(candles: List[Dict], threshold_multiplier: float = 3.0,
                        lookback: int = 50) -> Tuple[bool, str]:
    """
    Detect volume climax (extremely high volume that often marks tops/bottoms)
    
    Args:
        candles: List of candle dictionaries
        threshold_multiplier: How many times above average to consider climax
        lookback: Period to calculate average
        
    Returns:
        tuple: (is_climax, climax_type) where climax_type is "buying" or "selling"
    """
    try:
        if len(candles) < lookback:
            return False, None
            
        volumes = get_volumes_array(candles[-lookback:])
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])
        
        if avg_volume <= 0:
            return False, None
            
        # Check if current volume is climactic
        if current_volume > avg_volume * threshold_multiplier:
            # Determine if it's buying or selling climax
            current_candle = candles[-1]
            close = float(current_candle['close'])
            open_price = float(current_candle['open'])
            
            if close > open_price:
                return True, "buying"
            else:
                return True, "selling"
                
        return False, None
        
    except Exception as e:
        log(f"Error detecting volume climax: {e}", level="ERROR")
        return False, None

def get_volume_weighted_average_price(candles: List[Dict]) -> float:
    """
    Calculate VWAP (Volume Weighted Average Price)
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        float: VWAP value
    """
    try:
        if not candles:
            return 0.0
            
        total_volume = 0.0
        total_value = 0.0
        
        for candle in candles:
            # Use typical price (high + low + close) / 3
            typical_price = (float(candle['high']) + float(candle['low']) + float(candle['close'])) / 3
            volume = float(candle['volume'])
            
            total_value += typical_price * volume
            total_volume += volume
            
        if total_volume <= 0:
            return 0.0
            
        return round(total_value / total_volume, 6)
        
    except Exception as e:
        log(f"Error calculating VWAP: {e}", level="ERROR")
        return 0.0

def analyze_volume_trend(candles: List[Dict], periods: List[int] = [5, 10, 20]) -> Dict[str, any]:
    """
    Comprehensive volume trend analysis
    
    Args:
        candles: List of candle dictionaries
        periods: List of periods to analyze
        
    Returns:
        dict: Comprehensive volume analysis
    """
    try:
        if len(candles) < max(periods):
            return {}
            
        volumes = get_volumes_array(candles)
        current_volume = volumes[-1] if len(volumes) > 0 else 0
        
        analysis = {
            'current_volume': current_volume,
            'averages': {},
            'ratios': {},
            'trend': None,
            'strength': 0
        }
        
        # Calculate averages and ratios for different periods
        for period in periods:
            if len(volumes) >= period:
                avg = np.mean(volumes[-period:])
                analysis['averages'][f'{period}ma'] = round(avg, 2)
                
                if avg > 0:
                    analysis['ratios'][f'{period}ma_ratio'] = round(current_volume / avg, 2)
        
        # Determine overall trend
        if len(analysis['ratios']) >= 2:
            ratios = list(analysis['ratios'].values())
            if all(r > 1.2 for r in ratios):
                analysis['trend'] = 'increasing'
                analysis['strength'] = min(ratios)
            elif all(r < 0.8 for r in ratios):
                analysis['trend'] = 'decreasing'
                analysis['strength'] = max(ratios)
            else:
                analysis['trend'] = 'neutral'
                analysis['strength'] = 1.0
                
        # Add volume momentum
        analysis['momentum'] = get_volume_momentum(candles)
        
        # Check for volume spike
        analysis['has_spike'] = is_volume_spike(candles)
        
        return analysis
        
    except Exception as e:
        log(f"Error analyzing volume trend: {e}", level="ERROR")
        return {}

# Backward compatibility wrapper
def is_volume_increasing(candles: List[Dict], lookback: int = 5) -> bool:
    """Check if volume is increasing over the lookback period"""
    return detect_volume_pattern(candles[-lookback:], "increasing")

def is_volume_decreasing(candles: List[Dict], lookback: int = 5) -> bool:
    """Check if volume is decreasing over the lookback period"""
    return detect_volume_pattern(candles[-lookback:], "decreasing")
