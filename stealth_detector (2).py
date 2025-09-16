# stealth_detector.py - Enhanced with better performance and advanced detection

import asyncio
import traceback
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
from datetime import datetime, timedelta
from error_handler import send_error_to_telegram
from logger import log

# Cache for stealth detection results
_stealth_cache = {}
_cache_ttl = 60  # 60 seconds cache TTL
_cache_timestamps = {}
_cache_max_size = 500

# Performance optimization: Pre-calculated thresholds
DEFAULT_MIN_GROWTH_RATIO = 1.2
DEFAULT_SLOW_BREAKOUT_WINDOW = 15
MIN_CANDLES_REQUIRED = 20

# Cache for volume calculations
_volume_stats_cache = {}

class StealthAccumulationDetector:
    """
    Advanced stealth accumulation detector with multiple detection methods
    """
    def __init__(self, symbol: str = None):
        self.symbol = symbol
        self.volume_history = deque(maxlen=100)
        self.price_history = deque(maxlen=100)
        self.detection_history = deque(maxlen=50)
        self.last_update = None
        
    def update(self, candle: Dict) -> Dict:
        """Update detector with new candle data"""
        try:
            volume = float(candle.get('volume', 0))
            close = float(candle.get('close', 0))
            high = float(candle.get('high', 0))
            low = float(candle.get('low', 0))
            
            if volume <= 0 or close <= 0:
                return {'detected': False}
            
            self.volume_history.append(volume)
            self.price_history.append({
                'close': close,
                'high': high,
                'low': low,
                'volume': volume,
                'timestamp': candle.get('timestamp', datetime.now())
            })
            self.last_update = datetime.now()
            
            # Run detection algorithms
            result = self._run_detection_algorithms()
            
            if result['detected']:
                self.detection_history.append({
                    'timestamp': datetime.now(),
                    'type': result['type'],
                    'strength': result['strength'],
                    'price': close,
                    'volume': volume
                })
                
            return result
            
        except Exception as e:
            log(f"Error updating stealth detector: {e}", level="ERROR")
            return {'detected': False}
    
    def _run_detection_algorithms(self) -> Dict:
        """Run multiple detection algorithms"""
        if len(self.volume_history) < MIN_CANDLES_REQUIRED:
            return {'detected': False}
        
        results = []
        
        # 1. Volume Divergence Detection
        vol_div = self._detect_volume_divergence_advanced()
        if vol_div['detected']:
            results.append(vol_div)
        
        # 2. Smart Money Accumulation
        smart_money = self._detect_smart_money_accumulation()
        if smart_money['detected']:
            results.append(smart_money)
        
        # 3. Wyckoff Accumulation
        wyckoff = self._detect_wyckoff_pattern()
        if wyckoff['detected']:
            results.append(wyckoff)
        
        # 4. Hidden Volume Surge
        hidden_surge = self._detect_hidden_volume_surge()
        if hidden_surge['detected']:
            results.append(hidden_surge)
        
        # Aggregate results
        if results:
            # Return the strongest signal
            strongest = max(results, key=lambda x: x['strength'])
            return strongest
        
        return {'detected': False}
    
    def _detect_volume_divergence_advanced(self) -> Dict:
        """Advanced volume divergence detection"""
        try:
            if len(self.price_history) < 10:
                return {'detected': False}
            
            # Get recent data
            recent_prices = [p['close'] for p in list(self.price_history)[-10:]]
            recent_volumes = list(self.volume_history)[-10:]
            
            # Calculate trends using numpy for performance
            x = np.arange(len(recent_prices))
            price_slope = np.polyfit(x, recent_prices, 1)[0]
            volume_slope = np.polyfit(x, recent_volumes, 1)[0]
            
            # Normalize slopes
            avg_price = np.mean(recent_prices)
            avg_volume = np.mean(recent_volumes)
            
            price_trend = price_slope / avg_price if avg_price > 0 else 0
            volume_trend = volume_slope / avg_volume if avg_volume > 0 else 0
            
            # Detect divergence: flat/down price with increasing volume
            if price_trend <= 0.001 and volume_trend >= 0.2:  # Price flat/down, volume up 20%+
                strength = min(volume_trend / 0.2, 2.0)  # Cap at 2.0
                return {
                    'detected': True,
                    'type': 'volume_divergence',
                    'strength': strength,
                    'price_trend': price_trend,
                    'volume_trend': volume_trend
                }
                
        except Exception as e:
            log(f"Error in volume divergence detection: {e}", level="ERROR")
            
        return {'detected': False}
    
    def _detect_smart_money_accumulation(self) -> Dict:
        """Detect smart money accumulation patterns"""
        try:
            if len(self.price_history) < 20:
                return {'detected': False}
            
            recent_data = list(self.price_history)[-20:]
            
            # Calculate average trade size
            total_volume = sum(d['volume'] for d in recent_data)
            avg_trade_size = total_volume / len(recent_data)
            
            # Look for consistent buying pressure
            buying_pressure = 0
            large_trades = 0
            
            for i in range(len(recent_data)):
                data = recent_data[i]
                
                # Check if close near high (buying pressure)
                range_size = data['high'] - data['low']
                if range_size > 0:
                    close_position = (data['close'] - data['low']) / range_size
                    if close_position > 0.7:  # Close in upper 30% of range
                        buying_pressure += 1
                
                # Check for large volume trades
                if data['volume'] > avg_trade_size * 1.5:
                    large_trades += 1
            
            # Calculate accumulation score
            buying_ratio = buying_pressure / len(recent_data)
            large_trade_ratio = large_trades / len(recent_data)
            
            if buying_ratio > 0.6 and large_trade_ratio > 0.3:
                strength = (buying_ratio + large_trade_ratio) / 2
                return {
                    'detected': True,
                    'type': 'smart_money_accumulation',
                    'strength': strength,
                    'buying_pressure': buying_ratio,
                    'large_trades': large_trade_ratio
                }
                
        except Exception as e:
            log(f"Error in smart money detection: {e}", level="ERROR")
            
        return {'detected': False}
    
    def _detect_wyckoff_pattern(self) -> Dict:
        """Detect Wyckoff accumulation patterns"""
        try:
            if len(self.price_history) < 30:
                return {'detected': False}
            
            prices = [p['close'] for p in list(self.price_history)[-30:]]
            volumes = list(self.volume_history)[-30:]
            
            # Find trading range
            price_high = max(prices)
            price_low = min(prices)
            price_range = price_high - price_low
            
            if price_range <= 0:
                return {'detected': False}
            
            # Check if price is in accumulation range (middle 60% of range)
            current_price = prices[-1]
            position_in_range = (current_price - price_low) / price_range
            
            if 0.2 <= position_in_range <= 0.8:
                # Check for volume characteristics
                recent_vol_avg = np.mean(volumes[-10:])
                older_vol_avg = np.mean(volumes[-30:-10])
                
                # Look for decreasing volume (accumulation phase)
                if recent_vol_avg < older_vol_avg * 0.8:
                    # Check for spring pattern (false breakdown)
                    min_price_idx = prices.index(min(prices[-10:]))
                    if min_price_idx > 0 and prices[min_price_idx] < price_low * 1.01:
                        # Price tested low and recovered
                        if prices[-1] > prices[min_price_idx] * 1.02:
                            return {
                                'detected': True,
                                'type': 'wyckoff_accumulation',
                                'strength': 1.5,
                                'phase': 'spring',
                                'position_in_range': position_in_range
                            }
                
                # General accumulation phase
                return {
                    'detected': True,
                    'type': 'wyckoff_accumulation',
                    'strength': 1.0,
                    'phase': 'accumulation',
                    'position_in_range': position_in_range
                }
                
        except Exception as e:
            log(f"Error in Wyckoff detection: {e}", level="ERROR")
            
        return {'detected': False}
    
    def _detect_hidden_volume_surge(self) -> Dict:
        """Detect hidden volume surges in small timeframes"""
        try:
            if len(self.volume_history) < 5:
                return {'detected': False}
            
            recent_volumes = list(self.volume_history)[-5:]
            older_volumes = list(self.volume_history)[-20:-5] if len(self.volume_history) >= 20 else []
            
            if not older_volumes:
                return {'detected': False}
            
            # Calculate volume metrics
            recent_avg = np.mean(recent_volumes)
            older_avg = np.mean(older_volumes)
            
            # Check for sudden volume increase
            if recent_avg > older_avg * 2.5:  # 150% increase
                # Verify it's not just one spike
                spikes = sum(1 for v in recent_volumes if v > older_avg * 2)
                if spikes >= 3:  # At least 3 out of 5 candles have high volume
                    strength = min(recent_avg / older_avg / 2.5, 2.0)
                    return {
                        'detected': True,
                        'type': 'hidden_volume_surge',
                        'strength': strength,
                        'volume_ratio': recent_avg / older_avg,
                        'spike_count': spikes
                    }
                    
        except Exception as e:
            log(f"Error in hidden volume surge detection: {e}", level="ERROR")
            
        return {'detected': False}

# Global detector instances
_stealth_detectors = {}

def _get_cache_key(candles_hash: int, func_name: str, **kwargs) -> str:
    """Generate cache key for stealth detection"""
    params = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return f"{candles_hash}_{func_name}_{params}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached result is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with new value"""
    global _stealth_cache, _cache_timestamps
    
    _stealth_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Clean up old cache entries if needed
    if len(_stealth_cache) > _cache_max_size:
        # Remove oldest 20% of entries
        sorted_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])
        for key, _ in sorted_keys[:int(_cache_max_size * 0.2)]:
            _stealth_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def detect_volume_divergence(candles: List[Dict], min_growth_ratio: float = DEFAULT_MIN_GROWTH_RATIO,
                           use_cache: bool = True) -> bool:
    """
    Optimized volume divergence detection with caching
    
    Args:
        candles: List of candle dictionaries
        min_growth_ratio: Minimum volume growth ratio to consider divergence
        use_cache: Whether to use caching
        
    Returns:
        bool: True if volume divergence detected
    """
    try:
        if len(candles) < MIN_CANDLES_REQUIRED:
            return False
        
        # Generate cache key
        cache_key = None
        if use_cache:
            candles_str = str([(c.get('timestamp', ''), c.get('close', ''), c.get('volume', '')) 
                             for c in candles[-MIN_CANDLES_REQUIRED:]])
            cache_key = _get_cache_key(hash(candles_str), "volume_divergence", 
                                     min_growth_ratio=min_growth_ratio)
            
            if cache_key in _stealth_cache and _is_cache_valid(cache_key):
                return _stealth_cache[cache_key]
        
        # Use numpy for performance
        recent = candles[-10:]
        prices = np.array([float(c.get('close', 0)) for c in recent])
        volumes = np.array([float(c.get('volume', 0)) for c in recent])
        
        # Skip if any invalid data
        if np.any(prices <= 0) or np.any(volumes <= 0):
            result = False
        else:
            # Calculate price change and volume growth efficiently
            price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0
            volume_growth = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] != 0 else 0
            
            # Detect divergence: price flat/down but volume increasing
            result = price_change <= 0 and volume_growth >= (min_growth_ratio - 1)
        
        # Update cache
        if use_cache and cache_key:
            _update_cache(cache_key, result)
        
        return result

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Volume Divergence Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return False

def detect_slow_breakout(candles: List[Dict], window: int = DEFAULT_SLOW_BREAKOUT_WINDOW,
                        use_cache: bool = True) -> bool:
    """
    Optimized slow breakout detection with caching
    
    Args:
        candles: List of candle dictionaries
        window: Number of candles to analyze
        use_cache: Whether to use caching
        
    Returns:
        bool: True if slow breakout pattern detected
    """
    try:
        if len(candles) < window:
            return False
        
        # Generate cache key
        cache_key = None
        if use_cache:
            candles_str = str([(c.get('timestamp', ''), c.get('close', '')) 
                             for c in candles[-window:]])
            cache_key = _get_cache_key(hash(candles_str), "slow_breakout", window=window)
            
            if cache_key in _stealth_cache and _is_cache_valid(cache_key):
                return _stealth_cache[cache_key]
        
        # Use numpy for performance
        closes = np.array([float(c.get('close', 0)) for c in candles[-window:]])
        
        if np.any(closes <= 0):
            result = False
        else:
            # Calculate metrics efficiently
            avg = np.mean(closes)
            last_close = closes[-1]
            recent_3 = closes[-3:]
            
            # Check if last 3 candles are consistently above average
            # and last close is at least 1% above average
            result = bool(np.all(recent_3 > avg) and last_close > avg * 1.01)
        
        # Update cache
        if use_cache and cache_key:
            _update_cache(cache_key, result)
        
        return result

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>Slow Breakout Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return False

def detect_stealth_accumulation_advanced(candles: List[Dict], symbol: str = None) -> Dict:
    """
    Advanced stealth accumulation detection using multiple algorithms
    
    Args:
        candles: List of candle dictionaries
        symbol: Trading symbol for tracking
        
    Returns:
        dict: Detection results with pattern details
    """
    try:
        if len(candles) < MIN_CANDLES_REQUIRED:
            return {
                'detected': False,
                'patterns': [],
                'strength': 0,
                'recommendation': 'insufficient_data'
            }
        
        # Get or create detector instance
        if symbol and symbol not in _stealth_detectors:
            _stealth_detectors[symbol] = StealthAccumulationDetector(symbol)
        
        detector = _stealth_detectors.get(symbol) if symbol else StealthAccumulationDetector()
        
        # Update detector with recent candles
        results = []
        for candle in candles[-MIN_CANDLES_REQUIRED:]:
            result = detector.update(candle)
            if result['detected']:
                results.append(result)
        
        # Also run quick checks
        patterns = []
        
        # Volume divergence check
        if detect_volume_divergence(candles):
            patterns.append('volume_divergence')
        
        # Slow breakout check
        if detect_slow_breakout(candles):
            patterns.append('slow_breakout')
        
        # Analyze detection history
        if detector.detection_history:
            recent_detections = list(detector.detection_history)[-5:]
            pattern_types = [d['type'] for d in recent_detections]
            max_strength = max(d['strength'] for d in recent_detections)
            
            # Determine recommendation
            if len(recent_detections) >= 3:
                recommendation = 'strong_accumulation'
            elif max_strength > 1.5:
                recommendation = 'moderate_accumulation'
            else:
                recommendation = 'weak_signal'
            
            return {
                'detected': True,
                'patterns': list(set(pattern_types + patterns)),
                'strength': max_strength,
                'recommendation': recommendation,
                'recent_signals': len(recent_detections),
                'details': recent_detections[-1] if recent_detections else None
            }
        
        # Fallback to simple pattern detection
        if patterns:
            return {
                'detected': True,
                'patterns': patterns,
                'strength': 1.0,
                'recommendation': 'monitor',
                'recent_signals': 0
            }
        
        return {
            'detected': False,
            'patterns': [],
            'strength': 0,
            'recommendation': 'no_signal'
        }
        
    except Exception as e:
        log(f"❌ Error in advanced stealth detection: {e}", level="ERROR")
        return {
            'detected': False,
            'patterns': [],
            'strength': 0,
            'recommendation': 'error'
        }

def get_stealth_statistics(symbol: str = None) -> Dict:
    """
    Get statistics about stealth accumulation activity
    
    Args:
        symbol: Trading symbol
        
    Returns:
        dict: Statistics about stealth patterns
    """
    try:
        if not symbol or symbol not in _stealth_detectors:
            return {'status': 'no_data'}
        
        detector = _stealth_detectors[symbol]
        
        if not detector.detection_history:
            return {'status': 'no_stealth_activity'}
        
        # Calculate statistics
        detections = list(detector.detection_history)
        
        # Group by type
        type_counts = {}
        for detection in detections:
            pattern_type = detection.get('type', 'unknown')
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1
        
        # Calculate average strength
        avg_strength = np.mean([d.get('strength', 0) for d in detections])
        
        # Find most recent detection
        last_detection = detections[-1] if detections else None
        
        return {
            'status': 'active',
            'total_detections': len(detections),
            'pattern_types': type_counts,
            'average_strength': round(avg_strength, 2),
            'last_detection': last_detection,
            'most_common': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }
        
    except Exception as e:
        log(f"❌ Error getting stealth statistics: {e}", level="ERROR")
        return {'status': 'error'}

def calculate_accumulation_score(candles: List[Dict]) -> float:
    """
    Calculate overall accumulation score (0-1)
    
    Args:
        candles: List of candle dictionaries
        
    Returns:
        float: Accumulation score
    """
    try:
        if len(candles) < MIN_CANDLES_REQUIRED:
            return 0.0
        
        score = 0.0
        
        # Volume divergence contributes 40%
        if detect_volume_divergence(candles):
            score += 0.4
        
        # Slow breakout contributes 30%
        if detect_slow_breakout(candles):
            score += 0.3
        
        # Advanced detection contributes 30%
        advanced = detect_stealth_accumulation_advanced(candles)
        if advanced['detected']:
            score += 0.3 * min(advanced['strength'], 1.0)
        
        return round(score, 2)
        
    except Exception as e:
        log(f"❌ Error calculating accumulation score: {e}", level="ERROR")
        return 0.0

# Periodic cleanup task
async def cleanup_stealth_cache():
    """Periodically clean up expired cache entries"""
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in _cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > _cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            _stealth_cache.pop(key, None)
            _cache_timestamps.pop(key, None)
        
        # Also clean up old detectors
        symbols_to_remove = []
        for symbol, detector in _stealth_detectors.items():
            if detector.last_update and (current_time - detector.last_update).total_seconds() > 3600:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del _stealth_detectors[symbol]
        
        if expired_keys or symbols_to_remove:
            log(f"🧹 Cleaned {len(expired_keys)} cache entries and {len(symbols_to_remove)} detectors")

# Backward compatibility
def detect_accumulation_pattern(candles: List[Dict]) -> bool:
    """Backward compatibility wrapper"""
    return detect_volume_divergence(candles)

def detect_distribution_pattern(candles: List[Dict]) -> bool:
    """Backward compatibility wrapper"""
    # Inverse of accumulation - price up but volume down
    try:
        if len(candles) < 10:
            return False
        
        recent = candles[-10:]
        prices = [float(c.get('close', 0)) for c in recent]
        volumes = [float(c.get('volume', 0)) for c in recent]
        
        if not all(p > 0 for p in prices) or not all(v > 0 for v in volumes):
            return False
        
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_change = (volumes[-1] - volumes[0]) / volumes[0]
        
        # Distribution: price up but volume decreasing
        return price_change > 0.01 and volume_change < -0.2
        
    except Exception:
        return False

# Export main functions and constants
__all__ = [
    'detect_volume_divergence',
    'detect_slow_breakout',
    'detect_stealth_accumulation_advanced',
    'get_stealth_statistics',
    'calculate_accumulation_score',
    'cleanup_stealth_cache',
    'DEFAULT_MIN_GROWTH_RATIO',
    'DEFAULT_SLOW_BREAKOUT_WINDOW'
]
