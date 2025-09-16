# whale_detector.py - Enhanced with better performance and advanced whale detection

import asyncio
import traceback
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from collections import deque
from datetime import datetime, timedelta
from error_handler import send_error_to_telegram
from logger import log

# Cache for whale detection results
_whale_cache = {}
_cache_ttl = 60  # 60 seconds cache TTL
_cache_timestamps = {}
_cache_max_size = 500

# Configuration constants
DEFAULT_THRESHOLD_RATIO = 1.8
DEFAULT_LOOKBACK = 6
MIN_VOLUME_FOR_WHALE = 10000  # Minimum volume to consider whale activity

class WhaleDetector:
    """
    Advanced whale detection with multiple algorithms and caching
    """
    def __init__(self, symbol: str, lookback_periods: int = 20):
        self.symbol = symbol
        self.lookback_periods = lookback_periods
        self.volume_history = deque(maxlen=lookback_periods * 2)
        self.price_history = deque(maxlen=lookback_periods * 2)
        self.whale_events = deque(maxlen=50)
        self.last_update = None
        
    def update(self, candle: Dict) -> Dict:
        """Update detector with new candle data"""
        try:
            volume = float(candle['volume'])
            close = float(candle['close'])
            
            self.volume_history.append(volume)
            self.price_history.append(close)
            self.last_update = datetime.now()
            
            # Detect whale activity
            result = self._detect_whale_patterns()
            
            if result['detected']:
                self.whale_events.append({
                    'timestamp': candle.get('timestamp', datetime.now()),
                    'type': result['type'],
                    'strength': result['strength'],
                    'volume': volume,
                    'price': close
                })
                
            return result
            
        except Exception as e:
            log(f"Error updating whale detector: {e}", level="ERROR")
            return {'detected': False}
    
    def _detect_whale_patterns(self) -> Dict:
        """Detect various whale patterns"""
        if len(self.volume_history) < self.lookback_periods:
            return {'detected': False}
            
        results = {
            'detected': False,
            'type': None,
            'strength': 0,
            'patterns': []
        }
        
        # Check multiple whale patterns
        patterns = []
        
        # 1. Volume spike pattern
        spike_result = self._check_volume_spike()
        if spike_result['detected']:
            patterns.append(spike_result)
            
        # 2. Accumulation pattern
        accum_result = self._check_accumulation()
        if accum_result['detected']:
            patterns.append(accum_result)
            
        # 3. Distribution pattern
        dist_result = self._check_distribution()
        if dist_result['detected']:
            patterns.append(dist_result)
            
        # 4. Iceberg order pattern
        iceberg_result = self._check_iceberg_orders()
        if iceberg_result['detected']:
            patterns.append(iceberg_result)
        
        # Aggregate results
        if patterns:
            results['detected'] = True
            results['patterns'] = patterns
            # Use the strongest pattern as primary
            strongest = max(patterns, key=lambda x: x['strength'])
            results['type'] = strongest['type']
            results['strength'] = strongest['strength']
            
        return results
    
    def _check_volume_spike(self) -> Dict:
        """Check for sudden volume spikes"""
        volumes = np.array(list(self.volume_history))
        if len(volumes) < self.lookback_periods:
            return {'detected': False}
            
        recent = volumes[-3:]
        historical = volumes[:-3]
        
        avg_historical = np.mean(historical)
        if avg_historical < MIN_VOLUME_FOR_WHALE:
            return {'detected': False}
            
        max_recent = np.max(recent)
        ratio = max_recent / avg_historical if avg_historical > 0 else 0
        
        if ratio > DEFAULT_THRESHOLD_RATIO:
            return {
                'detected': True,
                'type': 'volume_spike',
                'strength': min(ratio / DEFAULT_THRESHOLD_RATIO, 2.0),
                'ratio': ratio
            }
            
        return {'detected': False}
    
    def _check_accumulation(self) -> Dict:
        """Check for whale accumulation pattern"""
        if len(self.volume_history) < self.lookback_periods:
            return {'detected': False}
            
        volumes = np.array(list(self.volume_history))
        prices = np.array(list(self.price_history))
        
        # Look for high volume with minimal price movement (accumulation)
        recent_volumes = volumes[-5:]
        recent_prices = prices[-5:]
        
        avg_volume = np.mean(volumes[:-5])
        volume_increase = np.mean(recent_volumes) / avg_volume if avg_volume > 0 else 0
        
        price_range = (np.max(recent_prices) - np.min(recent_prices)) / np.mean(recent_prices)
        
        # High volume with low price movement suggests accumulation
        if volume_increase > 1.5 and price_range < 0.01:  # Less than 1% price movement
            return {
                'detected': True,
                'type': 'accumulation',
                'strength': min(volume_increase / 1.5, 2.0),
                'volume_increase': volume_increase,
                'price_stability': 1 - price_range
            }
            
        return {'detected': False}
    
    def _check_distribution(self) -> Dict:
        """Check for whale distribution pattern"""
        if len(self.volume_history) < self.lookback_periods:
            return {'detected': False}
            
        volumes = np.array(list(self.volume_history))
        prices = np.array(list(self.price_history))
        
        # Look for high volume at resistance levels
        recent_volumes = volumes[-5:]
        recent_prices = prices[-5:]
        historical_prices = prices[:-5]
        
        # Check if we're near historical highs
        historical_high = np.percentile(historical_prices, 95)
        current_price = prices[-1]
        
        near_resistance = current_price >= historical_high * 0.98
        high_volume = np.mean(recent_volumes) > np.mean(volumes[:-5]) * 1.5
        
        # Price stalling or declining with high volume
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if near_resistance and high_volume and price_momentum < 0.005:
            return {
                'detected': True,
                'type': 'distribution',
                'strength': 1.5,
                'near_resistance': True,
                'price_momentum': price_momentum
            }
            
        return {'detected': False}
    
    def _check_iceberg_orders(self) -> Dict:
        """Check for iceberg order patterns (consistent volume at specific price levels)"""
        if len(self.volume_history) < 10:
            return {'detected': False}
            
        volumes = list(self.volume_history)[-10:]
        prices = list(self.price_history)[-10:]
        
        # Group volumes by price levels (round to 0.1% precision)
        price_volume_map = {}
        for price, volume in zip(prices, volumes):
            price_level = round(price / 0.001) * 0.001  # Round to 0.1%
            if price_level not in price_volume_map:
                price_volume_map[price_level] = []
            price_volume_map[price_level].append(volume)
        
        # Look for consistent high volume at specific price levels
        for price_level, level_volumes in price_volume_map.items():
            if len(level_volumes) >= 3:  # At least 3 touches at this level
                avg_level_volume = np.mean(level_volumes)
                overall_avg = np.mean(volumes)
                
                if avg_level_volume > overall_avg * 1.5:
                    return {
                        'detected': True,
                        'type': 'iceberg_order',
                        'strength': 1.3,
                        'price_level': price_level,
                        'touches': len(level_volumes)
                    }
                    
        return {'detected': False}

# Global detector instances for different symbols
_whale_detectors = {}

def _get_cache_key(symbol: str, candle_count: int, threshold: float) -> str:
    """Generate cache key for whale detection"""
    # Use last candle timestamp if available
    return f"{symbol}_{candle_count}_{threshold}"

def _is_cache_valid(cache_key: str) -> bool:
    """Check if cached result is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    
    elapsed = (datetime.now() - _cache_timestamps[cache_key]).total_seconds()
    return elapsed < _cache_ttl

def _update_cache(cache_key: str, value: any) -> None:
    """Update cache with new value"""
    global _whale_cache, _cache_timestamps
    
    _whale_cache[cache_key] = value
    _cache_timestamps[cache_key] = datetime.now()
    
    # Clean up old cache entries if needed
    if len(_whale_cache) > _cache_max_size:
        # Remove oldest 20% of entries
        sorted_keys = sorted(_cache_timestamps.items(), key=lambda x: x[1])
        keys_to_remove = [k for k, _ in sorted_keys[:int(_cache_max_size * 0.2)]]
        
        for key in keys_to_remove:
            _whale_cache.pop(key, None)
            _cache_timestamps.pop(key, None)

def detect_whale_activity(candles: List[Dict], threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
                         use_cache: bool = True) -> bool:
    """
    Enhanced whale activity detection with caching and better performance
    
    Args:
        candles: List of candle dictionaries
        threshold_ratio: Volume threshold multiplier for detection
        use_cache: Whether to use caching
        
    Returns:
        bool: True if whale activity detected
    """
    try:
        if not candles or len(candles) < DEFAULT_LOOKBACK:
            return False
        
        # Generate cache key
        cache_key = _get_cache_key("default", len(candles), threshold_ratio)
        
        # Check cache
        if use_cache and cache_key in _whale_cache and _is_cache_valid(cache_key):
            return _whale_cache[cache_key]
        
        # Use numpy for better performance
        volumes = np.array([float(c['volume']) for c in candles[-DEFAULT_LOOKBACK:]])
        closes = np.array([float(c['close']) for c in candles[-DEFAULT_LOOKBACK:]])
        opens = np.array([float(c['open']) for c in candles[-DEFAULT_LOOKBACK:]])
        
        # Split into recent and earlier periods
        split_point = DEFAULT_LOOKBACK // 2
        recent_volumes = volumes[split_point:]
        earlier_volumes = volumes[:split_point]
        
        # Calculate averages
        avg_early_volume = np.mean(earlier_volumes)
        avg_recent_volume = np.mean(recent_volumes)
        
        # Skip if volumes are too low
        if avg_early_volume < MIN_VOLUME_FOR_WHALE:
            result = False
        else:
            # Calculate body sizes for recent candles
            recent_closes = closes[split_point:]
            recent_opens = opens[split_point:]
            body_sizes = np.abs(recent_closes - recent_opens)
            avg_body = np.mean(body_sizes)
            
            # Detect whale activity
            volume_spike = avg_recent_volume > avg_early_volume * threshold_ratio
            significant_body = avg_body > np.mean(closes) * 0.005  # 0.5% average body size
            
            result = volume_spike and significant_body
        
        # Update cache
        if use_cache:
            _update_cache(cache_key, result)
        
        return result

    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"🐋 <b>Whale Detector Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return False

def detect_whale_activity_advanced(candles: List[Dict], symbol: str = None,
                                 min_volume: float = MIN_VOLUME_FOR_WHALE) -> Dict:
    """
    Advanced whale detection with multiple patterns and detailed analysis
    
    Args:
        candles: List of candle dictionaries
        symbol: Trading symbol (for tracking)
        min_volume: Minimum volume threshold
        
    Returns:
        dict: Detailed whale activity analysis
    """
    try:
        if not candles or len(candles) < 20:
            return {
                'detected': False,
                'patterns': [],
                'strength': 0,
                'recommendation': 'insufficient_data'
            }
        
        # Get or create detector instance
        if symbol and symbol not in _whale_detectors:
            _whale_detectors[symbol] = WhaleDetector(symbol)
        
        detector = _whale_detectors.get(symbol) if symbol else WhaleDetector("default")
        
        # Update detector with recent candles
        results = []
        for candle in candles[-20:]:
            result = detector.update(candle)
            if result['detected']:
                results.append(result)
        
        # Analyze patterns
        if not results:
            # Fallback to simple detection
            simple_detected = detect_whale_activity(candles)
            return {
                'detected': simple_detected,
                'patterns': ['volume_spike'] if simple_detected else [],
                'strength': 1.0 if simple_detected else 0,
                'recommendation': 'monitor' if simple_detected else 'no_action'
            }
        
        # Aggregate pattern analysis
        pattern_types = [r['type'] for r in results]
        max_strength = max(r['strength'] for r in results)
        
        # Determine recommendation based on patterns
        recommendation = 'no_action'
        if 'accumulation' in pattern_types:
            recommendation = 'potential_long'
        elif 'distribution' in pattern_types:
            recommendation = 'potential_short'
        elif 'volume_spike' in pattern_types and max_strength > 1.5:
            recommendation = 'high_alert'
        elif 'iceberg_order' in pattern_types:
            recommendation = 'hidden_interest'
        
        return {
            'detected': True,
            'patterns': list(set(pattern_types)),
            'strength': max_strength,
            'recommendation': recommendation,
            'recent_events': len(results),
            'details': results[-1] if results else None  # Most recent whale event
        }
        
    except Exception as e:
        log(f"❌ Advanced whale detection error: {e}", level="ERROR")
        return {
            'detected': False,
            'patterns': [],
            'strength': 0,
            'recommendation': 'error'
        }

def analyze_whale_impact(candles: List[Dict], whale_events: List[Dict]) -> Dict:
    """
    Analyze the impact of whale activity on price movement
    
    Args:
        candles: Price candles
        whale_events: List of detected whale events
        
    Returns:
        dict: Analysis of whale impact on price
    """
    try:
        if not whale_events or not candles:
            return {'impact': 'unknown'}
        
        impacts = []
        
        for event in whale_events:
            event_time = event.get('timestamp')
            event_price = event.get('price', 0)
            
            if not event_time or not event_price:
                continue
            
            # Find price movement after whale event
            future_prices = []
            for candle in candles:
                candle_time = candle.get('timestamp')
                if candle_time and candle_time > event_time:
                    future_prices.append(float(candle['close']))
                    
                    if len(future_prices) >= 10:  # Check next 10 candles
                        break
            
            if future_prices:
                price_change = (future_prices[-1] - event_price) / event_price * 100
                impacts.append({
                    'event_type': event.get('type'),
                    'price_change': price_change,
                    'direction': 'up' if price_change > 0 else 'down'
                })
        
        # Aggregate impact analysis
        if impacts:
            avg_impact = np.mean([abs(i['price_change']) for i in impacts])
            positive_impacts = sum(1 for i in impacts if i['price_change'] > 0)
            
            return {
                'impact': 'significant' if avg_impact > 2 else 'moderate' if avg_impact > 1 else 'minimal',
                'average_change': round(avg_impact, 2),
                'success_rate': round(positive_impacts / len(impacts), 2),
                'sample_size': len(impacts)
            }
        
        return {'impact': 'unknown'}
        
    except Exception as e:
        log(f"❌ Error analyzing whale impact: {e}", level="ERROR")
        return {'impact': 'error'}

def get_whale_statistics(symbol: str = None) -> Dict:
    """
    Get statistics about whale activity for a symbol
    
    Args:
        symbol: Trading symbol
        
    Returns:
        dict: Whale activity statistics
    """
    try:
        if not symbol or symbol not in _whale_detectors:
            return {'status': 'no_data'}
        
        detector = _whale_detectors[symbol]
        
        if not detector.whale_events:
            return {'status': 'no_whale_activity'}
        
        # Calculate statistics
        events = list(detector.whale_events)
        
        # Group by type
        type_counts = {}
        for event in events:
            event_type = event.get('type', 'unknown')
            type_counts[event_type] = type_counts.get(event_type, 0) + 1
        
        # Calculate time distribution
        if len(events) >= 2:
            timestamps = [e.get('timestamp') for e in events if e.get('timestamp')]
            if timestamps:
                time_diffs = []
                for i in range(1, len(timestamps)):
                    if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                        diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                        time_diffs.append(diff)
                
                avg_interval = np.mean(time_diffs) if time_diffs else 0
            else:
                avg_interval = 0
        else:
            avg_interval = 0
        
        return {
            'status': 'active',
            'total_events': len(events),
            'event_types': type_counts,
            'most_common': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            'average_interval_seconds': round(avg_interval, 2),
            'last_event': events[-1] if events else None
        }
        
    except Exception as e:
        log(f"❌ Error getting whale statistics: {e}", level="ERROR")
        return {'status': 'error'}

# Clear cache periodically
async def clear_whale_cache_periodically():
    """Clear whale detection cache every hour"""
    while True:
        await asyncio.sleep(3600)  # 1 hour
        _whale_cache.clear()
        _cache_timestamps.clear()
        
        # Also clean up old detectors
        current_time = datetime.now()
        symbols_to_remove = []
        
        for symbol, detector in _whale_detectors.items():
            if detector.last_update and (current_time - detector.last_update).total_seconds() > 7200:  # 2 hours
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del _whale_detectors[symbol]
        
        log(f"🧹 Cleared whale detection cache and {len(symbols_to_remove)} old detectors")

# Backward compatibility wrapper
def detect_whale_accumulation(candles: List[Dict]) -> bool:
    """Check for whale accumulation pattern (backward compatibility)"""
    result = detect_whale_activity_advanced(candles)
    return 'accumulation' in result.get('patterns', [])

def detect_whale_distribution(candles: List[Dict]) -> bool:
    """Check for whale distribution pattern (backward compatibility)"""
    result = detect_whale_activity_advanced(candles)
    return 'distribution' in result.get('patterns', [])
