# divergence_detector.py - Advanced Divergence Detection Module

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from logger import log

class DivergenceDetector:
    """Advanced divergence detection across multiple indicators"""
    
    def __init__(self):
        self.min_lookback = 8
        self.max_lookback = 20
        self.min_peak_distance = 3  # Minimum distance between peaks/troughs
        
    def detect_rsi_divergence(self, candles: List[Dict], rsi_values: List[float], 
                             lookback: int = 12) -> Optional[Dict[str, Any]]:
        """
        Detect RSI divergence patterns
        
        Args:
            candles: List of candle dictionaries
            rsi_values: List of RSI values
            lookback: Number of periods to analyze
            
        Returns:
            Dict with divergence details or None
        """
        if not candles or not rsi_values or len(candles) < lookback or len(rsi_values) < lookback:
            return None
        
        try:
            # Get recent data
            recent_candles = candles[-lookback:]
            recent_rsi = rsi_values[-lookback:]
            
            prices = [float(c['close']) for c in recent_candles]
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(prices)
            price_troughs = self._find_troughs(prices)
            rsi_peaks = self._find_peaks(recent_rsi)
            rsi_troughs = self._find_troughs(recent_rsi)
            
            # Check for bullish divergence (price lower low, RSI higher low)
            bullish_div = self._check_bullish_divergence(
                price_troughs, rsi_troughs, prices, recent_rsi
            )
            
            if bullish_div:
                return {
                    "type": "bullish",
                    "indicator": "rsi",
                    "strength": bullish_div["strength"],
                    "confidence": bullish_div["confidence"],
                    "price_points": bullish_div["price_points"],
                    "indicator_points": bullish_div["indicator_points"]
                }
            
            # Check for bearish divergence (price higher high, RSI lower high)
            bearish_div = self._check_bearish_divergence(
                price_peaks, rsi_peaks, prices, recent_rsi
            )
            
            if bearish_div:
                return {
                    "type": "bearish",
                    "indicator": "rsi",
                    "strength": bearish_div["strength"],
                    "confidence": bearish_div["confidence"],
                    "price_points": bearish_div["price_points"],
                    "indicator_points": bearish_div["indicator_points"]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error detecting RSI divergence: {e}", level="ERROR")
            return None
    
    def detect_macd_divergence(self, candles: List[Dict], macd_data: List[Dict], 
                              lookback: int = 14) -> Optional[Dict[str, Any]]:
        """
        Detect MACD divergence patterns
        
        Args:
            candles: List of candle dictionaries
            macd_data: List of MACD data dictionaries
            lookback: Number of periods to analyze
            
        Returns:
            Dict with divergence details or None
        """
        if not candles or not macd_data or len(candles) < lookback or len(macd_data) < lookback:
            return None
        
        try:
            # Get recent data
            recent_candles = candles[-lookback:]
            recent_macd = macd_data[-lookback:]
            
            prices = [float(c['close']) for c in recent_candles]
            macd_values = [m['macd'] for m in recent_macd]
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(prices)
            price_troughs = self._find_troughs(prices)
            macd_peaks = self._find_peaks(macd_values)
            macd_troughs = self._find_troughs(macd_values)
            
            # Check for bullish divergence
            bullish_div = self._check_bullish_divergence(
                price_troughs, macd_troughs, prices, macd_values
            )
            
            if bullish_div:
                return {
                    "type": "bullish",
                    "indicator": "macd",
                    "strength": bullish_div["strength"],
                    "confidence": bullish_div["confidence"],
                    "price_points": bullish_div["price_points"],
                    "indicator_points": bullish_div["indicator_points"]
                }
            
            # Check for bearish divergence
            bearish_div = self._check_bearish_divergence(
                price_peaks, macd_peaks, prices, macd_values
            )
            
            if bearish_div:
                return {
                    "type": "bearish",
                    "indicator": "macd",
                    "strength": bearish_div["strength"],
                    "confidence": bearish_div["confidence"],
                    "price_points": bearish_div["price_points"],
                    "indicator_points": bearish_div["indicator_points"]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error detecting MACD divergence: {e}", level="ERROR")
            return None
    
    def detect_volume_divergence(self, candles: List[Dict], 
                                lookback: int = 10) -> Optional[Dict[str, Any]]:
        """
        Detect volume divergence patterns
        
        Args:
            candles: List of candle dictionaries
            lookback: Number of periods to analyze
            
        Returns:
            Dict with divergence details or None
        """
        if not candles or len(candles) < lookback:
            return None
        
        try:
            # Get recent data
            recent_candles = candles[-lookback:]
            
            prices = [float(c['close']) for c in recent_candles]
            volumes = [float(c['volume']) for c in recent_candles]
            
            # Calculate volume trend using On-Balance Volume (OBV)
            obv_values = self._calculate_obv(recent_candles)
            
            if len(obv_values) < lookback:
                return None
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(prices)
            price_troughs = self._find_troughs(prices)
            obv_peaks = self._find_peaks(obv_values)
            obv_troughs = self._find_troughs(obv_values)
            
            # Check for bullish divergence (price lower low, OBV higher low)
            bullish_div = self._check_bullish_divergence(
                price_troughs, obv_troughs, prices, obv_values
            )
            
            if bullish_div:
                return {
                    "type": "bullish",
                    "indicator": "volume",
                    "strength": bullish_div["strength"],
                    "confidence": bullish_div["confidence"],
                    "price_points": bullish_div["price_points"],
                    "indicator_points": bullish_div["indicator_points"]
                }
            
            # Check for bearish divergence (price higher high, OBV lower high)
            bearish_div = self._check_bearish_divergence(
                price_peaks, obv_peaks, prices, obv_values
            )
            
            if bearish_div:
                return {
                    "type": "bearish",
                    "indicator": "volume",
                    "strength": bearish_div["strength"],
                    "confidence": bearish_div["confidence"],
                    "price_points": bearish_div["price_points"],
                    "indicator_points": bearish_div["indicator_points"]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error detecting volume divergence: {e}", level="ERROR")
            return None
    
    def detect_momentum_divergence(self, candles: List[Dict], 
                                  lookback: int = 12) -> Optional[Dict[str, Any]]:
        """
        Detect momentum divergence using Rate of Change (ROC)
        
        Args:
            candles: List of candle dictionaries
            lookback: Number of periods to analyze
            
        Returns:
            Dict with divergence details or None
        """
        if not candles or len(candles) < lookback + 5:
            return None
        
        try:
            # Calculate Rate of Change
            roc_period = 5
            roc_values = []
            
            for i in range(roc_period, len(candles)):
                current_close = float(candles[i]['close'])
                prev_close = float(candles[i - roc_period]['close'])
                
                if prev_close != 0:
                    roc = ((current_close - prev_close) / prev_close) * 100
                    roc_values.append(roc)
                else:
                    roc_values.append(0)
            
            if len(roc_values) < lookback:
                return None
            
            # Get recent data
            recent_candles = candles[-(lookback + roc_period):]
            recent_roc = roc_values[-lookback:]
            
            prices = [float(c['close']) for c in recent_candles[-lookback:]]
            
            # Find peaks and troughs
            price_peaks = self._find_peaks(prices)
            price_troughs = self._find_troughs(prices)
            roc_peaks = self._find_peaks(recent_roc)
            roc_troughs = self._find_troughs(recent_roc)
            
            # Check for bullish divergence
            bullish_div = self._check_bullish_divergence(
                price_troughs, roc_troughs, prices, recent_roc
            )
            
            if bullish_div:
                return {
                    "type": "bullish",
                    "indicator": "momentum",
                    "strength": bullish_div["strength"],
                    "confidence": bullish_div["confidence"],
                    "price_points": bullish_div["price_points"],
                    "indicator_points": bullish_div["indicator_points"]
                }
            
            # Check for bearish divergence
            bearish_div = self._check_bearish_divergence(
                price_peaks, roc_peaks, prices, recent_roc
            )
            
            if bearish_div:
                return {
                    "type": "bearish",
                    "indicator": "momentum",
                    "strength": bearish_div["strength"],
                    "confidence": bearish_div["confidence"],
                    "price_points": bearish_div["price_points"],
                    "indicator_points": bearish_div["indicator_points"]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error detecting momentum divergence: {e}", level="ERROR")
            return None
    
    def detect_all_divergences(self, candles: List[Dict], 
                              rsi_values: List[float] = None,
                              macd_data: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Detect all types of divergences
        
        Args:
            candles: List of candle dictionaries
            rsi_values: Optional RSI values
            macd_data: Optional MACD data
            
        Returns:
            List of detected divergences
        """
        divergences = []
        
        try:
            # RSI divergence
            if rsi_values:
                rsi_div = self.detect_rsi_divergence(candles, rsi_values)
                if rsi_div:
                    divergences.append(rsi_div)
            
            # MACD divergence
            if macd_data:
                macd_div = self.detect_macd_divergence(candles, macd_data)
                if macd_div:
                    divergences.append(macd_div)
            
            # Volume divergence
            vol_div = self.detect_volume_divergence(candles)
            if vol_div:
                divergences.append(vol_div)
            
            # Momentum divergence
            mom_div = self.detect_momentum_divergence(candles)
            if mom_div:
                divergences.append(mom_div)
            
            return divergences
            
        except Exception as e:
            log(f"❌ Error detecting all divergences: {e}", level="ERROR")
            return []
    
    def _find_peaks(self, data: List[float], min_distance: int = None) -> List[Tuple[int, float]]:
        """Find local peaks in data"""
        if min_distance is None:
            min_distance = self.min_peak_distance
        
        peaks = []
        
        for i in range(1, len(data) - 1):
            if data[i] > data[i-1] and data[i] > data[i+1]:
                # Check minimum distance from previous peak
                if not peaks or i - peaks[-1][0] >= min_distance:
                    peaks.append((i, data[i]))
        
        return peaks
    
    def _find_troughs(self, data: List[float], min_distance: int = None) -> List[Tuple[int, float]]:
        """Find local troughs in data"""
        if min_distance is None:
            min_distance = self.min_peak_distance
        
        troughs = []
        
        for i in range(1, len(data) - 1):
            if data[i] < data[i-1] and data[i] < data[i+1]:
                # Check minimum distance from previous trough
                if not troughs or i - troughs[-1][0] >= min_distance:
                    troughs.append((i, data[i]))
        
        return troughs
    
    def _check_bullish_divergence(self, price_troughs: List[Tuple[int, float]], 
                                 indicator_troughs: List[Tuple[int, float]],
                                 prices: List[float], 
                                 indicator_values: List[float]) -> Optional[Dict[str, Any]]:
        """Check for bullish divergence pattern"""
        if len(price_troughs) < 2 or len(indicator_troughs) < 2:
            return None
        
        try:
            # Get the two most recent troughs
            price_trough1 = price_troughs[-2]
            price_trough2 = price_troughs[-1]
            
            # Find corresponding indicator troughs (within reasonable time window)
            indicator_trough1 = None
            indicator_trough2 = None
            
            for trough in indicator_troughs:
                if abs(trough[0] - price_trough1[0]) <= 2:  # Within 2 periods
                    indicator_trough1 = trough
                if abs(trough[0] - price_trough2[0]) <= 2:
                    indicator_trough2 = trough
            
            if indicator_trough1 is None or indicator_trough2 is None:
                return None
            
            # Check for divergence: price lower low, indicator higher low
            price_lower = price_trough2[1] < price_trough1[1]
            indicator_higher = indicator_trough2[1] > indicator_trough1[1]
            
            if price_lower and indicator_higher:
                # Calculate strength and confidence
                price_decline = abs(price_trough2[1] - price_trough1[1]) / price_trough1[1]
                indicator_rise = abs(indicator_trough2[1] - indicator_trough1[1])
                
                strength = min(price_decline * 10 + indicator_rise * 0.1, 1.0)
                confidence = 0.7 if price_decline > 0.02 else 0.5  # 2% price decline
                
                return {
                    "strength": strength,
                    "confidence": confidence,
                    "price_points": [price_trough1, price_trough2],
                    "indicator_points": [indicator_trough1, indicator_trough2]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error checking bullish divergence: {e}", level="ERROR")
            return None
    
    def _check_bearish_divergence(self, price_peaks: List[Tuple[int, float]], 
                                 indicator_peaks: List[Tuple[int, float]],
                                 prices: List[float], 
                                 indicator_values: List[float]) -> Optional[Dict[str, Any]]:
        """Check for bearish divergence pattern"""
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return None
        
        try:
            # Get the two most recent peaks
            price_peak1 = price_peaks[-2]
            price_peak2 = price_peaks[-1]
            
            # Find corresponding indicator peaks (within reasonable time window)
            indicator_peak1 = None
            indicator_peak2 = None
            
            for peak in indicator_peaks:
                if abs(peak[0] - price_peak1[0]) <= 2:  # Within 2 periods
                    indicator_peak1 = peak
                if abs(peak[0] - price_peak2[0]) <= 2:
                    indicator_peak2 = peak
            
            if indicator_peak1 is None or indicator_peak2 is None:
                return None
            
            # Check for divergence: price higher high, indicator lower high
            price_higher = price_peak2[1] > price_peak1[1]
            indicator_lower = indicator_peak2[1] < indicator_peak1[1]
            
            if price_higher and indicator_lower:
                # Calculate strength and confidence
                price_rise = abs(price_peak2[1] - price_peak1[1]) / price_peak1[1]
                indicator_decline = abs(indicator_peak1[1] - indicator_peak2[1])
                
                strength = min(price_rise * 10 + indicator_decline * 0.1, 1.0)
                confidence = 0.7 if price_rise > 0.02 else 0.5  # 2% price rise
                
                return {
                    "strength": strength,
                    "confidence": confidence,
                    "price_points": [price_peak1, price_peak2],
                    "indicator_points": [indicator_peak1, indicator_peak2]
                }
            
            return None
            
        except Exception as e:
            log(f"❌ Error checking bearish divergence: {e}", level="ERROR")
            return None
    
    def _calculate_obv(self, candles: List[Dict]) -> List[float]:
        """Calculate On-Balance Volume"""
        if len(candles) < 2:
            return []
        
        try:
            obv = [0]  # Start with 0
            
            for i in range(1, len(candles)):
                prev_close = float(candles[i-1]['close'])
                curr_close = float(candles[i]['close'])
                curr_volume = float(candles[i]['volume'])
                
                if curr_close > prev_close:
                    obv.append(obv[-1] + curr_volume)
                elif curr_close < prev_close:
                    obv.append(obv[-1] - curr_volume)
                else:
                    obv.append(obv[-1])
            
            return obv
            
        except Exception as e:
            log(f"❌ Error calculating OBV: {e}", level="ERROR")
            return []

# Additional utility functions for divergence analysis

def calculate_divergence_score(divergences: List[Dict[str, Any]]) -> float:
    """Calculate overall divergence score from multiple divergences"""
    if not divergences:
        return 0.0
    
    try:
        total_score = 0
        weight_sum = 0
        
        # Weights for different indicators
        weights = {
            "rsi": 1.0,
            "macd": 1.2,
            "volume": 0.8,
            "momentum": 0.9
        }
        
        for div in divergences:
            indicator = div.get("indicator", "unknown")
            strength = div.get("strength", 0)
            confidence = div.get("confidence", 0)
            
            weight = weights.get(indicator, 0.5)
            score = strength * confidence * weight
            
            total_score += score
            weight_sum += weight
        
        return total_score / weight_sum if weight_sum > 0 else 0.0
        
    except Exception as e:
        log(f"❌ Error calculating divergence score: {e}", level="ERROR")
        return 0.0

def get_divergence_recommendation(divergences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get trading recommendation based on divergences"""
    if not divergences:
        return {"action": "none", "confidence": 0, "reason": "No divergences detected"}
    
    try:
        bullish_count = sum(1 for d in divergences if d.get("type") == "bullish")
        bearish_count = sum(1 for d in divergences if d.get("type") == "bearish")
        
        total_strength = sum(d.get("strength", 0) for d in divergences)
        avg_confidence = sum(d.get("confidence", 0) for d in divergences) / len(divergences)
        
        if bullish_count > bearish_count and total_strength > 0.6:
            return {
                "action": "buy",
                "confidence": avg_confidence,
                "reason": f"{bullish_count} bullish divergences detected",
                "strength": total_strength
            }
        elif bearish_count > bullish_count and total_strength > 0.6:
            return {
                "action": "sell", 
                "confidence": avg_confidence,
                "reason": f"{bearish_count} bearish divergences detected",
                "strength": total_strength
            }
        else:
            return {
                "action": "watch",
                "confidence": avg_confidence,
                "reason": "Mixed or weak divergence signals",
                "strength": total_strength
            }
        
    except Exception as e:
        log(f"❌ Error getting divergence recommendation: {e}", level="ERROR")
        return {"action": "none", "confidence": 0, "reason": "Error in analysis"}

# Global instance for easy import
divergence_detector = DivergenceDetector()

# Export main functions and classes
__all__ = [
    'DivergenceDetector',
    'divergence_detector',
    'calculate_divergence_score',
    'get_divergence_recommendation'
]
