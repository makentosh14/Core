# rsi.py - Enhanced RSI Indicator with Performance Optimizations

import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
import asyncio
from error_handler import send_error_to_telegram

class RSICalculator:
    """
    Optimized RSI calculator with caching and incremental updates
    """
    def __init__(self, period: int = 14):
        self.period = period
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.avg_gain = None
        self.avg_loss = None
        self.last_close = None
        self.rsi_values = deque(maxlen=100)  # Cache last 100 RSI values
        
    def update(self, close: float) -> Optional[float]:
        """
        Update RSI with new closing price (incremental calculation)
        """
        if self.last_close is None:
            self.last_close = close
            return None
            
        delta = close - self.last_close
        gain = max(delta, 0)
        loss = abs(min(delta, 0))
        
        self.gains.append(gain)
        self.losses.append(loss)
        
        if len(self.gains) < self.period:
            self.last_close = close
            return None
            
        # Calculate or update averages
        if self.avg_gain is None:
            self.avg_gain = sum(self.gains) / self.period
            self.avg_loss = sum(self.losses) / self.period
        else:
            # Wilder's smoothing method
            self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
            self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period
        
        # Calculate RSI
        if self.avg_loss == 0:
            rsi = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        self.last_close = close
        self.rsi_values.append(round(rsi, 2))
        return round(rsi, 2)

# Cache for RSI calculators by symbol and period
_rsi_cache = {}

def calculate_rsi(candles: List[Dict], period: int = 14) -> Optional[List[float]]:
    """
    Optimized RSI calculation using numpy for better performance
    Returns a list of RSI values starting from index [period]
    """
    try:
        if not candles or len(candles) < period + 1:
            return None
            
        # Use numpy for faster calculations
        closes = np.array([float(c['close']) for c in candles])
        
        # Calculate price changes
        deltas = np.diff(closes)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        
        # Calculate RSI using Wilder's smoothing
        for i in range(period, len(closes)):
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(round(rsi, 2))
            
            # Update averages using Wilder's smoothing
            if i < len(gains):
                avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
                avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        
        return rsi_values
        
    except Exception as e:
        import traceback
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>RSI Calculation Error</b>\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_rsi_with_bands(candles: List[Dict], period: int = 14, 
                            overbought: float = 70, oversold: float = 30) -> Optional[Dict]:
    """
    Calculate RSI with overbought/oversold bands and additional metrics
    """
    try:
        rsi_values = calculate_rsi(candles, period)
        if not rsi_values:
            return None
            
        current_rsi = rsi_values[-1]
        
        # Calculate RSI momentum (rate of change)
        rsi_momentum = None
        if len(rsi_values) >= 5:
            rsi_momentum = rsi_values[-1] - rsi_values[-5]
        
        # Detect divergences
        divergence = detect_rsi_divergence(candles[-len(rsi_values):], rsi_values)
        
        return {
            "rsi": current_rsi,
            "values": rsi_values,
            "overbought": current_rsi > overbought,
            "oversold": current_rsi < oversold,
            "momentum": rsi_momentum,
            "divergence": divergence,
            "trend": get_rsi_trend(rsi_values)
        }
        
    except Exception as e:
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>RSI Bands Calculation Error</b>\nError: <code>{str(e)}</code>"
        ))
        return None

def detect_rsi_divergence(candles: List[Dict], rsi_values: List[float], 
                         lookback: int = 10) -> Optional[str]:
    """
    Detect bullish or bearish RSI divergence
    """
    try:
        if len(candles) < lookback or len(rsi_values) < lookback:
            return None
            
        # Get price highs and lows
        prices = [float(c['close']) for c in candles[-lookback:]]
        recent_rsi = rsi_values[-lookback:]
        
        # Find local peaks and troughs
        price_peaks = []
        price_troughs = []
        rsi_peaks = []
        rsi_troughs = []
        
        for i in range(1, len(prices) - 1):
            # Price peaks and troughs
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                price_peaks.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                price_troughs.append((i, prices[i]))
                
            # RSI peaks and troughs
            if recent_rsi[i] > recent_rsi[i-1] and recent_rsi[i] > recent_rsi[i+1]:
                rsi_peaks.append((i, recent_rsi[i]))
            elif recent_rsi[i] < recent_rsi[i-1] and recent_rsi[i] < recent_rsi[i+1]:
                rsi_troughs.append((i, recent_rsi[i]))
        
        # Check for bearish divergence (price higher high, RSI lower high)
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            if (price_peaks[-1][1] > price_peaks[-2][1] and 
                rsi_peaks[-1][1] < rsi_peaks[-2][1]):
                return "bearish_divergence"
        
        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            if (price_troughs[-1][1] < price_troughs[-2][1] and 
                rsi_troughs[-1][1] > rsi_troughs[-2][1]):
                return "bullish_divergence"
                
        return None
        
    except Exception as e:
        return None

def get_rsi_trend(rsi_values: List[float], period: int = 5) -> str:
    """
    Determine RSI trend direction
    """
    if len(rsi_values) < period:
        return "neutral"
        
    recent = rsi_values[-period:]
    
    # Simple linear regression slope
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    
    if slope > 0.5:
        return "rising"
    elif slope < -0.5:
        return "falling"
    else:
        return "neutral"

def calculate_stoch_rsi(candles: List[Dict], rsi_period: int = 14, 
                       stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Optional[Dict]:
    """
    Calculate Stochastic RSI for additional confirmation
    """
    try:
        rsi_values = calculate_rsi(candles, rsi_period)
        if not rsi_values or len(rsi_values) < stoch_period:
            return None
            
        stoch_rsi_values = []
        k_values = []
        d_values = []
        
        # Calculate Stochastic RSI
        for i in range(stoch_period - 1, len(rsi_values)):
            rsi_window = rsi_values[i - stoch_period + 1:i + 1]
            min_rsi = min(rsi_window)
            max_rsi = max(rsi_window)
            
            if max_rsi - min_rsi > 0:
                stoch_rsi = ((rsi_values[i] - min_rsi) / (max_rsi - min_rsi)) * 100
            else:
                stoch_rsi = 50  # Default to middle if no range
                
            stoch_rsi_values.append(stoch_rsi)
        
        # Calculate %K (SMA of Stoch RSI)
        for i in range(k_period - 1, len(stoch_rsi_values)):
            k = np.mean(stoch_rsi_values[i - k_period + 1:i + 1])
            k_values.append(k)
        
        # Calculate %D (SMA of %K)
        for i in range(d_period - 1, len(k_values)):
            d = np.mean(k_values[i - d_period + 1:i + 1])
            d_values.append(d)
        
        if not k_values or not d_values:
            return None
            
        return {
            "k": round(k_values[-1], 2),
            "d": round(d_values[-1], 2),
            "overbought": k_values[-1] > 80,
            "oversold": k_values[-1] < 20,
            "cross": detect_stoch_rsi_cross(k_values, d_values)
        }
        
    except Exception as e:
        return None

def detect_stoch_rsi_cross(k_values: List[float], d_values: List[float]) -> Optional[str]:
    """
    Detect Stochastic RSI crossovers
    """
    if len(k_values) < 2 or len(d_values) < 2:
        return None
        
    # Check for bullish cross (K crosses above D)
    if k_values[-2] <= d_values[-2] and k_values[-1] > d_values[-1]:
        return "bullish_cross"
    # Check for bearish cross (K crosses below D)
    elif k_values[-2] >= d_values[-2] and k_values[-1] < d_values[-1]:
        return "bearish_cross"
        
    return None

def get_rsi_signal(rsi_data: Dict, price_trend: str = None) -> Tuple[str, float]:
    """
    Generate trading signal based on RSI data and price trend
    Returns: (signal, strength) where signal is 'buy', 'sell', or 'neutral'
    """
    if not rsi_data:
        return "neutral", 0.0
        
    rsi = rsi_data.get("rsi", 50)
    momentum = rsi_data.get("momentum", 0)
    divergence = rsi_data.get("divergence")
    rsi_trend = rsi_data.get("trend", "neutral")
    
    signal = "neutral"
    strength = 0.0
    
    # Strong oversold with positive momentum
    if rsi < 30 and momentum > 0:
        signal = "buy"
        strength = 0.8
        
    # Strong overbought with negative momentum
    elif rsi > 70 and momentum < 0:
        signal = "sell"
        strength = 0.8
        
    # Divergence signals
    elif divergence == "bullish_divergence":
        signal = "buy"
        strength = 0.7
        
    elif divergence == "bearish_divergence":
        signal = "sell"
        strength = 0.7
        
    # Moderate signals
    elif rsi < 40 and rsi_trend == "rising":
        signal = "buy"
        strength = 0.5
        
    elif rsi > 60 and rsi_trend == "falling":
        signal = "sell"
        strength = 0.5
    
    # Adjust strength based on price trend alignment
    if price_trend:
        if (signal == "buy" and price_trend == "up") or \
           (signal == "sell" and price_trend == "down"):
            strength = min(strength * 1.2, 1.0)  # Increase strength
        elif (signal == "buy" and price_trend == "down") or \
             (signal == "sell" and price_trend == "up"):
            strength *= 0.8  # Decrease strength
    
    return signal, strength

# Incremental RSI update for real-time processing
def update_rsi_incremental(symbol: str, close: float, period: int = 14) -> Optional[float]:
    """
    Update RSI incrementally for a symbol (more efficient for real-time)
    """
    cache_key = f"{symbol}_{period}"
    
    if cache_key not in _rsi_cache:
        _rsi_cache[cache_key] = RSICalculator(period)
    
    return _rsi_cache[cache_key].update(close)

def clear_rsi_cache(symbol: str = None, period: int = None):
    """
    Clear RSI cache for a specific symbol/period or all
    """
    if symbol and period:
        cache_key = f"{symbol}_{period}"
        if cache_key in _rsi_cache:
            del _rsi_cache[cache_key]
    else:
        _rsi_cache.clear()

# Multi-timeframe RSI analysis
def analyze_multi_timeframe_rsi(candles_by_tf: Dict[str, List[Dict]], 
                               periods: Dict[str, int] = None) -> Dict:
    """
    Analyze RSI across multiple timeframes for confluence
    """
    if periods is None:
        periods = {
            "1": 14,
            "5": 14,
            "15": 14,
            "60": 14
        }
    
    results = {}
    
    for tf, candles in candles_by_tf.items():
        if tf in periods and candles:
            rsi_data = calculate_rsi_with_bands(candles, periods[tf])
            if rsi_data:
                results[tf] = rsi_data
    
    # Calculate confluence score
    buy_signals = 0
    sell_signals = 0
    
    for tf, data in results.items():
        if data.get("oversold") or data.get("divergence") == "bullish_divergence":
            buy_signals += 1
        elif data.get("overbought") or data.get("divergence") == "bearish_divergence":
            sell_signals += 1
    
    return {
        "timeframes": results,
        "buy_confluence": buy_signals / len(results) if results else 0,
        "sell_confluence": sell_signals / len(results) if results else 0,
        "signal": "buy" if buy_signals > sell_signals else "sell" if sell_signals > buy_signals else "neutral"
    }
