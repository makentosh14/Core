# rsi.py - Professional RSI Implementation with Wilder's Smoothing and Multi-TF Support

import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
import asyncio
from logger import log
from error_handler import send_error_to_telegram

class RSICalculator:
    """
    Professional RSI calculator using Wilder's smoothing method
    Optimized for real-time incremental updates with proper logging
    """
    def __init__(self, period: int = 14, symbol: str = ""):
        self.period = period
        self.symbol = symbol
        self.gains = deque(maxlen=period)
        self.losses = deque(maxlen=period)
        self.avg_gain = None
        self.avg_loss = None
        self.last_close = None
        self.rsi_values = deque(maxlen=100)  # Cache last 100 RSI values
        self.is_initialized = False
        
    def update(self, close: float) -> Optional[float]:
        """
        Update RSI with new closing price using Wilder's smoothing
        """
        try:
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
                
            # Initialize with simple average for first calculation
            if not self.is_initialized:
                self.avg_gain = sum(self.gains) / self.period
                self.avg_loss = sum(self.losses) / self.period
                self.is_initialized = True
                log(f"🔧 RSI[{self.period}] initialized for {self.symbol}: avg_gain={self.avg_gain:.6f}, avg_loss={self.avg_loss:.6f}")
            else:
                # Use Wilder's smoothing (exponential moving average with alpha = 1/period)
                alpha = 1.0 / self.period
                self.avg_gain = (1 - alpha) * self.avg_gain + alpha * gain
                self.avg_loss = (1 - alpha) * self.avg_loss + alpha * loss
            
            # Calculate RSI
            if self.avg_loss == 0:
                rsi = 100.0
            else:
                rs = self.avg_gain / self.avg_loss
                rsi = 100 - (100 / (1 + rs))
                
            self.last_close = close
            rsi_rounded = round(rsi, 2)
            self.rsi_values.append(rsi_rounded)
            
            # Debug logging for extreme values
            if rsi_rounded <= 30 or rsi_rounded >= 70:
                log(f"📊 RSI[{self.period}] for {self.symbol} = {rsi_rounded} ({'OVERSOLD' if rsi_rounded <= 30 else 'OVERBOUGHT'})")
            
            return rsi_rounded
            
        except Exception as e:
            log(f"❌ RSI update error for {self.symbol}: {e}", level="ERROR")
            return None

# Cache for RSI calculators by symbol and period
_rsi_cache = {}

def calculate_rsi_wilder(candles: List[Dict], period: int = 14, symbol: str = "") -> Optional[List[float]]:
    """
    Calculate RSI using Wilder's smoothing method (industry standard)
    Returns a list of RSI values starting from index [period]
    
    Args:
        candles: List of candle dictionaries with 'close' field
        period: RSI period (default: 14)
        symbol: Symbol name for logging
        
    Returns:
        List of RSI values or None if insufficient data
    """
    try:
        # Guard clause for insufficient data
        if not candles or len(candles) < period + 1:
            log(f"⚠️ Insufficient data for RSI[{period}] on {symbol}: {len(candles) if candles else 0} candles < {period + 1} required", level="WARN")
            return None
            
        # Extract and validate closing prices
        closes = []
        for i, c in enumerate(candles):
            if not isinstance(c, dict):
                log(f"❌ Invalid candle format at index {i} for {symbol}", level="ERROR")
                return None
                
            close_val = c.get('close', 0)
            try:
                close_float = float(close_val)
                if close_float <= 0:
                    log(f"❌ Invalid close price at index {i} for {symbol}: {close_val}", level="ERROR")
                    return None
                closes.append(close_float)
            except (ValueError, TypeError):
                log(f"❌ Cannot convert close price to float at index {i} for {symbol}: {close_val}", level="ERROR")
                return None
        
        # Calculate price changes
        deltas = np.diff(closes)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialize with simple average for first period
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        alpha = 1.0 / period  # Wilder's smoothing factor
        
        log(f"🔧 Calculating RSI[{period}] for {symbol}: {len(closes)} candles, initial avg_gain={avg_gain:.6f}, avg_loss={avg_loss:.6f}")
        
        # Calculate RSI using Wilder's smoothing for remaining periods
        for i in range(period, len(closes)):
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_rounded = round(rsi, 2)
            rsi_values.append(rsi_rounded)
            
            # Update averages using Wilder's smoothing (except for the last iteration)
            if i < len(gains):
                gain = gains[i]
                loss = losses[i]
                avg_gain = (1 - alpha) * avg_gain + alpha * gain
                avg_loss = (1 - alpha) * avg_loss + alpha * loss
        
        # Log final RSI value and key levels
        final_rsi = rsi_values[-1] if rsi_values else None
        if final_rsi:
            level = "OVERSOLD" if final_rsi <= 30 else "OVERBOUGHT" if final_rsi >= 70 else "NEUTRAL"
            log(f"📊 RSI[{period}] for {symbol}: {final_rsi} ({level}) | {len(rsi_values)} values calculated")
        
        return rsi_values
        
    except Exception as e:
        import traceback
        log(f"❌ RSI calculation error for {symbol}: {e}", level="ERROR")
        log(f"Full traceback: {traceback.format_exc()}", level="ERROR")
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>RSI Calculation Error</b>\nSymbol: {symbol}\nError: <code>{str(e)}</code>\n<pre>{traceback.format_exc()}</pre>"
        ))
        return None

def calculate_rsi_with_scoring(candles: List[Dict], period: int = 14, symbol: str = "",
                              overbought: float = 70, oversold: float = 30) -> Optional[Dict]:
    """
    Calculate RSI with trading signals and score contributions for bot logic
    
    Returns:
        Dict with RSI data, signals, and score contributions
    """
    try:
        rsi_values = calculate_rsi_wilder(candles, period, symbol)
        if not rsi_values:
            return None
            
        current_rsi = rsi_values[-1]
        
        # Calculate score contribution based on RSI levels
        score_contribution = 0.0
        signal = "neutral"
        signal_strength = 0.0
        
        if current_rsi <= oversold:
            # Bullish signal - oversold condition
            signal = "bullish"
            signal_strength = min((oversold - current_rsi) / 15, 1.0)  # Stronger the lower it goes
            score_contribution = signal_strength * 1.0  # Positive score for buy signal
            log(f"🟢 RSI[{period}] BULLISH signal for {symbol}: {current_rsi} (oversold), strength={signal_strength:.2f}, score=+{score_contribution:.2f}")
            
        elif current_rsi >= overbought:
            # Bearish signal - overbought condition
            signal = "bearish"
            signal_strength = min((current_rsi - overbought) / 15, 1.0)  # Stronger the higher it goes
            score_contribution = -signal_strength * 1.0  # Negative score for sell signal
            log(f"🔴 RSI[{period}] BEARISH signal for {symbol}: {current_rsi} (overbought), strength={signal_strength:.2f}, score={score_contribution:.2f}")
            
        else:
            # Neutral zone - no strong signal
            # Slight bias based on position within neutral zone
            mid_point = (overbought + oversold) / 2
            if current_rsi > mid_point:
                score_contribution = -0.1  # Slight bearish bias
            elif current_rsi < mid_point:
                score_contribution = 0.1   # Slight bullish bias
            
            log(f"⚪ RSI[{period}] NEUTRAL for {symbol}: {current_rsi}, score={score_contribution:.2f}")
        
        # Calculate RSI momentum (rate of change over 5 periods)
        rsi_momentum = None
        momentum_score = 0.0
        if len(rsi_values) >= 5:
            rsi_momentum = rsi_values[-1] - rsi_values[-5]
            # Add momentum to score (momentum confirms direction)
            if signal == "bullish" and rsi_momentum > 0:
                momentum_score = 0.2
            elif signal == "bearish" and rsi_momentum < 0:
                momentum_score = -0.2
            elif signal != "neutral":
                # Momentum against signal - reduce confidence
                momentum_score = score_contribution * -0.3
        
        # Detect divergences
        candle_slice = candles[-len(rsi_values):] if len(candles) >= len(rsi_values) else candles
        divergence = detect_rsi_divergence(candle_slice, rsi_values, symbol)
        divergence_score = 0.0
        
        if divergence == "bullish_divergence":
            divergence_score = 0.5
            if signal != "bearish":  # Don't conflict with strong bearish signal
                score_contribution += divergence_score
            log(f"📈 RSI[{period}] BULLISH DIVERGENCE detected for {symbol}, score=+{divergence_score:.2f}")
            
        elif divergence == "bearish_divergence":
            divergence_score = -0.5
            if signal != "bullish":  # Don't conflict with strong bullish signal
                score_contribution += divergence_score
            log(f"📉 RSI[{period}] BEARISH DIVERGENCE detected for {symbol}, score={divergence_score:.2f}")
        
        total_score = score_contribution + momentum_score
        
        result = {
            "rsi": current_rsi,
            "values": rsi_values,
            "signal": signal,
            "signal_strength": signal_strength,
            "score_contribution": total_score,
            "breakdown": {
                "level_score": score_contribution,
                "momentum_score": momentum_score,
                "divergence_score": divergence_score
            },
            "overbought": current_rsi >= overbought,
            "oversold": current_rsi <= oversold,
            "momentum": rsi_momentum,
            "divergence": divergence,
            "trend": get_rsi_trend(rsi_values, symbol)
        }
        
        log(f"📊 RSI[{period}] Summary for {symbol}: RSI={current_rsi}, Signal={signal}, Total Score={total_score:.2f}")
        return result
        
    except Exception as e:
        log(f"❌ RSI scoring calculation error for {symbol}: {e}", level="ERROR")
        asyncio.create_task(send_error_to_telegram(
            f"❌ <b>RSI Scoring Error</b>\nSymbol: {symbol}\nError: <code>{str(e)}</code>"
        ))
        return None

def detect_rsi_divergence(candles: List[Dict], rsi_values: List[float], 
                         symbol: str = "", lookback: int = 10) -> Optional[str]:
    """
    Detect bullish or bearish RSI divergence with proper logging
    """
    try:
        if len(candles) < lookback or len(rsi_values) < lookback:
            return None
            
        # Get price highs and lows with safe indexing
        lookback_size = min(lookback, len(candles), len(rsi_values))
        prices = []
        for i in range(len(candles) - lookback_size, len(candles)):
            if i >= 0 and i < len(candles):
                close_val = candles[i].get('close', 0)
                prices.append(float(close_val))
        
        # Get recent RSI values with safe indexing
        recent_rsi = []
        for i in range(len(rsi_values) - lookback_size, len(rsi_values)):
            if i >= 0 and i < len(rsi_values):
                recent_rsi.append(rsi_values[i])
        
        if len(prices) < 3 or len(recent_rsi) < 3:
            return None
        
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
            price_hh = price_peaks[-1][1] > price_peaks[-2][1]
            rsi_lh = rsi_peaks[-1][1] < rsi_peaks[-2][1]
            if price_hh and rsi_lh:
                log(f"📉 BEARISH DIVERGENCE detected for {symbol}: Price HH ({price_peaks[-2][1]:.4f} -> {price_peaks[-1][1]:.4f}), RSI LH ({rsi_peaks[-2][1]:.2f} -> {rsi_peaks[-1][1]:.2f})")
                return "bearish_divergence"
        
        # Check for bullish divergence (price lower low, RSI higher low)
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            price_ll = price_troughs[-1][1] < price_troughs[-2][1]
            rsi_hl = rsi_troughs[-1][1] > rsi_troughs[-2][1]
            if price_ll and rsi_hl:
                log(f"📈 BULLISH DIVERGENCE detected for {symbol}: Price LL ({price_troughs[-2][1]:.4f} -> {price_troughs[-1][1]:.4f}), RSI HL ({rsi_troughs[-2][1]:.2f} -> {rsi_troughs[-1][1]:.2f})")
                return "bullish_divergence"
                
        return None
        
    except Exception as e:
        log(f"❌ Divergence detection error for {symbol}: {e}", level="ERROR")
        return None

def get_rsi_trend(rsi_values: List[float], symbol: str = "", period: int = 5) -> str:
    """
    Determine RSI trend direction using linear regression
    """
    try:
        if len(rsi_values) < period:
            return "neutral"
            
        # Get recent values with safe indexing
        start_idx = max(0, len(rsi_values) - period)
        recent = rsi_values[start_idx:]
        
        # Linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        trend = "neutral"
        if slope > 0.5:
            trend = "rising"
        elif slope < -0.5:
            trend = "falling"
        
        if symbol and trend != "neutral":
            log(f"📈 RSI trend for {symbol}: {trend} (slope: {slope:.3f})")
        
        return trend
        
    except Exception as e:
        log(f"❌ RSI trend calculation error for {symbol}: {e}", level="ERROR")
        return "neutral"

def analyze_multi_timeframe_rsi(candles_by_tf: Dict[str, List[Dict]], symbol: str = "",
                               periods: Dict[str, int] = None, 
                               overbought: float = 70, oversold: float = 30) -> Dict:
    """
    Analyze RSI across multiple timeframes for confluence scoring
    Designed for integration with main.py scoring logic
    """
    if periods is None:
        periods = {
            "1": 14,
            "3": 14, 
            "5": 14,
            "15": 14,
            "60": 14
        }
    
    results = {}
    total_score = 0.0
    timeframe_scores = {}
    
    log(f"🔍 Multi-timeframe RSI analysis for {symbol} across {list(candles_by_tf.keys())} timeframes")
    
    for tf, candles in candles_by_tf.items():
        if tf in periods and candles:
            rsi_data = calculate_rsi_with_scoring(candles, periods[tf], f"{symbol}_{tf}m", overbought, oversold)
            if rsi_data:
                results[tf] = rsi_data
                tf_score = rsi_data["score_contribution"]
                timeframe_scores[tf] = tf_score
                
                # Weight higher timeframes more heavily
                weight = 1.0
                if tf in ["15", "60"]:
                    weight = 1.5
                elif tf in ["5"]:
                    weight = 1.2
                
                weighted_score = tf_score * weight
                total_score += weighted_score
                
                log(f"📊 {tf}m RSI for {symbol}: {rsi_data['rsi']:.2f} | Signal: {rsi_data['signal']} | Score: {tf_score:.2f} (weighted: {weighted_score:.2f})")
    
    # Calculate confluence metrics
    bullish_signals = sum(1 for data in results.values() if data["signal"] == "bullish")
    bearish_signals = sum(1 for data in results.values() if data["signal"] == "bearish")
    total_signals = len(results)
    
    # Determine overall signal based on confluence
    overall_signal = "neutral"
    if bullish_signals > bearish_signals and bullish_signals >= total_signals * 0.6:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals and bearish_signals >= total_signals * 0.6:
        overall_signal = "bearish"
    
    confluence_strength = max(bullish_signals, bearish_signals) / total_signals if total_signals > 0 else 0
    
    result = {
        "timeframes": results,
        "timeframe_scores": timeframe_scores,
        "total_score": round(total_score, 2),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
        "confluence_strength": round(confluence_strength, 2),
        "overall_signal": overall_signal,
        "signal_count": total_signals
    }
    
    log(f"🎯 RSI Multi-TF Summary for {symbol}: Overall={overall_signal}, Score={total_score:.2f}, Confluence={confluence_strength:.2f} ({bullish_signals}B/{bearish_signals}Be/{total_signals}T)")
    
    return result

def calculate_stoch_rsi(candles: List[Dict], rsi_period: int = 14, 
                       stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Optional[Dict]:
    """
    Calculate Stochastic RSI - wrapper for backward compatibility
    """
    try:
        rsi_values = calculate_rsi_wilder(candles, rsi_period, "")
        if not rsi_values or len(rsi_values) < stoch_period:
            return None
            
        stoch_rsi_values = []
        k_values = []
        d_values = []
        
        # Calculate Stochastic RSI
        for i in range(stoch_period - 1, len(rsi_values)):
            start_idx = max(0, i - stoch_period + 1)
            end_idx = i + 1
            rsi_window = rsi_values[start_idx:end_idx]
            
            min_rsi = min(rsi_window)
            max_rsi = max(rsi_window)
            
            if max_rsi - min_rsi > 0:
                stoch_rsi = ((rsi_values[i] - min_rsi) / (max_rsi - min_rsi)) * 100
            else:
                stoch_rsi = 50
                
            stoch_rsi_values.append(stoch_rsi)
        
        # Calculate %K
        for i in range(k_period - 1, len(stoch_rsi_values)):
            start_idx = max(0, i - k_period + 1)
            end_idx = i + 1
            k = np.mean(stoch_rsi_values[start_idx:end_idx])
            k_values.append(k)
        
        # Calculate %D
        for i in range(d_period - 1, len(k_values)):
            start_idx = max(0, i - d_period + 1)
            end_idx = i + 1
            d = np.mean(k_values[start_idx:end_idx])
            d_values.append(d)
        
        if not k_values or not d_values:
            return None
            
        return {
            "k": round(k_values[-1], 2),
            "d": round(d_values[-1], 2),
            "overbought": k_values[-1] > 80,
            "oversold": k_values[-1] < 20,
        }
        
    except Exception as e:
        return None

# Incremental RSI update for real-time processing
def update_rsi_incremental(symbol: str, close: float, period: int = 14) -> Optional[float]:
    """
    Update RSI incrementally for a symbol (more efficient for real-time)
    """
    cache_key = f"{symbol}_{period}"
    
    if cache_key not in _rsi_cache:
        _rsi_cache[cache_key] = RSICalculator(period, symbol)
    
    return _rsi_cache[cache_key].update(close)

def clear_rsi_cache(symbol: str = None, period: int = None):
    """
    Clear RSI cache for a specific symbol/period or all
    """
    if symbol and period:
        cache_key = f"{symbol}_{period}"
        if cache_key in _rsi_cache:
            del _rsi_cache[cache_key]
            log(f"🗑️ Cleared RSI cache for {symbol} period {period}")
    else:
        cleared_count = len(_rsi_cache)
        _rsi_cache.clear()
        log(f"🗑️ Cleared all RSI cache ({cleared_count} entries)")

# Legacy function for backward compatibility
def calculate_rsi(candles: List[Dict], period: int = 14) -> Optional[List[float]]:
    """
    Legacy function - use calculate_rsi_wilder for new implementations
    """
    return calculate_rsi_wilder(candles, period, "")

def calculate_rsi_with_bands(candles: List[Dict], period: int = 14, 
                            overbought: float = 70, oversold: float = 30) -> Optional[Dict]:
    """
    Legacy function - use calculate_rsi_with_scoring for new implementations
    """
    return calculate_rsi_with_scoring(candles, period, "", overbought, oversold)

