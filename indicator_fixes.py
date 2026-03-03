"""
Indicator fixes to remove bearish bias
"""
from logger import log
import numpy as np

def rebalance_indicator_scores(indicator_scores, market_context):
    """Rebalance indicator scores to remove bearish bias"""
    balanced_scores = {}
    
    for indicator, score in indicator_scores.items():
        # Apply market context adjustment
        if market_context.get("btc_trend") == "uptrend":
            # In uptrend, reduce bearish signal strength
            if score < 0:
                score *= 0.7  # Reduce bearish signals by 30%
            else:
                score *= 1.1  # Boost bullish signals by 10%
        elif market_context.get("btc_trend") == "downtrend":
            # In downtrend, reduce bullish signal strength
            if score > 0:
                score *= 0.8
            else:
                score *= 1.1
        
        # Fix EMA bias - ensure symmetric scoring
        if "ema" in indicator.lower():
            if abs(score) == 1.5 and score < 0:
                # This was the biased bearish score
                score = -1.0  # Make it symmetric with bullish
        
        # Cap extreme scores
        score = max(min(score, 2.0), -2.0)
        
        balanced_scores[indicator] = score
    
    return balanced_scores

def analyze_volume_direction(candles):
    """Determine if volume spike is bullish or bearish"""
    if not candles or len(candles) < 2:
        return "neutral", 0
    
    last_candle = candles[-1]
    close = float(last_candle['close'])
    open_price = float(last_candle['open'])
    volume = float(last_candle['volume'])
    
    # Check previous candle for comparison
    prev_close = float(candles[-2]['close'])
    
    # Calculate volume average
    if len(candles) >= 20:
        avg_volume = sum(float(c['volume']) for c in candles[-20:-1]) / 19
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
    else:
        volume_ratio = 1
    
    # Buying vs selling volume
    if close > open_price and close > prev_close and volume_ratio > 1.5:
        return "bullish", min(volume_ratio / 2, 1.0)  # Bullish volume
    elif close < open_price and close < prev_close and volume_ratio > 1.5:
        return "bearish", min(volume_ratio / 2, 1.0)  # Bearish volume
    else:
        return "neutral", 0

def get_balanced_rsi_signal(rsi_data, market_trend="neutral"):
    if not rsi_data:
        return "neutral", 0
    
    rsi = rsi_data.get('rsi', 50)
    
    if market_trend == "uptrend":
        overbought = 80
        oversold = 35
    elif market_trend == "downtrend":
        overbought = 65
        oversold = 20
    else:
        overbought = 70
        oversold = 30

    if rsi > overbought:
        strength = min((rsi - overbought) / 20, 1.0)
        return "sell", strength
    elif rsi < oversold:
        strength = min((oversold - rsi) / 20, 1.0)
        return "buy", strength
    else:
        # BEFORE: flat neutral bias
        # if rsi > 50: return "neutral_bullish", (rsi - 50) / 50
        # else: return "neutral_bearish", (50 - rsi) / 50

        # AFTER: scale by proximity to oversold/overbought
        proximity_to_oversold = (oversold + 10 - rsi) / 10  # 0→1 as RSI drops toward oversold+10
        proximity_to_overbought = (rsi - (overbought - 10)) / 10  # 0→1 as RSI rises toward overbought-10

        if proximity_to_oversold > 0.5:  # RSI within 5 pts of oversold
            strength = round(proximity_to_oversold * 0.6, 2)  # max 0.6 — not a full signal but meaningful
            return "buy", strength
        elif proximity_to_overbought > 0.5:
            strength = round(proximity_to_overbought * 0.6, 2)
            return "sell", strength
        elif rsi > 50:
            return "neutral_bullish", round((rsi - 50) / 50, 2)
        else:
            return "neutral_bearish", round((50 - rsi) / 50, 2)
