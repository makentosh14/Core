# volume_utils.py - Advanced volume analysis utilities

import numpy as np
from typing import Dict, List, Tuple, Optional
from logger import log

def analyze_volume_profile(candles: List[Dict], lookback: int = 50) -> Dict:
    """Analyze volume profile for better quality assessment"""
    if len(candles) < lookback:
        return {"quality": "unknown", "score": 0.5}
    
    recent_candles = candles[-lookback:]
    volumes = [float(c.get('volume', 0)) for c in recent_candles]
    prices = [float(c.get('close', 0)) for c in recent_candles]
    
    # Calculate volume metrics
    avg_volume = np.mean(volumes)
    median_volume = np.median(volumes)
    volume_std = np.std(volumes)
    
    # Volume trend (increasing/decreasing)
    first_half_avg = np.mean(volumes[:lookback//2])
    second_half_avg = np.mean(volumes[lookback//2:])
    volume_trend = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
    
    # Price-volume correlation
    if len(prices) == len(volumes):
        correlation = np.corrcoef(prices, volumes)[0, 1]
    else:
        correlation = 0
    
    # Calculate quality score
    quality_score = 0.5  # Base score
    
    # Consistent volume is good
    if volume_std < avg_volume * 1.5:
        quality_score += 0.2
    
    # Increasing volume trend is good
    if volume_trend > 0.1:
        quality_score += 0.1
    
    # Some correlation between price and volume is healthy
    if 0.2 < abs(correlation) < 0.8:
        quality_score += 0.2
    
    # Determine quality category
    if quality_score >= 0.7:
        quality = "good"
    elif quality_score >= 0.5:
        quality = "acceptable"
    else:
        quality = "poor"

    if correlation < 0.2 and volume_trend > 0.05:
        log("⚠️ Volume trend lagging but increasing — potential stealth breakout.")
        quality_score += 0.05  # Slight reward, not punishment
    
    return {
        "quality": quality,
        "score": quality_score,
        "avg_volume": avg_volume,
        "median_volume": median_volume,
        "volume_trend": volume_trend,
        "correlation": correlation
    }

def get_position_size_multiplier(volume_analysis: Dict, trade_type: str) -> float:
    """Adjust position size based on volume quality"""
    quality_score = volume_analysis.get("score", 0.5)

    # 💥 NEW SKIP LOGIC
    if quality_score < 0.35:
        log(f"⛔ Skipping trade due to very weak volume (score={quality_score:.2f})")
        return 0.0  # SKIP TRADE
    
    # More conservative for scalping
    if trade_type == "Scalp":
        if quality_score >= 0.7:
            return 1.0
        elif quality_score >= 0.5:
            return 0.6
        else:
            return 0.4
    
    # Medium for intraday
    elif trade_type == "Intraday":
        if quality_score >= 0.6:
            return 1.0
        elif quality_score >= 0.4:
            return 0.8
        else:
            return 0.6
    
    # More lenient for swing
    else:  # Swing
        if quality_score >= 0.5:
            return 1.0
        elif quality_score >= 0.35:
            return 0.7
        else:
            return 0.5
