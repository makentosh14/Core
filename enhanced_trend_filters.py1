# enhanced_trend_filters.py - Advanced Trend Detection System
"""
Enhanced trend detection with:
1. Market Structure Analysis
2. Advanced Altseason Detection
3. Multi-source Sentiment Analysis
4. Volume Profile Engine
5. Institutional Activity Detection
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any
from bybit_api import signed_request
from logger import log
import aiohttp
import json

def normalize_klines(candles):
    """Ensure klines are ordered oldest -> newest.
    Each candle is [start, open, high, low, close, volume, ...].
    Detect order by timestamps and reverse if needed.
    """
    if not candles or len(candles) < 2:
        return candles
    try:
        t0, t1 = int(candles[0][0]), int(candles[1][0])
        newest_first = t0 > t1
    except Exception:
        newest_first = False
    return list(reversed(candles)) if newest_first else candles


class MarketStructureAnalyzer:
    """Advanced market structure analysis for trend detection"""
    
    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []
        self.market_structure = "neutral"
        self.structure_strength = 0.5
        
    async def analyze_market_structure(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Analyze market structure using advanced techniques:
        - Higher highs / Lower lows pattern
        - Support/Resistance levels
        - Breakout potential
        - Volume at key levels
        """
        try:
            # Get multiple timeframe data
            timeframes = ["1", "5", "15", "60", "240"]
            candle_data = {}
            
            for tf in timeframes:
                response = await signed_request("GET", "/v5/market/kline", {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": tf,
                    "limit": 100
                })
                
                if response.get("retCode") == 0:
                    candle_data[tf] = normalize_klines(response["result"]["list"])
            
            if not candle_data:
                return self._get_default_structure()
            
            # Analyze structure across timeframes
            structure_signals = {}
            
            for tf, candles in candle_data.items():
                if len(candles) >= 50:
                    tf_analysis = self._analyze_timeframe_structure(candles, tf)
                    structure_signals[tf] = tf_analysis
            
            # Combine timeframe analysis
            overall_structure = self._combine_structure_signals(structure_signals)
            
            # Detect key levels
            key_levels = await self._detect_key_levels(candle_data)
            
            # Calculate breakout probability
            breakout_analysis = self._analyze_breakout_potential(candle_data, key_levels)
            
            result = {
                "structure": overall_structure["trend"],
                "strength": overall_structure["strength"],
                "confidence": overall_structure["confidence"],
                "key_levels": key_levels,
                "breakout_probability": breakout_analysis,
                "timeframe_analysis": structure_signals,
                "market_phase": self._determine_market_phase(overall_structure, breakout_analysis)
            }
            
            self.market_structure = result["structure"]
            self.structure_strength = result["strength"]
            
            return result
            
        except Exception as e:
            log(f"❌ Error analyzing market structure: {e}", level="ERROR")
            return self._get_default_structure()
    
    def _analyze_timeframe_structure(self, candles: List, timeframe: str) -> Dict[str, Any]:
        """Analyze market structure for a specific timeframe"""
        try:
            # Convert candle data
            highs = [float(c[2]) for c in candles]
            lows = [float(c[3]) for c in candles]
            closes = [float(c[4]) for c in candles]
            volumes = [float(c[5]) for c in candles]
            
            # Find swing highs and lows
            swing_highs = self._find_swing_points(highs, "high")
            swing_lows = self._find_swing_points(lows, "low")
            
            # Analyze pattern
            hh_ll_pattern = self._analyze_hh_ll_pattern(swing_highs, swing_lows)
            
            # Volume analysis at key points
            volume_confirmation = self._analyze_volume_at_swings(
                swing_highs, swing_lows, volumes, closes
            )
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(
                hh_ll_pattern, volume_confirmation, closes
            )
            
            return {
                "trend": hh_ll_pattern["trend"],
                "strength": trend_strength,
                "swing_highs": swing_highs,
                "swing_lows": swing_lows,
                "volume_confirmation": volume_confirmation,
                "timeframe": timeframe
            }
            
        except Exception as e:
            log(f"❌ Error analyzing {timeframe} structure: {e}", level="ERROR")
            return {"trend": "neutral", "strength": 0.5, "timeframe": timeframe}
    
    def _find_swing_points(self, data: List[float], point_type: str) -> List[Tuple[int, float]]:
        """Find swing highs or lows in price data"""
        swing_points = []
        lookback = 5
        
        for i in range(lookback, len(data) - lookback):
            if point_type == "high":
                if all(data[i] >= data[j] for j in range(i-lookback, i+lookback+1) if j != i):
                    swing_points.append((i, data[i]))
            else:  # low
                if all(data[i] <= data[j] for j in range(i-lookback, i+lookback+1) if j != i):
                    swing_points.append((i, data[i]))
        
        return swing_points[-10:]  # Keep last 10 swing points
    
    def _analyze_hh_ll_pattern(self, swing_highs: List, swing_lows: List) -> Dict[str, Any]:
        """Analyze Higher High/Lower Low patterns"""
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return {"trend": "neutral", "pattern": "insufficient_data"}
        
        # Check recent highs pattern
        recent_highs = [h[1] for h in swing_highs[-3:]]
        recent_lows = [l[1] for l in swing_lows[-3:]]
        
        # Higher highs and higher lows = uptrend
        hh = all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs)))
        hl = all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        # Lower highs and lower lows = downtrend
        lh = all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs)))
        ll = all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows)))
        
        if hh and hl:
            return {"trend": "uptrend", "pattern": "hh_hl", "strength": 0.8}
        elif lh and ll:
            return {"trend": "downtrend", "pattern": "lh_ll", "strength": 0.8}
        elif hh and not hl:
            return {"trend": "weak_uptrend", "pattern": "hh_only", "strength": 0.6}
        elif hl and not hh:
            return {"trend": "weak_uptrend", "pattern": "hl_only", "strength": 0.6}
        elif lh and not ll:
            return {"trend": "weak_downtrend", "pattern": "lh_only", "strength": 0.6}
        elif ll and not lh:
            return {"trend": "weak_downtrend", "pattern": "ll_only", "strength": 0.6}
        else:
            return {"trend": "ranging", "pattern": "mixed", "strength": 0.4}
    
    def _analyze_volume_at_swings(self, swing_highs: List, swing_lows: List, 
                                 volumes: List, closes: List) -> Dict[str, Any]:
        """Analyze volume confirmation at swing points"""
        try:
            avg_volume = np.mean(volumes)
            volume_confirmations = []
            
            # Check volume at swing highs
            for idx, price in swing_highs[-3:]:
                if idx < len(volumes):
                    vol_ratio = volumes[idx] / avg_volume
                    volume_confirmations.append({
                        "type": "high",
                        "volume_ratio": vol_ratio,
                        "confirmed": vol_ratio > 1.2
                    })
            
            # Check volume at swing lows
            for idx, price in swing_lows[-3:]:
                if idx < len(volumes):
                    vol_ratio = volumes[idx] / avg_volume
                    volume_confirmations.append({
                        "type": "low", 
                        "volume_ratio": vol_ratio,
                        "confirmed": vol_ratio > 1.2
                    })
            
            confirmation_rate = sum(1 for v in volume_confirmations if v["confirmed"]) / len(volume_confirmations) if volume_confirmations else 0
            
            return {
                "confirmation_rate": confirmation_rate,
                "confirmations": volume_confirmations,
                "volume_trend": ("increasing" if (np.mean(volumes[-5:]) if len(volumes) >= 5 else np.mean(volumes)) > (np.mean(volumes[-10:-5]) if len(volumes) >= 10 else (np.mean(volumes[-5:]) if len(volumes) >= 5 else np.mean(volumes))) else "decreasing")
            }
            
        except Exception as e:
            return {"confirmation_rate": 0.5, "confirmations": [], "volume_trend": "neutral"}
    
    def _calculate_trend_strength(self, hh_ll_pattern: Dict, volume_confirmation: Dict, closes: List) -> float:
        """Calculate overall trend strength"""
        base_strength = hh_ll_pattern.get("strength", 0.5)
        volume_boost = volume_confirmation.get("confirmation_rate", 0.5) * 0.3
        
        # Price momentum component
        momentum = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        momentum_component = min(abs(momentum) * 2, 0.2)
        
        return min(base_strength + volume_boost + momentum_component, 1.0)
    
    async def _detect_key_levels(self, candle_data: Dict) -> Dict[str, List[float]]:
        """Detect key support and resistance levels"""
        try:
            all_highs = []
            all_lows = []
            
            # Collect data from multiple timeframes
            for tf, candles in candle_data.items():
                if tf in ["15", "60", "240"]:  # Focus on higher timeframes for key levels
                    highs = [float(c[2]) for c in candles]
                    lows = [float(c[3]) for c in candles]
                    all_highs.extend(highs)
                    all_lows.extend(lows)
            
            # Find resistance levels (areas where price has been rejected)
            resistance_levels = self._find_level_clusters(all_highs, "resistance")
            
            # Find support levels (areas where price has bounced)
            support_levels = self._find_level_clusters(all_lows, "support")
            
            return {
                "resistance": resistance_levels,
                "support": support_levels,
                "key_level_strength": len(resistance_levels) + len(support_levels)
            }
            
        except Exception as e:
            log(f"❌ Error detecting key levels: {e}", level="ERROR")
            return {"resistance": [], "support": [], "key_level_strength": 0}
    
    def _find_level_clusters(self, prices: List[float], level_type: str) -> List[float]:
        """Find clustered price levels that act as support/resistance"""
        if not prices:
            return []
        
        # Create price histogram
        min_price, max_price = min(prices), max(prices)
        price_range = max_price - min_price
        if price_range <= 0:
            return []
        bin_size = max(price_range / 100.0, 1e-9)  # 100 bins
        
        level_counts = defaultdict(int)
        
        for price in prices:
            bin_level = round(price / bin_size) * bin_size
            level_counts[bin_level] += 1
        
        # Find levels with high touch count
        significant_levels = []
        for level, count in level_counts.items():
            if count >= 3:  # Level touched at least 3 times
                significant_levels.append((level, count))
        
        # Sort by importance and return top levels
        significant_levels.sort(key=lambda x: x[1], reverse=True)
        return [level for level, count in significant_levels[:5]]
    
    def _analyze_breakout_potential(self, candle_data: Dict, key_levels: Dict) -> Dict[str, Any]:
        """Analyze potential for breakouts from current levels"""
        try:
            if "5" not in candle_data:
                return {"probability": 0.5, "direction": "neutral"}
            
            current_candles = candle_data["5"]
            current_price = float(current_candles[-1][4])  # Latest close
            current_volume = float(current_candles[-1][5])
            
            # Check proximity to key levels
            resistance_levels = key_levels.get("resistance", [])
            support_levels = key_levels.get("support", [])
            
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price)) if resistance_levels else None
            nearest_support = min(support_levels, key=lambda x: abs(x - current_price)) if support_levels else None
            
            # Calculate distance to key levels
            resistance_distance = abs(nearest_resistance - current_price) / current_price if nearest_resistance else 1
            support_distance = abs(nearest_support - current_price) / current_price if nearest_support else 1
            
            # Volume analysis for breakout
            avg_volume = np.mean([float(c[5]) for c in current_candles[-20:]])
            volume_surge = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Calculate breakout probability
            base_probability = 0.5
            
            # Higher probability near key levels
            # Proximity defaults (fade)
            if resistance_distance < 0.02:
                base_probability += 0.2
                direction_bias = "bearish"
            elif support_distance < 0.02:
                base_probability += 0.2
                direction_bias = "bullish"
            else:
                direction_bias = "neutral"

            # Flip if an actual break occurs with volume
            if volume_surge > 1.5:
                broke_res = (nearest_resistance is not None) and (current_price > nearest_resistance)
                broke_sup = (nearest_support is not None) and (current_price < nearest_support)
                if broke_res:
                    direction_bias = "bullish"
                elif broke_sup:
                    direction_bias = "bearish"
            
            # Price momentum component
            recent_prices = [float(c[4]) for c in current_candles[-10:]]
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(momentum) > 0.02:  # 2% momentum
                base_probability += 0.1
                if momentum > 0 and direction_bias != "bearish":
                    direction_bias = "bullish"
                elif momentum < 0 and direction_bias != "bullish":
                    direction_bias = "bearish"
            
            return {
                "probability": min(base_probability, 0.9),
                "direction": direction_bias,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "volume_surge": volume_surge,
                "momentum": momentum
            }
            
        except Exception as e:
            log(f"❌ Error analyzing breakout potential: {e}", level="ERROR")
            return {"probability": 0.5, "direction": "neutral"}
    
    def _combine_structure_signals(self, structure_signals: Dict) -> Dict[str, Any]:
        """Combine structure analysis from multiple timeframes"""
        if not structure_signals:
            return {"trend": "neutral", "strength": 0.5, "confidence": 30}
        
        # Weight timeframes by importance
        tf_weights = {"1": 0.1, "5": 0.2, "15": 0.3, "60": 0.25, "240": 0.15}
        
        trend_scores = {"uptrend": 0, "downtrend": 0, "neutral": 0, "ranging": 0}
        total_weight = 0
        strengths = []
        
        for tf, analysis in structure_signals.items():
            weight = tf_weights.get(tf, 0.1)
            trend = analysis.get("trend", "neutral")
            strength = analysis.get("strength", 0.5)
            
            # Map trend variations to main categories
            if "uptrend" in trend:
                trend_scores["uptrend"] += weight * strength
            elif "downtrend" in trend:
                trend_scores["downtrend"] += weight * strength
            elif trend == "ranging":
                trend_scores["ranging"] += weight * strength
            else:
                trend_scores["neutral"] += weight * strength
            
            total_weight += weight
            strengths.append(strength)
        
        # Normalize scores
        if total_weight > 0:
            for trend in trend_scores:
                trend_scores[trend] /= total_weight
        
        # Determine overall trend
        overall_trend = max(trend_scores.items(), key=lambda x: x[1])[0]
        overall_strength = trend_scores[overall_trend]
        
        # Calculate confidence based on agreement across timeframes
        confidence = np.mean(strengths) * 100 if strengths else 50
        confidence = max(30, min(95, confidence))
        
        return {
            "trend": overall_trend,
            "strength": overall_strength,
            "confidence": confidence,
            "trend_scores": trend_scores
        }
    
    def _determine_market_phase(self, structure: Dict, breakout: Dict) -> str:
        """Determine current market phase"""
        trend = structure.get("trend", "neutral")
        strength = structure.get("strength", 0.5)
        breakout_prob = breakout.get("probability", 0.5)
        
        if breakout_prob > 0.7:
            return "breakout_imminent"
        elif trend in ["uptrend", "downtrend"] and strength > 0.7:
            return "trending"
        elif trend == "ranging" and strength > 0.6:
            return "consolidation"
        elif strength < 0.4:
            return "accumulation"
        else:
            return "neutral"
    
    def _get_default_structure(self) -> Dict[str, Any]:
        """Return default structure analysis"""
        return {
            "structure": "neutral",
            "strength": 0.5,
            "confidence": 30,
            "key_levels": {"resistance": [], "support": [], "key_level_strength": 0},
            "breakout_probability": {"probability": 0.5, "direction": "neutral"},
            "timeframe_analysis": {},
            "market_phase": "neutral"
        }


class EnhancedAltseasonDetector:
    """Enhanced altseason detection with broader analysis"""
    
    def __init__(self):
        self.monitored_symbols = [
            # Large caps
            "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
            # Mid caps
            "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT",
            "UNIUSDT", "ATOMUSDT", "NEARUSDT", "FTMUSDT", "SANDUSDT",
            # Small caps / trending
            "APTUSDT", "OPUSDT", "ARBUSDT", "SUIUSDT", "SEIUSDT",
            "INJUSDT", "TIAUSDT", "STRKUSDT", "WUSDT", "PENDLEUSDT",
            # DeFi tokens
            "AAVEUSDT", "MKRUSDT", "COMPUSDT", "CRVUSDT", "SUSHIUSDT",
            # Layer 1s
            "ALGOUSDT", "HBARUSDT", "IOTAUSDT", "VETUSDT", "XLMUSDT",
            # Gaming/NFT
            "AXSUSDT", "MANAUSDT", "CHZUSDT", "ENJUSDT", "GALAUSDT"
        ]
        self.btc_dominance_history = deque(maxlen=30)
        self.alt_performance_data = {}
        self.market_cap_data = {}
        
    async def detect_enhanced_altseason(self) -> Dict[str, Any]:
        """Enhanced altseason detection with comprehensive analysis"""
        try:
            # 1. Alt performance analysis (expanded sample)
            alt_performance = await self._analyze_expanded_alt_performance()
            
            # 2. Market cap flow analysis
            mcap_analysis = await self._analyze_market_cap_flows()
            
            # 3. Sector rotation analysis
            sector_analysis = await self._analyze_sector_rotation()
            
            # 4. Social sentiment analysis
            social_sentiment = await self._analyze_social_sentiment()
            
            # 5. Institutional flow analysis
            institutional_flow = await self._analyze_institutional_flows()
            
            # Combine all signals
            altseason_score = self._calculate_altseason_score({
                "alt_performance": alt_performance,
                "market_cap": mcap_analysis,
                "sector_rotation": sector_analysis,
                "social_sentiment": social_sentiment,
                "institutional_flow": institutional_flow
            })
            
            return altseason_score
            
        except Exception as e:
            log(f"❌ Error in enhanced altseason detection: {e}", level="ERROR")
            return self._get_default_altseason()
    
    async def _analyze_expanded_alt_performance(self) -> Dict[str, Any]:
        """Analyze performance across expanded altcoin universe"""
        try:
            # Get BTC performance
            btc_response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            })
            
            if btc_response.get("retCode") != 0:
                return {"outperforming_ratio": 0.5, "avg_outperformance": 0, "strength": 0.5}
            
            btc_change = float(btc_response["result"]["list"][0]["price24hPcnt"])
            
            # Get alt performance in batches
            outperforming_alts = 0
            total_alts = 0
            performance_deltas = []
            
            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(self.monitored_symbols), batch_size):
                batch = self.monitored_symbols[i:i+batch_size]
                
                for symbol in batch:
                    try:
                        response = await signed_request("GET", "/v5/market/tickers", {
                            "category": "linear",
                            "symbol": symbol
                        })
                        
                        if response.get("retCode") == 0 and response["result"]["list"]:
                            alt_change = float(response["result"]["list"][0]["price24hPcnt"])
                            performance_delta = alt_change - btc_change
                            
                            performance_deltas.append(performance_delta)
                            total_alts += 1
                            
                            if performance_delta > 0:
                                outperforming_alts += 1
                                
                    except Exception as e:
                        log(f"❌ Error getting {symbol} performance: {e}", level="DEBUG")
                        continue
                
                # Rate limit protection
                await asyncio.sleep(0.1)
            
            if total_alts == 0:
                return {"outperforming_ratio": 0.5, "avg_outperformance": 0, "strength": 0.5}
            
            outperforming_ratio = outperforming_alts / total_alts
            avg_outperformance = np.mean(performance_deltas)
            
            # Calculate strength
            strength = min(outperforming_ratio * 2, 1.0) if outperforming_ratio > 0.6 else outperforming_ratio
            
            return {
                "outperforming_ratio": outperforming_ratio,
                "avg_outperformance": avg_outperformance,
                "strength": strength,
                "total_analyzed": total_alts,
                "btc_change": btc_change
            }
            
        except Exception as e:
            log(f"❌ Error analyzing alt performance: {e}", level="ERROR")
            return {"outperforming_ratio": 0.5, "avg_outperformance": 0, "strength": 0.5}
    
    async def _analyze_market_cap_flows(self) -> Dict[str, Any]:
        """Analyze market cap flows between BTC and alts"""
        try:
            # This would require CoinGecko or similar API for market cap data
            # For now, simulate based on volume and price action
            
            # Get volume data for BTC vs alt leaders
            btc_vol_response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            })
            
            alt_leaders = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]
            alt_volumes = []
            
            for symbol in alt_leaders:
                try:
                    response = await signed_request("GET", "/v5/market/tickers", {
                        "category": "linear",
                        "symbol": symbol
                    })
                    
                    if response.get("retCode") == 0 and response["result"]["list"]:
                        volume = float(response["result"]["list"][0]["volume24h"])
                        alt_volumes.append(volume)
                        
                except Exception:
                    continue
                
                await asyncio.sleep(0.05)
            
            if not alt_volumes or btc_vol_response.get("retCode") != 0:
                return {"flow_direction": "neutral", "strength": 0.5}
            
            btc_volume = float(btc_vol_response["result"]["list"][0]["volume24h"])
            total_alt_volume = sum(alt_volumes)
            
            # Calculate flow ratio
            flow_ratio = total_alt_volume / (btc_volume + total_alt_volume) if btc_volume > 0 else 0.5
            
            if flow_ratio > 0.6:
                flow_direction = "into_alts"
                strength = min(flow_ratio * 1.5, 1.0)
            elif flow_ratio < 0.3:
                flow_direction = "into_btc"
                strength = min((1 - flow_ratio) * 1.5, 1.0)
            else:
                flow_direction = "neutral"
                strength = 0.5
            
            return {
                "flow_direction": flow_direction,
                "strength": strength,
                "flow_ratio": flow_ratio,
                "btc_volume": btc_volume,
                "alt_volume": total_alt_volume
            }
            
        except Exception as e:
            log(f"❌ Error analyzing market cap flows: {e}", level="ERROR")
            return {"flow_direction": "neutral", "strength": 0.5}
    
    async def _analyze_sector_rotation(self) -> Dict[str, Any]:
        """Analyze sector rotation patterns"""
        try:
            sectors = {
                "defi": ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "COMPUSDT", "CRVUSDT"],
                "layer1": ["ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "AVAXUSDT"],
                "gaming": ["AXSUSDT", "MANAUSDT", "SANDUSDT", "ENJUSDT", "GALAUSDT"],
                "infrastructure": ["LINKUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT", "FTMUSDT"]
            }
            
            sector_performance = {}
            
            for sector, symbols in sectors.items():
                sector_changes = []
                
                for symbol in symbols:
                    try:
                        response = await signed_request("GET", "/v5/market/tickers", {
                            "category": "linear",
                            "symbol": symbol
                        })
                        
                        if response.get("retCode") == 0 and response["result"]["list"]:
                            change = float(response["result"]["list"][0]["price24hPcnt"])
                            sector_changes.append(change)
                            
                    except Exception:
                        continue
                    
                    await asyncio.sleep(0.05)
                
                if sector_changes:
                    sector_performance[sector] = {
                        "avg_change": np.mean(sector_changes),
                        "positive_ratio": sum(1 for c in sector_changes if c > 0) / len(sector_changes)
                    }
            
            # Find leading sector
            if sector_performance:
                leading_sector = max(sector_performance.items(), 
                                   key=lambda x: x[1]["avg_change"])[0]
                
                leading_performance = sector_performance[leading_sector]["avg_change"]
                rotation_strength = min(abs(leading_performance) / 5, 1.0)  # Normalize to 5% max
                
                return {
                    "leading_sector": leading_sector,
                    "rotation_strength": rotation_strength,
                    "sector_data": sector_performance
                }
            
            return {"leading_sector": "none", "rotation_strength": 0.5, "sector_data": {}}
            
        except Exception as e:
            log(f"❌ Error analyzing sector rotation: {e}", level="ERROR")
            return {"leading_sector": "none", "rotation_strength": 0.5, "sector_data": {}}
    
    async def _analyze_social_sentiment(self) -> Dict[str, Any]:
        """Analyze social sentiment indicators"""
        try:
            # This would integrate with Fear & Greed Index, social APIs, etc.
            # For now, simulate based on market behavior patterns
            
            # Get volatility as sentiment proxy
            btc_response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "1",
                "limit": 100
            })
            
            if btc_response.get("retCode") != 0:
                return {"sentiment": "neutral", "strength": 0.5}
            
            candles = normalize_klines(btc_response["result"]["list"])
            price_changes = []
            
            for candle in candles:
                open_price = float(candle[1])
                close_price = float(candle[4])
                change = (close_price - open_price) / open_price
                price_changes.append(change)
            
            # Calculate sentiment metrics
            volatility = np.std(price_changes)
            trend = np.mean(price_changes)
            
            # Sentiment scoring
            if volatility > 0.02:  # High volatility
                if trend > 0:
                    sentiment = "euphoric"
                    strength = min(volatility * 25, 1.0)
                else:
                    sentiment = "fearful"
                    strength = min(volatility * 25, 1.0)
            elif abs(trend) < 0.005:  # Low movement
                sentiment = "apathetic"
                strength = 0.3
            elif trend > 0:
                sentiment = "optimistic"
                strength = min(abs(trend) * 100, 0.8)
            else:
                sentiment = "pessimistic"
                strength = min(abs(trend) * 100, 0.8)
            
            return {
                "sentiment": sentiment,
                "strength": strength,
                "volatility": volatility,
                "trend": trend
            }
            
        except Exception as e:
            log(f"❌ Error analyzing social sentiment: {e}", level="ERROR")
            return {"sentiment": "neutral", "strength": 0.5}
    
    async def _analyze_institutional_flows(self) -> Dict[str, Any]:
        """Analyze institutional flow patterns"""
        try:
            # Analyze large transaction patterns and order book depth
            # This is a simplified version - would need access to institutional data
            
            large_cap_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
            institutional_signals = []
            
            for symbol in large_cap_symbols:
                try:
                    # Get recent trades to analyze for large transactions
                    response = await signed_request("GET", "/v5/market/recent-trade", {
                        "category": "linear",
                        "symbol": symbol,
                        "limit": 100
                    })
                    
                    if response.get("retCode") == 0:
                        trades = normalize_klines(response["result"]["list"])
                        
                        # Analyze trade sizes
                        trade_sizes = [float(trade["size"]) for trade in trades]
                        avg_size = np.mean(trade_sizes)
                        large_trades = [size for size in trade_sizes if size > avg_size * 3]
                        
                        # Calculate institutional activity score
                        if trade_sizes:
                            large_trade_ratio = len(large_trades) / len(trade_sizes)
                            institutional_signals.append(large_trade_ratio)
                        
                except Exception:
                    continue
                
                await asyncio.sleep(0.1)
            
            if institutional_signals:
                avg_institutional_activity = np.mean(institutional_signals)
                
                if avg_institutional_activity > 0.15:
                    flow_type = "institutional_accumulation"
                    strength = min(avg_institutional_activity * 4, 1.0)
                elif avg_institutional_activity < 0.05:
                    flow_type = "retail_dominated"
                    strength = 0.4
                else:
                    flow_type = "mixed_flow"
                    strength = 0.6
            else:
                flow_type = "unknown"
                strength = 0.5
            
            return {
                "flow_type": flow_type,
                "strength": strength,
                "institutional_activity": avg_institutional_activity if institutional_signals else 0
            }
            
        except Exception as e:
            log(f"❌ Error analyzing institutional flows: {e}", level="ERROR")
            return {"flow_type": "unknown", "strength": 0.5}
    
    def _calculate_altseason_score(self, analyses: Dict) -> Dict[str, Any]:
        """Calculate overall altseason score from all analyses"""
        try:
            # Weighted scoring system
            weights = {
                "alt_performance": 0.35,
                "market_cap": 0.25,
                "sector_rotation": 0.15,
                "social_sentiment": 0.15,
                "institutional_flow": 0.10
            }
            
            total_score = 0
            total_weight = 0
            details = {}
            
            for analysis_type, analysis_data in analyses.items():
                if analysis_type in weights:
                    weight = weights[analysis_type]
                    strength = analysis_data.get("strength", 0.5)
                    
                    total_score += strength * weight
                    total_weight += weight
                    details[analysis_type] = analysis_data
            
            final_score = total_score / total_weight if total_weight > 0 else 0.5
            
            # Determine altseason status
            if final_score >= 0.75:
                season = "strong_altseason"
                is_altseason = True
            elif final_score >= 0.6:
                season = "altseason"
                is_altseason = True
            elif final_score <= 0.3:
                season = "btc_season"
                is_altseason = False
            else:
                season = "neutral"
                is_altseason = False
            
            return {
                "is_altseason": is_altseason,
                "season": season,
                "strength": final_score,
                "confidence": min(final_score * 100 + 20, 95),
                "details": details,
                "components": {
                    "alt_outperformance": analyses["alt_performance"].get("outperforming_ratio", 0.5),
                    "market_cap_flow": analyses["market_cap"].get("flow_ratio", 0.5),
                    "sector_strength": analyses["sector_rotation"].get("rotation_strength", 0.5),
                    "sentiment_score": analyses["social_sentiment"].get("strength", 0.5),
                    "institutional_score": analyses["institutional_flow"].get("strength", 0.5)
                }
            }
            
        except Exception as e:
            log(f"❌ Error calculating altseason score: {e}", level="ERROR")
            return self._get_default_altseason()
    
    def _get_default_altseason(self) -> Dict[str, Any]:
        """Return default altseason analysis"""
        return {
            "is_altseason": False,
            "season": "neutral",
            "strength": 0.5,
            "confidence": 50,
            "details": {},
            "components": {}
        }


class MultiSourceSentimentAnalyzer:
    """Multi-source sentiment analysis aggregator"""
    
    def __init__(self):
        self.sentiment_sources = {
            "fear_greed": {"weight": 0.3, "last_value": 50},
            "social_volume": {"weight": 0.2, "last_value": 0.5},
            "news_sentiment": {"weight": 0.25, "last_value": 0.5},
            "options_flow": {"weight": 0.25, "last_value": 0.5}
        }
    
    async def get_aggregated_sentiment(self) -> Dict[str, Any]:
        """Get aggregated sentiment from multiple sources"""
        try:
            sentiment_scores = {}
            
            # 1. Fear & Greed Index (simulated)
            fear_greed = await self._get_fear_greed_index()
            sentiment_scores["fear_greed"] = fear_greed
            
            # 2. Social volume analysis
            social_volume = await self._analyze_social_volume()
            sentiment_scores["social_volume"] = social_volume
            
            # 3. News sentiment (simulated)
            news_sentiment = await self._analyze_news_sentiment()
            sentiment_scores["news_sentiment"] = news_sentiment
            
            # 4. Options flow analysis (simulated)
            options_flow = await self._analyze_options_sentiment()
            sentiment_scores["options_flow"] = options_flow
            
            # Aggregate all scores
            aggregated = self._aggregate_sentiment_scores(sentiment_scores)
            
            return aggregated
            
        except Exception as e:
            log(f"❌ Error getting aggregated sentiment: {e}", level="ERROR")
            return self._get_default_sentiment()
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index (with smart fallback)"""
        try:
            # Try to import and use the API manager
            from api_config import get_fear_greed
            
            fg_value = await get_fear_greed()
            
            # Map to sentiment
            if fg_value <= 20:
                sentiment = "extreme_fear"
            elif fg_value <= 40:
                sentiment = "fear"
            elif fg_value <= 60:
                sentiment = "neutral"
            elif fg_value <= 80:
                sentiment = "greed"
            else:
                sentiment = "extreme_greed"
            
            return {
                "value": fg_value,
                "sentiment": sentiment,
                "strength": min(abs(fg_value - 50) / 50, 1.0)
            }
            
        except ImportError:
            log("⚠️ API config not available, using built-in fallback", level="WARNING")
            return await self._builtin_fear_greed_fallback()
        except Exception as e:
            log(f"❌ Error getting fear & greed index: {e}", level="ERROR")
            return await self._builtin_fear_greed_fallback()
    
    async def _builtin_fear_greed_fallback(self) -> Dict[str, Any]:
        """Built-in fallback that works with just Bybit API"""
        try:
            # Get BTC volatility as proxy for fear/greed
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 24
            })
            
            if response.get("retCode") != 0:
                return {"value": 50, "sentiment": "neutral", "strength": 0.5}
            
            candles = normalize_klines(response["result"]["list"])
            price_changes = []
            
            for candle in candles:
                open_price = float(candle[1])
                close_price = float(candle[4])
                change = abs((close_price - open_price) / open_price)
                price_changes.append(change)
            
            volatility = np.mean(price_changes)
            
            # Convert volatility to fear/greed score
            if volatility > 0.03:  # High volatility
                fg_value = 25  # Fear
                sentiment = "extreme_fear"
            elif volatility > 0.02:
                fg_value = 40
                sentiment = "fear"
            elif volatility < 0.005:  # Low volatility
                fg_value = 75  # Greed
                sentiment = "greed"
            else:
                fg_value = 50
                sentiment = "neutral"
            
            return {
                "value": fg_value,
                "sentiment": sentiment,
                "strength": min(abs(fg_value - 50) / 50, 1.0)
            }
            
        except Exception as e:
            log(f"❌ Built-in fear & greed fallback failed: {e}", level="ERROR")
            return {"value": 50, "sentiment": "neutral", "strength": 0.5}
    
    async def _analyze_social_volume(self) -> Dict[str, Any]:
        """Analyze social media volume patterns"""
        try:
            # This would integrate with Twitter/Reddit APIs
            # For now, simulate based on trading volume
            
            response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            })
            
            if response.get("retCode") != 0:
                return {"volume_trend": "neutral", "strength": 0.5}
            
            current_volume = float(response["result"]["list"][0]["volume24h"])
            
            # Get historical volume for comparison
            hist_response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear", 
                "symbol": "BTCUSDT",
                "interval": "240",
                "limit": 7
            })
            
            if hist_response.get("retCode") != 0:
                return {"volume_trend": "neutral", "strength": 0.5}
            
            hist_volumes = [float(candle[5]) for candle in hist_response["result"]["list"]]
            avg_volume = np.mean(hist_volumes)
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:
                volume_trend = "high_interest"
                strength = min((volume_ratio - 1) / 2, 1.0)
            elif volume_ratio < 0.7:
                volume_trend = "low_interest"
                strength = min((1 - volume_ratio) / 0.5, 1.0)
            else:
                volume_trend = "normal"
                strength = 0.5
            
            return {
                "volume_trend": volume_trend,
                "strength": strength,
                "volume_ratio": volume_ratio
            }
            
        except Exception as e:
            log(f"❌ Error analyzing social volume: {e}", level="ERROR")
            return {"volume_trend": "neutral", "strength": 0.5}
    
    async def _analyze_news_sentiment(self) -> Dict[str, Any]:
        """Analyze news sentiment (simulated)"""
        try:
            # This would integrate with news APIs
            # For now, simulate based on price momentum
            
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT", 
                "interval": "240",
                "limit": 10
            })
            
            if response.get("retCode") != 0:
                return {"sentiment": "neutral", "strength": 0.5}
            
            candles = normalize_klines(response["result"]["list"])
            first = float(candles[0][4])
            last  = float(candles[-1][4])
            price_trend = (last - first) / first if first > 0 else 0
            
            if price_trend > 0.05:
                sentiment = "very_positive"
                strength = min(price_trend * 10, 1.0)
            elif price_trend > 0.02:
                sentiment = "positive"
                strength = min(price_trend * 15, 0.8)
            elif price_trend < -0.05:
                sentiment = "very_negative"
                strength = min(abs(price_trend) * 10, 1.0)
            elif price_trend < -0.02:
                sentiment = "negative"
                strength = min(abs(price_trend) * 15, 0.8)
            else:
                sentiment = "neutral"
                strength = 0.5
            
            return {
                "sentiment": sentiment,
                "strength": strength,
                "price_trend": price_trend
            }
            
        except Exception as e:
            log(f"❌ Error analyzing news sentiment: {e}", level="ERROR")
            return {"sentiment": "neutral", "strength": 0.5}
    
    async def _analyze_options_sentiment(self) -> Dict[str, Any]:
        """Analyze options sentiment (simulated)"""
        try:
            # This would analyze put/call ratios, open interest, etc.
            # For now, simulate based on volatility patterns
            
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 48
            })
            
            if response.get("retCode") != 0:
                return {"options_sentiment": "neutral", "strength": 0.5}
            
            candles = normalize_klines(response["result"]["list"])
            
            # Calculate implied volatility proxy
            price_ranges = []
            for candle in candles:
                high = float(candle[2])
                low = float(candle[3])
                close = float(candle[4])
                price_range = (high - low) / close if close > 0 else 0
                price_ranges.append(price_range)
            
            avg_range = np.mean(price_ranges)
            recent_range = np.mean(price_ranges[-12:])  # Last 12 hours
            
            volatility_trend = recent_range / avg_range if avg_range > 0 else 1
            
            if volatility_trend > 1.3:
                options_sentiment = "high_vol_expected"
                strength = min((volatility_trend - 1) / 1, 1.0)
            elif volatility_trend < 0.7:
                options_sentiment = "low_vol_expected"
                strength = min((1 - volatility_trend) / 0.5, 1.0)
            else:
                options_sentiment = "stable_vol"
                strength = 0.5
            
            return {
                "options_sentiment": options_sentiment,
                "strength": strength,
                "volatility_trend": volatility_trend
            }
            
        except Exception as e:
            log(f"❌ Error analyzing options sentiment: {e}", level="ERROR")
            return {"options_sentiment": "neutral", "strength": 0.5}
    
    def _aggregate_sentiment_scores(self, sentiment_scores: Dict) -> Dict[str, Any]:
        """Aggregate sentiment scores from all sources"""
        try:
            total_score = 0
            total_weight = 0
            component_details = {}
            
            for source, data in sentiment_scores.items():
                if source in self.sentiment_sources:
                    weight = self.sentiment_sources[source]["weight"]
                    strength = data.get("strength", 0.5)
                    
                    total_score += strength * weight
                    total_weight += weight
                    component_details[source] = data
            
            final_score = total_score / total_weight if total_weight > 0 else 0.5
            
            # Determine overall sentiment
            if final_score >= 0.8:
                overall_sentiment = "extremely_bullish"
            elif final_score >= 0.65:
                overall_sentiment = "bullish"
            elif final_score >= 0.35:
                overall_sentiment = "neutral"
            elif final_score >= 0.2:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "extremely_bearish"
            
            return {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": final_score,
                "confidence": min(final_score * 100 + 20, 95),
                "components": component_details,
                "market_mood": self._determine_market_mood(final_score, component_details)
            }
            
        except Exception as e:
            log(f"❌ Error aggregating sentiment scores: {e}", level="ERROR")
            return self._get_default_sentiment()
    
    def _determine_market_mood(self, score: float, components: Dict) -> str:
        """Determine overall market mood"""
        fear_greed = components.get("fear_greed", {}).get("sentiment", "neutral")
        social = components.get("social_volume", {}).get("volume_trend", "normal")
        news = components.get("news_sentiment", {}).get("sentiment", "neutral")
        
        if score > 0.7 and social == "high_interest":
            return "euphoric"
        elif score > 0.6 and "positive" in news:
            return "optimistic"
        elif score < 0.3 and "fear" in fear_greed:
            return "panic"
        elif score < 0.4 and social == "low_interest":
            return "apathetic"
        else:
            return "cautious"
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default sentiment analysis"""
        return {
            "overall_sentiment": "neutral",
            "sentiment_score": 0.5,
            "confidence": 50,
            "components": {},
            "market_mood": "cautious"
        }


class VolumeProfileEngine:
    """Advanced volume profile analysis for institutional activity detection"""
    
    def __init__(self):
        self.volume_nodes = []
        self.poc_levels = []  # Point of Control levels
        self.value_areas = []
        
    async def analyze_volume_profile(self, symbol: str = "BTCUSDT", 
                                   timeframe: str = "15", 
                                   lookback: int = 100) -> Dict[str, Any]:
        """Analyze volume profile for institutional activity"""
        try:
            # Get candle data
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": symbol,
                "interval": timeframe,
                "limit": lookback
            })
            
            if response.get("retCode") != 0:
                return self._get_default_volume_profile()
            
            candles = normalize_klines(response["result"]["list"])
            
            # Build volume profile
            volume_profile = self._build_volume_profile(candles)
            
            # Analyze institutional patterns
            institutional_analysis = self._analyze_institutional_patterns(volume_profile, candles)
            
            # Detect accumulation/distribution
            accumulation_analysis = self._detect_accumulation_distribution(volume_profile, candles)
            
            # Find high-volume nodes
            volume_nodes = self._find_high_volume_nodes(volume_profile)
            
            # Calculate support/resistance strength
            sr_analysis = self._analyze_support_resistance_strength(volume_nodes, candles)
            
            return {
                "volume_profile": volume_profile,
                "institutional_activity": institutional_analysis,
                "accumulation_distribution": accumulation_analysis,
                "volume_nodes": volume_nodes,
                "support_resistance": sr_analysis,
                "poc_level": volume_nodes[0]["price"] if volume_nodes else None,
                "value_area": self._calculate_value_area(volume_profile)
            }
            
        except Exception as e:
            log(f"❌ Error analyzing volume profile: {e}", level="ERROR")
            return self._get_default_volume_profile()
    
    def _build_volume_profile(self, candles: List) -> Dict[float, float]:
        """Build volume profile from candle data"""
        volume_profile = defaultdict(float)
        
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            volume = float(candle[5])
            
            # Distribute volume across price range
            price_range = high - low
            if price_range > 0:
                # Use 10 price levels per candle
                levels = 10
                volume_per_level = volume / levels
                price_step = price_range / levels
                
                for i in range(levels):
                    price_level = low + (i * price_step)
                    # Round to reasonable precision
                    price_key = round(price_level, 2)
                    volume_profile[price_key] += volume_per_level
        
        return dict(volume_profile)
    
    def _analyze_institutional_patterns(self, volume_profile: Dict, candles: List) -> Dict[str, Any]:
        """Analyze volume profile for institutional trading patterns"""
        try:
            # Sort by volume to find highest activity areas
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_levels:
                return {"activity_level": "low", "pattern": "none", "strength": 0.5}
            
            # Analyze top volume areas
            top_10_percent = int(len(sorted_levels) * 0.1) or 1
            high_volume_levels = sorted_levels[:top_10_percent]
            
            total_volume = sum(volume_profile.values())
            high_volume_concentration = sum(vol for _, vol in high_volume_levels) / total_volume
            
            # Get current price for context
            current_price = float(candles[-1][4])
            
            # Analyze volume distribution relative to current price
            above_current = sum(vol for price, vol in high_volume_levels if price > current_price)
            below_current = sum(vol for price, vol in high_volume_levels if price < current_price)
            
            volume_imbalance = abs(above_current - below_current) / (above_current + below_current) if (above_current + below_current) > 0 else 0
            
            # Determine institutional activity type
            if high_volume_concentration > 0.6:
                if volume_imbalance > 0.3:
                    if above_current > below_current:
                        pattern = "institutional_resistance"
                    else:
                        pattern = "institutional_support"
                    activity_level = "high"
                else:
                    pattern = "institutional_accumulation"
                    activity_level = "very_high"
            elif high_volume_concentration > 0.4:
                pattern = "moderate_institutional"
                activity_level = "medium"
            else:
                pattern = "retail_dominated"
                activity_level = "low"
            
            return {
                "activity_level": activity_level,
                "pattern": pattern,
                "strength": high_volume_concentration,
                "volume_imbalance": volume_imbalance,
                "concentration_ratio": high_volume_concentration
            }
            
        except Exception as e:
            log(f"❌ Error analyzing institutional patterns: {e}", level="ERROR")
            return {"activity_level": "low", "pattern": "none", "strength": 0.5}
    
    def _detect_accumulation_distribution(self, volume_profile: Dict, candles: List) -> Dict[str, Any]:
        """Detect accumulation/distribution phases"""
        try:
            if len(candles) < 20:
                return {"phase": "unknown", "strength": 0.5}
            
            # Get price trend
            prices = [float(c[4]) for c in candles[-20:]]
            price_trend = (prices[-1] - prices[0]) / prices[0]
            
            # Get volume trend
            volumes = [float(c[5]) for c in candles[-20:]]
            early_vol = np.mean(volumes[:10])
            recent_vol = np.mean(volumes[-10:])
            volume_trend = (recent_vol - early_vol) / early_vol if early_vol > 0 else 0
            
            # Analyze volume profile distribution
            current_price = float(candles[-1][4])
            
            # Calculate volume above and below current price
            total_volume = sum(volume_profile.values())
            volume_below = sum(vol for price, vol in volume_profile.items() if price < current_price)
            volume_above = sum(vol for price, vol in volume_profile.items() if price > current_price)
            
            volume_balance = (volume_below - volume_above) / total_volume if total_volume > 0 else 0
            
            # Determine accumulation/distribution
            if volume_trend > 0.2 and abs(price_trend) < 0.05:
                # High volume, low price movement = accumulation
                if volume_balance > 0.1:
                    phase = "accumulation_below"
                else:
                    phase = "accumulation_above"
                strength = min(volume_trend * 2, 1.0)
            elif volume_trend < -0.2 and abs(price_trend) > 0.05:
                # Decreasing volume with price movement = distribution
                phase = "distribution"
                strength = min(abs(volume_trend) * 2, 1.0)
            elif price_trend > 0.05 and volume_trend > 0.1:
                # Rising price with rising volume = markup
                phase = "markup"
                strength = min((price_trend + volume_trend) * 2, 1.0)
            elif price_trend < -0.05 and volume_trend > 0.1:
                # Falling price with rising volume = markdown
                phase = "markdown"
                strength = min((abs(price_trend) + volume_trend) * 2, 1.0)
            else:
                phase = "neutral"
                strength = 0.5
            
            return {
                "phase": phase,
                "strength": strength,
                "price_trend": price_trend,
                "volume_trend": volume_trend,
                "volume_balance": volume_balance
            }
            
        except Exception as e:
            log(f"❌ Error detecting accumulation/distribution: {e}", level="ERROR")
            return {"phase": "unknown", "strength": 0.5}
    
    def _find_high_volume_nodes(self, volume_profile: Dict) -> List[Dict[str, Any]]:
        """Find high-volume nodes that act as support/resistance"""
        try:
            if not volume_profile:
                return []
            
            # Sort by volume
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            
            # Get top volume nodes
            total_volume = sum(volume_profile.values())
            nodes = []
            
            for price, volume in sorted_levels[:10]:  # Top 10 nodes
                volume_percentage = volume / total_volume
                
                if volume_percentage > 0.03:  # At least 3% of total volume
                    nodes.append({
                        "price": price,
                        "volume": volume,
                        "volume_percentage": volume_percentage,
                        "strength": min(volume_percentage * 20, 1.0)  # Normalize to 0-1
                    })
            
            return nodes
            
        except Exception as e:
            log(f"❌ Error finding high volume nodes: {e}", level="ERROR")
            return []
    
    def _analyze_support_resistance_strength(self, volume_nodes: List, candles: List) -> Dict[str, Any]:
        """Analyze support/resistance strength based on volume"""
        try:
            if not volume_nodes or not candles:
                return {"support_levels": [], "resistance_levels": [], "strength": "weak"}
            
            current_price = float(candles[-1][4])
            
            support_levels = []
            resistance_levels = []
            
            for node in volume_nodes:
                price = node["price"]
                strength = node["strength"]
                
                if price < current_price:
                    support_levels.append({
                        "price": price,
                        "strength": strength,
                        "distance_pct": abs(price - current_price) / current_price * 100
                    })
                else:
                    resistance_levels.append({
                        "price": price,
                        "strength": strength,
                        "distance_pct": abs(price - current_price) / current_price * 100
                    })
            
            # Sort by proximity to current price
            support_levels.sort(key=lambda x: x["distance_pct"])
            resistance_levels.sort(key=lambda x: x["distance_pct"])
            
            # Determine overall strength
            avg_strength = np.mean([node["strength"] for node in volume_nodes])
            
            if avg_strength > 0.7:
                overall_strength = "very_strong"
            elif avg_strength > 0.5:
                overall_strength = "strong"
            elif avg_strength > 0.3:
                overall_strength = "moderate"
            else:
                overall_strength = "weak"
            
            return {
                "support_levels": support_levels[:3],  # Top 3 nearest support
                "resistance_levels": resistance_levels[:3],  # Top 3 nearest resistance
                "strength": overall_strength,
                "avg_strength": avg_strength
            }
            
        except Exception as e:
            log(f"❌ Error analyzing support/resistance strength: {e}", level="ERROR")
            return {"support_levels": [], "resistance_levels": [], "strength": "weak"}
    
    def _calculate_value_area(self, volume_profile: Dict) -> Dict[str, Any]:
        """Calculate value area (70% of volume)"""
        try:
            if not volume_profile:
                return {"high": None, "low": None, "poc": None}
            
            # Sort by price
            sorted_by_price = sorted(volume_profile.items())
            total_volume = sum(volume_profile.values())
            target_volume = total_volume * 0.7
            
            # Find POC (Point of Control) - highest volume level
            poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
            
            # Find value area around POC
            accumulated_volume = volume_profile[poc_price]
            value_area_prices = [poc_price]
            
            # Expand around POC until 70% volume is captured
            price_list = [price for price, _ in sorted_by_price]
            poc_index = price_list.index(poc_price)
            
            left_idx = poc_index - 1
            right_idx = poc_index + 1
            
            while accumulated_volume < target_volume and (left_idx >= 0 or right_idx < len(price_list)):
                left_vol = volume_profile.get(price_list[left_idx], 0) if left_idx >= 0 else 0
                right_vol = volume_profile.get(price_list[right_idx], 0) if right_idx < len(price_list) else 0
                
                if left_vol >= right_vol and left_idx >= 0:
                    accumulated_volume += left_vol
                    value_area_prices.append(price_list[left_idx])
                    left_idx -= 1
                elif right_idx < len(price_list):
                    accumulated_volume += right_vol
                    value_area_prices.append(price_list[right_idx])
                    right_idx += 1
                else:
                    break
            
            return {
                "high": max(value_area_prices),
                "low": min(value_area_prices),
                "poc": poc_price,
                "volume_percentage": accumulated_volume / total_volume
            }
            
        except Exception as e:
            log(f"❌ Error calculating value area: {e}", level="ERROR")
            return {"high": None, "low": None, "poc": None}
    
    def _get_default_volume_profile(self) -> Dict[str, Any]:
        """Return default volume profile analysis"""
        return {
            "volume_profile": {},
            "institutional_activity": {"activity_level": "low", "pattern": "none", "strength": 0.5},
            "accumulation_distribution": {"phase": "unknown", "strength": 0.5},
            "volume_nodes": [],
            "support_resistance": {"support_levels": [], "resistance_levels": [], "strength": "weak"},
            "poc_level": None,
            "value_area": {"high": None, "low": None, "poc": None}
        }


class EnhancedTrendOrchestrator:
    """Main orchestrator for enhanced trend detection system"""
    
    def __init__(self):
        self.market_structure = MarketStructureAnalyzer()
        self.altseason_detector = EnhancedAltseasonDetector()
        self.sentiment_analyzer = MultiSourceSentimentAnalyzer()
        self.volume_engine = VolumeProfileEngine()
        
        # Cache for performance
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_update = {}
    
    async def get_enhanced_trend_context(self) -> Dict[str, Any]:
        """Get comprehensive trend context using all enhanced analyzers"""
        try:
            current_time = datetime.now()
            
            # Check cache
            if self._is_cache_valid("trend_context", current_time):
                return self.cache["trend_context"]
            
            log("🔍 Running enhanced trend analysis...")
            
            # Run all analyses concurrently for speed
            analyses = await asyncio.gather(
                self.market_structure.analyze_market_structure(),
                self.altseason_detector.detect_enhanced_altseason(),
                self.sentiment_analyzer.get_aggregated_sentiment(),
                self.volume_engine.analyze_volume_profile(),
                return_exceptions=True
            )
            
            structure_analysis, altseason_analysis, sentiment_analysis, volume_analysis = analyses
            
            # Handle any exceptions
            if isinstance(structure_analysis, Exception):
                log(f"❌ Market structure analysis failed: {structure_analysis}", level="WARNING")
                structure_analysis = self.market_structure._get_default_structure()
            
            if isinstance(altseason_analysis, Exception):
                log(f"❌ Altseason analysis failed: {altseason_analysis}", level="WARNING")
                altseason_analysis = self.altseason_detector._get_default_altseason()
            
            if isinstance(sentiment_analysis, Exception):
                log(f"❌ Sentiment analysis failed: {sentiment_analysis}", level="WARNING")
                sentiment_analysis = self.sentiment_analyzer._get_default_sentiment()
            
            if isinstance(volume_analysis, Exception):
                log(f"❌ Volume analysis failed: {volume_analysis}", level="WARNING")
                volume_analysis = self.volume_engine._get_default_volume_profile()
            
            # Combine all analyses
            enhanced_context = self._combine_trend_analyses(
                structure_analysis, altseason_analysis, sentiment_analysis, volume_analysis
            )
            
            # Cache result
            self.cache["trend_context"] = enhanced_context
            self.last_update["trend_context"] = current_time
            
            # Log comprehensive summary
            self._log_trend_summary(enhanced_context)
            
            return enhanced_context
            
        except Exception as e:
            log(f"❌ Error getting enhanced trend context: {e}", level="ERROR")
            return self._get_fallback_context()
    
    def _combine_trend_analyses(self, structure: Dict, altseason: Dict, 
                               sentiment: Dict, volume: Dict) -> Dict[str, Any]:
        """Combine all trend analyses into unified context"""
        try:
            # Base trend from market structure
            base_trend = structure.get("structure", "neutral")
            structure_strength = structure.get("strength", 0.5)
            structure_confidence = structure.get("confidence", 50)
            
            # Altseason impact
            altseason_active = altseason.get("is_altseason", False)
            altseason_strength = altseason.get("strength", 0.5)
            
            # Sentiment impact
            sentiment_score = sentiment.get("sentiment_score", 0.5)
            overall_sentiment = sentiment.get("overall_sentiment", "neutral")
            
            # Volume profile impact
            institutional_activity = volume.get("institutional_activity", {})
            accumulation_phase = volume.get("accumulation_distribution", {}).get("phase", "unknown")
            
            # Calculate enhanced trend confidence
            confidence_factors = [
                structure_confidence / 100,
                altseason.get("confidence", 50) / 100,
                sentiment.get("confidence", 50) / 100,
                institutional_activity.get("strength", 0.5)
            ]
            
            enhanced_confidence = np.mean(confidence_factors) * 100
            enhanced_confidence = max(30, min(95, enhanced_confidence))
            
            # Determine market regime
            market_regime = self._determine_enhanced_regime(
                structure, altseason, sentiment, volume
            )
            
            # Calculate trend strength with all factors
            enhanced_strength = self._calculate_enhanced_strength(
                structure_strength, altseason_strength, sentiment_score,
                institutional_activity.get("strength", 0.5)
            )
            
            # Generate trading recommendations
            recommendations = self._generate_trading_recommendations(
                base_trend, enhanced_strength, enhanced_confidence, market_regime
            )
            
            return {
                # Core trend data
                "trend": base_trend,
                "strength": enhanced_strength,
                "confidence": enhanced_confidence,
                "regime": market_regime,
                
                # Enhanced components
                "market_structure": structure,
                "altseason": altseason,
                "sentiment": sentiment,
                "volume_profile": volume,
                
                # Derived insights
                "market_phase": structure.get("market_phase", "neutral"),
                "institutional_activity": institutional_activity.get("activity_level", "low"),
                "accumulation_phase": accumulation_phase,
                "breakout_probability": structure.get("breakout_probability", {}),
                
                # Trading context
                "recommendations": recommendations,
                "risk_level": self._assess_risk_level(enhanced_confidence, market_regime),
                "opportunity_score": self._calculate_opportunity_score(structure, sentiment, volume),
                
                # Key levels
                "support_levels": volume.get("support_resistance", {}).get("support_levels", []),
                "resistance_levels": volume.get("support_resistance", {}).get("resistance_levels", []),
                "poc_level": volume.get("poc_level"),
                "value_area": volume.get("value_area", {}),
                
                # Metadata
                "timestamp": datetime.now().isoformat(),
                "analysis_quality": enhanced_confidence,
                "data_sources": ["market_structure", "altseason", "sentiment", "volume_profile"]
            }
            
        except Exception as e:
            log(f"❌ Error combining trend analyses: {e}", level="ERROR")
            return self._get_fallback_context()
    
    def _determine_enhanced_regime(self, structure: Dict, altseason: Dict, 
                                  sentiment: Dict, volume: Dict) -> str:
        """Determine enhanced market regime"""
        try:
            market_phase = structure.get("market_phase", "neutral")
            breakout_prob = structure.get("breakout_probability", {}).get("probability", 0.5)
            sentiment_score = sentiment.get("sentiment_score", 0.5)
            institutional_activity = volume.get("institutional_activity", {}).get("activity_level", "low")
            
            # Enhanced regime logic
            if breakout_prob > 0.8:
                return "breakout_imminent"
            elif market_phase == "trending" and sentiment_score > 0.7:
                return "strong_trending"
            elif market_phase == "consolidation" and institutional_activity == "very_high":
                return "institutional_accumulation"
            elif altseason.get("is_altseason") and altseason.get("strength", 0) > 0.7:
                return "altseason_active"
            elif sentiment_score < 0.3 and institutional_activity in ["high", "very_high"]:
                return "capitulation_buying"
            elif sentiment_score > 0.8 and breakout_prob < 0.3:
                return "euphoric_distribution"
            elif market_phase == "ranging":
                return "range_bound"
            elif institutional_activity == "low" and sentiment_score < 0.4:
                return "low_conviction"
            else:
                return "transitional"
                
        except Exception as e:
            log(f"❌ Error determining enhanced regime: {e}", level="ERROR")
            return "unknown"
    
    def _calculate_enhanced_strength(self, structure_strength: float, altseason_strength: float,
                                   sentiment_score: float, institutional_strength: float) -> float:
        """Calculate enhanced trend strength"""
        try:
            # Weighted combination
            weights = {
                "structure": 0.4,
                "sentiment": 0.25,
                "institutional": 0.2,
                "altseason": 0.15
            }
            
            enhanced_strength = (
                structure_strength * weights["structure"] +
                sentiment_score * weights["sentiment"] +
                institutional_strength * weights["institutional"] +
                altseason_strength * weights["altseason"]
            )
            
            return max(0.0, min(1.0, enhanced_strength))
            
        except Exception as e:
            log(f"❌ Error calculating enhanced strength: {e}", level="ERROR")
            return 0.5
    
    def _generate_trading_recommendations(self, trend: str, strength: float, 
                                        confidence: float, regime: str) -> Dict[str, Any]:
        """Generate trading recommendations based on analysis"""
        try:
            recommendations = {
                "primary_strategy": "neutral",
                "risk_allocation": "moderate",
                "timeframe_preference": "mixed",
                "position_sizing": "normal",
                "entry_conditions": [],
                "risk_management": []
            }
            
            # Strategy recommendations
            if regime == "strong_trending" and confidence > 70:
                recommendations["primary_strategy"] = "trend_following"
                recommendations["risk_allocation"] = "aggressive"
                recommendations["timeframe_preference"] = "medium_to_long"
                recommendations["entry_conditions"] = ["momentum_confirmation", "pullback_entry"]
                
            elif regime == "breakout_imminent" and strength > 0.7:
                recommendations["primary_strategy"] = "breakout_trading"
                recommendations["risk_allocation"] = "moderate_aggressive"
                recommendations["timeframe_preference"] = "short_to_medium"
                recommendations["entry_conditions"] = ["volume_confirmation", "level_break"]
                
            elif regime == "range_bound":
                recommendations["primary_strategy"] = "mean_reversion"
                recommendations["risk_allocation"] = "conservative"
                recommendations["timeframe_preference"] = "short"
                recommendations["entry_conditions"] = ["support_resistance_touch", "oversold_overbought"]
                
            elif regime == "institutional_accumulation":
                recommendations["primary_strategy"] = "accumulation_following"
                recommendations["risk_allocation"] = "moderate"
                recommendations["timeframe_preference"] = "long"
                recommendations["entry_conditions"] = ["institutional_confirmation", "value_area_entry"]
                
            elif regime == "altseason_active":
                recommendations["primary_strategy"] = "alt_momentum"
                recommendations["risk_allocation"] = "aggressive"
                recommendations["timeframe_preference"] = "short_to_medium"
                recommendations["entry_conditions"] = ["alt_breakout", "sector_rotation"]
                
            else:
                recommendations["primary_strategy"] = "wait_and_see"
                recommendations["risk_allocation"] = "conservative"
                
            # Risk management based on confidence
            if confidence < 50:
                recommendations["risk_management"] = ["tight_stops", "small_position", "quick_exits"]
            elif confidence < 70:
                recommendations["risk_management"] = ["normal_stops", "moderate_position"]
            else:
                recommendations["risk_management"] = ["wider_stops", "larger_position", "let_winners_run"]
            
            return recommendations
            
        except Exception as e:
            log(f"❌ Error generating recommendations: {e}", level="ERROR")
            return {"primary_strategy": "neutral", "risk_allocation": "conservative"}
    
    def _assess_risk_level(self, confidence: float, regime: str) -> str:
        """Assess overall market risk level"""
        high_risk_regimes = ["euphoric_distribution", "capitulation_buying", "breakout_imminent"]
        moderate_risk_regimes = ["strong_trending", "altseason_active", "transitional"]
        low_risk_regimes = ["range_bound", "institutional_accumulation"]
        
        if regime in high_risk_regimes or confidence < 40:
            return "high"
        elif regime in moderate_risk_regimes and confidence > 60:
            return "moderate"
        elif regime in low_risk_regimes and confidence > 70:
            return "low"
        else:
            return "moderate"
    
    def _calculate_opportunity_score(self, structure: Dict, sentiment: Dict, volume: Dict) -> float:
        """Calculate overall opportunity score"""
        try:
            # Factors that increase opportunity
            breakout_prob = structure.get("breakout_probability", {}).get("probability", 0.5)
            sentiment_extremes = abs(sentiment.get("sentiment_score", 0.5) - 0.5) * 2  # 0-1 scale
            institutional_activity = volume.get("institutional_activity", {}).get("strength", 0.5)
            
            # Combine factors
            opportunity_score = (breakout_prob * 0.4 + sentiment_extremes * 0.3 + institutional_activity * 0.3)
            
            return max(0.0, min(1.0, opportunity_score))
            
        except Exception as e:
            log(f"❌ Error calculating opportunity score: {e}", level="ERROR")
            return 0.5
    
    def _is_cache_valid(self, key: str, current_time: datetime) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache or key not in self.last_update:
            return False
        
        elapsed = (current_time - self.last_update[key]).total_seconds()
        return elapsed < self.cache_ttl
    
    def _log_trend_summary(self, context: Dict) -> None:
        """Log comprehensive trend summary"""
        try:
            trend = context.get("trend", "neutral")
            strength = context.get("strength", 0.5)
            confidence = context.get("confidence", 50)
            regime = context.get("regime", "unknown")
            opportunity = context.get("opportunity_score", 0.5)
            
            altseason = "ACTIVE" if context.get("altseason", {}).get("is_altseason") else "INACTIVE"
            institutional = context.get("institutional_activity", "unknown").upper()
            
            summary = (
                f"📊 ENHANCED TREND ANALYSIS\n"
                f"Trend: {trend.upper()} (strength: {strength:.2f}, confidence: {confidence:.1f}%)\n"
                f"Regime: {regime.upper()} | Risk: {context.get('risk_level', 'unknown').upper()}\n"
                f"Altseason: {altseason} | Institutional: {institutional}\n"
                f"Opportunity Score: {opportunity:.2f} | Strategy: {context.get('recommendations', {}).get('primary_strategy', 'unknown').upper()}"
            )
            
            log(summary)
            
            # Log key levels if available
            poc = context.get("poc_level")
            if poc:
                log(f"🎯 Key Levels - POC: {poc:.2f}")
            
        except Exception as e:
            log(f"❌ Error logging trend summary: {e}", level="ERROR")
    
    def _get_fallback_context(self) -> Dict[str, Any]:
        """Return fallback context if all analyses fail"""
        return {
            "trend": "neutral",
            "strength": 0.5,
            "confidence": 30,
            "regime": "unknown",
            "market_structure": {},
            "altseason": {"is_altseason": False, "strength": 0.5},
            "sentiment": {"overall_sentiment": "neutral", "sentiment_score": 0.5},
            "volume_profile": {},
            "recommendations": {"primary_strategy": "wait_and_see", "risk_allocation": "conservative"},
            "risk_level": "high",
            "opportunity_score": 0.3,
            "support_levels": [],
            "resistance_levels": [],
            "timestamp": datetime.now().isoformat(),
            "analysis_quality": 30,
            "error": "fallback_mode"
        }


# Global enhanced trend orchestrator
enhanced_trend_orchestrator = EnhancedTrendOrchestrator()

# Main enhanced trend function for backward compatibility
async def get_enhanced_trend_context() -> Dict[str, Any]:
    """Main function to get enhanced trend context"""
    return await enhanced_trend_orchestrator.get_enhanced_trend_context()

# Export enhanced functions
__all__ = [
    'MarketStructureAnalyzer',
    'EnhancedAltseasonDetector', 
    'MultiSourceSentimentAnalyzer',
    'VolumeProfileEngine',
    'EnhancedTrendOrchestrator',
    'enhanced_trend_orchestrator',
    'get_enhanced_trend_context'
]
