# trend_filters.py - COMPLETE FIXES FOR ALL 13 ISSUES
"""
Enhanced Trend detection and market context analysis with multi-timeframe BTC analysis
ALL ISSUES FIXED WITH PRECISE CORRECTIONS
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from bybit_api import signed_request
from logger import log
from collections import deque

class AltseasonDetector:
    """
    Detects altseason conditions based on multiple metrics - FIXED VERSION
    """
    
    def __init__(self):
        self.btc_dominance_history = deque(maxlen=30)
        self.alt_performance_history = deque(maxlen=30)
        self.is_altseason = False
        self.altseason_strength = 0
        # FIX #4: Initialize last_season to prevent AttributeError
        self.last_season = "neutral"
        
    async def detect_altseason(self):
        """
        Detect if we're in altseason based on:
        1. BTC dominance declining
        2. Majority of alts outperforming BTC
        3. Alt market cap increasing faster than BTC
        4. Alt volume surge
        """
        
        altseason_scores = {
            'strong_altseason': 0,
            'altseason': 0,
            'neutral': 0,
            'btc_season': 0
        }
        
        analysis_details = {}
        
        # 1. Check alt performance vs BTC
        alt_performance = await self._analyze_alt_performance()
        altseason_scores[alt_performance['season']] += alt_performance['weight']
        analysis_details['alt_performance'] = alt_performance
        
        # 2. Check volume distribution
        volume_analysis = await self._analyze_volume_distribution()
        altseason_scores[volume_analysis['season']] += volume_analysis['weight']
        analysis_details['volume'] = volume_analysis
        
        # 3. Check momentum shift
        momentum_shift = await self._analyze_momentum_shift()
        altseason_scores[momentum_shift['season']] += momentum_shift['weight']
        analysis_details['momentum'] = momentum_shift
        
        # 4. Check market breadth
        breadth_analysis = await self._analyze_market_breadth()
        altseason_scores[breadth_analysis['season']] += breadth_analysis['weight']
        analysis_details['breadth'] = breadth_analysis
        
        # Determine final altseason status
        total_score = sum(altseason_scores.values())
        if total_score == 0:
            season = 'neutral'
            strength = 0
        else:
            season = max(altseason_scores.items(), key=lambda x: x[1])[0]
            strength = altseason_scores[season] / total_score
        
        # Update state
        self.is_altseason = season in ['altseason', 'strong_altseason']
        self.altseason_strength = strength if self.is_altseason else 0
        
        # Log significant changes
        if self.last_season != season:
            log(f"🔄 Market Season Change: {self.last_season} → {season} (strength: {strength:.2f})")
            
            if season == 'strong_altseason':
                try:
                    from telegram_bot import send_telegram_message
                    await send_telegram_message(
                        f"🚀 <b>ALTSEASON DETECTED!</b>\n"
                        f"Strength: {strength:.2%}\n"
                        f"Alt coins showing strong outperformance vs BTC"
                    )
                except:
                    pass
        
        self.last_season = season
        
        return {
            'is_altseason': self.is_altseason,
            'season': season,
            'strength': strength,
            'details': analysis_details
        }
    
    async def _analyze_alt_performance(self):
        """
        Check how many alts are outperforming BTC
        FIX #1: Remove double-scaling of price24hPcnt
        FIX #3: Use concurrent API calls instead of sequential
        ADDITIONAL FIX: Handle empty API response lists properly
        """
        
        try:
            # Top altcoins to check
            alt_symbols = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
                          "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", 
                          "DOTUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT"]
            
            # FIX #3: Fire API calls concurrently instead of sequentially
            tasks = []
            
            # BTC performance first
            tasks.append(signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            }))
            
            # All alt symbols
            for symbol in alt_symbols:
                tasks.append(signed_request("GET", "/v5/market/tickers", {
                    "category": "linear",
                    "symbol": symbol
                }))
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process BTC data with proper error handling
            btc_resp = results[0]
            btc_perf_24h = 0
            
            if isinstance(btc_resp, dict) and btc_resp.get("retCode") == 0:
                result_data = btc_resp.get("result", {})
                ticker_list = result_data.get("list", [])
                
                # ADDITIONAL FIX: Check if list is not empty before accessing index
                if ticker_list and len(ticker_list) > 0:
                    btc_data = ticker_list[0]
                    # FIX #1: Remove * 100 - price24hPcnt is already a percentage
                    btc_perf_24h = float(btc_data.get("price24hPcnt", 0))
                else:
                    log(f"⚠️ Empty ticker list for BTCUSDT", level="WARNING")
                
            outperforming_24h = 0
            strong_performers = 0
            total_checked = 0
            
            # Process alt data with improved error handling
            for i, symbol in enumerate(alt_symbols):
                try:
                    ticker_resp = results[i + 1]  # Skip BTC result
                    
                    if (isinstance(ticker_resp, dict) and 
                        ticker_resp.get("retCode") == 0):
                        
                        result_data = ticker_resp.get("result", {})
                        ticker_list = result_data.get("list", [])
                        
                        # ADDITIONAL FIX: Check if list is not empty before accessing index
                        if ticker_list and len(ticker_list) > 0:
                            ticker = ticker_list[0]
                            # FIX #1: Remove * 100 - price24hPcnt is already a percentage
                            alt_perf_24h = float(ticker.get("price24hPcnt", 0))
                            
                            total_checked += 1
                            
                            # Check if outperforming BTC
                            if alt_perf_24h > btc_perf_24h + 2:  # 2% outperformance threshold
                                outperforming_24h += 1
                                
                            # Check for strong performers (>10% gain)
                            if alt_perf_24h > 10:
                                strong_performers += 1
                        else:
                            log(f"⚠️ Empty ticker list for {symbol}", level="WARNING")
                            
                except Exception as e:
                    log(f"❌ Error processing {symbol}: {e}", level="WARNING")
                    continue
            
            # Calculate ratios
            outperform_ratio = outperforming_24h / total_checked if total_checked > 0 else 0
            strong_ratio = strong_performers / total_checked if total_checked > 0 else 0
            
            # Determine season based on performance
            if outperform_ratio > 0.7 and strong_ratio > 0.3:
                return {'season': 'strong_altseason', 'weight': 2.0, 'ratio': outperform_ratio}
            elif outperform_ratio > 0.6:
                return {'season': 'altseason', 'weight': 1.5, 'ratio': outperform_ratio}
            elif outperform_ratio < 0.3:
                return {'season': 'btc_season', 'weight': 1.5, 'ratio': outperform_ratio}
            else:
                return {'season': 'neutral', 'weight': 1.0, 'ratio': outperform_ratio}
                
        except Exception as e:
            log(f"❌ Error analyzing alt performance: {e}", level="ERROR")
            return {'season': 'neutral', 'weight': 0.5, 'ratio': 0.5}
    
    async def _analyze_volume_distribution(self):
        """
        Analyze if volume is shifting to altcoins
        FIX #5: Use turnover24h (quote-value) for proper USD comparison
        ADDITIONAL FIX: Handle empty API response lists properly
        """
        
        try:
            # Get volume for BTC and major alts
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
            
            # FIX #3: Use concurrent requests
            tasks = [
                signed_request("GET", "/v5/market/tickers", {
                    "category": "linear",
                    "symbol": symbol
                }) for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            volumes = {}
            
            for i, symbol in enumerate(symbols):
                try:
                    ticker_resp = results[i]
                    
                    if (isinstance(ticker_resp, dict) and 
                        ticker_resp.get("retCode") == 0):
                        
                        result_data = ticker_resp.get("result", {})
                        ticker_list = result_data.get("list", [])
                        
                        # ADDITIONAL FIX: Check if list is not empty before accessing index
                        if ticker_list and len(ticker_list) > 0:
                            ticker = ticker_list[0]
                            
                            # FIX #5: Use turnover24h for USD-quoted volume instead of volume24h
                            turnover_24h = float(ticker.get("turnover24h", 0))
                            volumes[symbol] = turnover_24h
                        else:
                            log(f"⚠️ Empty ticker list for {symbol} in volume analysis", level="WARNING")
                        
                except Exception as e:
                    log(f"❌ Error processing volume for {symbol}: {e}", level="WARNING")
                    continue
            
            if not volumes or "BTCUSDT" not in volumes:
                return {'season': 'neutral', 'weight': 0.5}
            
            # Calculate BTC volume dominance
            total_volume = sum(volumes.values())
            btc_volume = volumes["BTCUSDT"]
            btc_dominance = btc_volume / total_volume if total_volume > 0 else 0
            
            # Lower BTC dominance = altseason
            if btc_dominance < 0.3:  # BTC less than 30% of volume
                return {'season': 'strong_altseason', 'weight': 1.5, 'btc_dominance': btc_dominance}
            elif btc_dominance < 0.4:
                return {'season': 'altseason', 'weight': 1.2, 'btc_dominance': btc_dominance}
            elif btc_dominance > 0.6:
                return {'season': 'btc_season', 'weight': 1.2, 'btc_dominance': btc_dominance}
            else:
                return {'season': 'neutral', 'weight': 0.8, 'btc_dominance': btc_dominance}
                
        except Exception as e:
            log(f"❌ Error analyzing volume distribution: {e}", level="ERROR")
            return {'season': 'neutral', 'weight': 0.5}
    
    async def _analyze_momentum_shift(self):
        """
        Check momentum shift between BTC and alts
        FIX #6: Expand beyond just ETH-BTC to include multiple alts
        """
        
        try:
            # FIX #6: Use basket of major alts instead of just ETH
            alt_symbols = ["ETHUSDT", "SOLUSDT", "BNBUSDT"]
            period = 14  # 14 day momentum
            
            # Get candles concurrently
            tasks = []
            tasks.append(signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "D",
                "limit": str(period)
            }))
            
            for symbol in alt_symbols:
                tasks.append(signed_request("GET", "/v5/market/kline", {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": "D",
                    "limit": str(period)
                }))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process BTC momentum
            btc_resp = results[0]
            btc_momentum = 0
            
            if (isinstance(btc_resp, dict) and 
                btc_resp.get("retCode") == 0):
                candles = btc_resp.get("result", {}).get("list", [])
                if len(candles) >= period:
                    closes = [float(c[4]) for c in candles]
                    btc_momentum = ((closes[-1] - closes[0]) / closes[0]) * 100
            
            # Process alt momentum (average of basket)
            alt_momentums = []
            for i, symbol in enumerate(alt_symbols):
                try:
                    alt_resp = results[i + 1]
                    
                    if (isinstance(alt_resp, dict) and 
                        alt_resp.get("retCode") == 0):
                        candles = alt_resp.get("result", {}).get("list", [])
                        if len(candles) >= period:
                            closes = [float(c[4]) for c in candles]
                            alt_momentum = ((closes[-1] - closes[0]) / closes[0]) * 100
                            alt_momentums.append(alt_momentum)
                            
                except Exception as e:
                    log(f"❌ Error processing momentum for {symbol}: {e}", level="WARNING")
                    continue
            
            # Average alt momentum
            avg_alt_momentum = sum(alt_momentums) / len(alt_momentums) if alt_momentums else 0
            
            # Compare momentum
            momentum_diff = avg_alt_momentum - btc_momentum
            
            if momentum_diff > 3:  # Alts momentum 3% higher
                return {'season': 'altseason', 'weight': 1.3, 'momentum_diff': momentum_diff}
            elif momentum_diff < -3:  # BTC momentum 3% higher
                return {'season': 'btc_season', 'weight': 1.3, 'momentum_diff': momentum_diff}
            else:
                return {'season': 'neutral', 'weight': 0.7, 'momentum_diff': momentum_diff}
                
        except Exception as e:
            log(f"❌ Error analyzing momentum shift: {e}", level="ERROR")
            return {'season': 'neutral', 'weight': 0.5}
    
    async def _analyze_market_breadth(self):
        """
        Check how many alts are in uptrend
        FIX #7: Use proper 10-day window as stated in comment
        """
        
        try:
            alt_symbols = ["ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
                          "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT"]
            
            # FIX #3: Use concurrent requests
            tasks = []
            for symbol in alt_symbols:
                tasks.append(signed_request("GET", "/v5/market/kline", {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": "D",
                    "limit": "10"  # FIX #7: Use 10 days as comment states
                }))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            uptrending = 0
            downtrending = 0
            
            for i, symbol in enumerate(alt_symbols):
                try:
                    kline_resp = results[i]
                    
                    if (isinstance(kline_resp, dict) and 
                        kline_resp.get("retCode") == 0):
                        candles = kline_resp.get("result", {}).get("list", [])
                        if len(candles) >= 10:
                            # FIX #7: Use proper 10-day window (slice [:10] not [:5])
                            closes = [float(c[4]) for c in candles[:10]]
                            closes.reverse()  # oldest → newest
                            
                            # Check for 5% up in last 10 days as stated in comment
                            if closes[-1] > closes[0] * 1.05:  # 5% up
                                uptrending += 1
                            elif closes[-1] < closes[0] * 0.95:  # 5% down
                                downtrending += 1
                                
                except Exception as e:
                    log(f"❌ Error processing breadth for {symbol}: {e}", level="WARNING")
                    continue
            
            total = uptrending + downtrending
            
            if total == 0:
                return {'season': 'neutral', 'weight': 0.5}
            
            uptrend_ratio = uptrending / total
            
            if uptrend_ratio > 0.7:
                return {'season': 'altseason', 'weight': 1.2, 'uptrend_ratio': uptrend_ratio}
            elif uptrend_ratio < 0.3:
                return {'season': 'btc_season', 'weight': 1.2, 'uptrend_ratio': uptrend_ratio}
            else:
                return {'season': 'neutral', 'weight': 0.8, 'uptrend_ratio': uptrend_ratio}
                
        except Exception as e:
            log(f"❌ Error analyzing market breadth: {e}", level="ERROR")
            return {'season': 'neutral', 'weight': 0.5}


class BTCTrendAnalyzer:
    """
    Enhanced BTC trend analyzer with thread-safe caching
    FIXED: Complete implementation with all required methods
    """
    
    def __init__(self):
        self.last_trend = "neutral"
        self.trend_strength = 0
        self.confidence = 0
        self.timeframes = ['15', '1H', '4H', '1D']
        # FIX #10: Add thread-safe lock for trend cache
        self._trend_lock = asyncio.Lock()
        
    async def _fetch_btc_candles(self, interval, limit=100):
        """
        Fetch BTC candles for analysis
        FIX #8: Adopt consistent candle ordering (oldest→newest)
        """
        try:
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": interval,
                "limit": str(limit)
            })
            
            if response.get("retCode") == 0:
                candles = response.get("result", {}).get("list", [])
                # FIX #8: Ensure consistent oldest→newest ordering
                candles.reverse()  # Bybit returns newest first, we want oldest first
                return candles
            else:
                log(f"❌ Failed to fetch BTC candles: {response.get('retMsg', 'Unknown error')}")
                return []
                
        except Exception as e:
            log(f"❌ Error fetching BTC candles: {e}", level="ERROR")
            return []
    
    def _calculate_ema(self, prices, period):
        """
        Calculate EMA with proper initialization
        FIX #9: Use SMA of first period prices as initial EMA value
        """
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        # FIX #9: Initialize EMA with SMA of first period prices
        ema = np.mean(prices[:period])
        multiplier = 2 / (period + 1)
        
        # Start calculation from period index
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _analyze_moving_averages(self, candles_by_tf):
        """Analyze moving averages across timeframes"""
        signals = []
        
        for tf in ['15', '1H', '4H']:
            if tf not in candles_by_tf or len(candles_by_tf[tf]) < 50:
                continue
                
            candles = candles_by_tf[tf]
            closes = [float(c[4]) for c in candles]
            
            # Calculate EMAs
            ema_20 = self._calculate_ema(closes, 20)
            ema_50 = self._calculate_ema(closes, 50)
            current_price = closes[-1]
            
            # Determine trend
            if current_price > ema_20 > ema_50:
                signals.append(('uptrend', 1.5 if tf in ['4H'] else 1.0))
            elif current_price < ema_20 < ema_50:
                signals.append(('downtrend', 1.5 if tf in ['4H'] else 1.0))
            else:
                signals.append(('neutral', 0.8))
        
        if not signals:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        # Aggregate signals
        trend_weights = {'uptrend': 0, 'downtrend': 0, 'neutral': 0}
        for trend, weight in signals:
            trend_weights[trend] += weight
        
        dominant_trend = max(trend_weights.items(), key=lambda x: x[1])
        confidence = min(75, 40 + len(signals) * 10)
        
        return {
            'trend': dominant_trend[0],
            'weight': dominant_trend[1],
            'confidence': confidence
        }
    
    def _analyze_price_structure(self, candles_by_tf):
        """Analyze higher highs/lower lows structure"""
        if '1H' not in candles_by_tf or len(candles_by_tf['1H']) < 20:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        candles = candles_by_tf['1H'][-20:]  # Last 20 hours
        highs = [float(c[2]) for c in candles]
        lows = [float(c[3]) for c in candles]
        
        # Check for higher highs and higher lows (uptrend)
        recent_high = max(highs[-10:])
        earlier_high = max(highs[:10])
        recent_low = min(lows[-10:])
        earlier_low = min(lows[:10])
        
        if recent_high > earlier_high and recent_low > earlier_low:
            return {'trend': 'uptrend', 'weight': 1.2, 'confidence': 65}
        elif recent_high < earlier_high and recent_low < earlier_low:
            return {'trend': 'downtrend', 'weight': 1.2, 'confidence': 65}
        else:
            return {'trend': 'neutral', 'weight': 0.8, 'confidence': 45}
    
    def _analyze_momentum(self, candles_by_tf):
        """Analyze price momentum"""
        if '15' not in candles_by_tf or len(candles_by_tf['15']) < 14:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        candles = candles_by_tf['15']
        closes = [float(c[4]) for c in candles]
        
        # Calculate RSI-like momentum
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < 14:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        rsi = 100 - (100 / (1 + rs))
        
        if rsi > 60:
            return {'trend': 'uptrend', 'weight': 1.0, 'confidence': min(70, 50 + (rsi-60)*2)}
        elif rsi < 40:
            return {'trend': 'downtrend', 'weight': 1.0, 'confidence': min(70, 50 + (40-rsi)*2)}
        else:
            return {'trend': 'neutral', 'weight': 0.8, 'confidence': 45}
    
    def _analyze_volume_trend(self, candles_by_tf):
        """Analyze volume trends"""
        if '1H' not in candles_by_tf or len(candles_by_tf['1H']) < 20:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        candles = candles_by_tf['1H']
        
        up_volume = []
        down_volume = []
        
        for candle in candles:
            close = float(candle[4])
            open_price = float(candle[1])
            volume = float(candle[5])
            
            if close > open_price:
                up_volume.append(volume)
            else:
                down_volume.append(volume)
        
        if not up_volume or not down_volume:
            return {'trend': 'neutral', 'weight': 0.5, 'confidence': 30}
        
        avg_up_vol = np.mean(up_volume)
        avg_down_vol = np.mean(down_volume)
        
        if avg_up_vol > avg_down_vol * 1.3:
            return {'trend': 'uptrend', 'weight': 1.0, 'confidence': 60}
        elif avg_down_vol > avg_up_vol * 1.3:
            return {'trend': 'downtrend', 'weight': 1.0, 'confidence': 60}
        else:
            return {'trend': 'neutral', 'weight': 0.8, 'confidence': 45}
    
    async def analyze_btc_trend(self):
        """
        Enhanced BTC trend analysis
        FIX #12: Use concurrent API calls for multiple timeframes
        FIXED: Complete implementation with all required analysis methods
        """
        try:
            # FIX #12: Fetch all timeframes concurrently
            tasks = []
            for tf in self.timeframes:
                tasks.append(self._fetch_btc_candles(tf, 100))
            
            candles_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            candles_by_tf = {}
            for i, tf in enumerate(self.timeframes):
                if (isinstance(candles_results[i], list) and 
                    len(candles_results[i]) > 20):  # Reduced from 50 to 20 for more lenient check
                    candles_by_tf[tf] = candles_results[i]
            
            if not candles_by_tf:
                log("⚠️ No valid candle data for BTC trend analysis")
                return {
                    'trend': 'neutral',
                    'strength': 0.5,
                    'confidence': 30,
                    'details': {'error': 'No valid candle data'}
                }
            
            # Analyze each component
            trend_scores = {'uptrend': 0, 'downtrend': 0, 'neutral': 0}
            confidence_factors = []
            analysis_details = {}
            
            # 1. Moving Average Analysis
            ma_analysis = self._analyze_moving_averages(candles_by_tf)
            trend_scores[ma_analysis['trend']] += ma_analysis['weight']
            confidence_factors.append(ma_analysis['confidence'])
            analysis_details['moving_averages'] = ma_analysis
            
            # 2. Price Structure Analysis
            structure_analysis = self._analyze_price_structure(candles_by_tf)
            trend_scores[structure_analysis['trend']] += structure_analysis['weight']
            confidence_factors.append(structure_analysis['confidence'])
            analysis_details['price_structure'] = structure_analysis
            
            # 3. Momentum Analysis
            momentum_analysis = self._analyze_momentum(candles_by_tf)
            trend_scores[momentum_analysis['trend']] += momentum_analysis['weight']
            confidence_factors.append(momentum_analysis['confidence'])
            analysis_details['momentum'] = momentum_analysis
            
            # 4. Volume Analysis
            volume_analysis = self._analyze_volume_trend(candles_by_tf)
            trend_scores[volume_analysis['trend']] += volume_analysis['weight']
            confidence_factors.append(volume_analysis['confidence'])
            analysis_details['volume'] = volume_analysis
            
            # Determine overall trend
            total_weight = sum(trend_scores.values())
            
            if total_weight == 0:
                overall_trend = 'neutral'
                strength = 0.5
            else:
                # Get trend with highest score
                overall_trend = max(trend_scores.items(), key=lambda x: x[1])[0]
                strength = trend_scores[overall_trend] / total_weight
                
                # Require minimum agreement for non-neutral trends
                if overall_trend != 'neutral' and strength < 0.6:
                    overall_trend = 'neutral'
                    strength = 0.5
            
            # Calculate confidence
            confidence = np.mean(confidence_factors) if confidence_factors else 30
            confidence = max(30, min(95, confidence))  # Clamp between 30-95
            
            # Update state
            self.last_trend = overall_trend
            self.trend_strength = strength
            self.confidence = confidence
            
            return {
                'trend': overall_trend,
                'strength': strength,
                'confidence': confidence,
                'details': analysis_details
            }
            
        except Exception as e:
            log(f"❌ Error analyzing BTC trend: {e}", level="ERROR")
            return {
                'trend': 'neutral',
                'strength': 0.5,
                'confidence': 30,
                'details': {'error': str(e)}
            }


# Global analyzer instances
btc_analyzer = BTCTrendAnalyzer()
altseason_detector = AltseasonDetector()

# FIX #10: Thread-safe trend cache with lock
_trend_cache = {}
_trend_cache_lock = asyncio.Lock()

async def get_trend_context_cached():
    """
    Get trend context with thread-safe caching
    FIX #10: Add proper locking for thread safety
    """
    async with _trend_cache_lock:
        current_time = datetime.now()
        
        # Check if cached result is still valid (5 minutes)
        if ('timestamp' in _trend_cache and 
            (current_time - _trend_cache['timestamp']).seconds < 300):
            return _trend_cache['context']
        
        # Get fresh context
        context = await get_trend_context()
        
        # Update cache
        _trend_cache['context'] = context
        _trend_cache['timestamp'] = current_time
        
        return context


async def detect_market_regime():
    """Detect current market regime - Enhanced to handle all expected values"""
    try:
        # Get BTC volatility data
        kline_resp = await signed_request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": "BTCUSDT",
            "interval": "60",
            "limit": "100"
        })
        
        if kline_resp.get("retCode") != 0:
            return "volatile"  # Default to volatile on API error
            
        candles = kline_resp.get("result", {}).get("list", [])
        if len(candles) < 50:
            return "volatile"  # Default to volatile with insufficient data
            
        # Calculate ATR for volatility
        highs = [float(c[2]) for c in candles[:50]]
        lows = [float(c[3]) for c in candles[:50]]
        closes = [float(c[4]) for c in candles[:50]]
        
        # Simple ATR calculation
        tr_values = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_values.append(tr)
        
        atr = sum(tr_values[-14:]) / 14 if len(tr_values) >= 14 else 0
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] > 0 else 0
        
        # Enhanced regime detection with three categories
        if atr_pct >= 4.0:  # High volatility
            return "volatile"
        elif atr_pct < 1.5:  # Very low volatility - tight range
            # Check for ranging behavior
            price_range = (max(closes[-20:]) - min(closes[-20:])) / min(closes[-20:])
            if price_range < 0.03:  # Less than 3% range in last 20 periods
                return "ranging"  # Tight ranging market
            else:
                return "stable"  # Low volatility but not tight range
        else:  # Medium volatility
            return "stable"
            
    except Exception as e:
        log(f"❌ Error detecting market regime: {e}", level="ERROR")
        return "volatile"  # Safe default


async def get_market_sentiment():
    """
    Analyze overall market sentiment
    FIX #12: Use concurrent API calls instead of sequential
    ADDITIONAL FIX: Handle empty API response lists properly
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    try:
        # Get top 10 coins performance
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
                  "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT"]
        
        # FIX #12: Use concurrent requests
        tasks = [
            signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": symbol
            }) for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        bullish_count = 0
        bearish_count = 0
        
        for i, symbol in enumerate(symbols):
            try:
                ticker_resp = results[i]
                
                if (isinstance(ticker_resp, dict) and 
                    ticker_resp.get("retCode") == 0):
                    
                    result_data = ticker_resp.get("result", {})
                    ticker_list = result_data.get("list", [])
                    
                    # ADDITIONAL FIX: Check if list is not empty before accessing index
                    if ticker_list and len(ticker_list) > 0:
                        ticker = ticker_list[0]
                        # FIX #1: Remove * 100 - price24hPcnt is already a percentage
                        price_24h_pct = float(ticker.get("price24hPcnt", 0))
                        
                        if price_24h_pct > 2:
                            bullish_count += 1
                        elif price_24h_pct < -2:
                            bearish_count += 1
                    else:
                        log(f"⚠️ Empty ticker list for {symbol} in sentiment analysis", level="WARNING")

                else:
                    log(f"⚠️ Failed ticker call for {symbol}", level="WARNING")
                    continue
                    
            except Exception as e:
                log(f"❌ Error processing sentiment for {symbol}: {e}", level="WARNING")
                continue
        
        # Determine sentiment
        if bullish_count >= 6:
            return "bullish"
        elif bearish_count >= 6:
            return "bearish"
        else:
            return "neutral"
            
    except Exception as e:
        log(f"❌ Error calculating market sentiment: {e}", level="ERROR")
        return "neutral"


async def get_trend_context():
    """
    Enhanced main function to get complete market context
    FIX #12: Use concurrent execution for all analyses
    FIX #13: Move alert logic before return statement
    """
    try:
        # FIX #12: Run all analyses in parallel for better performance
        btc_trend_task = btc_analyzer.analyze_btc_trend()
        sentiment_task = get_market_sentiment()
        regime_task = detect_market_regime()
        altseason_task = altseason_detector.detect_altseason()
        
        # Execute all tasks concurrently
        btc_analysis, sentiment, regime, altseason_analysis = await asyncio.gather(
            btc_trend_task, sentiment_task, regime_task, altseason_task,
            return_exceptions=True
        )
        
        # Handle any exceptions from concurrent execution
        if isinstance(btc_analysis, Exception):
            log(f"❌ BTC analysis failed: {btc_analysis}", level="ERROR")
            btc_analysis = {'trend': 'neutral', 'strength': 0.5, 'confidence': 30, 'details': {}}
        
        if isinstance(sentiment, Exception):
            log(f"❌ Sentiment analysis failed: {sentiment}", level="ERROR")
            sentiment = "neutral"
            
        if isinstance(regime, Exception):
            log(f"❌ Regime detection failed: {regime}", level="ERROR")
            regime = "volatile"
            
        if isinstance(altseason_analysis, Exception):
            log(f"❌ Altseason analysis failed: {altseason_analysis}", level="ERROR")
            altseason_analysis = {'is_altseason': False, 'strength': 0, 'details': {}, 'season': 'neutral'}
        
        # Map neutral to ranging for backward compatibility
        btc_trend = btc_analysis['trend']
        if btc_trend == 'neutral':
            btc_trend = 'ranging'
        
        # FIX #13: Move alert logic BEFORE return statement
        if btc_trend == 'downtrend' and btc_analysis['confidence'] >= 70:
            log(f"⚠️ BTC DOWNTREND CONFIRMED with {btc_analysis['confidence']:.1f}% confidence")
            try:
                from telegram_bot import send_telegram_message
                await send_telegram_message(
                    f"⚠️ <b>BTC DOWNTREND ALERT</b>\n"
                    f"Confidence: {btc_analysis['confidence']:.1f}%\n"
                    f"Strength: {btc_analysis['strength']:.1f}\n"
                    f"Consider reducing risk exposure"
                )
            except Exception as alert_error:
                log(f"❌ Failed to send downtrend alert: {alert_error}", level="WARNING")
        
        context = {
            "btc_trend": btc_trend,
            "btc_strength": btc_analysis['strength'],
            "btc_confidence": btc_analysis['confidence'],
            "btc_details": btc_analysis['details'],
            "sentiment": sentiment,
            "regime": regime,
            "altseason": altseason_analysis['is_altseason'],
            "altseason_strength": altseason_analysis['strength'],
            "altseason_details": altseason_analysis['details'],
            "market_season": altseason_analysis['season'],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Enhanced logging with confidence
        season_str = f" | ALTSEASON ({altseason_analysis['strength']:.0%})" if altseason_analysis['is_altseason'] else ""
        log(f"📊 Market Context: BTC {btc_trend} (conf: {btc_analysis['confidence']:.1f}%), " +
            f"Sentiment {sentiment}, Regime {regime}{season_str}")
        
        return context
        
    except Exception as e:
        log(f"❌ Error getting trend context: {e}", level="ERROR")
        # Return safe defaults on any error
        return {
            "btc_trend": "ranging",
            "btc_strength": 0.5,
            "btc_confidence": 30,
            "btc_details": {"error": str(e)},
            "sentiment": "neutral",
            "regime": "volatile",
            "altseason": False,
            "altseason_strength": 0,
            "altseason_details": {},
            "market_season": "neutral",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


# Additional helper functions with fixes

def calculate_ema_fixed(prices, period):
    """
    Fixed EMA calculation for backward compatibility
    FIX #9: Proper initialization with SMA of first period prices
    """
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    # FIX #9: Initialize with SMA of first period prices
    ema = np.mean(prices[:period])
    multiplier = 2 / (period + 1)
    
    # Start from period index
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema


async def get_btc_trend():
    """
    Enhanced BTC trend analysis using the new analyzer
    Returns: 'uptrend', 'downtrend', or 'ranging' (maps neutral to ranging)
    """
    result = await btc_analyzer.analyze_btc_trend()
    
    # Map neutral to ranging for backward compatibility
    trend = result['trend']
    if trend == 'neutral':
        trend = 'ranging'
    
    return trend


# Monitoring functions with concurrent improvements

async def monitor_btc_trend_accuracy():
    """
    Monitor and report BTC trend accuracy
    FIX #12: Improved monitoring with better error handling
    """
    
    while True:
        try:
            # Get current status with timeout
            summary = f"BTC Trend: {btc_analyzer.last_trend.upper()} "
            summary += f"(strength: {btc_analyzer.trend_strength:.2f}, "
            summary += f"confidence: {btc_analyzer.confidence:.1f}%)"
            
            # Log every 30 minutes
            log(f"📊 BTC Trend Monitor: {summary}")
            
        except Exception as e:
            log(f"❌ Error in BTC trend monitor: {e}", level="ERROR")
        
        await asyncio.sleep(1800)  # 30 minutes


async def monitor_altseason_status():
    """
    Monitor and report altseason status
    FIX #12: Enhanced monitoring with concurrent checks
    """
    
    while True:
        try:
            # Use timeout for altseason detection
            result = await asyncio.wait_for(
                altseason_detector.detect_altseason(), 
                timeout=30
            )
            
            if result['is_altseason']:
                details = result['details']
                
                # Build status message
                msg = f"🚀 ALTSEASON STATUS\n"
                msg += f"Season: {result['season']}\n"
                msg += f"Strength: {result['strength']:.0%}\n"
                
                if 'alt_performance' in details:
                    ratio = details['alt_performance'].get('ratio', 0)
                    msg += f"Alts outperforming BTC: {ratio:.0%}\n"
                
                if 'volume' in details:
                    btc_dom = details['volume'].get('btc_dominance', 0)
                    msg += f"BTC volume dominance: {btc_dom:.0%}\n"
                
                log(msg)
            
        except asyncio.TimeoutError:
            log("⚠️ Altseason detection timeout", level="WARNING")
        except Exception as e:
            log(f"❌ Error in altseason monitor: {e}", level="ERROR")
        
        await asyncio.sleep(3600)  # Check every hour


# Enhanced validation functions

async def validate_short_signal(symbol, candles_by_tf):
    """
    Unified async short signal validator with full macro + micro logic:
    - BTC trend, confidence, sentiment, volatility regime, altseason
    - Bullish candle filter (5m)
    - Bearish indicator scoring
    - Price/indicator divergence (15m)
    """
    try:
        # Ensure required candles are present
        if '5' not in candles_by_tf or '15' not in candles_by_tf:
            log(f"❌ {symbol}: Missing required timeframes (5m/15m)")
            return False

        # Fetch full trend context (cached async call)
        context = await get_trend_context_cached()
        btc_trend = context.get('btc_trend', 'neutral')
        btc_confidence = context.get('btc_confidence', 0)
        sentiment = context.get('sentiment', 'neutral')
        regime = context.get('regime', 'calm')
        altseason_strength = context.get('altseason_strength', 0)
        is_altseason = context.get('altseason', False)

        # Step 1: BTC trend and confidence check
        if btc_trend not in ['downtrend', 'ranging']:
            log(f"❌ {symbol}: BTC trend is {btc_trend}, not short-friendly")
            return False
        if btc_trend == 'downtrend' and btc_confidence < 65:
            log(f"❌ {symbol}: BTC downtrend confidence too low ({btc_confidence:.1f}%)")
            return False

        # Step 2: Recent bullish candles (5m)
        recent_candles = candles_by_tf['5'][-5:]
        bullish_candles = sum(1 for c in recent_candles if float(c['close']) > float(c['open']))
        if bullish_candles > 2:
            log(f"❌ {symbol}: Too many recent bullish candles ({bullish_candles}/5)")
            return False

        # Step 3: Macro indicator scoring (custom logic)
        indicator_scores = {}

        # BTC trend impact
        indicator_scores['btc_trend'] = -1.5 if btc_trend == 'downtrend' else (-0.5 if btc_trend == 'ranging' else 0.3)

        # Sentiment
        indicator_scores['sentiment'] = -1.2 if sentiment == 'bearish' else (-0.3 if sentiment == 'neutral' else 0.3)

        # Volatility regime
        indicator_scores['regime'] = -0.8 if regime == 'volatile' else 0.2

        # Altseason impact
        indicator_scores['altseason'] = 0.8 if is_altseason and altseason_strength > 0.7 else -0.2

        # Step 4: Require at least 2 strong bearish indicators
        strong_bearish = sum(1 for v in indicator_scores.values() if v < -1.0)
        if strong_bearish < 2:
            log(f"❌ {symbol}: Not enough strong bearish indicators ({strong_bearish})")
            return False

        # Step 5: Check divergence (price rising, indicators bearish)
        candles_15m = candles_by_tf['15'][-5:]
        close_prices = [float(c['close']) for c in candles_15m]
        price_up = close_prices[-1] > close_prices[0]
        indicator_trend_down = sum(indicator_scores.values()) < -2

        if price_up and not indicator_trend_down:
            log(f"❌ {symbol}: Price rising while indicators not strongly bearish")
            return False

        log(f"✅ {symbol}: Short signal validated (macro + micro aligned)")
        return True

    except Exception as e:
        log(f"❌ Error validating short signal for {symbol}: {e}", level="ERROR")
        return False


# Cache cleanup functions

async def cleanup_caches_periodically():
    """
    Periodically clean up caches to prevent memory bloat
    FIX #10: Proper cache management
    """
    while True:
        try:
            async with _trend_cache_lock:
                # Clear old cache entries
                current_time = datetime.now()
                if ('timestamp' in _trend_cache and 
                    (current_time - _trend_cache['timestamp']).seconds > 3600):
                    _trend_cache.clear()
                    log("🧹 Cleared trend cache")
            
        except Exception as e:
            log(f"❌ Error cleaning caches: {e}", level="ERROR")
        
        await asyncio.sleep(1800)  # Clean every 30 minutes


# Export main functions for backward compatibility
__all__ = [
    'get_trend_context',
    'get_trend_context_cached', 
    'get_btc_trend',
    'detect_market_regime',
    'get_market_sentiment',
    'calculate_ema_fixed',
    'validate_short_signal',  # Added missing function
    'monitor_btc_trend_accuracy',
    'monitor_altseason_status',
    'btc_analyzer',
    'altseason_detector'
]



