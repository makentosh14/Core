# api_config.py - External API configuration and fallback system
"""
Configuration for external APIs used by enhanced trend system.
Includes fallback methods that work without external APIs.
"""

import aiohttp
import asyncio
import numpy as np
from typing import Dict, Any, Optional
from logger import log

# API Configuration
API_CONFIG = {
    "fear_greed": {
        "enabled": True,
        "url": "https://api.alternative.me/fng/",
        "timeout": 10,
        "fallback": True
    },
    "coingecko": {
        "enabled": True,
        "url": "https://api.coingecko.com/api/v3",
        "timeout": 10,
        "fallback": True
    },
    "social_sentiment": {
        "enabled": False,  # Disable by default
        "fallback": True
    }
}

class ExternalAPIManager:
    """Manages external API calls with intelligent fallbacks"""
    
    def __init__(self):
        self.session = None
        self.api_status = {}
        self.fallback_mode = {}
        
    async def ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={'User-Agent': 'TradingBot/1.0'}
            )
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index with fallback"""
        api_name = "fear_greed"
        
        # Check if we should use fallback
        if not API_CONFIG[api_name]["enabled"] or self.fallback_mode.get(api_name, False):
            return await self._fallback_fear_greed()
        
        try:
            await self.ensure_session()
            
            async with self.session.get(API_CONFIG[api_name]["url"]) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "data" in data and len(data["data"]) > 0:
                        fg_data = data["data"][0]
                        
                        result = {
                            "value": int(fg_data.get("value", 50)),
                            "classification": fg_data.get("value_classification", "Neutral"),
                            "timestamp": fg_data.get("timestamp"),
                            "source": "api"
                        }
                        
                        # Reset fallback mode on success
                        self.fallback_mode[api_name] = False
                        log(f"✅ Fear & Greed Index: {result['value']}")
                        return result
                    
        except Exception as e:
            log(f"⚠️ Fear & Greed API error: {e}, using fallback", level="WARNING")
            
        # Use fallback on any error
        self.fallback_mode[api_name] = True
        return await self._fallback_fear_greed()
    
    async def _fallback_fear_greed(self) -> Dict[str, Any]:
        """Fallback Fear & Greed calculation based on BTC volatility"""
        try:
            from bybit_api import signed_request
            
            # Get BTC volatility as proxy for fear/greed
            response = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": "BTCUSDT",
                "interval": "60",
                "limit": 24
            })
            
            if response.get("retCode") == 0:
                candles = response["result"]["list"]
                
                # Calculate volatility
                price_changes = []
                for candle in candles:
                    open_price = float(candle[1])
                    close_price = float(candle[4])
                    change = abs((close_price - open_price) / open_price)
                    price_changes.append(change)
                
                volatility = np.mean(price_changes)
                
                # Map volatility to fear/greed (inverse relationship)
                if volatility > 0.05:  # Very high volatility = fear
                    fg_value = 20
                    classification = "Extreme Fear"
                elif volatility > 0.03:
                    fg_value = 35
                    classification = "Fear"
                elif volatility < 0.01:  # Low volatility = greed
                    fg_value = 75
                    classification = "Greed"
                elif volatility < 0.015:
                    fg_value = 60
                    classification = "Neutral"
                else:
                    fg_value = 50
                    classification = "Neutral"
                
                return {
                    "value": fg_value,
                    "classification": classification,
                    "timestamp": None,
                    "source": "fallback_volatility"
                }
        
        except Exception as e:
            log(f"❌ Fallback fear & greed failed: {e}", level="ERROR")
        
        # Ultimate fallback
        return {
            "value": 50,
            "classification": "Neutral",
            "timestamp": None,
            "source": "default"
        }
    
    async def get_market_dominance(self) -> Dict[str, Any]:
        """Get BTC dominance with fallback"""
        api_name = "coingecko"
        
        if not API_CONFIG[api_name]["enabled"] or self.fallback_mode.get(api_name, False):
            return await self._fallback_market_dominance()
        
        try:
            await self.ensure_session()
            
            url = f"{API_CONFIG[api_name]['url']}/global"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "data" in data:
                        global_data = data["data"]
                        
                        result = {
                            "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 50),
                            "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 15),
                            "total_market_cap": global_data.get("total_market_cap", {}).get("usd", 0),
                            "source": "api"
                        }
                        
                        self.fallback_mode[api_name] = False
                        log(f"✅ BTC Dominance: {result['btc_dominance']:.1f}%")
                        return result
                        
        except Exception as e:
            log(f"⚠️ CoinGecko API error: {e}, using fallback", level="WARNING")
        
        self.fallback_mode[api_name] = True
        return await self._fallback_market_dominance()
    
    async def _fallback_market_dominance(self) -> Dict[str, Any]:
        """Fallback market dominance calculation"""
        try:
            from bybit_api import signed_request
            
            # Get BTC and ETH volumes as proxy for dominance
            btc_response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            })
            
            eth_response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear", 
                "symbol": "ETHUSDT"
            })
            
            if (btc_response.get("retCode") == 0 and 
                eth_response.get("retCode") == 0):
                
                btc_volume = float(btc_response["result"]["list"][0]["volume24h"])
                eth_volume = float(eth_response["result"]["list"][0]["volume24h"])
                
                # Estimate dominance based on volume (rough approximation)
                total_volume = btc_volume + eth_volume  # Simplified
                btc_dominance = (btc_volume / total_volume) * 100 if total_volume > 0 else 50
                eth_dominance = (eth_volume / total_volume) * 100 if total_volume > 0 else 15
                
                # Adjust to realistic ranges
                btc_dominance = max(40, min(70, btc_dominance * 0.6 + 45))  # Scale to 40-70%
                eth_dominance = max(10, min(25, eth_dominance * 0.4 + 15))  # Scale to 10-25%
                
                return {
                    "btc_dominance": btc_dominance,
                    "eth_dominance": eth_dominance,
                    "total_market_cap": 0,
                    "source": "fallback_volume"
                }
                
        except Exception as e:
            log(f"❌ Fallback dominance calculation failed: {e}", level="ERROR")
        
        # Ultimate fallback - realistic defaults
        return {
            "btc_dominance": 52.0,  # Typical BTC dominance
            "eth_dominance": 17.0,  # Typical ETH dominance
            "total_market_cap": 0,
            "source": "default"
        }
    
    async def get_social_sentiment(self) -> Dict[str, Any]:
        """Get social sentiment (currently fallback only)"""
        # This would integrate with Twitter/Reddit APIs in the future
        # For now, use volume-based sentiment
        
        try:
            from bybit_api import signed_request
            
            # Get volume data for sentiment proxy
            response = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": "BTCUSDT"
            })
            
            if response.get("retCode") == 0:
                ticker_data = response["result"]["list"][0]
                volume_24h = float(ticker_data["volume24h"])
                price_change = float(ticker_data["price24hPcnt"])
                
                # Calculate sentiment based on volume and price
                if volume_24h > 1000000000 and price_change > 2:  # High volume + green
                    sentiment = "very_bullish"
                    score = 0.8
                elif volume_24h > 500000000 and price_change > 0:
                    sentiment = "bullish"
                    score = 0.65
                elif volume_24h > 1000000000 and price_change < -2:  # High volume + red
                    sentiment = "very_bearish"
                    score = 0.2
                elif price_change < -1:
                    sentiment = "bearish"
                    score = 0.35
                else:
                    sentiment = "neutral"
                    score = 0.5
                
                return {
                    "sentiment": sentiment,
                    "score": score,
                    "volume_24h": volume_24h,
                    "price_change": price_change,
                    "source": "fallback_volume"
                }
                
        except Exception as e:
            log(f"❌ Social sentiment fallback failed: {e}", level="ERROR")
        
        return {
            "sentiment": "neutral",
            "score": 0.5,
            "source": "default"
        }

# Global API manager instance
api_manager = ExternalAPIManager()

# Convenience functions for easy use
async def get_fear_greed() -> int:
    """Get Fear & Greed Index value (0-100)"""
    try:
        result = await api_manager.get_fear_greed_index()
        return result.get("value", 50)
    except Exception:
        return 50

async def get_btc_dominance() -> float:
    """Get BTC dominance percentage"""
    try:
        result = await api_manager.get_market_dominance()
        return result.get("btc_dominance", 52.0)
    except Exception:
        return 52.0

async def get_sentiment_score() -> float:
    """Get sentiment score (0.0-1.0)"""
    try:
        result = await api_manager.get_social_sentiment()
        return result.get("score", 0.5)
    except Exception:
        return 0.5

# Test function
async def test_all_apis():
    """Test all external APIs"""
    print("🧪 Testing external APIs...")
    
    try:
        # Test Fear & Greed
        fg_result = await api_manager.get_fear_greed_index()
        print(f"✅ Fear & Greed: {fg_result['value']} ({fg_result['source']})")
        
        # Test Market Dominance
        dom_result = await api_manager.get_market_dominance()
        print(f"✅ BTC Dominance: {dom_result['btc_dominance']:.1f}% ({dom_result['source']})")
        
        # Test Social Sentiment
        sent_result = await api_manager.get_social_sentiment()
        print(f"✅ Sentiment: {sent_result['sentiment']} ({sent_result['source']})")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False
    
    finally:
        await api_manager.close_session()

# Cleanup function
async def cleanup_apis():
    """Cleanup API resources"""
    await api_manager.close_session()

# Export main functions
__all__ = [
    'ExternalAPIManager',
    'api_manager',
    'get_fear_greed',
    'get_btc_dominance', 
    'get_sentiment_score',
    'test_all_apis',
    'cleanup_apis'
]
