#!/usr/bin/env python3
"""
test_apis.py - Simple script to test if external APIs work from your server
"""

import asyncio
import json
import time
from datetime import datetime

def print_header(title):
    """Print a nice header"""
    print("\n" + "=" * 60)
    print(f"🧪 {title}")
    print("=" * 60)

def print_test_result(name, success, data=None, error=None):
    """Print test result in a nice format"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {name}")
    
    if success and data:
        if isinstance(data, dict):
            for key, value in list(data.items())[:3]:  # Show first 3 items
                print(f"   {key}: {value}")
            if len(data) > 3:
                print(f"   ... and {len(data) - 3} more fields")
        else:
            print(f"   Data: {str(data)[:100]}...")
    elif error:
        print(f"   Error: {error}")
    print()

async def test_basic_connectivity():
    """Test basic internet connectivity"""
    print_header("BASIC CONNECTIVITY TEST")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            # Test basic HTTP connectivity
            try:
                async with session.get("https://httpbin.org/ip") as response:
                    if response.status == 200:
                        data = await response.json()
                        print_test_result("Internet Connectivity", True, {"your_ip": data.get("origin", "unknown")})
                        return True
                    else:
                        print_test_result("Internet Connectivity", False, error=f"HTTP {response.status}")
                        return False
            except Exception as e:
                print_test_result("Internet Connectivity", False, error=str(e))
                return False
                
    except ImportError:
        print_test_result("aiohttp Library", False, error="Not installed. Run: pip install aiohttp")
        return False

async def test_fear_greed_api():
    """Test Fear & Greed Index API"""
    print_header("FEAR & GREED INDEX API")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            url = "https://api.alternative.me/fng/"
            
            start_time = time.time()
            async with session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "data" in data and len(data["data"]) > 0:
                        fg_data = data["data"][0]
                        result_data = {
                            "fear_greed_value": fg_data.get("value", "unknown"),
                            "classification": fg_data.get("value_classification", "unknown"),
                            "response_time": f"{response_time:.2f}s"
                        }
                        print_test_result("Fear & Greed API", True, result_data)
                        return True, fg_data
                    else:
                        print_test_result("Fear & Greed API", False, error="Invalid response format")
                        return False, None
                else:
                    print_test_result("Fear & Greed API", False, error=f"HTTP {response.status}")
                    return False, None
                    
    except Exception as e:
        print_test_result("Fear & Greed API", False, error=str(e))
        return False, None

async def test_coingecko_api():
    """Test CoinGecko API"""
    print_header("COINGECKO API")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            # Test global data endpoint
            url = "https://api.coingecko.com/api/v3/global"
            
            start_time = time.time()
            async with session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    if "data" in data:
                        global_data = data["data"]
                        market_cap_pct = global_data.get("market_cap_percentage", {})
                        
                        result_data = {
                            "btc_dominance": f"{market_cap_pct.get('btc', 'unknown')}%",
                            "eth_dominance": f"{market_cap_pct.get('eth', 'unknown')}%",
                            "total_market_cap": f"${global_data.get('total_market_cap', {}).get('usd', 0):,.0f}",
                            "active_cryptos": global_data.get("active_cryptocurrencies", "unknown"),
                            "response_time": f"{response_time:.2f}s"
                        }
                        print_test_result("CoinGecko Global Data", True, result_data)
                        
                        # Test trending endpoint
                        trending_url = "https://api.coingecko.com/api/v3/search/trending"
                        async with session.get(trending_url) as trending_response:
                            if trending_response.status == 200:
                                trending_data = await trending_response.json()
                                trending_coins = [coin["item"]["name"] for coin in trending_data.get("coins", [])[:3]]
                                print_test_result("CoinGecko Trending", True, {"top_3_trending": trending_coins})
                                return True, global_data
                            else:
                                print_test_result("CoinGecko Trending", False, error=f"HTTP {trending_response.status}")
                                return True, global_data  # Still count as success for global data
                    else:
                        print_test_result("CoinGecko Global Data", False, error="Invalid response format")
                        return False, None
                else:
                    print_test_result("CoinGecko Global Data", False, error=f"HTTP {response.status}")
                    return False, None
                    
    except Exception as e:
        print_test_result("CoinGecko API", False, error=str(e))
        return False, None

async def test_bybit_api():
    """Test your existing Bybit API"""
    print_header("BYBIT API (Your Existing Connection)")
    
    try:
        # Try to import your bybit_api
        from bybit_api import signed_request
        
        # Test a simple API call
        response = await signed_request("GET", "/v5/market/tickers", {
            "category": "linear",
            "symbol": "BTCUSDT"
        })
        
        if response.get("retCode") == 0:
            ticker_data = response["result"]["list"][0]
            result_data = {
                "btc_price": f"${float(ticker_data['lastPrice']):,.2f}",
                "24h_change": f"{float(ticker_data['price24hPcnt']):.2f}%",
                "24h_volume": f"${float(ticker_data['volume24h']):,.0f}"
            }
            print_test_result("Bybit API", True, result_data)
            return True, ticker_data
        else:
            print_test_result("Bybit API", False, error=f"API Error: {response.get('retMsg', 'Unknown')}")
            return False, None
            
    except ImportError:
        print_test_result("Bybit API", False, error="bybit_api module not found")
        return False, None
    except Exception as e:
        print_test_result("Bybit API", False, error=str(e))
        return False, None

async def test_rate_limits():
    """Test API rate limits"""
    print_header("RATE LIMIT TEST")
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test multiple rapid calls to see rate limiting
            success_count = 0
            total_calls = 5
            
            print("Making 5 rapid API calls to test rate limits...")
            
            for i in range(total_calls):
                try:
                    async with session.get("https://api.alternative.me/fng/") as response:
                        if response.status == 200:
                            success_count += 1
                        elif response.status == 429:
                            print(f"   Call {i+1}: Rate limited (429)")
                        else:
                            print(f"   Call {i+1}: HTTP {response.status}")
                except Exception as e:
                    print(f"   Call {i+1}: Error - {e}")
                
                await asyncio.sleep(0.5)  # Small delay between calls
            
            rate_limit_data = {
                "successful_calls": f"{success_count}/{total_calls}",
                "rate_limit_hit": success_count < total_calls
            }
            
            if success_count >= 4:  # Allow for 1 failure
                print_test_result("Rate Limit Test", True, rate_limit_data)
                return True
            else:
                print_test_result("Rate Limit Test", False, rate_limit_data)
                return False
                
    except Exception as e:
        print_test_result("Rate Limit Test", False, error=str(e))
        return False

def create_api_test_report(results):
    """Create a summary report"""
    print_header("TEST SUMMARY REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result["success"])
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print()
    
    # Categorize results
    critical_apis = ["internet", "bybit"]
    enhanced_apis = ["fear_greed", "coingecko"]
    performance_tests = ["rate_limits"]
    
    print("🔑 Critical APIs (Required):")
    for api in critical_apis:
        if api in results:
            status = "✅" if results[api]["success"] else "❌"
            print(f"   {status} {api.replace('_', ' ').title()}")
    
    print("\n🚀 Enhanced APIs (Optional but Recommended):")
    for api in enhanced_apis:
        if api in results:
            status = "✅" if results[api]["success"] else "❌"
            print(f"   {status} {api.replace('_', ' ').title()}")
    
    print("\n⚡ Performance Tests:")
    for api in performance_tests:
        if api in results:
            status = "✅" if results[api]["success"] else "❌"
            print(f"   {status} {api.replace('_', ' ').title()}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if results.get("internet", {}).get("success") and results.get("bybit", {}).get("success"):
        print("✅ Your bot's basic functionality is working perfectly!")
        
        enhanced_working = sum(1 for api in enhanced_apis if results.get(api, {}).get("success"))
        
        if enhanced_working == len(enhanced_apis):
            print("✅ All enhanced APIs working - you can get 100% functionality!")
            print("🚀 Run: python3 api_integration_patch.py")
        elif enhanced_working > 0:
            print(f"⚠️ {enhanced_working}/{len(enhanced_apis)} enhanced APIs working")
            print("📈 You'll get partial enhanced functionality with fallbacks")
            print("🚀 Run: python3 setup_enhanced_trends.py")
        else:
            print("⚠️ Enhanced APIs not accessible from your server")
            print("🛡️ System will work with intelligent fallbacks")
            print("🚀 Run: python3 setup_enhanced_trends.py")
    else:
        print("❌ Critical connectivity issues detected")
        if not results.get("internet", {}).get("success"):
            print("🌐 Check your internet connection")
        if not results.get("bybit", {}).get("success"):
            print("🔑 Check your Bybit API configuration")
    
    print("\n📋 Next Steps:")
    if passed_tests >= total_tests * 0.6:  # 60% success rate
        print("1. Run enhanced trend setup: python3 setup_enhanced_trends.py")
        print("2. Your bot will automatically use the best available data sources")
        print("3. Monitor logs for 'Enhanced trend analysis' messages")
    else:
        print("1. Fix connectivity issues above")
        print("2. Ensure aiohttp is installed: pip install aiohttp")
        print("3. Check firewall/proxy settings if needed")

async def main():
    """Run all API tests"""
    print("🔬 API CONNECTIVITY TEST SUITE")
    print("Testing all external APIs for enhanced trend detection...")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Basic connectivity
    results["internet"] = {"success": await test_basic_connectivity()}
    
    # Test 2: Your existing Bybit API
    success, data = await test_bybit_api()
    results["bybit"] = {"success": success, "data": data}
    
    # Test 3: Fear & Greed API
    success, data = await test_fear_greed_api()
    results["fear_greed"] = {"success": success, "data": data}
    
    # Test 4: CoinGecko API
    success, data = await test_coingecko_api()
    results["coingecko"] = {"success": success, "data": data}
    
    # Test 5: Rate limits
    results["rate_limits"] = {"success": await test_rate_limits()}
    
    # Generate report
    create_api_test_report(results)
    
    return results

if __name__ == "__main__":
    try:
        print("Starting API tests...\n")
        results = asyncio.run(main())
        
        # Save results to file
        with open("api_test_results.json", "w") as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                json_results[key] = {
                    "success": value["success"],
                    "timestamp": datetime.now().isoformat()
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: api_test_results.json")
        print("🎯 Run this test anytime with: python3 test_apis.py")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test cancelled by user")
    except Exception as e:
        print(f"\n💥 Test suite error: {e}")
        import traceback
        traceback.print_exc()
