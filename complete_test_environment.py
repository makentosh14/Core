#!/usr/bin/env python3
"""
Complete test environment - runs websockets AND scanning together
"""

import asyncio
import time
from datetime import datetime

async def run_complete_test():
    """Run websockets + scanning together for testing"""
    
    print("🔧 COMPLETE TEST ENVIRONMENT")
    print("=" * 50)
    print("Starting websockets + scanning together...")
    
    try:
        # Import required modules
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        from main import core_strategy_scan
        from trend_upgrade_integration import get_trend_context_cached
        
        # Get symbols
        symbols = await fetch_symbols()
        print(f"✅ Fetched {len(symbols)} symbols")
        
        # Start websocket data collection
        print("🔌 Starting websocket streams...")
        websocket_task = asyncio.create_task(stream_candles(symbols[:20]))  # Test with 20 symbols
        
        # Wait for initial data collection
        print("⏳ Waiting for websocket data collection...")
        for i in range(30):  # Wait up to 30 seconds
            await asyncio.sleep(1)
            
            if live_candles and len(live_candles) > 0:
                symbol = list(live_candles.keys())[0]
                candle_counts = []
                for tf in ['1', '5', '15']:
                    if tf in live_candles[symbol]:
                        count = len(live_candles[symbol][tf])
                        candle_counts.append(f"{tf}m:{count}")
                
                print(f"⏳ Second {i+1}: {len(live_candles)} symbols, Sample: {candle_counts}")
                
                # Check if we have enough data to test
                if len(live_candles) >= 5:  # At least 5 symbols
                    enough_data = False
                    for sym in list(live_candles.keys())[:5]:
                        tf_with_data = 0
                        for tf in ['1', '5', '15']:
                            if tf in live_candles[sym] and len(live_candles[sym][tf]) >= 5:
                                tf_with_data += 1
                        if tf_with_data >= 2:  # At least 2 timeframes with 5+ candles
                            enough_data = True
                            break
                    
                    if enough_data:
                        print(f"✅ Sufficient data after {i+1} seconds!")
                        break
            else:
                print(f"⏳ Second {i+1}: No data yet...")
        
        # Check final data status
        if not live_candles or len(live_candles) == 0:
            print("❌ No websocket data collected after 30 seconds")
            print("   This indicates a websocket connection problem")
            return
        
        print(f"\n📊 Data collection results:")
        print(f"   Total symbols with data: {len(live_candles)}")
        
        # Show data quality for first 5 symbols
        for symbol in list(live_candles.keys())[:5]:
            print(f"   {symbol}:")
            for tf in ['1', '5', '15']:
                if tf in live_candles[symbol]:
                    count = len(live_candles[symbol][tf])
                    print(f"     {tf}m: {count} candles")
        
        # Now test the scanning
        print(f"\n🔍 Testing core strategy scanning...")
        
        # Apply relaxed filtering for testing
        import main
        original_filter = main.filter_core_symbols
        
        async def test_filter(symbols):
            """Test filter with current data"""
            filtered = []
            for symbol in symbols[:10]:  # Test first 10
                if symbol in live_candles:
                    tf_count = 0
                    for tf in ['1', '5', '15']:
                        if tf in live_candles[symbol]:
                            candles = live_candles[symbol][tf]
                            if candles and len(candles) >= 3:  # Very relaxed: 3 candles
                                tf_count += 1
                    
                    if tf_count >= 2:  # At least 2 timeframes
                        filtered.append(symbol)
                        print(f"     ✅ {symbol}: {tf_count} timeframes ready")
            
            print(f"   Filter result: {len(filtered)} symbols passed")
            return filtered
        
        # Temporarily replace filter
        main.filter_core_symbols = test_filter
        
        # Get trend context
        trend_context = await get_trend_context_cached()
        print(f"   Trend context: {trend_context}")
        
        # Run the scan
        await core_strategy_scan(symbols[:20], trend_context)
        
        print("\n✅ Complete test finished!")
        
        # Cleanup
        websocket_task.cancel()
        
    except KeyboardInterrupt:
        print("\n⏹️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Complete test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting complete test environment...")
    print("This will run websockets + scanning together")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(run_complete_test())
