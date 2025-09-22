#!/usr/bin/env python3
"""
Simple check of what data is actually available
"""

import asyncio
import time

async def check_actual_data():
    """Check what data is actually in live_candles"""
    
    print("📊 CHECKING ACTUAL WEBSOCKET DATA")
    print("=" * 50)
    
    try:
        # Start websockets
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        
        symbols = await fetch_symbols()
        websocket_task = asyncio.create_task(stream_candles(symbols[:10]))
        
        print("Collecting data for 15 seconds...")
        await asyncio.sleep(15)
        
        print(f"\n📊 LIVE_CANDLES ANALYSIS:")
        print(f"Type: {type(live_candles)}")
        print(f"Total symbols: {len(live_candles) if live_candles else 0}")
        
        if not live_candles:
            print("❌ live_candles is empty")
            return
        
        # Check first 5 symbols in detail
        for symbol in list(live_candles.keys())[:5]:
            print(f"\n🔍 {symbol}:")
            symbol_data = live_candles[symbol]
            print(f"   Type: {type(symbol_data)}")
            print(f"   Keys: {list(symbol_data.keys()) if hasattr(symbol_data, 'keys') else 'No keys'}")
            
            for tf in ['1', '5', '15']:
                if tf in symbol_data:
                    candles = symbol_data[tf]
                    print(f"   {tf}m: {len(candles) if candles else 0} candles")
                    
                    if candles and len(candles) > 0:
                        # Show sample candle
                        sample = candles[0] if len(candles) > 0 else None
                        if sample:
                            print(f"      Sample: {type(sample)} - {list(sample.keys()) if hasattr(sample, 'keys') else sample}")
                else:
                    print(f"   {tf}m: missing")
        
        # Test the exact filter logic
        print(f"\n🔍 TESTING FILTER LOGIC:")
        
        filtered_count = 0
        for symbol in list(live_candles.keys())[:10]:
            print(f"\n   Testing {symbol}:")
            
            if symbol in live_candles:
                print(f"     ✅ Symbol exists in live_candles")
                
                tf_count = 0
                for tf in ['1', '5', '15']:
                    if tf in live_candles[symbol]:
                        candles = live_candles[symbol][tf]
                        if candles:
                            candle_count = len(candles)
                            print(f"     {tf}m: {candle_count} candles")
                            if candle_count >= 3:
                                tf_count += 1
                        else:
                            print(f"     {tf}m: empty list")
                    else:
                        print(f"     {tf}m: missing key")
                
                print(f"     Timeframes with 3+ candles: {tf_count}")
                
                if tf_count >= 2:
                    filtered_count += 1
                    print(f"     ✅ WOULD PASS FILTER")
                else:
                    print(f"     ❌ WOULD FAIL FILTER")
            else:
                print(f"     ❌ Symbol not in live_candles")
        
        print(f"\n📊 FILTER RESULTS:")
        print(f"   Symbols tested: {min(10, len(live_candles))}")
        print(f"   Would pass filter: {filtered_count}")
        
        websocket_task.cancel()
        
    except Exception as e:
        print(f"❌ Data check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_actual_data())
