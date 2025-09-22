#!/usr/bin/env python3
"""
Temporary patch to make the bot testable with limited data
REMOVE THIS FOR PRODUCTION - USE ONLY FOR TESTING
"""

def apply_testing_patch():
    """Apply temporary patches for testing with limited data"""
    
    import main
    
    print("🧪 APPLYING TESTING PATCHES")
    print("WARNING: These patches lower quality requirements!")
    print("Remove for production use!")
    
    # Save originals
    original_filter = main.filter_core_symbols
    original_min_scalp = main.MIN_SCALP_SCORE
    original_min_intraday = main.MIN_INTRADAY_SCORE
    original_min_swing = main.MIN_SWING_SCORE
    
    # Patch 1: Relaxed filtering
    async def testing_filter_core_symbols(symbols):
        """TESTING ONLY: Relaxed filter requirements"""
        from main import live_candles, fix_live_candles_structure
        
        live_candles = fix_live_candles_structure(live_candles)
        filtered = []
        
        print(f"🧪 Testing filter with {len(symbols)} symbols...")
        
        for symbol in symbols[:50]:  # Test more symbols
            if symbol in live_candles:
                core_candles = {}
                for tf in ['1', '5', '15']:
                    if tf in live_candles.get(symbol, {}):
                        candles = live_candles[symbol][tf]
                        # TESTING: Require only 5 candles (was 30)
                        if candles and len(candles) >= 5:
                            core_candles[tf] = candles
                
                # TESTING: Require only 2 timeframes (was 3)
                if len(core_candles) >= 2:
                    filtered.append(symbol)
        
        print(f"🧪 Testing filter found: {len(filtered)} symbols")
        return filtered[:10]  # Limit to 10 for testing
    
    # Patch 2: Lower score requirements
    main.MIN_SCALP_SCORE = 4.0      # Was 9.0
    main.MIN_INTRADAY_SCORE = 5.0   # Was 10.0  
    main.MIN_SWING_SCORE = 6.0      # Was 14.0
    
    # Apply patches
    main.filter_core_symbols = testing_filter_core_symbols
    
    print(f"✅ Applied testing patches:")
    print(f"   Candle requirement: 30 → 5")
    print(f"   Timeframe requirement: 3 → 2") 
    print(f"   Min scalp score: {original_min_scalp} → {main.MIN_SCALP_SCORE}")
    print(f"   Min intraday score: {original_min_intraday} → {main.MIN_INTRADAY_SCORE}")
    print(f"   Min swing score: {original_min_swing} → {main.MIN_SWING_SCORE}")
    
    return {
        'filter': original_filter,
        'min_scalp': original_min_scalp,
        'min_intraday': original_min_intraday,
        'min_swing': original_min_swing
    }

def remove_testing_patch(originals):
    """Remove testing patches and restore production settings"""
    import main
    
    main.filter_core_symbols = originals['filter']
    main.MIN_SCALP_SCORE = originals['min_scalp']
    main.MIN_INTRADAY_SCORE = originals['min_intraday'] 
    main.MIN_SWING_SCORE = originals['min_swing']
    
    print("✅ Removed testing patches - back to production settings")

async def run_testing_scan():
    """Run a scan with testing patches applied"""
    
    try:
        # Apply patches
        originals = apply_testing_patch()
        
        # Wait for some data
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        from main import core_strategy_scan
        from trend_upgrade_integration import get_trend_context_cached
        
        symbols = await fetch_symbols()
        websocket_task = asyncio.create_task(stream_candles(symbols[:20]))
        
        print("\n⏳ Collecting data for 30 seconds...")
        await asyncio.sleep(30)
        
        if live_candles and len(live_candles) > 0:
            print(f"✅ Got data for {len(live_candles)} symbols")
            
            trend_context = await get_trend_context_cached()
            print(f"📈 Trend context: {trend_context}")
            
            print(f"\n🔍 Running core strategy scan with testing patches...")
            await core_strategy_scan(symbols[:30], trend_context)
            
        else:
            print("❌ Still no websocket data available")
        
        websocket_task.cancel()
        
        # Remove patches
        remove_testing_patch(originals)
        
        print(f"\n🏁 Testing scan complete!")
        
    except Exception as e:
        print(f"❌ Testing scan failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    print("🧪 PRODUCTION READY TESTING")
    print("This will apply temporary patches to test with limited data")
    print("=" * 60)
    
    asyncio.run(run_testing_scan())
