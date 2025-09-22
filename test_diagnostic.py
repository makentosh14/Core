#!/usr/bin/env python3
"""
Quick test mode - reduces data requirements for immediate testing
"""

def patch_for_quick_testing():
    """
    Temporarily patch the core strategy to work with minimal data
    FOR TESTING ONLY - remove for production
    """
    
    import main
    
    # Save original filter function
    original_filter = main.filter_core_symbols
    
    async def quick_test_filter(symbols):
        """Relaxed filter for testing with minimal data"""
        try:
            from main import live_candles, fix_live_candles_structure
            
            # Fix structure
            global live_candles
            live_candles = fix_live_candles_structure(live_candles)
            
            filtered = []
            for symbol in symbols[:20]:  # Test first 20 only
                if symbol in live_candles:
                    # RELAXED: Accept any data (was 30+ candles)
                    timeframe_count = 0
                    for tf in ['1', '5', '15']:
                        if tf in live_candles[symbol]:
                            candles = live_candles[symbol][tf]
                            if candles and len(candles) >= 5:  # RELAXED: 5 candles (was 30)
                                timeframe_count += 1
                    
                    if timeframe_count >= 2:  # RELAXED: 2 TFs (was 3)
                        filtered.append(symbol)
            
            print(f"🧪 QUICK TEST: {len(filtered)} symbols passed relaxed filter")
            return filtered
            
        except Exception as e:
            print(f"Quick test filter error: {e}")
            return []
    
    # Patch the filter function
    main.filter_core_symbols = quick_test_filter
    print("🧪 Applied quick test patches (relaxed data requirements)")

async def run_quick_test():
    """Run core strategy with relaxed requirements"""
    
    print("🧪 QUICK TEST MODE")
    print("=" * 40)
    
    try:
        # Apply patches
        patch_for_quick_testing()
        
        # Import after patching
        from main import core_strategy_scan
        from scanner import fetch_symbols
        from trend_upgrade_integration import get_trend_context_cached
        
        # Get symbols and trend context
        symbols = await fetch_symbols()
        trend_context = await get_trend_context_cached()
        
        print(f"Testing with {len(symbols)} symbols...")
        print(f"Trend context: {trend_context}")
        
        # Run the scan
        await core_strategy_scan(symbols[:20], trend_context)  # Test first 20 symbols only
        
        print("Quick test completed!")
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_quick_test())
