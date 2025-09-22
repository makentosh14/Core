#!/usr/bin/env python3
"""
Test using the exact same logic as main.py core_strategy_scan
"""

import asyncio
import time

async def test_exact_main_logic():
    """Test with the exact same steps as main.py"""
    
    print("🎯 TESTING WITH EXACT MAIN BOT LOGIC")
    print("=" * 50)
    
    try:
        # Start websockets first
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        
        symbols = await fetch_symbols()
        websocket_task = asyncio.create_task(stream_candles(symbols[:10]))
        
        print("Collecting data for 20 seconds...")
        await asyncio.sleep(20)
        
        print(f"Got data for {len(live_candles)} symbols")
        
        # Import everything exactly like main.py
        from main import (
            core_strategy_scan,
            filter_core_symbols, 
            calculate_core_score,
            determine_core_direction,
            validate_core_conditions,
            determine_core_strategy_type,
            check_strategy_position_limits,
            get_core_confirmations,
            MIN_SCALP_SCORE,
            active_trades,
            recent_exits,
            EXIT_COOLDOWN
        )
        from trend_upgrade_integration import get_trend_context_cached
        from score import score_symbol, calculate_confidence
        
        trend_context = await get_trend_context_cached()
        print(f"Trend context: {trend_context}")
        
        # Step 1: Use the actual filter_core_symbols function
        print(f"\n🔍 STEP 1: Using real filter_core_symbols()")
        try:
            quality_symbols = await filter_core_symbols(symbols[:20])
            print(f"Filter returned: {len(quality_symbols) if quality_symbols else 0} symbols")
            if quality_symbols:
                print(f"Filtered symbols: {quality_symbols[:5]}...")
            else:
                print("❌ Filter returned no symbols - this is the issue!")
                
                # Let's debug the real filter
                print("\n🔧 DEBUGGING REAL FILTER:")
                from main import live_candles as main_live_candles, fix_live_candles_structure
                
                print(f"live_candles before fix: {len(live_candles)} symbols")
                fixed_candles = fix_live_candles_structure(live_candles)
                print(f"live_candles after fix: {len(fixed_candles)} symbols")
                
                # Manual check with real filter criteria
                manual_filtered = []
                for symbol in symbols[:10]:
                    if symbol in fixed_candles:
                        # Check exact filter criteria from main.py
                        core_candles = {}
                        for tf in ['1', '5', '15']:
                            if tf in fixed_candles.get(symbol, {}):
                                candles = fixed_candles[symbol][tf]
                                if candles and len(candles) >= 30:  # Real requirement: 30 candles
                                    core_candles[tf] = candles
                        
                        if len(core_candles) >= 3:  # Real requirement: all 3 TFs
                            manual_filtered.append(symbol)
                            print(f"  ✅ {symbol}: {len(core_candles)} TFs with 30+ candles")
                        else:
                            tf_counts = {}
                            for tf in ['1', '5', '15']:
                                if tf in fixed_candles[symbol]:
                                    count = len(fixed_candles[symbol][tf])
                                    tf_counts[tf] = count
                            print(f"  ❌ {symbol}: {tf_counts} (need 30+ each)")
                
                print(f"Manual filter with REAL criteria: {len(manual_filtered)} symbols")
                
                if len(manual_filtered) == 0:
                    print("\n💡 ROOT CAUSE FOUND:")
                    print("   Your filter requires 30 candles per timeframe")
                    print("   But you only have 3-5 candles per timeframe")  
                    print("   The bot needs to run for 30+ minutes to collect enough data")
                    print("   OR lower the candle requirement for testing")
                    return
        except Exception as e:
            print(f"❌ Filter test failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 2: If we got symbols, test the pipeline manually
        if quality_symbols and len(quality_symbols) > 0:
            print(f"\n🔍 STEP 2: Testing pipeline with {len(quality_symbols)} symbols")
            
            for symbol in quality_symbols[:3]:
                print(f"\n--- Testing {symbol} ---")
                
                try:
                    # Get candles exactly like main.py
                    core_candles = {}
                    for tf in ['1', '5', '15']:
                        if tf in live_candles.get(symbol, {}):
                            candles = live_candles[symbol][tf]
                            if candles and len(candles) >= 30:  # Real requirement
                                core_candles[tf] = candles
                    
                    if len(core_candles) < 3:
                        print(f"❌ Not enough timeframes: {len(core_candles)}")
                        continue
                    
                    # Test each step
                    score_result = score_symbol(symbol, core_candles, trend_context)
                    score, tf_scores, trade_type, indicator_scores, used_indicators = score_result
                    print(f"Regular score: {score}")
                    
                    core_score = await calculate_core_score(symbol, core_candles, trend_context)
                    print(f"Core score: {core_score} (need >= {MIN_SCALP_SCORE})")
                    
                    if core_score < MIN_SCALP_SCORE:
                        print(f"❌ Score too low")
                        continue
                    
                    direction = determine_core_direction(core_candles, trend_context)
                    print(f"Direction: {direction}")
                    
                    if not direction:
                        print(f"❌ No direction")
                        continue
                    
                    confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                    print(f"Confidence: {confidence}%")
                    
                    if confidence < 60:
                        print(f"❌ Low confidence")
                        continue
                    
                    conditions_valid = await validate_core_conditions(symbol, core_candles, direction, trend_context)
                    print(f"Conditions: {conditions_valid}")
                    
                    if conditions_valid:
                        print(f"🎉 {symbol} WOULD GENERATE A SIGNAL!")
                    else:
                        print(f"❌ Conditions failed")
                        
                except Exception as e:
                    print(f"❌ {symbol} failed: {e}")
        
        websocket_task.cancel()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

async def create_test_with_relaxed_requirements():
    """Create a temporary test that lowers the candle requirements"""
    
    print(f"\n🧪 CREATING TEST WITH RELAXED REQUIREMENTS")
    print("=" * 50)
    
    try:
        # Patch main.py temporarily
        import main
        
        # Save original values
        original_filter = main.filter_core_symbols
        
        async def relaxed_filter(symbols):
            """Filter with relaxed requirements for testing"""
            from main import live_candles, fix_live_candles_structure
            
            live_candles = fix_live_candles_structure(live_candles)
            filtered = []
            
            for symbol in symbols[:20]:
                if symbol in live_candles:
                    core_candles = {}
                    for tf in ['1', '5', '15']:
                        if tf in live_candles.get(symbol, {}):
                            candles = live_candles[symbol][tf]
                            if candles and len(candles) >= 3:  # RELAXED: 3 candles (was 30)
                                core_candles[tf] = candles
                    
                    if len(core_candles) >= 3:  # Still need all 3 TFs
                        filtered.append(symbol)
            
            print(f"🧪 Relaxed filter: {len(filtered)} symbols")
            return filtered
        
        # Apply patch
        main.filter_core_symbols = relaxed_filter
        
        # Now run the scan
        from scanner import fetch_symbols
        from trend_upgrade_integration import get_trend_context_cached
        
        symbols = await fetch_symbols()
        trend_context = await get_trend_context_cached()
        
        print(f"Running core_strategy_scan with relaxed requirements...")
        await main.core_strategy_scan(symbols[:20], trend_context)
        
        # Restore original
        main.filter_core_symbols = original_filter
        
    except Exception as e:
        print(f"❌ Relaxed test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_exact_main_logic())
    print(f"\n" + "="*50)
    asyncio.run(create_test_with_relaxed_requirements())
