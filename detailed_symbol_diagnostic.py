#!/usr/bin/env python3
"""
Detailed diagnostic to see exactly where each symbol fails in the scanning pipeline
"""

import asyncio
import traceback

async def diagnose_symbol_processing():
    """Diagnose exactly where each symbol fails in the core strategy pipeline"""
    
    print("🔬 DETAILED SYMBOL PROCESSING DIAGNOSTIC")
    print("=" * 60)
    
    try:
        # Start websockets and collect data first
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        from trend_upgrade_integration import get_trend_context_cached
        
        symbols = await fetch_symbols()
        print(f"Starting websockets for {len(symbols[:20])} symbols...")
        
        # Start websocket task
        websocket_task = asyncio.create_task(stream_candles(symbols[:20]))
        
        # Wait for data
        for i in range(20):
            await asyncio.sleep(1)
            if live_candles and len(live_candles) >= 5:
                break
            print(f"Waiting for data... {i+1}/20")
        
        if not live_candles or len(live_candles) == 0:
            print("❌ No websocket data - cannot proceed")
            return
        
        print(f"✅ Got data for {len(live_candles)} symbols")
        
        # Import core strategy functions
        from main import (
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
        from score import score_symbol, calculate_confidence
        
        trend_context = await get_trend_context_cached()
        
        # Test each filtered symbol through the complete pipeline
        test_symbols = []
        
        # Quick filter to get test symbols
        for symbol in list(live_candles.keys())[:10]:
            tf_count = 0
            for tf in ['1', '5', '15']:
                if tf in live_candles[symbol]:
                    candles = live_candles[symbol][tf]
                    if candles and len(candles) >= 3:
                        tf_count += 1
            if tf_count >= 2:
                test_symbols.append(symbol)
        
        print(f"\nTesting {len(test_symbols)} symbols through complete pipeline:")
        print("=" * 60)
        
        for i, symbol in enumerate(test_symbols[:5], 1):  # Test first 5
            print(f"\n🔍 SYMBOL {i}: {symbol}")
            print("-" * 40)
            
            try:
                # Step 1: Check active trades
                if symbol in active_trades and not active_trades[symbol].get("exited", False):
                    print("❌ Step 1: Already have active position")
                    continue
                print("✅ Step 1: No active position")
                
                # Step 2: Check recent exits
                if symbol in recent_exits:
                    time_diff = time.time() - recent_exits[symbol]
                    if time_diff < EXIT_COOLDOWN:
                        print(f"❌ Step 2: In exit cooldown ({time_diff:.0f}s < {EXIT_COOLDOWN}s)")
                        continue
                print("✅ Step 2: Not in exit cooldown")
                
                # Step 3: Get candles
                core_candles = {}
                for tf in ['1', '5', '15']:
                    if tf in live_candles.get(symbol, {}):
                        candles = live_candles[symbol][tf]
                        if candles and len(candles) >= 3:  # Relaxed for testing
                            core_candles[tf] = candles
                
                print(f"✅ Step 3: Got {len(core_candles)} timeframes: {list(core_candles.keys())}")
                
                if len(core_candles) < 2:  # Relaxed requirement
                    print("❌ Step 3: Not enough timeframes")
                    continue
                
                # Step 4: Regular scoring
                try:
                    score_result = score_symbol(symbol, core_candles, trend_context)
                    score, tf_scores, trade_type, indicator_scores, used_indicators = score_result
                    print(f"✅ Step 4: Regular score = {score}")
                except Exception as e:
                    print(f"❌ Step 4: Regular scoring failed: {e}")
                    continue
                
                # Step 5: Core score
                try:
                    core_score = await calculate_core_score(symbol, core_candles, trend_context)
                    print(f"✅ Step 5: Core score = {core_score:.1f} (need >= {MIN_SCALP_SCORE})")
                    
                    if core_score < MIN_SCALP_SCORE:
                        print(f"❌ Step 5: Core score too low ({core_score:.1f} < {MIN_SCALP_SCORE})")
                        continue
                except Exception as e:
                    print(f"❌ Step 5: Core score calculation failed: {e}")
                    continue
                
                # Step 6: Direction
                try:
                    direction = determine_core_direction(core_candles, trend_context)
                    if not direction:
                        print("❌ Step 6: No direction determined")
                        continue
                    print(f"✅ Step 6: Direction = {direction}")
                except Exception as e:
                    print(f"❌ Step 6: Direction determination failed: {e}")
                    continue
                
                # Step 7: Confidence
                try:
                    confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                    print(f"✅ Step 7: Confidence = {confidence}% (need >= 60%)")
                    
                    if confidence < 60:
                        print(f"❌ Step 7: Confidence too low ({confidence}% < 60%)")
                        continue
                except Exception as e:
                    print(f"❌ Step 7: Confidence calculation failed: {e}")
                    continue
                
                # Step 8: Validate conditions
                try:
                    conditions_valid = await validate_core_conditions(symbol, core_candles, direction, trend_context)
                    if not conditions_valid:
                        print("❌ Step 8: Core conditions validation failed")
                        continue
                    print("✅ Step 8: Core conditions valid")
                except Exception as e:
                    print(f"❌ Step 8: Conditions validation error: {e}")
                    continue
                
                # Step 9: Strategy type
                try:
                    trend_strength = trend_context.get("trend_strength", 0.5)
                    strategy_type = determine_core_strategy_type(core_score, confidence, trend_strength)
                    if not strategy_type:
                        print("❌ Step 9: No strategy type determined")
                        continue
                    print(f"✅ Step 9: Strategy type = {strategy_type}")
                except Exception as e:
                    print(f"❌ Step 9: Strategy type determination failed: {e}")
                    continue
                
                # Step 10: Position limits
                try:
                    limits_ok = check_strategy_position_limits(strategy_type)
                    if not limits_ok:
                        print("❌ Step 10: Position limits exceeded")
                        continue
                    print("✅ Step 10: Position limits OK")
                except Exception as e:
                    print(f"❌ Step 10: Position limits check failed: {e}")
                    continue
                
                # Step 11: Confirmations
                try:
                    core_confirmations = await get_core_confirmations(symbol, core_candles, direction, trend_context)
                    confirmation_count = len(core_confirmations)
                    print(f"✅ Step 11: Got {confirmation_count} confirmations (need >= 2)")
                    
                    if confirmation_count < 2:
                        print(f"❌ Step 11: Not enough confirmations ({confirmation_count} < 2)")
                        continue
                except Exception as e:
                    print(f"❌ Step 11: Confirmations check failed: {e}")
                    continue
                
                # SUCCESS!
                print(f"🎉 SUCCESS: {symbol} passed all checks!")
                print(f"   Final: Score={core_score:.1f}, Direction={direction}, Type={strategy_type}")
                print(f"   Confidence={confidence}%, Confirmations={confirmation_count}")
                
            except Exception as e:
                print(f"❌ SYMBOL ERROR: {symbol} failed with exception: {e}")
                traceback.print_exc()
        
        # Cleanup
        websocket_task.cancel()
        
        print(f"\n🏁 DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("Check which step is consistently failing across symbols")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(diagnose_symbol_processing())
