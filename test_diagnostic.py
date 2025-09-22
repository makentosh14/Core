#!/usr/bin/env python3
"""
Diagnostic script to identify why core strategy scanning finds 0 scanned symbols
"""

import asyncio
import traceback
from datetime import datetime

async def diagnose_scanning_issue():
    """Step-by-step diagnosis of scanning pipeline"""
    
    print("🔬 CORE STRATEGY SCANNING DIAGNOSIS")
    print("=" * 60)
    
    try:
        # Import main functions
        from main import (
            core_strategy_scan,
            filter_core_symbols,
            live_candles,
            active_trades,
            recent_exits,
            EXIT_COOLDOWN,
            MAX_CORE_POSITIONS
        )
        from scanner import fetch_symbols
        from trend_upgrade_integration import get_trend_context_cached
        
        # Step 1: Check if we can get symbols
        print("\n📊 STEP 1: Symbol Fetching")
        try:
            all_symbols = await fetch_symbols()
            print(f"✅ Fetched {len(all_symbols)} total symbols")
            print(f"   First 10: {all_symbols[:10]}")
        except Exception as e:
            print(f"❌ Symbol fetching failed: {e}")
            return
        
        # Step 2: Check trend context
        print("\n📈 STEP 2: Trend Context")
        try:
            trend_context = await get_trend_context_cached()
            print(f"✅ Trend context: {trend_context}")
        except Exception as e:
            print(f"❌ Trend context failed: {e}")
            trend_context = {"trend": "neutral", "trend_strength": 0.5}
        
        # Step 3: Check position limits
        print("\n🚫 STEP 3: Position Limits Check")
        current_positions = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        print(f"   Current positions: {current_positions}")
        print(f"   Max positions: {MAX_CORE_POSITIONS}")
        print(f"   Active trades: {len(active_trades)}")
        print(f"   Recent exits: {len(recent_exits)}")
        
        if current_positions >= MAX_CORE_POSITIONS:
            print(f"❌ Max positions reached! This would stop scanning.")
            return
        else:
            print(f"✅ Position limits OK")
        
        # Step 4: Check live_candles data
        print("\n📊 STEP 4: Live Candles Data")
        if not live_candles:
            print("❌ live_candles is empty or None!")
            print("   This means websocket data isn't available yet")
            print("   The bot needs to run for a few minutes to collect data")
            return
        else:
            print(f"✅ live_candles contains {len(live_candles)} symbols")
            
            # Check data quality
            symbols_with_all_tf = 0
            symbols_with_sufficient_data = 0
            
            for symbol in list(live_candles.keys())[:5]:  # Check first 5
                tf_count = len(live_candles[symbol])
                print(f"   {symbol}: {tf_count} timeframes")
                
                if tf_count >= 3:  # Has 1m, 5m, 15m
                    symbols_with_all_tf += 1
                    
                    # Check data sufficiency
                    sufficient = True
                    for tf in ['1', '5', '15']:
                        if tf in live_candles[symbol]:
                            candles = live_candles[symbol][tf]
                            if not candles or len(candles) < 30:
                                sufficient = False
                                print(f"     {tf}m: {len(candles) if candles else 0} candles (need 30)")
                            else:
                                print(f"     {tf}m: {len(candles)} candles ✅")
                        else:
                            sufficient = False
                            print(f"     {tf}m: missing ❌")
                    
                    if sufficient:
                        symbols_with_sufficient_data += 1
            
            print(f"   Symbols with all timeframes: {symbols_with_all_tf}")
            print(f"   Symbols with sufficient data: {symbols_with_sufficient_data}")
        
        # Step 5: Test filtering function directly
        print("\n🔍 STEP 5: Filter Testing")
        try:
            test_symbols = all_symbols[:50]  # Test with first 50 symbols
            print(f"   Testing filter with {len(test_symbols)} symbols...")
            
            filtered_symbols = await filter_core_symbols(test_symbols)
            print(f"   Filter returned: {type(filtered_symbols)}")
            print(f"   Filter result length: {len(filtered_symbols) if filtered_symbols else 'None/Empty'}")
            
            if filtered_symbols:
                print(f"   Filtered symbols: {filtered_symbols[:10]}...")
            else:
                print("❌ Filter returned empty result!")
                print("   This explains why scanned_count = 0")
                
                # Let's check what's failing in the filter
                print("\n🔍 STEP 5b: Manual Filter Check")
                manual_filtered = []
                
                for symbol in test_symbols[:10]:  # Check first 10 manually
                    try:
                        if symbol in live_candles:
                            core_candles = {}
                            for tf in ['1', '5', '15']:
                                if tf in live_candles[symbol]:
                                    candles = live_candles[symbol][tf]
                                    if candles and len(candles) >= 30:
                                        core_candles[tf] = candles
                            
                            if len(core_candles) >= 3:
                                manual_filtered.append(symbol)
                                print(f"     ✅ {symbol}: All TFs available")
                            else:
                                print(f"     ❌ {symbol}: Only {len(core_candles)} TFs")
                        else:
                            print(f"     ❌ {symbol}: Not in live_candles")
                    except Exception as e:
                        print(f"     ❌ {symbol}: Error - {e}")
                
                print(f"   Manual filter found: {len(manual_filtered)} symbols")
                
        except Exception as e:
            print(f"❌ Filter testing failed: {e}")
            traceback.print_exc()
        
        # Step 6: Test one symbol through full pipeline
        print("\n🔬 STEP 6: Full Pipeline Test")
        if filtered_symbols and len(filtered_symbols) > 0:
            test_symbol = filtered_symbols[0]
            print(f"   Testing {test_symbol} through full pipeline...")
            
            try:
                from main import (
                    calculate_core_score,
                    determine_core_direction,
                    validate_core_conditions,
                    MIN_SCALP_SCORE
                )
                from score import score_symbol, calculate_confidence
                
                # Get candles
                core_candles = {}
                for tf in ['1', '5', '15']:
                    if tf in live_candles[test_symbol]:
                        candles = live_candles[test_symbol][tf]
                        if candles and len(candles) >= 30:
                            core_candles[tf] = candles
                
                print(f"     Candles extracted: {list(core_candles.keys())}")
                
                # Test scoring
                score_result = score_symbol(test_symbol, core_candles, trend_context)
                score, tf_scores, trade_type, indicator_scores, used_indicators = score_result
                print(f"     Regular score: {score}")
                
                # Test core score
                core_score = await calculate_core_score(test_symbol, core_candles, trend_context)
                print(f"     Core score: {core_score} (need >= {MIN_SCALP_SCORE})")
                
                if core_score < MIN_SCALP_SCORE:
                    print(f"     ❌ Core score too low")
                else:
                    print(f"     ✅ Core score sufficient")
                    
                    # Test direction
                    direction = determine_core_direction(core_candles, trend_context)
                    print(f"     Direction: {direction}")
                    
                    if direction:
                        # Test confidence
                        confidence = calculate_confidence(score, tf_scores, trend_context, trade_type)
                        print(f"     Confidence: {confidence}% (need >= 60%)")
                        
                        if confidence >= 60:
                            # Test validation
                            conditions_valid = await validate_core_conditions(test_symbol, core_candles, direction, trend_context)
                            print(f"     Conditions valid: {conditions_valid}")
                            
                            if conditions_valid:
                                print(f"     🎉 {test_symbol} would generate a signal!")
                            else:
                                print(f"     ❌ Conditions validation failed")
                        else:
                            print(f"     ❌ Confidence too low")
                    else:
                        print(f"     ❌ No direction determined")
                
            except Exception as e:
                print(f"     ❌ Pipeline test failed: {e}")
                traceback.print_exc()
        
        # Final diagnosis
        print("\n🏁 DIAGNOSIS COMPLETE")
        print("=" * 60)
        
        if not live_candles:
            print("❌ ROOT CAUSE: No websocket data available")
            print("   SOLUTION: Wait for websocket to collect data (2-3 minutes)")
        elif not filtered_symbols or len(filtered_symbols) == 0:
            print("❌ ROOT CAUSE: Filter returns no symbols")
            print("   SOLUTION: Check data quality or relax filtering criteria")
        else:
            print("✅ Pipeline appears functional")
            print("   Issue may be: High quality thresholds or market conditions")
        
    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        print("   Make sure you're running from the correct directory")
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        traceback.print_exc()

async def quick_live_data_check():
    """Quick check of live data availability"""
    print("\n🚀 QUICK LIVE DATA CHECK")
    print("=" * 30)
    
    try:
        from main import live_candles
        
        if not live_candles:
            print("❌ No live_candles data")
            print("   The bot needs to run with websockets active")
            return False
        
        print(f"✅ Found data for {len(live_candles)} symbols")
        
        # Check a few symbols
        working_symbols = []
        for symbol in list(live_candles.keys())[:10]:
            tf_count = 0
            for tf in ['1', '5', '15']:
                if tf in live_candles[symbol]:
                    candles = live_candles[symbol][tf]
                    if candles and len(candles) >= 30:
                        tf_count += 1
            
            if tf_count >= 3:
                working_symbols.append(symbol)
        
        print(f"✅ {len(working_symbols)} symbols have sufficient data")
        
        if len(working_symbols) == 0:
            print("❌ No symbols have enough data yet")
            print("   Wait for more websocket data collection")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Quick check failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting diagnostic...")
    
    # Quick check first
    asyncio.run(quick_live_data_check())
    
    print("\nRunning full diagnosis...")
    asyncio.run(diagnose_scanning_issue())
