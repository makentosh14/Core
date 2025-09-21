# test_scanning.py - Test the core strategy scanning pipeline

import asyncio
import sys
import traceback
from collections import defaultdict, deque

# Mock the live_candles structure
def create_mock_live_candles():
    """Create realistic mock live_candles data"""
    mock_candles = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
    
    # Create mock candle data for several symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        for tf in ['1', '5', '15']:
            # Create 50 realistic candles
            candles = []
            base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 1.0
            
            for i in range(50):
                price_change = (i % 10 - 5) * 0.01  # Small price movements
                current_price = base_price * (1 + price_change)
                
                candle = {
                    'timestamp': 1700000000000 + i * 60000,  # Sequential timestamps
                    'open': str(current_price * 0.999),
                    'high': str(current_price * 1.002),
                    'low': str(current_price * 0.998),
                    'close': str(current_price),
                    'volume': str(1000000 + i * 10000)  # Decent volume
                }
                candles.append(candle)
            
            # Store as list (not deque) to match your fixed structure
            mock_candles[symbol][tf] = candles
    
    return dict(mock_candles)

async def test_filtering():
    """Test the symbol filtering process"""
    print("=" * 60)
    print("TESTING SYMBOL FILTERING")
    print("=" * 60)
    
    try:
        # Import your functions
        from main import filter_core_symbols, fix_live_candles_structure
        
        # Set up mock data
        global live_candles
        live_candles = create_mock_live_candles()
        
        # Test symbols
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT", "INVALID", "NOUSDT"]
        
        print(f"Input symbols: {test_symbols}")
        print(f"Mock live_candles keys: {list(live_candles.keys())}")
        
        # Test filtering
        filtered = await filter_core_symbols(test_symbols)
        
        print(f"Filtered symbols: {filtered}")
        print(f"Filter success: {len(filtered)} symbols passed")
        
        return filtered
        
    except Exception as e:
        print(f"FILTERING TEST FAILED: {e}")
        traceback.print_exc()
        return []

async def test_scoring():
    """Test the scoring pipeline"""
    print("\n" + "=" * 60)
    print("TESTING SCORING PIPELINE")
    print("=" * 60)
    
    try:
        from score import score_symbol
        
        # Test with mock candles
        symbol = "BTCUSDT"
        live_candles = create_mock_live_candles()
        
        candles_by_tf = {
            '1': live_candles[symbol]['1'],
            '5': live_candles[symbol]['5'], 
            '15': live_candles[symbol]['15']
        }
        
        print(f"Testing scoring for {symbol}")
        print(f"Candles available: {[f'{tf}:{len(candles)}' for tf, candles in candles_by_tf.items()]}")
        
        # Test scoring
        result = score_symbol(symbol, candles_by_tf, {})
        score, tf_scores, trade_type, indicator_scores, used_indicators = result
        
        print(f"Score result: {score}")
        print(f"TF scores: {tf_scores}")
        print(f"Trade type: {trade_type}")
        print(f"Indicators used: {len(used_indicators)}")
        
        return result
        
    except Exception as e:
        print(f"SCORING TEST FAILED: {e}")
        traceback.print_exc()
        return None

async def test_core_conditions():
    """Test core strategy conditions"""
    print("\n" + "=" * 60)  
    print("TESTING CORE CONDITIONS")
    print("=" * 60)
    
    try:
        from main import (
            calculate_core_score, 
            determine_core_direction,
            validate_core_conditions,
            determine_core_strategy_type
        )
        
        symbol = "BTCUSDT"
        live_candles = create_mock_live_candles()
        
        core_candles = {
            '1': live_candles[symbol]['1'],
            '5': live_candles[symbol]['5'],
            '15': live_candles[symbol]['15']
        }
        
        trend_context = {
            "trend": "neutral",
            "trend_strength": 0.5,
            "btc_trend": "neutral"
        }
        
        print(f"Testing core conditions for {symbol}")
        
        # Test core score calculation
        try:
            core_score = await calculate_core_score(symbol, core_candles, trend_context)
            print(f"Core score: {core_score}")
        except Exception as e:
            print(f"Core score calculation failed: {e}")
            core_score = 0
        
        # Test direction determination
        try:
            direction = determine_core_direction(core_candles, trend_context)
            print(f"Direction: {direction}")
        except Exception as e:
            print(f"Direction determination failed: {e}")
            direction = None
        
        # Test validation
        try:
            if direction:
                conditions_valid = await validate_core_conditions(symbol, core_candles, direction, trend_context)
                print(f"Conditions valid: {conditions_valid}")
            else:
                conditions_valid = False
                print("Cannot validate conditions without direction")
        except Exception as e:
            print(f"Conditions validation failed: {e}")
            conditions_valid = False
        
        # Test strategy type determination
        try:
            strategy_type = determine_core_strategy_type(core_score, 75, 0.5)
            print(f"Strategy type: {strategy_type}")
        except Exception as e:
            print(f"Strategy type determination failed: {e}")
            strategy_type = None
        
        return {
            "core_score": core_score,
            "direction": direction, 
            "conditions_valid": conditions_valid,
            "strategy_type": strategy_type
        }
        
    except ImportError as e:
        print(f"Cannot import core functions: {e}")
        return None
    except Exception as e:
        print(f"CORE CONDITIONS TEST FAILED: {e}")
        traceback.print_exc()
        return None

async def test_full_scanning_pipeline():
    """Test the complete scanning pipeline"""
    print("\n" + "=" * 60)
    print("TESTING FULL SCANNING PIPELINE")
    print("=" * 60)
    
    try:
        # Mock the global variables
        import main
        main.live_candles = create_mock_live_candles()
        main.active_trades = {}
        main.recent_exits = {}
        
        # Test symbols
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
        
        trend_context = {
            "trend": "neutral",
            "trend_strength": 0.5,
            "btc_trend": "neutral"
        }
        
        print(f"Testing full pipeline with {len(test_symbols)} symbols")
        
        # Run the core strategy scan
        await main.core_strategy_scan(test_symbols, trend_context)
        
        print("Full pipeline test completed")
        
    except Exception as e:
        print(f"FULL PIPELINE TEST FAILED: {e}")
        traceback.print_exc()

async def test_individual_symbol():
    """Test scanning a single symbol step by step"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL SYMBOL STEP BY STEP")
    print("=" * 60)
    
    symbol = "BTCUSDT"
    
    try:
        # Import required functions
        from main import (
            fix_live_candles_structure,
            calculate_core_score,
            determine_core_direction,
            validate_core_conditions,
            determine_core_strategy_type,
            check_strategy_position_limits,
            MIN_SCALP_SCORE
        )
        
        # Set up data
        live_candles = create_mock_live_candles()
        live_candles = fix_live_candles_structure(live_candles)
        
        trend_context = {
            "trend": "neutral", 
            "trend_strength": 0.5,
            "btc_trend": "neutral"
        }
        
        print(f"Step-by-step test for {symbol}")
        
        # Step 1: Check if symbol has data
        if symbol not in live_candles:
            print(f"❌ Step 1 FAILED: {symbol} not in live_candles")
            return
        print(f"✅ Step 1 PASSED: {symbol} found in live_candles")
        
        # Step 2: Get candles for all timeframes
        core_candles = {}
        for tf in ['1', '5', '15']:
            if tf in live_candles.get(symbol, {}):
                candles = live_candles[symbol][tf]
                if candles and len(candles) >= 30:
                    core_candles[tf] = candles
        
        if len(core_candles) < 3:
            print(f"❌ Step 2 FAILED: Only {len(core_candles)} timeframes available, need 3")
            print(f"Available TFs: {list(core_candles.keys())}")
            return
        print(f"✅ Step 2 PASSED: All 3 timeframes available")
        
        # Step 3: Calculate core score
        try:
            core_score = await calculate_core_score(symbol, core_candles, trend_context)
            print(f"✅ Step 3 PASSED: Core score = {core_score}")
            
            if core_score < MIN_SCALP_SCORE:
                print(f"❌ Step 3 THRESHOLD: Score {core_score} < minimum {MIN_SCALP_SCORE}")
                return
            print(f"✅ Step 3 THRESHOLD PASSED: Score above minimum")
            
        except Exception as e:
            print(f"❌ Step 3 FAILED: Core score calculation error: {e}")
            return
        
        # Step 4: Determine direction
        try:
            direction = determine_core_direction(core_candles, trend_context)
            if not direction:
                print(f"❌ Step 4 FAILED: No direction determined")
                return
            print(f"✅ Step 4 PASSED: Direction = {direction}")
        except Exception as e:
            print(f"❌ Step 4 FAILED: Direction determination error: {e}")
            return
        
        # Step 5: Validate conditions
        try:
            conditions_valid = await validate_core_conditions(symbol, core_candles, direction, trend_context)
            if not conditions_valid:
                print(f"❌ Step 5 FAILED: Core conditions not valid")
                return
            print(f"✅ Step 5 PASSED: Core conditions valid")
        except Exception as e:
            print(f"❌ Step 5 FAILED: Conditions validation error: {e}")
            return
        
        # Step 6: Determine strategy type
        try:
            strategy_type = determine_core_strategy_type(core_score, 75, 0.5)
            if not strategy_type:
                print(f"❌ Step 6 FAILED: No strategy type determined")
                return
            print(f"✅ Step 6 PASSED: Strategy type = {strategy_type}")
        except Exception as e:
            print(f"❌ Step 6 FAILED: Strategy type determination error: {e}")
            return
        
        # Step 7: Check position limits
        try:
            limits_ok = check_strategy_position_limits(strategy_type)
            if not limits_ok:
                print(f"❌ Step 7 FAILED: Position limits exceeded")
                return
            print(f"✅ Step 7 PASSED: Position limits OK")
        except Exception as e:
            print(f"❌ Step 7 FAILED: Position limits check error: {e}")
            return
        
        print(f"🎉 ALL STEPS PASSED for {symbol}!")
        print(f"Final result: Score={core_score}, Direction={direction}, Type={strategy_type}")
        
    except Exception as e:
        print(f"INDIVIDUAL SYMBOL TEST FAILED: {e}")
        traceback.print_exc()

async def run_all_tests():
    """Run all scanning tests"""
    print("🔬 CORE STRATEGY SCANNING TESTS")
    print("=" * 60)
    
    # Test 1: Filtering
    filtered_symbols = await test_filtering()
    
    # Test 2: Scoring
    scoring_result = await test_scoring()
    
    # Test 3: Core conditions
    conditions_result = await test_core_conditions()
    
    # Test 4: Individual symbol
    await test_individual_symbol()
    
    # Test 5: Full pipeline
    await test_full_scanning_pipeline()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Filtering: {'✅ PASS' if filtered_symbols else '❌ FAIL'}")
    print(f"Scoring: {'✅ PASS' if scoring_result else '❌ FAIL'}")
    print(f"Core conditions: {'✅ PASS' if conditions_result else '❌ FAIL'}")
    print("Check individual symbol and full pipeline results above")

if __name__ == "__main__":
    print("Starting scanning tests...")
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️ Tests cancelled by user")
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        traceback.print_exc()
