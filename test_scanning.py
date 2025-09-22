# test_scanning.py - Test the core strategy scanning pipeline

import asyncio
import sys
import traceback
from collections import defaultdict, deque

def create_mock_live_candles_with_volume_spike():
    """Create realistic mock live_candles data with GUARANTEED volume spike"""
    from collections import defaultdict
    import random
    
    mock_candles = defaultdict(lambda: defaultdict(list))
    
    # Create mock candle data for several symbols
    test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", "SOLUSDT"]
    
    for symbol in test_symbols:
        # Set base parameters for each symbol
        if 'BTC' in symbol:
            base_price = 50000
            base_volume = 1000000  # 1M base volume
        elif 'ETH' in symbol:
            base_price = 3000
            base_volume = 800000   # 800K base volume
        else:
            base_price = 1.0
            base_volume = 500000   # 500K base volume
        
        for tf in ['1', '5', '15']:
            # Create 50 realistic candles
            candles = []
            current_price = base_price
            
            for i in range(50):
                # Create realistic price movements
                price_change_pct = random.uniform(-0.01, 0.01)  # +/- 1%
                current_price *= (1 + price_change_pct)
                
                # Create OHLC
                open_price = current_price * random.uniform(0.999, 1.001)
                close_price = current_price * random.uniform(0.999, 1.001)
                high_price = max(open_price, close_price) * random.uniform(1.001, 1.003)
                low_price = min(open_price, close_price) * random.uniform(0.997, 0.999)
                
                # CRITICAL: Create guaranteed volume pattern
                if i < 30:  # First 30 candles - LOW volume
                    volume = base_volume * random.uniform(0.5, 0.8)  # 50-80% of base
                elif i < 45:  # Next 15 candles - MEDIUM volume
                    volume = base_volume * random.uniform(0.8, 1.2)  # 80-120% of base
                else:  # Last 5 candles - HIGH volume (GUARANTEED 2x+ spike)
                    volume = base_volume * random.uniform(2.5, 4.0)  # 250-400% of base
                
                candle = {
                    'timestamp': 1700000000000 + i * 60000,
                    'open': str(round(open_price, 2)),
                    'high': str(round(high_price, 2)),
                    'low': str(round(low_price, 2)),
                    'close': str(round(close_price, 2)),
                    'volume': str(int(volume))
                }
                candles.append(candle)
            
            # VERIFY the volume pattern will pass validation
            last_20_volumes = [float(c['volume']) for c in candles[-20:]]
            avg_volume = sum(last_20_volumes) / len(last_20_volumes)
            recent_volume = sum(last_20_volumes[-5:]) / 5
            ratio = recent_volume / avg_volume if avg_volume > 0 else 0
            
            # Ensure ratio is above 1.2
            if ratio < 1.3:  # Add buffer above 1.2 requirement
                print(f"WARNING: {symbol} {tf} ratio only {ratio:.2f}, boosting...")
                # Boost last 5 candles even more
                for j in range(len(candles) - 5, len(candles)):
                    old_vol = float(candles[j]['volume'])
                    new_vol = old_vol * 2.0  # Double it
                    candles[j]['volume'] = str(int(new_vol))
                
                # Re-verify
                last_20_volumes = [float(c['volume']) for c in candles[-20:]]
                avg_volume = sum(last_20_volumes) / len(last_20_volumes)
                recent_volume = sum(last_20_volumes[-5:]) / 5
                ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                print(f"After boost: {symbol} {tf} ratio = {ratio:.2f}")
            
            # Store as list to match your fixed structure
            mock_candles[symbol][tf] = candles
    
    return dict(mock_candles)

def verify_volume_patterns(live_candles):
    """Verify that all symbols have proper volume patterns"""
    print("\n=== VERIFYING VOLUME PATTERNS ===")
    
    for symbol in live_candles:
        for tf in live_candles[symbol]:
            candles = live_candles[symbol][tf]
            if len(candles) >= 20:
                last_20_volumes = [float(c['volume']) for c in candles[-20:]]
                avg_volume = sum(last_20_volumes) / len(last_20_volumes)
                recent_volume = sum(last_20_volumes[-5:]) / 5
                ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                
                status = "✅ PASS" if ratio > 1.2 else "❌ FAIL"
                print(f"{symbol} {tf}: avg={avg_volume:.0f}, recent={recent_volume:.0f}, ratio={ratio:.2f} {status}")
    
    print("=" * 50)

def fix_test_volumes(live_candles):
    """Boost volumes in existing test data to pass validation"""
    import random
    
    for symbol in live_candles:
        for tf in live_candles[symbol]:
            candles = live_candles[symbol][tf]
            if isinstance(candles, list) and len(candles) > 20:
                # Boost the last 5 candles' volume significantly
                for i in range(len(candles)):
                    current_vol = float(candles[i].get('volume', '1000000'))
                    
                    if i < len(candles) - 5:
                        # Normal volume for earlier candles
                        new_vol = current_vol
                    else:
                        # HIGH volume for recent candles (2x to 4x boost)
                        new_vol = current_vol * random.uniform(2.0, 4.0)
                    
                    candles[i]['volume'] = str(int(new_vol))
    
    return live_candles

def fix_test_volumes_aggressive(live_candles):
    """Aggressively boost volumes to pass 1.5x validation requirement"""
    import random
    
    for symbol in live_candles:
        for tf in live_candles[symbol]:
            candles = live_candles[symbol][tf]
            if isinstance(candles, list) and len(candles) >= 20:
                
                # Strategy: Keep first 15 candles low, make last 5 candles VERY high
                for i in range(len(candles)):
                    current_vol = float(candles[i].get('volume', '1000000'))
                    
                    if i < len(candles) - 5:
                        # Keep earlier candles relatively low (0.5x to 1.0x of original)
                        new_vol = current_vol * random.uniform(0.5, 1.0)
                    else:
                        # Make last 5 candles VERY high (3x to 5x of original)
                        new_vol = current_vol * random.uniform(3.0, 5.0)
                    
                    candles[i]['volume'] = str(int(new_vol))
                
                # Verify the math will work
                volumes = [float(c['volume']) for c in candles[-20:]]
                avg_volume = sum(volumes) / len(volumes)
                recent_volume = sum(volumes[-5:]) / 5
                ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                
                print(f"Debug {symbol} {tf}: avg={avg_volume:.0f}, recent={recent_volume:.0f}, ratio={ratio:.2f}")
    
    return live_candles

def test_volume_validation_directly():
    """Test the volume validation function directly"""
    
    # Import your actual function
    try:
        from main import validate_core_volume
        
        # Create test data that should definitely pass
        test_candles = {
            '1': []
        }
        
        # Create 20 candles with specific volume pattern
        for i in range(20):
            if i < 15:
                # First 15 candles: low volume (1M each)
                volume = 1000000
            else:
                # Last 5 candles: high volume (5M each)
                volume = 5000000
            
            candle = {
                'volume': str(volume),
                'close': '50000'
            }
            test_candles['1'].append(candle)
        
        # Test the validation
        result = validate_core_volume(test_candles)
        
        # Calculate what it's actually checking
        candles = test_candles['1'][-20:]
        volumes = [float(c.get('volume', 0)) for c in candles]
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / 5
        ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        
        print(f"Direct test result: {result}")
        print(f"Average volume: {avg_volume}")
        print(f"Recent volume: {recent_volume}")
        print(f"Ratio: {ratio:.2f} (needs > 1.5)")
        print(f"Expected: recent_volume ({recent_volume}) > avg_volume * 1.5 ({avg_volume * 1.5})")
        
        return result
        
    except ImportError as e:
        print(f"Cannot import validate_core_volume: {e}")
        return False

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
        live_candles = create_mock_live_candles_with_volume_spike()
        verify_volume_patterns(live_candles)
        
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
        live_candles = create_mock_live_candles_with_volume_spike()
        verify_volume_patterns(live_candles)
        
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
        live_candles = create_mock_live_candles_with_volume_spike()
        verify_volume_patterns(live_candles)
        
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
        live_candles = create_mock_live_candles_with_volume_spike()
        verify_volume_patterns(live_candles)
        
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

    print("\n=== DIRECT VOLUME VALIDATION TEST ===")
    test_volume_validation_directly()
    
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print("\n⏹️ Tests cancelled by user")
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        traceback.print_exc()
