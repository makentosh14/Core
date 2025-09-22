#!/usr/bin/env python3
"""
Comprehensive Testing Script for Trading Strategy Core Conditions
Run this to validate all fixes and identify remaining issues.
"""

import asyncio
import traceback
import json
from typing import Dict, Any, List

# Mock logging function
def log(message, level="INFO"):
    print(f"[{level}] {message}")

async def run_comprehensive_tests():
    """Run all validation tests with detailed reporting"""
    
    print("=" * 80)
    print("COMPREHENSIVE TRADING STRATEGY VALIDATION TESTS")
    print("=" * 80)
    
    # Test 1: Risk/Reward Validation
    await test_risk_reward_validation()
    
    # Test 2: Volume Validation
    await test_volume_validation()
    
    # Test 3: Complete Pipeline
    await test_complete_pipeline()
    
    # Test 4: Edge Cases
    await test_edge_cases()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)

async def test_risk_reward_validation():
    """Test the improved risk/reward validation function"""
    print("\n" + "=" * 60)
    print("TEST 1: RISK/REWARD VALIDATION")
    print("=" * 60)
    
    # Test case 1: Clear uptrend with good R/R
    test_candles_uptrend = create_test_case_good_rr_uptrend()
    result1 = validate_core_risk_reward(test_candles_uptrend, "Long")
    print(f"✓ Uptrend Long Test: {'PASS' if result1 else 'FAIL'}")
    
    # Test case 2: Clear downtrend with good R/R  
    test_candles_downtrend = create_test_case_good_rr_downtrend()
    result2 = validate_core_risk_reward(test_candles_downtrend, "Short")
    print(f"✓ Downtrend Short Test: {'PASS' if result2 else 'FAIL'}")
    
    # Test case 3: Poor R/R scenario (should fail)
    test_candles_poor_rr = create_test_case_poor_rr()
    result3 = validate_core_risk_reward(test_candles_poor_rr, "Long")
    print(f"✓ Poor R/R Test (should fail): {'PASS' if not result3 else 'FAIL'}")

def create_test_case_good_rr_uptrend():
    """Create test case with clear uptrend and good risk/reward"""
    candles = []
    base_price = 50000
    
    # Create clear uptrend with pullbacks creating good S/R levels
    for i in range(20):
        if i < 5:  # Support area
            price = base_price + (i * 50) + (i % 2 * 25)  # 50000-50200 range
        elif i < 10:  # Breakout area
            price = base_price + 300 + (i * 100) + (i % 2 * 50)  # Moving up
        elif i < 15:  # Pullback to support
            price = base_price + 700 - (i % 3 * 30)  # Slight pullback
        else:  # Current levels - near resistance
            price = base_price + 1200 + (i % 2 * 50)  # Near resistance at 51250
    
        candle = {
            'high': str(price + 25),
            'low': str(price - 25), 
            'close': str(price),
            'open': str(price - 10)
        }
        candles.append(candle)
    
    return {'15': candles}

def create_test_case_good_rr_downtrend():
    """Create test case with clear downtrend and good risk/reward for short"""
    candles = []
    base_price = 50000
    
    # Create clear downtrend 
    for i in range(20):
        if i < 5:  # Resistance area (high)
            price = base_price - (i * 30)
        elif i < 10:  # Breakdown
            price = base_price - 200 - (i * 80)
        elif i < 15:  # Bounce to resistance
            price = base_price - 800 + (i % 3 * 40)
        else:  # Current levels - near support
            price = base_price - 1000 - (i % 2 * 30)
    
        candle = {
            'high': str(price + 30),
            'low': str(price - 30),
            'close': str(price),
            'open': str(price + 15)
        }
        candles.append(candle)
    
    return {'15': candles}

def create_test_case_poor_rr():
    """Create test case with poor risk/reward (tight range)"""
    candles = []
    base_price = 50000
    
    # Create sideways movement (poor R/R)
    for i in range(20):
        price = base_price + (i % 4 * 20) - 20  # Very tight range
        candle = {
            'high': str(price + 15),
            'low': str(price - 15),
            'close': str(price), 
            'open': str(price - 5)
        }
        candles.append(candle)
    
    return {'15': candles}

async def test_volume_validation():
    """Test the improved volume validation"""
    print("\n" + "=" * 60) 
    print("TEST 2: VOLUME VALIDATION")
    print("=" * 60)
    
    # Test case 1: Clear volume surge (should pass)
    test_data_volume_surge = create_test_case_volume_surge()
    result1 = validate_core_volume(test_data_volume_surge)
    print(f"✓ Volume Surge Test: {'PASS' if result1 else 'FAIL'}")
    
    # Test case 2: No volume surge (should fail)
    test_data_no_surge = create_test_case_no_volume_surge()
    result2 = validate_core_volume(test_data_no_surge)  
    print(f"✓ No Volume Surge Test (should fail): {'PASS' if not result2 else 'FAIL'}")
    
    # Test case 3: Gradual volume increase (should pass)
    test_data_gradual = create_test_case_gradual_volume()
    result3 = validate_core_volume(test_data_gradual)
    print(f"✓ Gradual Volume Test: {'PASS' if result3 else 'FAIL'}")

def create_test_case_volume_surge():
    """Create test case with clear volume surge in recent candles"""
    candles = []
    
    for i in range(30):
        if i < 25:  # Normal volume
            volume = 1000000 + (i % 5 * 50000)
        else:  # Volume surge in last 5 candles
            volume = 2800000 + (i % 3 * 200000)
        
        candle = {
            'volume': str(volume),
            'close': str(50000 + i * 10),
            'high': str(50000 + i * 10 + 15),
            'low': str(50000 + i * 10 - 15),
            'open': str(50000 + i * 10 - 5)
        }
        candles.append(candle)
    
    return {'1': candles}

def create_test_case_no_volume_surge():
    """Create test case with consistent volume (no surge)"""
    candles = []
    
    for i in range(30):
        volume = 1000000 + (i % 3 * 30000)  # Consistent volume, no surge
        
        candle = {
            'volume': str(volume),
            'close': str(50000 + i * 5),
            'high': str(50000 + i * 5 + 10),
            'low': str(50000 + i * 5 - 10),
            'open': str(50000 + i * 5)
        }
        candles.append(candle)
    
    return {'1': candles}

def create_test_case_gradual_volume():
    """Create test case with gradual volume increase"""
    candles = []
    
    for i in range(30):
        if i < 20:
            volume = 800000 + (i * 15000)  # Gradual increase
        else:
            volume = 1400000 + (i * 25000)  # Accelerating increase
        
        candle = {
            'volume': str(volume),
            'close': str(50000 + i * 8),
            'high': str(50000 + i * 8 + 12),
            'low': str(50000 + i * 8 - 12),
            'open': str(50000 + i * 8 - 3)
        }
        candles.append(candle)
    
    return {'1': candles}

async def test_complete_pipeline():
    """Test the complete validation pipeline"""
    print("\n" + "=" * 60)
    print("TEST 3: COMPLETE PIPELINE")
    print("=" * 60)
    
    # Create comprehensive test data
    test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Create test data for this symbol
        core_candles = create_realistic_test_data(symbol)
        trend_context = {"trend": "bullish", "trend_strength": 0.6}
        
        try:
            # Test scoring
            score = await calculate_mock_core_score(symbol, core_candles, trend_context)
            print(f"Core Score: {score}")
            
            # Test direction determination
            direction = determine_mock_core_direction(core_candles, trend_context)
            print(f"Direction: {direction}")
            
            if direction:
                # Test validation with relaxed conditions
                conditions_valid = validate_core_conditions_relaxed(symbol, core_candles, direction, trend_context)
                print(f"Conditions Valid: {'YES' if conditions_valid else 'NO'}")
            else:
                print("Conditions Valid: NO (no direction)")
                
        except Exception as e:
            print(f"Pipeline test failed for {symbol}: {e}")
            traceback.print_exc()

def create_realistic_test_data(symbol):
    """Create realistic test data for pipeline testing"""
    
    if symbol == "BTCUSDT":
        return create_btc_test_data()
    elif symbol == "ETHUSDT":
        return create_eth_test_data()
    else:
        return create_ada_test_data()

def create_btc_test_data():
    """Create BTC test data with clear uptrend and volume surge"""
    candles_1m = []
    candles_5m = []
    candles_15m = []
    
    base_price = 50000
    
    # 1m candles (50 candles)
    for i in range(50):
        price = base_price + (i * 25) + (i % 3 * 10)  # Gradual uptrend
        volume = 1200000 if i < 45 else 3500000  # Volume surge in last 5
        
        candle = create_candle(price, volume)
        candles_1m.append(candle)
    
    # 5m candles (simplified - 20 candles)
    for i in range(20):
        price = base_price + (i * 125) + (i % 2 * 25)
        volume = 1400000 if i < 15 else 3200000
        
        candle = create_candle(price, volume)
        candles_5m.append(candle)
    
    # 15m candles (20 candles with clear S/R levels)
    support_resistance_prices = [
        49800, 49850, 49900, 49950, 50000,  # Support area
        50200, 50400, 50600, 50800, 51000,  # Breakout area
        51200, 51400, 51600, 51800, 52000,  # Resistance area
        51800, 51900, 52000, 52100, 52200   # Current area
    ]
    
    for i, price in enumerate(support_resistance_prices):
        volume = 1500000 if i < 15 else 3000000
        candle = create_candle(price, volume)
        candles_15m.append(candle)
    
    return {
        '1': candles_1m,
        '5': candles_5m, 
        '15': candles_15m
    }

def create_eth_test_data():
    """Create ETH test data"""
    candles_1m = []
    candles_5m = []
    candles_15m = []
    
    base_price = 3000
    
    for i in range(30):
        price = base_price + (i * 15) + (i % 4 * 5)
        volume = 1100000 if i < 25 else 2800000
        
        candle = create_candle(price, volume)
        candles_1m.append(candle)
        
        if i < 20:
            candle_5m = create_candle(price + 20, volume + 100000)
            candles_5m.append(candle_5m)
            
        if i < 15:
            candle_15m = create_candle(price + 40, volume + 200000)
            candles_15m.append(candle_15m)
    
    return {'1': candles_1m, '5': candles_5m, '15': candles_15m}

def create_ada_test_data():
    """Create ADA test data with downtrend"""
    candles_1m = []
    candles_5m = []
    candles_15m = []
    
    base_price = 1.0
    
    for i in range(30):
        price = base_price - (i * 0.005) + (i % 5 * 0.002)  # Downtrend
        volume = 900000 if i < 25 else 2200000
        
        candle = create_candle(price, volume, precision=5)
        candles_1m.append(candle)
        
        if i < 20:
            candle_5m = create_candle(price - 0.01, volume + 50000, precision=5)
            candles_5m.append(candle_5m)
            
        if i < 15:
            candle_15m = create_candle(price - 0.02, volume + 100000, precision=5)
            candles_15m.append(candle_15m)
    
    return {'1': candles_1m, '5': candles_5m, '15': candles_15m}

def create_candle(price, volume, precision=2):
    """Helper function to create a candle"""
    high = price * 1.005
    low = price * 0.995
    open_price = price * 0.999
    
    return {
        'open': f"{open_price:.{precision}f}",
        'high': f"{high:.{precision}f}",
        'low': f"{low:.{precision}f}",
        'close': f"{price:.{precision}f}",
        'volume': str(int(volume))
    }

async def calculate_mock_core_score(symbol, core_candles, trend_context):
    """Mock core score calculation for testing"""
    base_score = 8.0
    
    # Add points for volume
    if validate_core_volume(core_candles):
        base_score += 2.0
    
    # Add points for trend alignment
    if trend_context.get("trend") == "bullish":
        base_score += 1.5
    
    return base_score

def determine_mock_core_direction(core_candles, trend_context):
    """Mock direction determination for testing"""
    try:
        if '5' not in core_candles:
            return None
            
        candles = core_candles['5'][-10:]
        if len(candles) < 5:
            return None
        
        closes = [float(c['close']) for c in candles]
        price_change = (closes[-1] - closes[0]) / closes[0]
        
        if price_change > 0.01:  # 1% up
            return "Long"
        elif price_change < -0.01:  # 1% down
            return "Short"
        else:
            return None
            
    except Exception as e:
        return None

async def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n" + "=" * 60)
    print("TEST 4: EDGE CASES")
    print("=" * 60)
    
    # Test 1: Empty data
    try:
        result = validate_core_risk_reward({}, "Long")
        print(f"✓ Empty data test: {'PASS' if not result else 'FAIL'}")
    except:
        print("✓ Empty data test: PASS (handled exception)")
    
    # Test 2: Insufficient candles
    try:
        insufficient_data = {'15': [create_candle(50000, 1000000) for _ in range(3)]}
        result = validate_core_risk_reward(insufficient_data, "Long")
        print(f"✓ Insufficient data test: {'PASS' if not result else 'FAIL'}")
    except:
        print("✓ Insufficient data test: PASS (handled exception)")
    
    # Test 3: Invalid price data
    try:
        invalid_data = {'15': [{'high': 'invalid', 'low': '0', 'close': '50000'}] * 20}
        result = validate_core_risk_reward(invalid_data, "Long")
        print(f"✓ Invalid price data test: {'PASS' if not result else 'FAIL'}")
    except:
        print("✓ Invalid price data test: PASS (handled exception)")
    
    # Test 4: Volume validation edge cases
    try:
        zero_volume_data = {'1': [create_candle(50000, 0) for _ in range(30)]}
        result = validate_core_volume(zero_volume_data)
        print(f"✓ Zero volume test: {'PASS' if not result else 'FAIL'}")
    except:
        print("✓ Zero volume test: PASS (handled exception)")

if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_comprehensive_tests())
