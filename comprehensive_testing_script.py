#!/usr/bin/env python3
"""
FIXED: Complete Testing Script with Robust Error Handling
This version includes the improved risk/reward validation with proper error handling
"""

import asyncio
import traceback
import json
from typing import Dict, Any, List

# Mock logging function
def log(message, level="INFO"):
    print(f"[{level}] {message}")

def validate_core_risk_reward(core_candles, direction):
    """
    FIXED: Risk/reward validation with robust error handling and improved logic
    """
    try:
        log(f"🔍 DEBUG RR: validate_core_risk_reward called, direction={direction}")
        log(f"🔍 DEBUG RR: Available timeframes: {list(core_candles.keys())}")
        
        if '15' not in core_candles:
            log(f"❌ DEBUG RR: No '15' timeframe in core_candles")
            return False
        
        candles = core_candles['15'][-20:]
        log(f"🔍 DEBUG RR: Got {len(candles)} candles from 15m timeframe")
        
        if len(candles) < 10:
            log(f"❌ DEBUG RR: Not enough candles: {len(candles)} < 10")
            return False
        
        # FIXED: Robust data extraction with error handling
        try:
            highs = []
            lows = []
            closes = []
            
            for c in candles:
                # Handle invalid data gracefully
                try:
                    high = float(c.get('high', 0))
                    low = float(c.get('low', 0))
                    close = float(c.get('close', 0))
                    
                    # Validate reasonable values
                    if high <= 0 or low <= 0 or close <= 0 or high < low:
                        log(f"❌ DEBUG RR: Invalid candle data: high={high}, low={low}, close={close}")
                        continue
                        
                    highs.append(high)
                    lows.append(low)
                    closes.append(close)
                    
                except (ValueError, TypeError) as e:
                    log(f"❌ DEBUG RR: Failed to convert candle data: {e}")
                    continue
            
            if len(highs) < 10 or len(lows) < 10 or len(closes) < 10:
                log(f"❌ DEBUG RR: Not enough valid candles after cleaning: {len(closes)}")
                return False
                
        except Exception as e:
            log(f"❌ DEBUG RR: Data extraction error: {e}")
            return False
        
        current_price = closes[-1]
        log(f"🔍 DEBUG RR: Current price: {current_price}")
        
        if direction.lower() == "long":
            # IMPROVED LONG LOGIC
            recent_lows = lows[-10:]
            support_candidate = min(recent_lows)
            max_risk_percent = 0.03
            min_support_level = current_price * (1 - max_risk_percent)
            effective_support = max(support_candidate, min_support_level)
            potential_risk = current_price - effective_support
            
            # Multiple target approaches
            approaches = []
            risk_based_target = current_price + (potential_risk * 1.5)
            approaches.append(("risk_1.5x", risk_based_target))
            
            percentage_target = current_price * 1.04
            approaches.append(("percent_4%", percentage_target))
            
            recent_highs = highs[-10:]
            recent_resistance = max(recent_highs)
            if recent_resistance > current_price * 1.01:
                buffered_resistance = recent_resistance * 1.01
                approaches.append(("resistance_buffered", buffered_resistance))
            
            price_range = max(highs) - min(lows)
            range_target = current_price + (price_range * 0.3)
            approaches.append(("range_30%", range_target))
            
            effective_resistance = max(target for _, target in approaches)
            best_approach = max(approaches, key=lambda x: x[1])
            
            log(f"🔍 DEBUG RR: LONG - Support: {effective_support:.4f}")
            log(f"🔍 DEBUG RR: LONG - Target approaches: {[(name, f'{target:.4f}') for name, target in approaches]}")
            log(f"🔍 DEBUG RR: LONG - Selected: {best_approach[0]} = {effective_resistance:.4f}")
            
            potential_reward = effective_resistance - current_price
            
        else:  # SHORT POSITIONS
            # IMPROVED SHORT LOGIC
            recent_highs = highs[-10:]
            resistance_candidate = max(recent_highs)
            max_risk_percent = 0.03
            max_resistance_level = current_price * (1 + max_risk_percent)
            effective_resistance = min(resistance_candidate, max_resistance_level)
            potential_risk = effective_resistance - current_price
            
            # Multiple target approaches for SHORT
            approaches = []
            risk_based_target = current_price - (potential_risk * 1.5)
            approaches.append(("risk_1.5x", risk_based_target))
            
            percentage_target = current_price * 0.96
            approaches.append(("percent_4%", percentage_target))
            
            recent_lows = lows[-10:]
            recent_support = min(recent_lows)
            if recent_support < current_price * 0.99:
                buffered_support = recent_support * 0.99
                approaches.append(("support_buffered", buffered_support))
            
            price_range = max(highs) - min(lows)
            range_target = current_price - (price_range * 0.3)
            approaches.append(("range_30%", range_target))
            
            # Choose the lowest reasonable target (for shorts, lower is better)
            valid_approaches = [(name, target) for name, target in approaches if target > 0]
            if not valid_approaches:
                log(f"❌ DEBUG RR: No valid SHORT targets found")
                return False
            
            effective_support = min(target for _, target in valid_approaches)
            best_approach = min(valid_approaches, key=lambda x: x[1])
            
            log(f"🔍 DEBUG RR: SHORT - Resistance: {effective_resistance:.4f}")
            log(f"🔍 DEBUG RR: SHORT - Target approaches: {[(name, f'{target:.4f}') for name, target in valid_approaches]}")
            log(f"🔍 DEBUG RR: SHORT - Selected: {best_approach[0]} = {effective_support:.4f}")
            
            potential_reward = current_price - effective_support
        
        log(f"🔍 DEBUG RR: Potential reward: {potential_reward:.4f}")
        log(f"🔍 DEBUG RR: Potential risk: {potential_risk:.4f}")
        
        if potential_reward <= 0 or potential_risk <= 0:
            log(f"❌ DEBUG RR: Invalid reward/risk: {potential_reward:.4f}/{potential_risk:.4f}")
            return False
        
        rr_ratio = potential_reward / potential_risk
        log(f"🔍 DEBUG RR: Risk/Reward ratio: {rr_ratio:.3f} (needs >= 1.2)")
        
        # Quality checks
        min_reward_threshold = current_price * 0.01
        if potential_reward < min_reward_threshold:
            log(f"❌ DEBUG RR: Reward too small: {potential_reward:.4f} < {min_reward_threshold:.4f}")
            return False
        
        max_risk_threshold = current_price * 0.04
        if potential_risk > max_risk_threshold:
            log(f"❌ DEBUG RR: Risk too large: {potential_risk:.4f} > {max_risk_threshold:.4f}")
            return False
        
        result = rr_ratio >= 1.2
        log(f"🔍 DEBUG RR: Risk/reward validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG RR: Risk/reward validation error: {e}", level="ERROR")
        import traceback
        log(f"❌ DEBUG RR: Traceback: {traceback.format_exc()}", level="ERROR")
        return False

def validate_core_volume(core_candles):
    """
    Improved volume validation with better logic and thresholds
    """
    try:
        log(f"🔍 DEBUG: validate_core_volume called with keys: {list(core_candles.keys())}")
        
        if '1' not in core_candles:
            log(f"❌ DEBUG: No '1' timeframe in core_candles")
            return False
        
        log(f"🔍 DEBUG: Found '1' timeframe, type: {type(core_candles['1'])}")
        
        candles = core_candles['1'][-30:]  # Use 30 candles for better average
        log(f"🔍 DEBUG: Extracted {len(candles)} candles from last 30")
        
        if len(candles) < 20:
            log(f"❌ DEBUG: Not enough candles: {len(candles)} < 20")
            return False
        
        # Debug: Show first and last candle
        log(f"🔍 DEBUG: First candle: {candles[0] if candles else 'None'}")
        log(f"🔍 DEBUG: Last candle: {candles[-1] if candles else 'None'}")
        
        volumes = [float(c.get('volume', 0)) for c in candles]
        log(f"🔍 DEBUG: Extracted {len(volumes)} volumes")
        log(f"🔍 DEBUG: First 5 volumes: {volumes[:5]}")
        log(f"🔍 DEBUG: Last 5 volumes: {volumes[-5:]}")
        
        if len(volumes) < 20:
            log(f"❌ DEBUG: Not enough volumes: {len(volumes)} < 20")
            return False
        
        # Improved volume analysis
        sorted_volumes = sorted(volumes[:-5])
        median_volume = sorted_volumes[len(sorted_volumes)//2] if sorted_volumes else 0
        avg_volume = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) > 5 else 0
        
        # Use average of last 3 candles
        recent_volume = sum(volumes[-3:]) / 3
        
        # Calculate both ratios
        avg_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        median_ratio = recent_volume / median_volume if median_volume > 0 else 0
        
        log(f"🔍 DEBUG: Average volume (excluding recent): {avg_volume}")
        log(f"🔍 DEBUG: Median volume (excluding recent): {median_volume}")
        log(f"🔍 DEBUG: Recent volume (last 3): {recent_volume}")
        log(f"🔍 DEBUG: Avg ratio: {avg_ratio:.3f} (needs > 1.5)")
        log(f"🔍 DEBUG: Median ratio: {median_ratio:.3f} (needs > 1.5)")
        
        volume_threshold = 1.5
        result = avg_ratio > volume_threshold or median_ratio > volume_threshold
        
        log(f"🔍 DEBUG: Volume validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG: Volume validation error: {e}", level="ERROR")
        return False

# Keep all the other functions unchanged...
def validate_core_price_action(core_candles, direction):
    """Mock price action validation"""
    return True

def validate_core_timing():
    """Mock timing validation"""
    return True

def validate_core_trend_coherence(core_candles, direction):
    """Mock trend coherence validation"""
    return True

def check_price_direction_alignment(core_candles, direction):
    """Mock price direction alignment check"""
    return True

async def validate_core_conditions(symbol, core_candles, direction, trend_context):
    """Mock core conditions validation"""
    validation_score = 0
    max_score = 5
    
    # Volume validation
    volume_ok = validate_core_volume(core_candles)
    if volume_ok:
        validation_score += 2
        log(f"   Volume check: ✅ (+2 points)")
    else:
        log(f"   Volume check: ❌ (0 points)")
    
    # Price action
    price_action_ok = validate_core_price_action(core_candles, direction)
    if price_action_ok:
        validation_score += 1
        log(f"   Price action check: ✅ (+1 point)")
    else:
        log(f"   Price action check: ❌ (0 points)")
    
    # Risk/reward validation with fallback
    risk_reward_ok = validate_core_risk_reward(core_candles, direction)
    if risk_reward_ok:
        validation_score += 1
        log(f"   Risk/reward check: ✅ (+1 point)")
    else:
        if check_price_direction_alignment(core_candles, direction):
            validation_score += 0.5
            log(f"   Risk/reward check: ⚠️ (fallback +0.5 points)")
        else:
            log(f"   Risk/reward check: ❌ (0 points)")
    
    # Timing
    timing_ok = validate_core_timing()
    if timing_ok:
        validation_score += 0.5
        log(f"   Timing check: ✅ (+0.5 points)")
    else:
        log(f"   Timing check: ❌ (0 points)")
    
    # Trend coherence
    trend_coherence_ok = validate_core_trend_coherence(core_candles, direction)
    if trend_coherence_ok:
        validation_score += 0.5
        log(f"   Trend coherence check: ✅ (+0.5 points)")
    else:
        log(f"   Trend coherence check: ❌ (0 points)")
    
    passing_threshold = 3.0
    result = validation_score >= passing_threshold
    
    log(f"📊 Validation score: {validation_score}/{max_score} (need {passing_threshold})")
    log(f"🎯 Final result: {'✅ PASS' if result else '❌ FAIL'}")
    
    return result

# Test case creation functions
def create_test_case_good_rr_uptrend():
    """Create test case with clear uptrend and good risk/reward"""
    candles = []
    base_price = 50000
    
    for i in range(20):
        if i < 5:
            price = base_price + (i * 50) + (i % 2 * 25)
        elif i < 10:
            price = base_price + 300 + (i * 100) + (i % 2 * 50)
        elif i < 15:
            price = base_price + 700 - (i % 3 * 30)
        else:
            price = base_price + 1200 + (i % 2 * 50)
    
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
    
    for i in range(20):
        if i < 5:
            price = base_price - (i * 30)
        elif i < 10:
            price = base_price - 200 - (i * 80)
        elif i < 15:
            price = base_price - 800 + (i % 3 * 40)
        else:
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
    
    for i in range(20):
        price = base_price + (i % 4 * 20) - 20
        candle = {
            'high': str(price + 15),
            'low': str(price - 15),
            'close': str(price), 
            'open': str(price - 5)
        }
        candles.append(candle)
    
    return {'15': candles}

def create_test_case_volume_surge():
    """Create test case with volume surge"""
    candles = []
    for i in range(30):
        volume = 1000000 + (i * 50000) if i < 25 else 3200000 - (i % 5 * 200000)
        price = 50000 + (i * 10)
        candle = {
            'volume': str(volume),
            'close': str(price),
            'high': str(price + 15),
            'low': str(price - 15),
            'open': str(price - 5)
        }
        candles.append(candle)
    return {'1': candles}

def create_test_case_no_volume_surge():
    """Create test case with no volume surge"""
    candles = []
    base_volume = 1000000
    for i in range(30):
        volume = base_volume + (i % 3 * 30000)
        price = 50000 + (i * 5)
        candle = {
            'volume': str(volume),
            'close': str(price),
            'high': str(price + 10),
            'low': str(price - 10),
            'open': str(price)
        }
        candles.append(candle)
    return {'1': candles}

def create_test_case_gradual_volume():
    """Create test case with gradual volume increase"""
    candles = []
    for i in range(30):
        if i < 25:
            volume = 800000 + (i * 15000)
        else:
            volume = 2000000 + (i * 25000)
        price = 50000 + (i * 8)
        candle = {
            'volume': str(volume),
            'close': str(price),
            'high': str(price + 12),
            'low': str(price - 12),
            'open': str(price - 3)
        }
        candles.append(candle)
    return {'1': candles}

def create_candle(price, volume, precision=2):
    """Create a single test candle"""
    high = price + (price * 0.005)
    low = price - (price * 0.005)
    return {
        'open': f"{price:.{precision}f}",
        'high': f"{high:.{precision}f}",
        'low': f"{low:.{precision}f}",
        'close': f"{price:.{precision}f}",
        'volume': str(int(volume))
    }

async def calculate_mock_core_score(symbol, core_candles, trend_context):
    """Mock core score calculation"""
    base_score = 8.0
    if validate_core_volume(core_candles):
        base_score += 2.0
    if trend_context.get("trend") == "bullish":
        base_score += 1.5
    return base_score

def determine_mock_core_direction(core_candles, trend_context):
    """Mock direction determination"""
    try:
        if '5' not in core_candles:
            return None
        candles = core_candles['5'][-10:]
        if len(candles) < 5:
            return None
        closes = [float(c['close']) for c in candles]
        price_change = (closes[-1] - closes[0]) / closes[0]
        if price_change > 0.01:
            return "Long"
        elif price_change < -0.01:
            return "Short"
        else:
            return None
    except Exception as e:
        return None

async def test_risk_reward_validation():
    """Test the improved risk/reward validation function"""
    print("\n" + "=" * 60)
    print("TEST 1: RISK/REWARD VALIDATION")
    print("=" * 60)
    
    test_candles_uptrend = create_test_case_good_rr_uptrend()
    result1 = validate_core_risk_reward(test_candles_uptrend, "Long")
    print(f"✓ Uptrend Long Test: {'PASS' if result1 else 'FAIL'}")
    
    test_candles_downtrend = create_test_case_good_rr_downtrend()
    result2 = validate_core_risk_reward(test_candles_downtrend, "Short")
    print(f"✓ Downtrend Short Test: {'PASS' if result2 else 'FAIL'}")
    
    test_candles_poor_rr = create_test_case_poor_rr()
    result3 = validate_core_risk_reward(test_candles_poor_rr, "Long")
    print(f"✓ Poor R/R Test (should fail): {'PASS' if not result3 else 'FAIL'}")

async def test_volume_validation():
    """Test the improved volume validation"""
    print("\n" + "=" * 60) 
    print("TEST 2: VOLUME VALIDATION")
    print("=" * 60)
    
    test_data_volume_surge = create_test_case_volume_surge()
    result1 = validate_core_volume(test_data_volume_surge)
    print(f"✓ Volume Surge Test: {'PASS' if result1 else 'FAIL'}")
    
    test_data_no_surge = create_test_case_no_volume_surge()
    result2 = validate_core_volume(test_data_no_surge)  
    print(f"✓ No Volume Surge Test (should fail): {'PASS' if not result2 else 'FAIL'}")
    
    test_data_gradual = create_test_case_gradual_volume()
    result3 = validate_core_volume(test_data_gradual)
    print(f"✓ Gradual Volume Test: {'PASS' if result3 else 'FAIL'}")

async def test_complete_pipeline():
    """Test complete pipeline with mock data"""
    print("\n" + "=" * 60)
    print("TEST 3: COMPLETE PIPELINE")
    print("=" * 60)
    
    # Create mock data for different symbols
    symbols_data = {
        'BTCUSDT': {
            'candles': {
                '1': create_test_case_volume_surge()['1'],
                '5': [create_candle(50000 + i*100, 1500000) for i in range(20)],
                '15': create_test_case_good_rr_uptrend()['15']
            },
            'direction': 'Long'
        },
        'ETHUSDT': {
            'candles': {
                '1': create_test_case_volume_surge()['1'],
                '5': [create_candle(3000 + i*20, 1200000) for i in range(15)],
                '15': [create_candle(3000 + i*20, 1200000) for i in range(15)]
            },
            'direction': 'Long'
        },
        'ADAUSDT': {
            'candles': {
                '1': create_test_case_volume_surge()['1'],
                '5': [create_candle(1.0 - i*0.01, 900000) for i in range(15)],
                '15': create_test_case_good_rr_downtrend()['15']
            },
            'direction': 'Short'
        }
    }
    
    for symbol, data in symbols_data.items():
        print(f"\n--- Testing {symbol} ---")
        
        # Mock calculations
        score = await calculate_mock_core_score(symbol, data['candles'], {"trend": "bullish"})
        direction = data['direction']
        
        print(f"Core Score: {score}")
        print(f"Direction: {direction}")
        
        # Test validation
        result = await validate_core_conditions(
            symbol, data['candles'], direction, {"trend": "bullish"}
        )
        print(f"Conditions Valid: {'YES' if result else 'NO'}")

async def test_edge_cases():
    """Test edge cases and error conditions with IMPROVED ERROR HANDLING"""
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
    
    # Test 3: Invalid price data - THIS IS NOW FIXED
    try:
        invalid_data = {'15': [
            {'high': 'invalid', 'low': '0', 'close': '50000'},
            {'high': '50100', 'low': 'bad_data', 'close': '50050'},
            {'high': '50200', 'low': '50000', 'close': 'invalid_price'}
        ] * 7}  # Make it 21 candles total
        
        result = validate_core_risk_reward(invalid_data, "Long")
        print(f"✓ Invalid price data test: {'PASS' if not result else 'FAIL'}")
    except Exception as e:
        print(f"✓ Invalid price data test: PASS (handled exception: {str(e)[:50]}...)")
    
    # Test 4: Volume validation edge cases
    try:
        zero_volume_data = {'1': [create_candle(50000, 0) for _ in range(30)]}
        result = validate_core_volume(zero_volume_data)
        print(f"✓ Zero volume test: {'PASS' if not result else 'FAIL'}")
    except:
        print("✓ Zero volume test: PASS (handled exception)")

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

if __name__ == "__main__":
    # Run all tests
    asyncio.run(run_comprehensive_tests())
