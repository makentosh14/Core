#!/usr/bin/env python3
"""
Complete Testing Script with All Required Functions
This is a standalone script that includes all necessary validation functions
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
    Fixed risk/reward validation with proper support/resistance calculation
    Uses wider lookback period and better level identification
    """
    try:
        log(f"🔍 DEBUG RR: validate_core_risk_reward called, direction={direction}")
        log(f"🔍 DEBUG RR: Available timeframes: {list(core_candles.keys())}")
        
        if '15' not in core_candles:
            log(f"❌ DEBUG RR: No '15' timeframe in core_candles")
            return False
        
        # Use more candles for better S/R identification
        candles = core_candles['15'][-20:]  # Increased from 5 to 20
        log(f"🔍 DEBUG RR: Got {len(candles)} candles from 15m timeframe")
        
        if len(candles) < 10:
            log(f"❌ DEBUG RR: Not enough candles: {len(candles)} < 10")
            return False
        
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        closes = [float(c.get('close', 0)) for c in candles]
        
        current_price = closes[-1]
        log(f"🔍 DEBUG RR: Current price: {current_price}")
        
        # Better support/resistance calculation
        resistance_levels = []
        support_levels = []
        
        # Find local highs and lows for S/R levels
        for i in range(2, len(candles) - 2):
            # Local high (resistance)
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistance_levels.append(highs[i])
            
            # Local low (support)  
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                support_levels.append(lows[i])
        
        # If no clear S/R levels found, use simple min/max with buffer
        if not resistance_levels:
            resistance_levels = [max(highs)]
        if not support_levels:
            support_levels = [min(lows)]
        
        if direction.lower() == "long":
            # For long: find nearest resistance above and support below
            nearby_resistance = min([r for r in resistance_levels if r > current_price], 
                                  default=current_price * 1.02)  # 2% above if none found
            nearby_support = max([s for s in support_levels if s < current_price], 
                               default=current_price * 0.98)  # 2% below if none found
            
            log(f"🔍 DEBUG RR: LONG - Resistance: {nearby_resistance}, Support: {nearby_support}")
            
            potential_reward = nearby_resistance - current_price
            potential_risk = current_price - nearby_support
            
            log(f"🔍 DEBUG RR: Potential reward: {potential_reward}")
            log(f"🔍 DEBUG RR: Potential risk: {potential_risk}")
            
            if potential_reward <= 0 or potential_risk <= 0:
                log(f"❌ DEBUG RR: Invalid reward/risk: {potential_reward}/{potential_risk}")
                return False
            
            rr_ratio = potential_reward / potential_risk
            log(f"🔍 DEBUG RR: Risk/Reward ratio: {rr_ratio:.3f} (needs >= 1.2)")
            
            result = rr_ratio >= 1.2
            log(f"🔍 DEBUG RR: Risk/reward validation result: {result}")
            
            return result
            
        else:  # SHORT
            # For short: find nearest support below and resistance above
            nearby_resistance = min([r for r in resistance_levels if r > current_price], 
                                  default=current_price * 1.02)
            nearby_support = max([s for s in support_levels if s < current_price], 
                               default=current_price * 0.98)
            
            log(f"🔍 DEBUG RR: SHORT - Resistance: {nearby_resistance}, Support: {nearby_support}")
            
            potential_reward = current_price - nearby_support
            potential_risk = nearby_resistance - current_price
            
            log(f"🔍 DEBUG RR: Potential reward: {potential_reward}")
            log(f"🔍 DEBUG RR: Potential risk: {potential_risk}")
            
            if potential_reward <= 0 or potential_risk <= 0:
                log(f"❌ DEBUG RR: Invalid reward/risk: {potential_reward}/{potential_risk}")
                return False
            
            rr_ratio = potential_reward / potential_risk
            log(f"🔍 DEBUG RR: Risk/Reward ratio: {rr_ratio:.3f} (needs >= 1.2)")
            
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
        # Use median instead of mean to avoid skew from outliers
        sorted_volumes = sorted(volumes[:-5])  # Exclude recent 5 candles
        median_volume = sorted_volumes[len(sorted_volumes)//2] if sorted_volumes else 0
        avg_volume = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) > 5 else 0
        
        # Use average of last 3 candles instead of 5 to be more sensitive
        recent_volume = sum(volumes[-3:]) / 3
        
        # Calculate both ratios
        avg_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        median_ratio = recent_volume / median_volume if median_volume > 0 else 0
        
        log(f"🔍 DEBUG: Average volume (excluding recent): {avg_volume}")
        log(f"🔍 DEBUG: Median volume (excluding recent): {median_volume}")
        log(f"🔍 DEBUG: Recent volume (last 3): {recent_volume}")
        log(f"🔍 DEBUG: Avg ratio: {avg_ratio:.3f} (needs > 1.5)")
        log(f"🔍 DEBUG: Median ratio: {median_ratio:.3f} (needs > 1.5)")
        
        # More lenient volume requirement - either ratio needs to be good
        volume_threshold = 1.5  # Relaxed from 2.0
        result = avg_ratio > volume_threshold or median_ratio > volume_threshold
        
        log(f"🔍 DEBUG: Volume validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG: Volume validation error: {e}", level="ERROR")
        import traceback
        log(f"❌ DEBUG: Traceback: {traceback.format_exc()}", level="ERROR")
        return False

def validate_core_price_action(core_candles, direction):
    """Validate clean price action for core strategy"""
    try:
        if '5' not in core_candles:
            return False
        
        candles = core_candles['5'][-10:]
        closes = [float(c.get('close', 0)) for c in candles]
        
        if len(closes) < 10:
            return False
        
        # RELAXED: Check for general directional movement
        if direction.lower() == "long":
            # For long signals, require some upward movement
            upward_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
            return upward_moves >= 4  # At least 40% upward moves (was 60%)
        else:
            # For short signals, require some downward movement
            downward_moves = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
            return downward_moves >= 4  # At least 40% downward moves
        
    except Exception as e:
        log(f"Price action validation error: {e}", level="WARN")
        return False

def validate_core_timing():
    """Validate market timing for core strategy"""
    try:
        from datetime import datetime
        current_hour = datetime.utcnow().hour
        
        # RELAXED: Extended trading hours (was 8-23, now 6-23)
        peak_hours = list(range(6, 24))  # 6 AM to 11 PM UTC
        
        return current_hour in peak_hours
        
    except Exception as e:
        log(f"Timing validation error: {e}", level="WARN")
        return True  # Default to True on error

def validate_core_trend_coherence(core_candles, direction):
    """FIXED: Validate trend coherence across AVAILABLE timeframes only"""
    try:
        trend_scores = {}
        
        # FIXED: Only check timeframes that are actually available
        available_timeframes = [tf for tf in ['1', '5', '15'] if tf in core_candles]
        
        if len(available_timeframes) < 2:  # Need at least 2 timeframes
            return False
        
        for tf in available_timeframes:
            candles = core_candles[tf][-20:]
            closes = [float(c.get('close', 0)) for c in candles]
            
            if len(closes) < 10:  # RELAXED: Reduced from 20 to 10
                continue
            
            # Calculate trend direction for this timeframe
            start_price = sum(closes[:3]) / 3  # RELAXED: First 3 candles (was 5)
            end_price = sum(closes[-3:]) / 3   # RELAXED: Last 3 candles (was 5)
            
            trend_scores[tf] = (end_price - start_price) / start_price
        
        # RELAXED: Majority of timeframes should agree on direction
        if direction.lower() == "long":
            positive_trends = sum(1 for score in trend_scores.values() if score > 0.001)  # RELAXED: 0.1% threshold
            return positive_trends >= len(trend_scores) * 0.6  # 60% agreement
        else:
            negative_trends = sum(1 for score in trend_scores.values() if score < -0.001)
            return negative_trends >= len(trend_scores) * 0.6  # 60% agreement
        
    except Exception as e:
        log(f"Trend coherence validation error: {e}", level="WARN")
        return False

def validate_core_conditions_relaxed(symbol, core_candles, direction, trend_context):
    """
    More relaxed validation for testing with fallback options
    """
    try:
        log(f"🔍 Validating core conditions for {symbol} {direction}")
        
        validation_score = 0
        max_score = 5
        
        # 1. Volume validation (weight: 2 points)
        volume_ok = validate_core_volume(core_candles)
        if volume_ok:
            validation_score += 2
            log(f"   Volume check: ✅ (+2 points)")
        else:
            log(f"   Volume check: ❌ (0 points)")
        
        # 2. Price action quality (weight: 1 point)
        price_action_ok = validate_core_price_action(core_candles, direction)
        if price_action_ok:
            validation_score += 1
            log(f"   Price action check: ✅ (+1 point)")
        else:
            log(f"   Price action check: ❌ (0 points)")
        
        # 3. Risk/reward validation (weight: 1 point) - with fallback
        risk_reward_ok = validate_core_risk_reward(core_candles, direction)
        if risk_reward_ok:
            validation_score += 1
            log(f"   Risk/reward check: ✅ (+1 point)")
        else:
            # Fallback: Check if at least price is trending in right direction
            if check_price_direction_alignment(core_candles, direction):
                validation_score += 0.5
                log(f"   Risk/reward check: ⚠️ (fallback +0.5 points)")
            else:
                log(f"   Risk/reward check: ❌ (0 points)")
        
        # 4. Market timing validation (weight: 0.5 points)
        timing_ok = validate_core_timing()
        if timing_ok:
            validation_score += 0.5
            log(f"   Timing check: ✅ (+0.5 points)")
        else:
            log(f"   Timing check: ❌ (0 points)")
        
        # 5. Trend coherence (weight: 0.5 points)
        trend_coherence_ok = validate_core_trend_coherence(core_candles, direction)
        if trend_coherence_ok:
            validation_score += 0.5
            log(f"   Trend coherence check: ✅ (+0.5 points)")
        else:
            log(f"   Trend coherence check: ❌ (0 points)")
        
        # Pass if score >= 3.0 out of 5.0 (60% threshold)
        passing_threshold = 3.0
        result = validation_score >= passing_threshold
        
        log(f"📊 Validation score: {validation_score}/{max_score} (need {passing_threshold})")
        log(f"🎯 Final result: {'✅ PASS' if result else '❌ FAIL'}")
        
        return result
        
    except Exception as e:
        log(f"❌ CORE STRATEGY: Error validating conditions for {symbol}: {e}", level="ERROR")
        return False

def check_price_direction_alignment(core_candles, direction):
    """
    Fallback check: Ensure price is at least moving in the right direction
    """
    try:
        if '5' not in core_candles:
            return False
        
        candles = core_candles['5'][-10:]
        if len(candles) < 5:
            return False
        
        closes = [float(c.get('close', 0)) for c in candles]
        
        # Simple price direction check
        start_price = closes[0]
        end_price = closes[-1]
        price_change = (end_price - start_price) / start_price
        
        if direction.lower() == "long":
            return price_change > -0.01  # Allow small pullback but not big drop
        else:
            return price_change < 0.01   # Allow small bounce but not big rally
        
    except Exception as e:
        log(f"Price direction alignment check error: {e}", level="WARN")
        return False

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
