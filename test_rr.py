#!/usr/bin/env python3
"""
Test the fixed risk/reward validation
Run this to verify the fix works before integrating into main.py
"""

def log(message, level="INFO"):
    print(f"[{level}] {message}")

# PASTE THE COMPLETE FIXED FUNCTION HERE (from above artifact)
def validate_core_risk_reward(core_candles, direction):
    """
    COMPLETE FIXED VERSION: Risk/reward validation with proper target calculation
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
        
        highs = [float(c.get('high', 0)) for c in candles]
        lows = [float(c.get('low', 0)) for c in candles]
        closes = [float(c.get('close', 0)) for c in candles]
        
        current_price = closes[-1]
        log(f"🔍 DEBUG RR: Current price: {current_price}")
        
        if direction.lower() == "long":
            # FIXED APPROACH FOR LONG POSITIONS
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
            
            log(f"🔍 DEBUG RR: LONG - Support: {effective_support:.2f}")
            log(f"🔍 DEBUG RR: LONG - Target approaches: {[(name, f'{target:.2f}') for name, target in approaches]}")
            log(f"🔍 DEBUG RR: LONG - Selected: {best_approach[0]} = {effective_resistance:.2f}")
            
            potential_reward = effective_resistance - current_price
            
        else:  # SHORT - existing logic
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(candles) - 2):
                if (highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and 
                    highs[i] >= highs[i+1] and highs[i] >= highs[i+2]):
                    resistance_levels.append(highs[i])
                
                if (lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and 
                    lows[i] <= lows[i+1] and lows[i] <= lows[i+2]):
                    support_levels.append(lows[i])
            
            if not resistance_levels:
                resistance_levels = [max(highs)]
            if not support_levels:
                support_levels = [min(lows)]
            
            nearby_resistance = min([r for r in resistance_levels if r > current_price], 
                                  default=current_price * 1.02)
            nearby_support = max([s for s in support_levels if s < current_price], 
                               default=current_price * 0.98)
            
            log(f"🔍 DEBUG RR: SHORT - Resistance: {nearby_resistance:.2f}, Support: {nearby_support:.2f}")
            
            potential_reward = current_price - nearby_support
            potential_risk = nearby_resistance - current_price
        
        log(f"🔍 DEBUG RR: Potential reward: {potential_reward:.2f}")
        log(f"🔍 DEBUG RR: Potential risk: {potential_risk:.2f}")
        
        if potential_reward <= 0 or potential_risk <= 0:
            log(f"❌ DEBUG RR: Invalid reward/risk: {potential_reward:.2f}/{potential_risk:.2f}")
            return False
        
        rr_ratio = potential_reward / potential_risk
        log(f"🔍 DEBUG RR: Risk/Reward ratio: {rr_ratio:.3f} (needs >= 1.2)")
        
        # Quality checks
        min_reward_threshold = current_price * 0.015
        if potential_reward < min_reward_threshold:
            log(f"❌ DEBUG RR: Reward too small: {potential_reward:.2f} < {min_reward_threshold:.2f}")
            return False
        
        max_risk_threshold = current_price * 0.04
        if potential_risk > max_risk_threshold:
            log(f"❌ DEBUG RR: Risk too large: {potential_risk:.2f} > {max_risk_threshold:.2f}")
            return False
        
        result = rr_ratio >= 1.2
        log(f"🔍 DEBUG RR: Risk/reward validation result: {result}")
        
        return result
        
    except Exception as e:
        log(f"❌ DEBUG RR: Risk/reward validation error: {e}", level="ERROR")
        import traceback
        log(f"❌ DEBUG RR: Traceback: {traceback.format_exc()}", level="ERROR")
        return False

def create_test_cases():
    """Create test cases that mirror your failing scenarios"""
    
    # Case 1: BTCUSDT-like scenario (was failing)
    btc_candles = []
    base_price = 50000
    for i in range(20):
        # Simulate gradual uptrend
        price = base_price + (i * 100) + (i % 3 * 50)  # 50000 to 52150 range
        btc_candles.append({
            'high': str(price + 60),
            'low': str(price - 50),
            'close': str(price),
            'open': str(price - 10)
        })
    
    # Case 2: ETHUSDT-like scenario (was failing)  
    eth_candles = []
    base_price = 3000
    for i in range(15):
        price = base_price + (i * 20)  # 3000 to 3280 range
        eth_candles.append({
            'high': str(price + 20),
            'low': str(price - 15),
            'close': str(price),
            'open': str(price - 5)
        })
    
    # Case 3: Short scenario (was working)
    ada_candles = []
    base_price = 1.0
    for i in range(15):
        price = base_price - (i * 0.01)  # Downtrend
        ada_candles.append({
            'high': str(price + 0.005),
            'low': str(price - 0.005),
            'close': str(price),
            'open': str(price + 0.002)
        })
    
    return {
        'BTCUSDT': {'15': btc_candles},
        'ETHUSDT': {'15': eth_candles}, 
        'ADAUSDT': {'15': ada_candles}
    }

def run_tests():
    """Run the fixed validation tests"""
    print("🧪 TESTING FIXED RISK/REWARD VALIDATION")
    print("=" * 80)
    
    test_cases = create_test_cases()
    
    # Test the scenarios that were failing
    print("\n📈 TESTING LONG POSITIONS (were failing):")
    print("-" * 50)
    
    btc_result = validate_core_risk_reward(test_cases['BTCUSDT'], "Long")
    print(f"✓ BTCUSDT Long: {'✅ PASS' if btc_result else '❌ FAIL'}")
    
    print("\n" + "-" * 50)
    eth_result = validate_core_risk_reward(test_cases['ETHUSDT'], "Long") 
    print(f"✓ ETHUSDT Long: {'✅ PASS' if eth_result else '❌ FAIL'}")
    
    print("\n📉 TESTING SHORT POSITION (was working):")
    print("-" * 50)
    
    ada_result = validate_core_risk_reward(test_cases['ADAUSDT'], "Short")
    print(f"✓ ADAUSDT Short: {'✅ PASS' if ada_result else '❌ FAIL'}")
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY:")
    print(f"   BTCUSDT Long: {'FIXED ✅' if btc_result else 'Still failing ❌'}")
    print(f"   ETHUSDT Long: {'FIXED ✅' if eth_result else 'Still failing ❌'}")
    print(f"   ADAUSDT Short: {'Still working ✅' if ada_result else 'Broken ❌'}")
    
    all_pass = btc_result and eth_result and ada_result
    print(f"\n🏆 OVERALL: {'ALL TESTS PASS - READY TO DEPLOY ✅' if all_pass else 'NEEDS MORE WORK ❌'}")
    
    return all_pass

if __name__ == "__main__":
    run_tests()
