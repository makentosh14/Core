#!/usr/bin/env python3
"""
Quick test to validate the Risk/Reward fix works for your specific case
Run this to verify the fix before updating main.py
"""

def log(message, level="INFO"):
    print(f"[{level}] {message}")

def validate_core_risk_reward_fixed(core_candles, direction):
    """Fixed version - same as the one for main.py integration"""
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
            # FIXED: Better logic for LONG positions
            
            # Find recent support levels (focus on last 10 candles, not all 20)
            recent_lows = lows[-10:]
            recent_highs = highs[-10:]
            
            # Use recent swing low as support
            recent_support = min(recent_lows)
            recent_resistance = max(recent_highs)
            
            # CRITICAL FIX: Ensure support is not too far from current price
            max_risk_percent = 0.04  # Maximum 4% risk
            min_support = current_price * (1 - max_risk_percent)
            
            # Use the higher of: recent support OR minimum acceptable support
            effective_support = max(recent_support, min_support)
            
            # For resistance, ensure reasonable target (at least 2% above current)
            min_resistance = current_price * 1.02
            effective_resistance = max(recent_resistance, min_resistance)
            
            log(f"🔍 DEBUG RR: LONG - Recent Support: {recent_support}, Effective Support: {effective_support}")
            log(f"🔍 DEBUG RR: LONG - Recent Resistance: {recent_resistance}, Effective Resistance: {effective_resistance}")
            
            potential_reward = effective_resistance - current_price
            potential_risk = current_price - effective_support
            
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
            
        else:  # SHORT - keep original logic
            resistance_levels = []
            support_levels = []
            
            for i in range(2, len(candles) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(highs[i])
                
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(lows[i])
            
            if not resistance_levels:
                resistance_levels = [max(highs)]
            if not support_levels:
                support_levels = [min(lows)]
            
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
        return False

def create_btc_like_test_case():
    """Create a test case similar to your failing BTC scenario"""
    candles = []
    
    # Simulate the pattern from your test: trending from ~49500 to 52200
    prices = [
        49800, 49850, 49900, 49950, 50000,  # Initial support area
        50200, 50400, 50600, 50800, 51000,  # Breakout area
        51200, 51400, 51600, 51800, 52000,  # Resistance building
        51800, 51900, 52000, 52100, 52200   # Current price at 52200
    ]
    
    for price in prices:
        high = price + 60   # Add realistic wicks
        low = price - 50
        open_price = price - 10
        
        candle = {
            'high': str(high),
            'low': str(low),
            'close': str(price),
            'open': str(open_price)
        }
        candles.append(candle)
    
    return {'15': candles}

def test_fix():
    """Test the fixed function against your failing scenario"""
    print("=" * 60)
    print("TESTING FIXED RISK/REWARD FUNCTION")
    print("=" * 60)
    
    # Test with BTC-like scenario that was failing
    btc_test_data = create_btc_like_test_case()
    
    print("\n--- Testing BTC-like Long Position (was failing) ---")
    result = validate_core_risk_reward_fixed(btc_test_data, "Long")
    print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
    
    print("\n" + "=" * 60)
    print("If this shows PASS, you can update your main.py file!")
    print("=" * 60)

if __name__ == "__main__":
    test_fix()
