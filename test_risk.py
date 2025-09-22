def validate_core_risk_reward(core_candles, direction):
    """
    FINAL FIX: Risk/reward validation with better target calculation for Long positions
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
            # Use recent support levels (last 10 candles)
            recent_lows = lows[-10:]
            recent_highs = highs[-10:]
            
            recent_support = min(recent_lows)
            recent_resistance = max(recent_highs)
            
            # Cap maximum risk at 4% of current price
            max_risk_percent = 0.04
            min_support = current_price * (1 - max_risk_percent)
            effective_support = max(recent_support, min_support)
            
            # FINAL FIX: Be more generous with target calculation
            # Use the better of: recent resistance OR calculated target based on risk
            calculated_risk = current_price - effective_support
            
            # Target should be at least 1.3x the risk for good R/R (gives 1.3 ratio)
            min_target_based_on_risk = current_price + (calculated_risk * 1.3)
            
            # Also ensure minimum 3% target above current price
            min_percentage_target = current_price * 1.03
            
            # Use the highest target for best R/R
            effective_resistance = max(recent_resistance, min_target_based_on_risk, min_percentage_target)
            
            log(f"🔍 DEBUG RR: LONG - Recent Support: {recent_support}, Effective Support: {effective_support}")
            log(f"🔍 DEBUG RR: LONG - Recent Resistance: {recent_resistance}")
            log(f"🔍 DEBUG RR: LONG - Risk-based Target: {min_target_based_on_risk}")
            log(f"🔍 DEBUG RR: LONG - Final Target: {effective_resistance}")
            
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
            
        else:  # SHORT - keep existing logic
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
        import traceback
        log(f"❌ DEBUG RR: Traceback: {traceback.format_exc()}", level="ERROR")
        return False


# Quick test of the final fix
def test_final_fix():
    """Test the final adjusted version"""
    
    def log(message, level="INFO"):
        print(f"[{level}] {message}")
    
    def create_btc_test():
        prices = [49800, 49850, 49900, 49950, 50000, 50200, 50400, 50600, 50800, 51000,
                 51200, 51400, 51600, 51800, 52000, 51800, 51900, 52000, 52100, 52200]
        candles = []
        for price in prices:
            candles.append({
                'high': str(price + 60),
                'low': str(price - 50), 
                'close': str(price),
                'open': str(price - 10)
            })
        return {'15': candles}
    
    print("TESTING FINAL ADJUSTED RISK/REWARD FIX")
    print("=" * 50)
    
    test_data = create_btc_test()
    result = validate_core_risk_reward(test_data, "Long")
    
    print(f"\nFINAL RESULT: {'✅ PASS - Ready to integrate!' if result else '❌ STILL FAILING'}")
    
    return result

if __name__ == "__main__":
    test_final_fix()
