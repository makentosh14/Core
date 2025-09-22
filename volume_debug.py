# volume_debug.py - Standalone script to debug volume validation

import sys
import traceback

def create_debug_candles():
    """Create candles with guaranteed volume spike"""
    candles = []
    
    # First 15 candles: 1M volume each
    for i in range(15):
        candles.append({
            'volume': '1000000',
            'close': '50000',
            'open': '49900',
            'high': '50100',
            'low': '49800'
        })
    
    # Last 5 candles: 4M volume each (should give 4x ratio)
    for i in range(5):
        candles.append({
            'volume': '4000000',
            'close': '50000',
            'open': '49900', 
            'high': '50100',
            'low': '49800'
        })
    
    return candles

def test_manual_calculation():
    """Manually test the volume calculation logic"""
    print("=== MANUAL VOLUME CALCULATION TEST ===")
    
    candles = create_debug_candles()
    print(f"Created {len(candles)} candles")
    
    # Extract last 20 candles (should be all of them)
    test_candles = candles[-20:]
    print(f"Using last {len(test_candles)} candles")
    
    # Calculate volumes
    volumes = [float(c.get('volume', 0)) for c in test_candles]
    print(f"Volumes array length: {len(volumes)}")
    print(f"First 5 volumes: {volumes[:5]}")
    print(f"Last 5 volumes: {volumes[-5:]}")
    
    if len(volumes) < 20:
        print(f"❌ Not enough volumes: {len(volumes)} < 20")
        return False
    
    # Calculate averages
    avg_volume = sum(volumes) / len(volumes)
    recent_volume = sum(volumes[-5:]) / 5
    
    print(f"Average volume (all 20): {avg_volume}")
    print(f"Recent volume (last 5): {recent_volume}")
    print(f"Ratio: {recent_volume / avg_volume:.2f}")
    print(f"Required ratio: > 1.5")
    print(f"Test passes: {recent_volume > avg_volume * 1.5}")
    
    return recent_volume > avg_volume * 1.5

def test_your_function():
    """Test your actual validate_core_volume function"""
    print("\n=== YOUR FUNCTION TEST ===")
    
    try:
        # Try to import your function
        from main import validate_core_volume
        print("✅ Successfully imported validate_core_volume")
        
        # Create test data
        core_candles = {
            '1': create_debug_candles()
        }
        print(f"Created core_candles with {len(core_candles['1'])} candles")
        
        # Test the function
        result = validate_core_volume(core_candles)
        print(f"Function result: {result}")
        
        return result
        
    except ImportError as e:
        print(f"❌ Cannot import validate_core_volume: {e}")
        return False
    except Exception as e:
        print(f"❌ Error calling validate_core_volume: {e}")
        traceback.print_exc()
        return False

def test_function_source():
    """Print the actual source of your function"""
    print("\n=== FUNCTION SOURCE INSPECTION ===")
    
    try:
        from main import validate_core_volume
        import inspect
        
        # Get the source code
        source = inspect.getsource(validate_core_volume)
        print("Your validate_core_volume function:")
        print("-" * 50)
        print(source)
        print("-" * 50)
        
    except Exception as e:
        print(f"Cannot inspect function source: {e}")

if __name__ == "__main__":
    print("🔍 VOLUME VALIDATION DEBUG SCRIPT")
    print("=" * 50)
    
    # Test 1: Manual calculation
    manual_result = test_manual_calculation()
    
    # Test 2: Your actual function
    function_result = test_your_function()
    
    # Test 3: Function source
    test_function_source()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Manual calculation: {'✅ PASS' if manual_result else '❌ FAIL'}")
    print(f"Your function: {'✅ PASS' if function_result else '❌ FAIL'}")
    
    if manual_result and not function_result:
        print("❌ Your function has a bug - manual calc works but function fails")
    elif not manual_result and function_result:
        print("❌ Logic error - function passes but manual calc fails")
    elif not manual_result and not function_result:
        print("❌ Volume spike not high enough - need more aggressive volumes")
    else:
        print("✅ Everything working correctly")
