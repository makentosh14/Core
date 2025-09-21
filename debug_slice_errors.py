# debug_slice_errors.py - Test file to find all slice errors in your bot

import sys
import traceback
import json
from typing import Any, Dict, List

# Mock data for testing
def create_mock_candles():
    """Create mock candle data to test with"""
    mock_candle = {
        'open': '50000.0',
        'high': '51000.0', 
        'low': '49000.0',
        'close': '50500.0',
        'volume': '1000.0'
    }
    return [mock_candle for _ in range(50)]

def create_problematic_data():
    """Create data types that might cause slice errors"""
    return {
        'dict_as_sequence': {'0': 'data', '1': 'more_data'},  # Dict that looks like sequence
        'string_data': 'not_a_list',  # String instead of list
        'none_data': None,  # None value
        'generator': (x for x in range(10)),  # Generator object
        'range_object': range(10),  # Range object
        'set_data': {1, 2, 3, 4, 5},  # Set instead of list
        'nested_weird': [{'data': [1, 2, 3]}, {'data': 'string'}],  # Mixed nested data
    }

def test_slice_operations():
    """Test various slice operations that might fail"""
    print("🔍 Testing slice operations...")
    
    # Good data
    good_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    good_tuple = tuple(good_list)
    
    # Problematic data
    problematic = create_problematic_data()
    
    test_cases = {
        'good_list': good_list,
        'good_tuple': good_tuple,
        **problematic
    }
    
    for name, data in test_cases.items():
        print(f"\n--- Testing {name} ---")
        print(f"Type: {type(data)}")
        print(f"Data: {data}")
        
        # Test various slice operations
        slice_tests = [
            ('[-20:]', lambda x: x[-20:]),
            ('[-10:]', lambda x: x[-10:]), 
            ('[0:5]', lambda x: x[0:5]),
            ('list(x)[-20:]', lambda x: list(x)[-20:]),
            ('len(x)', lambda x: len(x)),
            ('x[0]', lambda x: x[0]),
        ]
        
        for test_name, operation in slice_tests:
            try:
                result = operation(data)
                print(f"  ✅ {test_name}: {type(result)} - {str(result)[:50]}...")
            except Exception as e:
                print(f"  ❌ {test_name}: {type(e).__name__}: {e}")

def test_your_rsi_functions():
    """Test RSI functions with problematic data"""
    print("\n🔍 Testing RSI functions...")
    
    try:
        # Import your RSI module
        from rsi import calculate_rsi, calculate_rsi_with_bands, detect_rsi_divergence
        
        # Test with good data
        good_candles = create_mock_candles()
        print(f"✅ Created {len(good_candles)} mock candles")
        
        # Test RSI calculation
        try:
            rsi_values = calculate_rsi(good_candles)
            print(f"✅ RSI calculation: Got {len(rsi_values) if rsi_values else 0} values")
        except Exception as e:
            print(f"❌ RSI calculation error: {e}")
            traceback.print_exc()
        
        # Test RSI with bands
        try:
            rsi_data = calculate_rsi_with_bands(good_candles)
            print(f"✅ RSI with bands: {rsi_data['rsi'] if rsi_data else 'None'}")
        except Exception as e:
            print(f"❌ RSI with bands error: {e}")
            traceback.print_exc()
        
        # Test divergence detection
        if 'rsi_values' in locals() and rsi_values:
            try:
                divergence = detect_rsi_divergence(good_candles, rsi_values)
                print(f"✅ RSI divergence: {divergence}")
            except Exception as e:
                print(f"❌ RSI divergence error: {e}")
                traceback.print_exc()
        
        # Test with problematic data
        print("\n--- Testing with problematic candle data ---")
        problematic_candles = [
            [],  # Empty list
            [{}],  # List with empty dict
            [{'close': 'not_a_number'}],  # Invalid close price
            [{'open': '1', 'high': '2'}],  # Missing close
            None,  # None
        ]
        
        for i, bad_candles in enumerate(problematic_candles):
            print(f"\nTesting problematic case {i+1}: {type(bad_candles)} - {bad_candles}")
            try:
                rsi_result = calculate_rsi(bad_candles)
                print(f"  RSI result: {rsi_result}")
            except Exception as e:
                print(f"  ❌ RSI error: {e}")
        
    except ImportError as e:
        print(f"❌ Cannot import RSI module: {e}")

def test_score_functions():
    """Test scoring functions"""
    print("\n🔍 Testing score functions...")
    
    try:
        from score import score_symbol
        
        # Create mock candles by timeframe
        mock_candles_by_tf = {
            '1': create_mock_candles(),
            '5': create_mock_candles(),
            '15': create_mock_candles(),
        }
        
        print("✅ Created mock candles by timeframe")
        
        # Test scoring
        try:
            result = score_symbol("TESTUSDT", mock_candles_by_tf, {})
            print(f"✅ Score result: {result}")
        except Exception as e:
            print(f"❌ Score error: {e}")
            traceback.print_exc()
            
        # Test with problematic data
        print("\n--- Testing with problematic timeframe data ---")
        problematic_tf_data = {
            '1': None,
            '5': [],
            '15': [{}],
            '30': 'not_a_list',
        }
        
        try:
            result = score_symbol("TESTUSDT", problematic_tf_data, {})
            print(f"✅ Score with bad data: {result}")
        except Exception as e:
            print(f"❌ Score with bad data error: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Cannot import score module: {e}")

def test_divergence_detector():
    """Test divergence detector"""
    print("\n🔍 Testing divergence detector...")
    
    try:
        from divergence_detector import divergence_detector
        
        good_candles = create_mock_candles()
        mock_rsi = [30 + i for i in range(len(good_candles))]  # Mock RSI values
        
        try:
            div_result = divergence_detector.detect_rsi_divergence(good_candles, mock_rsi)
            print(f"✅ Divergence result: {div_result}")
        except Exception as e:
            print(f"❌ Divergence error: {e}")
            traceback.print_exc()
            
    except ImportError as e:
        print(f"❌ Cannot import divergence_detector: {e}")

def find_slice_in_code():
    """Find all potential slice operations in your code"""
    print("\n🔍 Scanning code for slice operations...")
    
    import os
    import re
    
    # Files to scan
    files_to_scan = ['score.py', 'rsi.py', 'divergence_detector.py', 'main.py']
    
    slice_pattern = re.compile(r'\[.*-\d+.*:\]|\[.*:\s*-?\d+\]|\[.*-\d+.*\]')
    
    for filename in files_to_scan:
        if os.path.exists(filename):
            print(f"\n--- Scanning {filename} ---")
            try:
                with open(filename, 'r') as f:
                    lines = f.readlines()
                    
                for i, line in enumerate(lines, 1):
                    if slice_pattern.search(line):
                        print(f"  Line {i}: {line.strip()}")
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
        else:
            print(f"  ❌ {filename} not found")

def test_live_candles_structure():
    """Test with a structure similar to live_candles"""
    print("\n🔍 Testing live_candles-like structure...")
    
    # Simulate different possible structures for live_candles
    test_structures = {
        'normal_structure': {
            'BTCUSDT': {
                '1': create_mock_candles(),
                '5': create_mock_candles(),
                '15': create_mock_candles(),
            }
        },
        'generator_structure': {
            'BTCUSDT': {
                '1': (candle for candle in create_mock_candles()),  # Generator
                '5': iter(create_mock_candles()),  # Iterator
            }
        },
        'range_structure': {
            'BTCUSDT': {
                '1': range(20),  # Range object
                '5': "not_iterable",  # String
            }
        },
        'nested_problem': {
            'BTCUSDT': {
                '1': [create_mock_candles()],  # Nested list
                '5': {'candles': create_mock_candles()},  # Dict with candles
            }
        }
    }
    
    for structure_name, live_candles in test_structures.items():
        print(f"\n--- Testing {structure_name} ---")
        
        symbol = 'BTCUSDT'
        
        try:
            # Test the problematic pattern
            candles = None
            for tf in ['1', '5', '15']:
                if tf in live_candles[symbol] and live_candles[symbol][tf]:
                    print(f"  TF {tf}: {type(live_candles[symbol][tf])}")
                    try:
                        candles = live_candles[symbol][tf][-20:]  # This is your problematic line
                        print(f"  ✅ Slice successful: {len(candles)} items")
                        break
                    except Exception as e:
                        print(f"  ❌ Slice failed: {e}")
                        continue
                        
        except Exception as e:
            print(f"  ❌ Structure access error: {e}")

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 COMPREHENSIVE SLICE ERROR DEBUG TEST")
    print("=" * 50)
    
    test_slice_operations()
    test_your_rsi_functions()
    test_score_functions()
    test_divergence_detector()
    test_live_candles_structure()
    find_slice_in_code()
    
    print("\n" + "=" * 50)
    print("🏁 Debug test completed!")
    print("Check the output above to identify where slice errors occur.")

if __name__ == "__main__":
    run_comprehensive_test()
