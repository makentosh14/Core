#!/usr/bin/env python3
"""
Diagnostic script to check why the bot isn't scanning after lock manager update
"""

import asyncio
import sys
import traceback

async def test_lock_manager():
    """Test if the lock manager is working correctly"""
    print("\n" + "="*60)
    print("🔍 TESTING TRADE LOCK MANAGER")
    print("="*60)
    
    try:
        from trade_lock_manager import trade_lock_manager
        print("✅ trade_lock_manager imported successfully")
        
        # Test basic functionality
        test_symbol = "BTCUSDT"
        
        # Test 1: Check if we can process symbol
        can_process, reason = await trade_lock_manager.can_process_symbol(test_symbol, check_exchange=False)
        print(f"\n📊 Can process {test_symbol}: {can_process}")
        print(f"   Reason: {reason}")
        
        # Test 2: Try to acquire lock
        if can_process:
            acquired = await trade_lock_manager.acquire_trade_lock(test_symbol)
            print(f"\n🔒 Lock acquired for {test_symbol}: {acquired}")
            
            if acquired:
                # Test 3: Check state
                print(f"   Pending trades: {trade_lock_manager.pending_trades}")
                print(f"   Confirmed trades: {trade_lock_manager.confirmed_trades}")
                
                # Release lock
                trade_lock_manager.release_trade_lock(test_symbol, False)
                print(f"🔓 Lock released for {test_symbol}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import trade_lock_manager: {e}")
        print("\n💡 Fix: Make sure trade_lock_manager.py exists in your bot directory")
        return False
    except Exception as e:
        print(f"❌ Error testing lock manager: {e}")
        traceback.print_exc()
        return False

async def test_main_imports():
    """Test if main.py imports are working"""
    print("\n" + "="*60)
    print("🔍 TESTING MAIN.PY IMPORTS")
    print("="*60)
    
    try:
        from main import core_strategy_scan, filter_core_symbols
        print("✅ main.py functions imported successfully")
        
        # Check if trade_lock_manager is imported in main
        import main
        if hasattr(main, 'trade_lock_manager'):
            print("✅ trade_lock_manager is imported in main.py")
        else:
            print("❌ trade_lock_manager NOT imported in main.py")
            print("💡 Fix: Add 'from trade_lock_manager import trade_lock_manager' to main.py imports")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import from main.py: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Error testing main imports: {e}")
        traceback.print_exc()
        return False

async def test_websocket_data():
    """Test if websocket is providing data"""
    print("\n" + "="*60)
    print("🔍 TESTING WEBSOCKET DATA")
    print("="*60)
    
    try:
        from websocket_candles import live_candles
        from scanner import fetch_symbols
        
        print(f"📊 Live candles has data for {len(live_candles)} symbols")
        
        if len(live_candles) == 0:
            print("⚠️ No live candle data available")
            print("💡 Fix: Wait for websocket to collect data (20-30 seconds)")
            
            # Try to fetch symbols
            symbols = await fetch_symbols()
            print(f"\n📋 Found {len(symbols)} symbols to scan")
            print(f"   First 5: {symbols[:5]}")
            
            return False
        else:
            # Show sample data
            sample_symbol = list(live_candles.keys())[0]
            print(f"\n✅ Sample data for {sample_symbol}:")
            if sample_symbol in live_candles:
                for tf, candles in live_candles[sample_symbol].items():
                    if candles:
                        print(f"   {tf}: {len(candles)} candles")
            return True
            
    except Exception as e:
        print(f"❌ Error testing websocket data: {e}")
        traceback.print_exc()
        return False

async def test_filter_core_symbols():
    """Test if filter_core_symbols is working"""
    print("\n" + "="*60)
    print("🔍 TESTING SYMBOL FILTERING")
    print("="*60)
    
    try:
        from main import filter_core_symbols
        from scanner import fetch_symbols
        from websocket_candles import live_candles
        
        symbols = await fetch_symbols()
        print(f"📋 Testing with {len(symbols[:10])} symbols")
        
        # Test the filter
        filtered = await filter_core_symbols(symbols[:10])
        
        if filtered:
            print(f"✅ Filter returned {len(filtered)} symbols")
            print(f"   Symbols: {filtered}")
        else:
            print("⚠️ Filter returned no symbols")
            print("💡 Possible issues:")
            print("   1. No websocket data yet")
            print("   2. Filter criteria too strict")
            print("   3. Error in filter function")
            
        return len(filtered) > 0
        
    except Exception as e:
        print(f"❌ Error testing filter: {e}")
        traceback.print_exc()
        return False

async def test_lock_acquisition():
    """Test if the lock acquisition logic is blocking everything"""
    print("\n" + "="*60)
    print("🔍 TESTING LOCK ACQUISITION LOGIC")
    print("="*60)
    
    try:
        from trade_lock_manager import trade_lock_manager
        
        test_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        for symbol in test_symbols:
            # Check if we can process
            can_process, reason = await trade_lock_manager.can_process_symbol(symbol, check_exchange=False)
            print(f"\n{symbol}:")
            print(f"  Can process: {can_process} - {reason}")
            
            if can_process:
                # Try to acquire lock
                acquired = await trade_lock_manager.acquire_trade_lock(symbol)
                print(f"  Lock acquired: {acquired}")
                
                if acquired:
                    # Check if we can acquire again (should fail)
                    acquired2 = await trade_lock_manager.acquire_trade_lock(symbol)
                    print(f"  Second acquire attempt: {acquired2} (should be False)")
                    
                    # Release
                    trade_lock_manager.release_trade_lock(symbol, False)
                    print(f"  Lock released")
                    
        return True
        
    except Exception as e:
        print(f"❌ Error testing lock acquisition: {e}")
        traceback.print_exc()
        return False

async def check_active_trades():
    """Check if active_trades is blocking everything"""
    print("\n" + "="*60)
    print("🔍 CHECKING ACTIVE TRADES")
    print("="*60)
    
    try:
        from monitor import active_trades
        
        print(f"📊 Active trades: {len(active_trades)}")
        
        if active_trades:
            print("\nActive positions:")
            for symbol, trade in active_trades.items():
                exited = trade.get("exited", False)
                print(f"  {symbol}: exited={exited}")
                
        # Check position limits
        from main import MAX_CORE_POSITIONS
        current_positions = sum(1 for trade in active_trades.values() if not trade.get("exited", False))
        print(f"\n📈 Current positions: {current_positions}/{MAX_CORE_POSITIONS}")
        
        if current_positions >= MAX_CORE_POSITIONS:
            print("⚠️ MAX POSITIONS REACHED - Bot won't scan for new trades")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error checking active trades: {e}")
        traceback.print_exc()
        return False

async def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("🚀 STARTING BOT DIAGNOSTICS")
    print("="*60)
    
    results = {}
    
    # Run tests
    results['lock_manager'] = await test_lock_manager()
    results['main_imports'] = await test_main_imports()
    results['active_trades'] = await check_active_trades()
    results['websocket'] = await test_websocket_data()
    results['filter'] = await test_filter_core_symbols()
    results['lock_acquisition'] = await test_lock_acquisition()
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    all_passed = True
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! The bot should be working.")
        print("💡 If still not scanning, check:")
        print("   1. Is the bot actually running? (python3 main.py)")
        print("   2. Check logs: tail -f /mnt/data/bot_logs/trading_bot_activity.log")
    else:
        print("\n❌ Some tests failed. Fix the issues above.")
        print("\n🔧 QUICK FIXES:")
        print("1. If lock_manager import fails:")
        print("   - Make sure trade_lock_manager.py exists")
        print("2. If websocket has no data:")
        print("   - Start the bot and wait 30 seconds")
        print("3. If max positions reached:")
        print("   - Close some positions or increase MAX_CORE_POSITIONS")

if __name__ == "__main__":
    asyncio.run(main())
