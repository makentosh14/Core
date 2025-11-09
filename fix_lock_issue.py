#!/usr/bin/env python3
"""
Quick fix for lock manager blocking issues
"""

import asyncio
import time

async def test_simple_scan():
    """Test if the bot can scan without the lock manager"""
    print("\n🔍 Testing simple scan without lock manager...")
    
    try:
        # Import required modules
        from scanner import fetch_symbols
        from websocket_candles import stream_candles, live_candles
        from main import filter_core_symbols, calculate_core_score, determine_core_direction
        from trend_upgrade_integration import get_trend_context_cached
        
        # Get symbols
        symbols = await fetch_symbols()
        print(f"✅ Found {len(symbols)} symbols")
        
        # Start websocket
        websocket_task = asyncio.create_task(stream_candles(symbols[:10]))
        print("⏳ Collecting websocket data for 20 seconds...")
        await asyncio.sleep(20)
        
        print(f"📊 Got data for {len(live_candles)} symbols")
        
        # Get trend context
        trend_context = await get_trend_context_cached()
        print(f"📈 Trend context: {trend_context.get('trend', 'unknown')}")
        
        # Try to filter symbols
        quality_symbols = await filter_core_symbols(symbols[:10])
        print(f"🎯 Filtered to {len(quality_symbols)} quality symbols")
        
        if not quality_symbols:
            print("❌ No symbols passed the filter!")
            print("💡 The filter might be too strict")
            return
        
        # Test scoring on first symbol
        symbol = quality_symbols[0]
        print(f"\n🧪 Testing {symbol}...")
        
        # Check if we have candle data
        if symbol in live_candles:
            print(f"✅ Has candle data")
            core_candles = {}
            for tf in ['1', '5', '15']:
                if tf in live_candles.get(symbol, {}):
                    candles = list(live_candles[symbol][tf])
                    if candles and len(candles) >= 30:
                        core_candles[tf] = candles
                        print(f"  {tf}: {len(candles)} candles")
            
            if len(core_candles) >= 3:
                # Calculate score
                score = await calculate_core_score(symbol, core_candles, trend_context)
                print(f"📊 Core score: {score}")
                
                # Get direction
                direction = determine_core_direction(core_candles, trend_context)
                print(f"➡️ Direction: {direction}")
                
                print("\n✅ Basic scanning logic is working!")
            else:
                print(f"❌ Not enough timeframe data (need 3, got {len(core_candles)})")
        else:
            print(f"❌ No candle data for {symbol}")
        
        # Cancel websocket task
        websocket_task.cancel()
        
    except Exception as e:
        print(f"❌ Error in simple scan test: {e}")
        import traceback
        traceback.print_exc()

async def reset_lock_manager():
    """Reset the lock manager state"""
    print("\n🔄 Resetting lock manager state...")
    
    try:
        from trade_lock_manager import trade_lock_manager
        
        # Clear all states
        trade_lock_manager.pending_trades.clear()
        trade_lock_manager.confirmed_trades.clear()
        trade_lock_manager.signal_cooldowns.clear()
        trade_lock_manager.failed_attempts.clear()
        
        # Release all locks
        for symbol, lock in trade_lock_manager.processing_locks.items():
            if lock.locked():
                try:
                    lock.release()
                    print(f"  Released lock for {symbol}")
                except:
                    pass
        
        trade_lock_manager.processing_locks.clear()
        
        print("✅ Lock manager state reset")
        
        # Sync with exchange
        await trade_lock_manager.sync_with_exchange()
        print(f"📊 Synced with exchange: {len(trade_lock_manager.confirmed_trades)} active positions")
        
    except ImportError:
        print("❌ trade_lock_manager not found")
    except Exception as e:
        print(f"❌ Error resetting lock manager: {e}")

async def fix_common_issues():
    """Try to fix common issues automatically"""
    print("\n" + "="*60)
    print("🔧 ATTEMPTING AUTOMATIC FIXES")
    print("="*60)
    
    # Fix 1: Reset lock manager
    await reset_lock_manager()
    
    # Fix 2: Clear stale active_trades
    try:
        from monitor import active_trades, save_active_trades
        
        # Mark all trades without proper exit status as exited
        fixed = 0
        for symbol, trade in active_trades.items():
            if "exited" not in trade:
                trade["exited"] = False
                fixed += 1
        
        if fixed > 0:
            save_active_trades()
            print(f"✅ Fixed {fixed} trades with missing exit status")
        
        # Count active positions
        active = sum(1 for t in active_trades.values() if not t.get("exited", False))
        print(f"📊 Active positions: {active}")
        
    except Exception as e:
        print(f"⚠️ Could not check active trades: {e}")
    
    # Fix 3: Test with a clean scan
    print("\n🧪 Testing clean scan...")
    await test_simple_scan()

async def main():
    """Main function"""
    print("🚀 LOCK MANAGER FIX UTILITY")
    print("="*60)
    
    # Run fixes
    await fix_common_issues()
    
    print("\n" + "="*60)
    print("📝 NEXT STEPS:")
    print("="*60)
    print("1. Run the diagnostic script:")
    print("   python3 diagnose_lock_issue.py")
    print("\n2. Check your bot logs:")
    print("   tail -f /mnt/data/bot_logs/trading_bot_activity.log")
    print("\n3. Try running your bot:")
    print("   python3 main.py")
    print("\n4. If still not working, check:")
    print("   - Is trade_lock_manager.py in your bot directory?")
    print("   - Did you add the import to main.py?")
    print("   - Are there any syntax errors?")

if __name__ == "__main__":
    asyncio.run(main())
