#!/usr/bin/env python3
"""
test_telegram.py - Telegram Bot Test Script
Simple script to test Telegram bot functionality
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_telegram_basic():
    """Test basic Telegram message sending"""
    print("🔍 Testing basic Telegram message sending...")
    
    try:
        from error_handler import send_telegram_message
        
        test_message = f"🤖 <b>Telegram Bot Test</b>\n\n✅ Bot connectivity test successful!\n📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await send_telegram_message(test_message)
        print("✅ Test message sent successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error sending test message: {e}")
        return False

async def test_telegram_config():
    """Test Telegram configuration"""
    print("🔍 Testing Telegram configuration...")
    
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        
        if not TELEGRAM_BOT_TOKEN:
            print("❌ TELEGRAM_BOT_TOKEN is not set!")
            return False
        
        if not TELEGRAM_CHAT_ID:
            print("❌ TELEGRAM_CHAT_ID is not set!")
            return False
        
        # Mask token for security
        masked_token = TELEGRAM_BOT_TOKEN[:10] + "..." + TELEGRAM_BOT_TOKEN[-10:] if len(TELEGRAM_BOT_TOKEN) > 20 else "***"
        
        print(f"✅ TELEGRAM_BOT_TOKEN: {masked_token}")
        print(f"✅ TELEGRAM_CHAT_ID: {TELEGRAM_CHAT_ID}")
        return True
        
    except ImportError as e:
        print(f"❌ Config import error: {e}")
        print("💡 Make sure config.py exists and has TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

async def test_telegram_error_handler():
    """Test Telegram error handler"""
    print("🔍 Testing Telegram error handler...")
    
    try:
        from error_handler import send_error_to_telegram
        
        test_error = "🧪 Test error message from telegram test script"
        await send_error_to_telegram(test_error)
        print("✅ Error handler test message sent!")
        return True
        
    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        return False

async def test_telegram_bot_commands():
    """Test if Telegram bot command handlers are available"""
    print("🔍 Testing Telegram bot command availability...")
    
    try:
        import telegram_bot
        
        # Check if key functions exist
        required_functions = ['send_telegram_message', 'format_trade_signal']
        missing_functions = []
        
        for func_name in required_functions:
            if not hasattr(telegram_bot, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        
        print("✅ Telegram bot module loaded successfully!")
        print("✅ Required functions are available!")
        return True
        
    except ImportError as e:
        print(f"❌ Telegram bot import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Telegram bot test error: {e}")
        return False

def test_telegram_bot_process():
    """Test if Telegram bot process can be started"""
    print("🔍 Testing Telegram bot process startup...")
    
    try:
        from telegram_bot import bot, dp
        print("✅ Telegram bot and dispatcher objects created successfully!")
        
        # Try to get bot info
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_bot_info():
            try:
                me = await bot.get_me()
                print(f"✅ Bot info: @{me.username} ({me.first_name})")
                return True
            except Exception as e:
                print(f"❌ Failed to get bot info: {e}")
                return False
        
        result = loop.run_until_complete(get_bot_info())
        loop.close()
        
        return result
        
    except ImportError as e:
        print(f"❌ Telegram bot objects import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Telegram bot process test error: {e}")
        return False

async def send_test_signal():
    """Send a test trading signal"""
    print("🔍 Testing trade signal formatting...")
    
    try:
        from telegram_bot import format_trade_signal, send_telegram_message
        
        # Create test signal data
        test_signal = format_trade_signal(
            symbol="BTCUSDT",
            score=8.5,
            tf_scores={"1": 2.1, "5": 3.2, "15": 3.2},
            trend={"btc_trend": "bullish", "altseason": "active"},
            entry_price=45000,
            sl=44000,
            tp1=47000,
            trade_type="Scalp",
            direction="Long",
            trailing_pct=1.5,
            leverage=10,
            risk_pct=2.0,
            confidence=75.5,
            sl_pct=2.22,
            tp1_pct=4.44
        )
        
        print("✅ Test signal formatted successfully!")
        print(f"📱 Signal preview:\n{test_signal}")
        
        # Send the test signal
        await send_telegram_message(f"🧪 <b>TEST SIGNAL</b>\n\n{test_signal}")
        print("✅ Test signal sent to Telegram!")
        return True
        
    except Exception as e:
        print(f"❌ Test signal error: {e}")
        return False

async def main():
    """Run all Telegram tests"""
    print("🚀 Starting Telegram Bot Tests...")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_telegram_config),
        ("Basic Messaging", test_telegram_basic),
        ("Error Handler", test_telegram_error_handler),
        ("Bot Commands", test_telegram_bot_commands),
        ("Bot Process", lambda: test_telegram_bot_process()),
        ("Test Signal", send_test_signal),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Telegram bot is ready!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n💡 Common fixes:")
        print("   • Make sure config.py has TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        print("   • Verify bot token is valid (from @BotFather)")
        print("   • Check if you've sent /start to your bot")
        print("   • Ensure internet connectivity")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test script error: {e}")
        import traceback
        traceback.print_exc()
