#!/usr/bin/env python3
"""
Emergency cleanup script to close all positions and cancel all orders
This version properly handles the Bybit API requirements
"""

import asyncio
import json
import os
from datetime import datetime
from bybit_api import signed_request

async def get_open_orders_by_symbol(symbol):
    """Get open orders for a specific symbol"""
    try:
        response = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear",
            "symbol": symbol
        })
        
        if response.get("retCode") == 0:
            return response.get("result", {}).get("list", [])
        else:
            print(f"❌ Failed to get orders for {symbol}: {response.get('retMsg')}")
            return []
    except Exception as e:
        print(f"❌ Error getting orders for {symbol}: {e}")
        return []

async def get_all_positions():
    """Get all open positions"""
    try:
        # Get positions by settleCoin instead of all at once
        response = await signed_request("GET", "/v5/position/list", {
            "category": "linear",
            "settleCoin": "USDT"  # This is required - we're using USDT perpetuals
        })
        
        if response.get("retCode") == 0:
            positions = response.get("result", {}).get("list", [])
            # Filter to only positions with size > 0
            open_positions = []
            for pos in positions:
                size = float(pos.get("size", "0"))
                if abs(size) > 0:
                    open_positions.append(pos)
            return open_positions
        else:
            print(f"❌ Failed to get positions: {response.get('retMsg')}")
            return []
    except Exception as e:
        print(f"❌ Error getting positions: {e}")
        return []

async def close_position(symbol, side, size):
    """Close a position by placing a market order"""
    try:
        # Determine the opposite side to close the position
        close_side = "Sell" if side == "Buy" else "Buy"
        
        print(f"🔄 Closing {side} position for {symbol}: {size} units")
        
        response = await signed_request("POST", "/v5/order/create", {
            "category": "linear",
            "symbol": symbol,
            "side": close_side,
            "orderType": "Market",
            "qty": str(abs(float(size))),
            "timeInForce": "IOC",
            "reduceOnly": True
        })
        
        if response.get("retCode") == 0:
            print(f"✅ Successfully closed position for {symbol}")
            return True
        else:
            print(f"❌ Failed to close position for {symbol}: {response.get('retMsg')}")
            return False
    except Exception as e:
        print(f"❌ Error closing position for {symbol}: {e}")
        return False

async def cancel_orders_for_symbol(symbol):
    """Cancel all orders for a specific symbol"""
    try:
        response = await signed_request("POST", "/v5/order/cancel-all", {
            "category": "linear",
            "symbol": symbol
        })
        
        if response.get("retCode") == 0:
            print(f"✅ Cancelled all orders for {symbol}")
            return True
        else:
            print(f"❌ Failed to cancel orders for {symbol}: {response.get('retMsg')}")
            return False
    except Exception as e:
        print(f"❌ Error cancelling orders for {symbol}: {e}")
        return False

async def get_symbols_with_activity():
    """Get all symbols that have either positions or orders"""
    symbols = set()
    
    # Get symbols from positions
    positions = await get_all_positions()
    for pos in positions:
        symbols.add(pos.get("symbol"))
    
    # Common trading symbols to check for orders
    common_symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "NEARUSDT", "FTMUSDT", "ALGOUSDT",
        "INJUSDT", "SUIUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "TIAUSDT"
    ]
    
    # Check each common symbol for orders
    for symbol in common_symbols:
        orders = await get_open_orders_by_symbol(symbol)
        if orders:
            symbols.add(symbol)
    
    return list(symbols)

async def emergency_cleanup():
    """Main emergency cleanup function"""
    print("🧹 Starting emergency cleanup...")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Get all symbols with activity
    print("\n📋 Step 1: Finding symbols with positions or orders...")
    active_symbols = await get_symbols_with_activity()
    
    if not active_symbols:
        print("✅ No active symbols found - nothing to clean up!")
        return
    
    print(f"🎯 Found {len(active_symbols)} active symbols: {', '.join(active_symbols)}")
    
    # Step 2: Close all positions
    print("\n🔄 Step 2: Closing all open positions...")
    positions = await get_all_positions()
    
    if positions:
        print(f"📊 Found {len(positions)} open positions")
        for pos in positions:
            symbol = pos.get("symbol")
            side = pos.get("side")
            size = pos.get("size")
            
            print(f"  - {symbol}: {side} {size}")
            await close_position(symbol, side, size)
            await asyncio.sleep(0.5)  # Small delay between operations
    else:
        print("✅ No open positions found")
    
    # Step 3: Cancel all orders
    print("\n❌ Step 3: Cancelling all orders...")
    total_orders_cancelled = 0
    
    for symbol in active_symbols:
        orders = await get_open_orders_by_symbol(symbol)
        if orders:
            print(f"📋 Found {len(orders)} orders for {symbol}")
            total_orders_cancelled += len(orders)
            await cancel_orders_for_symbol(symbol)
        await asyncio.sleep(0.5)  # Small delay between operations
    
    if total_orders_cancelled == 0:
        print("✅ No orders found to cancel")
    else:
        print(f"📊 Cancelled orders for {len(active_symbols)} symbols")
    
    # Step 4: Clean up bot's internal state
    print("\n🗑️ Step 4: Cleaning up bot's internal state...")
    
    # Clear active trades file
    trades_file = "monitor_active_trades.json"
    if os.path.exists(trades_file):
        try:
            with open(trades_file, 'w') as f:
                json.dump({}, f)
            print(f"✅ Cleared {trades_file}")
        except Exception as e:
            print(f"❌ Failed to clear {trades_file}: {e}")
    
    # Create a cleanup log entry
    cleanup_log = {
        "timestamp": datetime.now().isoformat(),
        "action": "emergency_cleanup",
        "positions_closed": len(positions) if positions else 0,
        "symbols_processed": len(active_symbols),
        "total_orders_cancelled": total_orders_cancelled
    }
    
    try:
        with open("emergency_cleanup_log.json", "w") as f:
            json.dump(cleanup_log, f, indent=2)
        print("✅ Created cleanup log")
    except Exception as e:
        print(f"⚠️ Failed to create cleanup log: {e}")
    
    print("\n🎉 Emergency cleanup completed!")
    print("📋 Summary:")
    print(f"  - Positions processed: {len(positions) if positions else 0}")
    print(f"  - Symbols checked: {len(active_symbols)}")
    print(f"  - Orders cancelled: {total_orders_cancelled}")
    print(f"  - Internal state: Cleared")

async def verify_cleanup():
    """Verify that cleanup was successful"""
    print("\n🔍 Verifying cleanup...")
    
    # Check positions
    positions = await get_all_positions()
    if positions:
        print(f"⚠️ Warning: {len(positions)} positions still open:")
        for pos in positions:
            print(f"  - {pos.get('symbol')}: {pos.get('side')} {pos.get('size')}")
    else:
        print("✅ No open positions found")
    
    # Check orders for common symbols
    total_orders = 0
    common_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT"]
    
    for symbol in common_symbols:
        orders = await get_open_orders_by_symbol(symbol)
        if orders:
            total_orders += len(orders)
            print(f"⚠️ {symbol} has {len(orders)} open orders")
    
    if total_orders == 0:
        print("✅ No orders found on common symbols")
    
    print("🔍 Verification complete")

if __name__ == "__main__":
    async def main():
        await emergency_cleanup()
        
        # Ask user if they want to verify
        verify = input("\n🔍 Would you like to verify the cleanup? (y/n): ")
        if verify.lower() == 'y':
            await verify_cleanup()
    
    asyncio.run(main())
