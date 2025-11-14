#!/usr/bin/env python3
"""
Fix for candle accumulation issue - ensures proper historical data loading
"""

import asyncio
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
import traceback

# Import your modules
from logger import log
from bybit_api import signed_request
from error_handler import send_error_to_telegram

# Enhanced live_candles structure with proper initialization
enhanced_live_candles = defaultdict(lambda: defaultdict(lambda: deque(maxlen=200)))

async def fetch_initial_candles(symbol: str, interval: str = '1', limit: int = 50, category: str = 'linear'):
    """
    Fetch initial historical candles for a symbol/interval
    """
    try:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": str(limit)
        }
        
        response = await signed_request("GET", "/v5/market/kline", params)
        
        if response.get("retCode") != 0:
            log(f"❌ Failed to fetch initial candles for {symbol} {interval}m: {response.get('retMsg')}", level="ERROR")
            return []
        
        raw_candles = response.get("result", {}).get("list", [])
        
        # Bybit returns newest first, we need oldest first
        candles = []
        for c in reversed(raw_candles):
            candles.append({
                "timestamp": int(c[0]),
                "open": str(c[1]),
                "high": str(c[2]),
                "low": str(c[3]),
                "close": str(c[4]),
                "volume": str(c[5])
            })
        
        return candles
        
    except Exception as e:
        log(f"❌ Error fetching initial candles for {symbol}: {e}", level="ERROR")
        return []

async def initialize_symbol_candles(symbol: str, intervals: list = ['1', '5', '15'], category: str = 'linear'):
    """
    Initialize a symbol with historical candle data for all intervals
    """
    log(f"📥 Initializing {symbol} with historical data...")
    
    for interval in intervals:
        # Fetch enough candles for RSI calculation (need at least 15 for RSI[14])
        candles = await fetch_initial_candles(symbol, interval, limit=50, category=category)
        
        if candles:
            # Clear any existing data and add historical candles
            enhanced_live_candles[symbol][interval].clear()
            enhanced_live_candles[symbol][interval].extend(candles)
            log(f"✅ {symbol} {interval}m initialized with {len(candles)} historical candles")
        else:
            log(f"⚠️ No historical data for {symbol} {interval}m", level="WARN")

async def enhanced_handle_stream(url, symbols, category, interval):
    """
    Enhanced websocket handler that ensures proper data accumulation
    """
    import websockets
    from scanner import symbol_category_map
    
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                # Filter symbols for this category
                category_symbols = [s for s in symbols if symbol_category_map.get(s) == category]
                
                if not category_symbols:
                    log(f"No symbols for category {category}", level="WARN")
                    return
                
                # Initialize all symbols with historical data BEFORE subscribing
                log(f"📥 Loading historical data for {len(category_symbols)} {category} symbols...")
                init_tasks = []
                for symbol in category_symbols:
                    init_tasks.append(initialize_symbol_candles(symbol, [interval], category))
                
                await asyncio.gather(*init_tasks)
                log(f"✅ Historical data loaded for {interval}m timeframe")
                
                # Now subscribe to live updates
                args = [f"kline.{interval}.{symbol}" for symbol in category_symbols]
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                log(f"📡 Subscribed to {len(args)} {category.upper()} @ {interval}m live updates")
                
                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(message)
                        
                        if "topic" not in data:
                            continue
                        
                        topic = data.get("topic", "")
                        symbol = topic.split(".")[-1] if topic else None
                        interval_from_topic = topic.split(".")[1] if topic and len(topic.split(".")) > 1 else None
                        
                        if not symbol or "data" not in data or not interval_from_topic:
                            continue
                        
                        candle_data = data["data"]
                        
                        # Process candle update
                        if isinstance(candle_data, list):
                            for k in candle_data:
                                update_candle(symbol, interval_from_topic, k)
                        elif isinstance(candle_data, dict):
                            update_candle(symbol, interval_from_topic, candle_data)
                        
                        current_count = len(enhanced_live_candles[symbol][interval_from_topic])
                        log(f"📈 {symbol} [{category}] @{interval_from_topic}m updated | total: {current_count} candles")
                        
                    except asyncio.TimeoutError:
                        log(f"⚠️ No data for {category} {interval}m in 30s", level="WARN")
                        # Don't break - just continue waiting
                        continue
                        
                    except Exception as e:
                        log(f"❌ Stream processing error: {e}", level="ERROR")
                        break
                
        except Exception as e:
            log(f"❌ Connection error for {category} {interval}m: {e}", level="ERROR")
            await asyncio.sleep(5)  # Reconnect delay

def update_candle(symbol: str, interval: str, candle_dict: dict):
    """
    Update or append candle to the deque
    """
    candle = {
        "timestamp": int(candle_dict["start"]),
        "open": candle_dict["open"],
        "high": candle_dict["high"],
        "low": candle_dict["low"],
        "close": candle_dict["close"],
        "volume": candle_dict["volume"]
    }
    
    candles = enhanced_live_candles[symbol][interval]
    
    # Check if this is an update to the last candle or a new candle
    if candles and candles[-1]["timestamp"] == candle["timestamp"]:
        # Update existing candle
        candles[-1] = candle
    else:
        # Append new candle
        candles.append(candle)

async def enhanced_stream_candles(symbols, intervals=['1', '5', '15']):
    """
    Enhanced streaming with proper initialization
    """
    futures_url = "wss://stream.bybit.com/v5/public/linear"
    
    # First, initialize ALL symbols with historical data
    log("🚀 Initializing all symbols with historical data...")
    init_tasks = []
    
    from scanner import symbol_category_map
    
    for symbol in symbols:
        category = symbol_category_map.get(symbol, 'linear')
        init_tasks.append(initialize_symbol_candles(symbol, intervals, category))
    
    # Initialize in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(init_tasks), batch_size):
        batch = init_tasks[i:i+batch_size]
        await asyncio.gather(*batch)
        if i + batch_size < len(init_tasks):
            await asyncio.sleep(1)  # Small delay between batches
    
    log(f"✅ Initialized {len(symbols)} symbols with historical data")
    
    # Now start websocket streams
    tasks = []
    for interval in intervals:
        tasks.append(asyncio.create_task(enhanced_handle_stream(futures_url, symbols, "linear", interval)))
    
    await asyncio.gather(*tasks)

async def test_enhanced_setup():
    """
    Test the enhanced setup to verify it fixes the RSI issue
    """
    print("\n🔧 TESTING ENHANCED CANDLE ACCUMULATION")
    print("=" * 60)
    
    try:
        from scanner import fetch_symbols
        from rsi import calculate_rsi_wilder
        
        # Get symbols
        symbols = await fetch_symbols()
        test_symbols = symbols[:5]  # Test with 5 symbols
        
        print(f"Testing with {len(test_symbols)} symbols: {test_symbols}")
        
        # Initialize with historical data
        for symbol in test_symbols:
            await initialize_symbol_candles(symbol)
        
        # Check candle counts and RSI calculation
        print("\n📊 Checking candle counts and RSI calculation:")
        for symbol in test_symbols:
            print(f"\n{symbol}:")
            for tf in ['1', '5', '15']:
                candles = list(enhanced_live_candles[symbol][tf])
                candle_count = len(candles)
                print(f"  {tf}m: {candle_count} candles", end="")
                
                if candle_count >= 15:
                    # Try RSI calculation
                    rsi_values = calculate_rsi_wilder(candles, period=14, symbol=f"{symbol}_{tf}m")
                    if rsi_values:
                        print(f" ✅ RSI: {rsi_values[-1]:.2f}")
                    else:
                        print(f" ❌ RSI calculation failed")
                else:
                    print(f" ⚠️ Not enough for RSI (need 15+)")
        
        print("\n✅ Enhanced setup complete! RSI warnings should be resolved.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

# Patch function to replace the original websocket handler
def apply_enhanced_candles_patch():
    """
    Apply the enhanced candles patch to fix RSI warnings
    """
    import websocket_candles
    
    # Replace the global live_candles with enhanced version
    websocket_candles.live_candles = enhanced_live_candles
    
    # Replace the stream function
    websocket_candles.stream_candles = enhanced_stream_candles
    
    log("✅ Applied enhanced candles patch - RSI warnings should be resolved")
    
    return enhanced_live_candles

if __name__ == "__main__":
    print("🚀 Running enhanced candle accumulation fix...")
    asyncio.run(test_enhanced_setup())
