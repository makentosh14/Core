# active_trade_scanner.py - FIXED VERSION
# Remove duplicate exit logic and use unified exit manager

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from logger import log, write_log
from bybit_api import signed_request
from error_handler import send_telegram_message

# IMPORT THE UNIFIED EXIT MANAGER - This replaces all duplicate exit logic
from unified_exit_manager import process_trade_exits

# Cache and state management
_active_trades_cache = {}
_cache_timestamp = 0
_cache_ttl = 10  # 10 seconds cache
_last_save_time = 0
_save_cooldown = 5  # 5 seconds between saves

PERSIST_PATH = "active_trades.json"

# Prevent duplicate processing
_processing_symbols = set()

def load_active_trades_directly():
    """Load active trades directly from file"""
    global _active_trades_cache, _cache_timestamp
    
    # Use cache if recent enough
    current_time = time.time()
    if _active_trades_cache and (current_time - _cache_timestamp) < _cache_ttl:
        return _active_trades_cache
    
    try:
        if os.path.exists(PERSIST_PATH):
            with open(PERSIST_PATH, 'r') as f:
                trades = json.load(f)
                
            # Filter out exited trades and return only active ones
            active_trades = {symbol: trade for symbol, trade in trades.items() 
                           if not trade.get("exited", False)}
            
            # Update cache
            _active_trades_cache = active_trades
            _cache_timestamp = current_time
            
            return active_trades
        else:
            return {}
            
    except Exception as e:
        log(f"❌ HF SCANNER: Error loading active trades: {e}", level="ERROR")
        return {}

async def scan_active_trades():
    """
    MAIN SCANNER FUNCTION - FIXED VERSION
    Uses unified exit manager to prevent double logic
    """

    global _active_trades_cache, _cache_timestamp

    try:
        # Load current active trades
        active_trades = load_active_trades_directly()
        
        if not active_trades:
            return
        
        log(f"🔍 HF SCANNER: Scanning {len(active_trades)} active trades...")
        
        # Process each trade
        for symbol, trade in list(active_trades.items()):
            try:
                # Skip if already being processed
                if symbol in _processing_symbols:
                    continue
                
                # Skip if trade is exited
                if trade.get("exited"):
                    continue
                
                # Add to processing set
                _processing_symbols.add(symbol)
                
                # Get current price
                current_price = await get_current_price(symbol)
                if current_price is None:
                    log(f"⚠️ Could not get price for {symbol}, skipping scan", level="WARN")
                    continue

                # Validate current_price
                if not isinstance(current_price, (int, float)) or current_price <= 0:
                    log(f"❌ Invalid price for {symbol}: {current_price}", level="ERROR")
                    continue

                # Calculate PnL safely (add this section)
                pnl_pct = await safe_calculate_pnl(symbol, trade, current_price)
                if pnl_pct is not None:
                    log(f"📊 {symbol} PnL: {pnl_pct:.2f}% (Price: {current_price})")
                
                # Get recent candles for analysis
                candles = await get_recent_candles(symbol)
                
                # USE UNIFIED EXIT MANAGER - Single source of truth
                trade_modified = await process_trade_exits(
                    symbol=symbol,
                    trade=trade,
                    current_price=current_price,
                    candles=candles
                )
                
                if trade_modified:
                    log(f"🔄 HF SCANNER: Trade {symbol} modified by exit manager")
                    # Update our local cache
                    if symbol in _active_trades_cache:
                        _active_trades_cache[symbol] = trade
                
            except Exception as e:
                log(f"❌ HF SCANNER: Error processing {symbol}: {e}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")
            finally:
                # Always remove from processing set
                _processing_symbols.discard(symbol)
        
        # Clear the cache to force reload next time
        _active_trades_cache.clear()
        _cache_timestamp = 0
        
    except Exception as e:
        log(f"❌ HF SCANNER: Error in scan_active_trades: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")

async def get_current_price(symbol):
    """
    FIXED VERSION - Get current price for symbol with comprehensive error handling
    """
    try:
        # Method 1: Try API first for reliability
        from bybit_api import signed_request
        result = await signed_request("GET", "/v5/market/tickers", {
            "category": "linear",
            "symbol": symbol
        })
        
        if result and result.get("retCode") == 0:
            tickers = result.get("result", {}).get("list", [])
            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                price = ticker.get("lastPrice")
                if price is not None:
                    price_float = float(price)
                    if price_float > 0:
                        return price_float
        
        # Method 2: Try live candles as backup
        try:
            from websocket_candles import live_candles
            if live_candles and symbol in live_candles:
                for tf in ['1', '5', '15']:
                    if tf in live_candles[symbol] and live_candles[symbol][tf]:
                        candles = live_candles[symbol][tf]
                        if candles and len(candles) > 0:
                            last_candle = candles[-1]
                            if 'close' in last_candle:
                                backup_price = float(last_candle['close'])
                                if backup_price > 0:
                                    log(f"🔄 Using live candles price for {symbol}: {backup_price}")
                                    return backup_price
        except Exception as backup_error:
            log(f"⚠️ Live candles backup failed for {symbol}: {backup_error}", level="WARN")
        
        log(f"❌ Failed to get price for {symbol}", level="ERROR")
        return None
        
    except Exception as e:
        log(f"❌ Error getting price for {symbol}: {e}", level="ERROR")
        return None

async def get_recent_candles(symbol, interval="1", limit=20):
    """Get recent candles for analysis"""
    try:
        result = await signed_request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
        
        if result.get("retCode") == 0:
            klines = result.get("result", {}).get("list", [])
            
            # Convert to standard format
            candles = []
            for kline in klines:
                candles.append({
                    "open": kline[1],
                    "high": kline[2],
                    "low": kline[3], 
                    "close": kline[4],
                    "volume": kline[5],
                    "timestamp": int(kline[0])
                })
            
            # Reverse to get chronological order (oldest first)
            return list(reversed(candles))
        
        return []
        
    except Exception as e:
        log(f"❌ HF SCANNER: Error getting candles for {symbol}: {e}", level="ERROR")
        return []

async def high_frequency_monitoring(live_candles=None):
    """Main high frequency monitoring loop"""
    while True:
        try:
            await scan_active_trades()
            
            # Wait 5 seconds before next scan
            await asyncio.sleep(5)
            
        except Exception as e:
            log(f"❌ HF SCANNER: Error in monitoring loop: {e}", level="ERROR")
            await asyncio.sleep(10)  # Wait longer on error

async def safe_calculate_pnl(symbol, trade, current_price):
    """
    FIXED VERSION - Calculate PnL with comprehensive null checks
    """
    try:
        # Validate inputs
        if not symbol:
            log(f"❌ Symbol is empty", level="ERROR")
            return None
            
        if not trade or not isinstance(trade, dict):
            log(f"❌ Invalid trade data for {symbol}", level="ERROR")
            return None
        
        if current_price is None:
            log(f"❌ Current price is None for {symbol}", level="ERROR")
            return None
        
        # Get and validate entry price
        entry_price = trade.get("entry_price")
        if entry_price is None:
            log(f"❌ Entry price is None for {symbol}", level="ERROR")
            return None
        
        # Convert to float and validate
        try:
            current_price = float(current_price)
            entry_price = float(entry_price)
        except (ValueError, TypeError) as e:
            log(f"❌ Price conversion error for {symbol}: {e}", level="ERROR")
            return None
        
        if current_price <= 0 or entry_price <= 0:
            log(f"❌ Invalid prices for {symbol}: current={current_price}, entry={entry_price}", level="ERROR")
            return None
        
        # Get direction
        direction = trade.get("direction", "").lower().strip()
        if direction not in ["long", "short"]:
            log(f"❌ Invalid direction for {symbol}: '{direction}'", level="ERROR")
            return None
        
        # Calculate PnL percentage
        if direction == "long":
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:  # short
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Validate result
        if pnl_pct is None:
            log(f"❌ PnL calculation returned None for {symbol}", level="ERROR")
            return None
        
        return float(pnl_pct)
        
    except Exception as e:
        log(f"❌ Error calculating PnL for {symbol}: {e}", level="ERROR")
        return None

# REMOVED FUNCTIONS - These are now handled by unified_exit_manager.py:
# - handle_trailing_sl_exit()
# - check_tp1_hit()
# - check_trailing_sl_hit()
# - handle_tp1_hit()
# - handle_trailing_stop()

if __name__ == "__main__":
    asyncio.run(high_frequency_monitoring())
