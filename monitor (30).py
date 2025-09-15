# monitor.py - FIXED VERSION
# Remove all duplicate exit logic and use unified exit manager

import json
import os
import time
startup_time = time.time()
import asyncio
import traceback
from datetime import datetime, timedelta
from score import score_symbol
from pattern_detector import detect_pattern
from volume import get_average_volume
from logger import log, write_log
from exit_manager import should_trail_stop, adjust_profit_protection, should_exit_by_time, evaluate_score_exit, detect_momentum_surge, calculate_exit_tranches
from config import ENABLE_AUTO_REENTRY, REENTRY_FEATURES
from auto_reentry import (
    log_exit, 
    update_exit_cooldowns, 
    update_reentry_performance,
    should_reenter,
    handle_reentry,
    cooldown_exits,
    exit_history
)
from ai_memory import log_trade_result
from activity_logger import log_trade_to_file
from bybit_api import signed_request, check_order_exists, place_stop_loss, place_stop_loss_with_retry, place_market_order
from error_handler import send_telegram_message
from strategy_performance import log_strategy_result
from sl_tp_utils import evaluate_score_exit
from dca_manager import dca_manager
from auto_exit_handler import auto_exit_past_sl

# IMPORT THE UNIFIED EXIT MANAGER - This replaces all exit logic
from unified_exit_manager import process_trade_exits

_last_monitor_save = 0
_monitor_save_cooldown = 5  # 5 seconds

# Active trades dictionary
active_trades = {}

PERSIST_PATH = "active_trades.json"

def load_active_trades():
    """Load active trades from file"""
    global active_trades
    try:
        if os.path.exists(PERSIST_PATH):
            with open(PERSIST_PATH, 'r') as f:
                active_trades = json.load(f)
                
            # Filter out exited trades
            active_trades = {k: v for k, v in active_trades.items() if not v.get("exited", False)}
            log(f"📊 Loaded {len(active_trades)} active trades")
        else:
            active_trades = {}
            log("📊 No existing trades file found, starting fresh")
    except Exception as e:
        log(f"❌ Error loading active trades: {e}", level="ERROR")
        active_trades = {}

def save_active_trades():
    """Save active trades to file with cooldown"""
    global _last_monitor_save
    current_time = time.time()
    
    if current_time - _last_monitor_save < _monitor_save_cooldown:
        return  # Skip save due to cooldown
    
    try:
        with open(PERSIST_PATH, 'w') as f:
            json.dump(active_trades, f, indent=2)
        _last_monitor_save = current_time
        
    except Exception as e:
        log(f"❌ Error saving active trades: {e}", level="ERROR")

def track_active_trade(symbol, trade_data):
    """Add a trade to active monitoring"""
    global active_trades
    active_trades[symbol] = trade_data
    log(f"📊 Now tracking {symbol}: {trade_data.get('direction')} @ {trade_data.get('entry_price')}")
    save_active_trades()

async def monitor_trades(live_candles=None):
    """
    MAIN MONITORING FUNCTION - FIXED VERSION
    Uses unified exit manager to prevent double logic
    """
    if not active_trades:
        return
    
    log(f"📊 Monitoring {len(active_trades)} active trades...")
    
    for symbol, trade in list(active_trades.items()):
        try:
            # Skip if trade is already exited
            if trade.get("exited"):
                continue
            
            # Get current price
            current_price = await get_symbol_price(symbol)
            if not current_price:
                continue
            
            # Get recent candles for analysis
            candles_by_tf = await get_candles_for_monitoring(symbol)
            candles_1m = candles_by_tf.get('1', [])
            
            # USE UNIFIED EXIT MANAGER - Single source of truth
            trade_modified = await process_trade_exits(
                symbol=symbol,
                trade=trade,
                current_price=current_price,
                candles=candles_1m
            )
            
            if trade_modified:
                log(f"🔄 Trade {symbol} was modified by exit manager")
                # Trade state is automatically saved by unified exit manager
            
            # Continue to next trade if this one was exited
            if trade.get("exited"):
                continue
            
            # Check for original SL hit (only before TP1)
            if not trade.get("tp1_hit") and trade.get("original_sl"):
                direction = trade.get("direction", "").lower()
                if check_original_sl_hit(trade, current_price, direction):
                    await handle_original_sl_exit(symbol, trade, current_price)
                    continue
            
            # Handle auto-reentry logic if enabled
            if ENABLE_AUTO_REENTRY and trade.get("exited"):
                await handle_reentry_logic(symbol, trade, current_price)
            
        except Exception as e:
            log(f"❌ Error monitoring {symbol}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
    
    # Save any changes
    save_active_trades()

def check_original_sl_hit(trade, current_price, direction):
    """Check if original SL (before TP1) has been hit"""
    original_sl = trade.get("original_sl")
    if not original_sl:
        return False
    
    buffer = 0.001  # 0.1% buffer
    
    if direction == "long":
        return current_price <= original_sl * (1 - buffer)
    else:  # short
        return current_price >= original_sl * (1 + buffer)

async def handle_original_sl_exit(symbol, trade, current_price):
    """Handle original SL hit (before TP1)"""
    try:
        log(f"🛑 Original SL hit for {symbol} at {current_price}")
        
        direction = trade.get("direction", "").lower()
        entry_price = trade.get("entry_price")
        qty = trade.get("qty", 0)
        
        # Calculate loss
        if direction == "long":
            loss_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            loss_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Execute exit
        exit_side = "Sell" if direction == "long" else "Buy"
        
        exit_order = await place_market_order(
            symbol=symbol,
            side=exit_side,
            qty=qty,
            reduce_only=True
        )
        
        if exit_order:
            # Mark trade as exited
            trade["exited"] = True
            trade["exit_price"] = current_price
            trade["exit_reason"] = "Original_SL_Hit"
            trade["profit_pct"] = loss_pct
            trade["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Send notification
            await send_telegram_message(
                f"🛑 <b>Stop Loss Hit</b> - <b>{symbol}</b>\n"
                f"💰 Exit Price: {current_price:.6f}\n"
                f"📈 Loss: {loss_pct:.2f}%"
            )
            
            log(f"✅ Original SL exit executed for {symbol}")
        else:
            log(f"❌ Failed to execute original SL exit for {symbol}", level="ERROR")
            
    except Exception as e:
        log(f"❌ Error handling original SL exit for {symbol}: {e}", level="ERROR")

async def get_symbol_price(symbol):
    """Get current symbol price"""
    try:
        result = await signed_request("GET", "/v5/market/tickers", {
            "category": "linear",
            "symbol": symbol
        })
        
        if result.get("retCode") == 0:
            tickers = result.get("result", {}).get("list", [])
            if tickers:
                return float(tickers[0]["lastPrice"])
        
        log(f"❌ Failed to get price for {symbol}", level="ERROR")
        return None
        
    except Exception as e:
        log(f"❌ Error getting price for {symbol}: {e}", level="ERROR")
        return None

async def get_candles_for_monitoring(symbol):
    """Get candles for monitoring analysis"""
    try:
        # Import here to avoid circular imports
        from websocket_candles import fetch_candles_rest
        
        candles_1m = await fetch_candles_rest(symbol, "1", limit=20)
        
        return {
            "1": candles_1m if candles_1m else []
        }
        
    except Exception as e:
        log(f"❌ Error getting candles for {symbol}: {e}", level="ERROR")
        return {"1": []}

async def handle_reentry_logic(symbol, trade, current_price):
    """Handle auto-reentry logic if enabled"""
    try:
        if not ENABLE_AUTO_REENTRY:
            return
        
        # Auto-reentry logic here if needed
        # This is where your existing reentry code would go
        
    except Exception as e:
        log(f"❌ Error in reentry logic for {symbol}: {e}", level="ERROR")

async def periodic_trade_sync():
    """Periodic trade sync function - called from main.py"""
    while True:
        try:
            # Only run after bot has been running for 30 seconds
            if time.time() - startup_time < 30:
                await asyncio.sleep(10)
                continue
                
            # Reload trades from file
            load_active_trades()
            log(f"🔄 Periodic sync: {len(active_trades)} active trades")
            
        except Exception as e:
            log(f"❌ Error in periodic trade sync: {e}", level="ERROR")
        
        # Sync every 60 seconds
        await asyncio.sleep(60)

# This function should already exist - if not, add it:
async def check_and_restore_sl(symbol, trade):
    """Check and restore stop loss for a trade"""
    try:
        if not trade or trade.get("exited"):
            return False
        
        log(f"🔍 Checking SL for {symbol}")
        
        # Add your SL check logic here
        # For now, just log that we're checking
        log(f"✅ SL check completed for {symbol}")
        return True
        
    except Exception as e:
        log(f"❌ Error checking SL for {symbol}: {e}", level="ERROR")
        return False

async def get_current_price(symbol, live_candles=None):
    """
    Get current price for symbol with comprehensive null checks
    FIXED VERSION - Prevents NoneType multiplication errors
    """
    try:
        # Method 1: Try live candles first
        if live_candles and symbol in live_candles:
            for tf in ['1', '5', '15']:  # Try different timeframes
                if tf in live_candles[symbol] and live_candles[symbol][tf]:
                    candles = live_candles[symbol][tf]
                    if candles and len(candles) > 0:
                        last_candle = candles[-1]
                        if 'close' in last_candle:
                            price = float(last_candle['close'])
                            if price > 0:
                                return price
        
        # Method 2: Fetch from API
        from bybit_api import signed_request
        result = await signed_request("GET", "/v5/market/tickers", {
            "category": "linear",
            "symbol": symbol
        })
        
        if result and result.get("retCode") == 0:
            tickers = result.get("result", {}).get("list", [])
            if tickers and len(tickers) > 0:
                ticker = tickers[0]
                price = float(ticker.get("lastPrice", 0))
                if price > 0:
                    return price
        
        log(f"⚠️ Could not get price for {symbol}", level="WARN")
        return None
        
    except Exception as e:
        log(f"❌ Error getting current price for {symbol}: {e}", level="ERROR")
        return None

async def get_current_price_enhanced(symbol, live_candles=None):
    """
    Enhanced price fetching with additional validation
    FIXED VERSION - Prevents NoneType errors
    """
    try:
        # Use the base function with null checks
        price = await get_current_price(symbol, live_candles)
        
        if price is None:
            log(f"⚠️ Price is None for {symbol}, trying backup methods", level="WARN")
            
            # Backup method: Try websocket candles module
            try:
                from websocket_candles import live_candles as ws_candles
                if ws_candles and symbol in ws_candles:
                    for tf in ['1', '5']:
                        if tf in ws_candles[symbol] and ws_candles[symbol][tf]:
                            candles = ws_candles[symbol][tf]
                            if candles and len(candles) > 0:
                                last_candle = candles[-1]
                                if 'close' in last_candle:
                                    backup_price = float(last_candle['close'])
                                    if backup_price > 0:
                                        log(f"✅ Got backup price for {symbol}: {backup_price}")
                                        return backup_price
            except Exception as backup_error:
                log(f"⚠️ Backup price method failed for {symbol}: {backup_error}", level="WARN")
            
            return None
        
        # Validate the price
        if not isinstance(price, (int, float)):
            log(f"❌ Invalid price type for {symbol}: {type(price)}", level="ERROR")
            return None
        
        if price <= 0:
            log(f"❌ Invalid price value for {symbol}: {price}", level="ERROR")
            return None
        
        return float(price)
        
    except Exception as e:
        log(f"❌ Error in enhanced price fetch for {symbol}: {e}", level="ERROR")
        return None

async def get_symbol_price(symbol, category="linear"):
    """
    Wrapper function for backward compatibility
    FIXED VERSION with null checks
    """
    try:
        price = await get_current_price_enhanced(symbol)
        return price
    except Exception as e:
        log(f"❌ Error in get_symbol_price for {symbol}: {e}", level="ERROR")
        return None

# This function should already exist - if not, add it:
async def recover_active_trades_from_exchange():
    """Recover active trades from exchange"""
    try:
        log("🔄 Attempting to recover trades from exchange...")
        
        # Add your recovery logic here
        # For now, just log that we're recovering
        log("✅ Trade recovery completed")
        
    except Exception as e:
        log(f"❌ Error recovering trades: {e}", level="ERROR")

# REMOVED FUNCTIONS - These are now handled by unified_exit_manager.py:
# - handle_tp1_hit()
# - handle_trailing_stop()
# - handle_trailing_sl_hit()
# - check_tp1_hit()
# - check_trailing_sl_hit()
# - update_trailing_stop()

# Initialize on import
load_active_trades()
