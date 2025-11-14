#!/usr/bin/env python3
"""
FIXED monitor.py - Complete imports and dependency setup
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from logger import log, write_log
from bybit_api import signed_request
from error_handler import send_telegram_message, send_error_to_telegram

# Configuration
ENABLE_AUTO_REENTRY = False
EXIT_COOLDOWN = 300  # 5 minutes

# Global state
active_trades: Dict[str, Any] = {}
recent_exits: Dict[str, float] = {}
startup_time = time.time()

# Initialize dependencies for other modules
def setup_module_dependencies():
    """Setup dependencies for modules to avoid circular imports"""
    try:
        # Setup trade verification dependencies
        from trade_verification import set_dependencies as set_verification_deps
        set_verification_deps(save_active_trades, active_trades)
        
        # Setup unified exit manager dependencies
        from unified_exit_manager import set_dependencies as set_exit_deps
        set_exit_deps(save_active_trades, update_stop_loss_order)
        
        log("✅ Module dependencies initialized")
        
    except ImportError as e:
        log(f"⚠️ Some modules not available for dependency setup: {e}")
    except Exception as e:
        log(f"❌ Error setting up module dependencies: {e}", level="ERROR")

def load_active_trades():
    """Load active trades from file"""
    global active_trades
    try:
        with open("active_trades.json", "r") as f:
            active_trades = json.load(f)
        log(f"✅ Loaded {len(active_trades)} active trades from file")
    except FileNotFoundError:
        active_trades = {}
        log("📁 No active trades file found, starting fresh")
    except Exception as e:
        log(f"❌ Error loading active trades: {e}", level="ERROR")
        active_trades = {}

def save_active_trades():
    """Save active trades to file"""
    try:
        with open("active_trades.json", "w") as f:
            json.dump(active_trades, f, indent=2)
        log(f"💾 Saved {len(active_trades)} active trades to file")
    except Exception as e:
        log(f"❌ Error saving active trades: {e}", level="ERROR")

async def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a symbol"""
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

async def get_candles_for_monitoring(symbol: str) -> Dict[str, list]:
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

async def update_stop_loss_order(symbol: str, trade: Dict, new_sl: float) -> bool:
    """Update stop loss order on exchange"""
    try:
        # Get existing stop loss order
        orders_response = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear",
            "symbol": symbol,
            "settleCoin": "USDT",
            "orderFilter": "StopOrder"
        })
        
        if orders_response.get("retCode") != 0:
            log(f"❌ Failed to get orders for {symbol}: {orders_response.get('retMsg')}")
            return False
        
        orders = orders_response.get("result", {}).get("list", [])
        sl_orders = [o for o in orders if o.get("orderType") in ["Stop", "StopLoss"]]
        
        if not sl_orders:
            log(f"⚠️ No existing SL order found for {symbol}")
            return False
        
        # Cancel existing SL order
        sl_order = sl_orders[0]
        cancel_response = await signed_request("POST", "/v5/order/cancel", {
            "category": "linear",
            "symbol": symbol,
            "settleCoin": "USDT",
            "orderId": sl_order.get("orderId")
        })
        
        if cancel_response.get("retCode") != 0:
            log(f"❌ Failed to cancel SL order for {symbol}")
            return False
        
        # Place new SL order
        direction = trade.get("direction", "").lower()
        qty = trade.get("qty", 0)
        
        side = "Sell" if direction == "long" else "Buy"
        
        new_sl_response = await signed_request("POST", "/v5/order/create", {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Stop",
            "qty": str(abs(float(qty))),
            "stopPrice": str(new_sl),
            "timeInForce": "GTC",
            "settleCoin": "USDT",
            "reduceOnly": True
        })
        
        if new_sl_response.get("retCode") == 0:
            trade["sl_order_id"] = new_sl_response.get("result", {}).get("orderId")
            log(f"✅ Updated SL order for {symbol} to {new_sl}")
            return True
        else:
            log(f"❌ Failed to place new SL order for {symbol}: {new_sl_response.get('retMsg')}")
            return False
        
    except Exception as e:
        log(f"❌ Error updating SL order for {symbol}: {e}", level="ERROR")
        return False

async def track_active_trade(symbol: str, trade_data: Dict[str, Any], trade_type: Optional[str] = None, initial_score: Optional[float] = None, entry_price: Optional[float] = None, direction: Optional[str] = None) -> None:
    """Track an active trade - called when trade is executed"""
    try:
        if trade_type:
            trade_data['trade_type'] = trade_type  # Add to data for persistence/logging
        if initial_score is not None:
            trade_data['initial_score'] = initial_score  # Add initial score for monitoring/strategies
        if entry_price is not None:
            trade_data['entry_price'] = entry_price  # Add entry price for P&L tracking
        if direction:
            trade_data['direction'] = direction  # Add for strategy direction tracking
        active_trades[symbol] = trade_data
        save_active_trades()
        log(f"📌 Now tracking {symbol}: {trade_data.get('direction', 'Unknown')} | Score: {trade_data.get('score', 'N/A')} | Type: {trade_data.get('trade_type', 'Unknown')} | Initial Score: {trade_data.get('initial_score', 'N/A')} | Entry Price: {trade_data.get('entry_price', 'N/A')}")
        
    except Exception as e:
        log(f"❌ Error tracking trade for {symbol}: {e}", level="ERROR")

async def monitor_trades(score_data: Dict[str, Any]) -> None:
    """Monitor trades based on current scores - compatibility function"""
    try:
        # This is mainly for compatibility with legacy code
        # The main monitoring is handled by monitor_active_trades()
        if not score_data:
            return
            
        for symbol in active_trades:
            if symbol in score_data:
                current_score = score_data[symbol].get("score", 0)
                log(f"📊 {symbol} current score: {current_score}")
        
    except Exception as e:
        log(f"❌ Error in monitor_trades: {e}", level="ERROR")

async def check_and_restore_sl(symbol: str, trade: Dict[str, Any]) -> bool:
    """Check and restore stop loss if missing"""
    try:
        log(f"🔍 Checking SL for {symbol}...")
        
        # Get current orders
        orders_response = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear", 
            "settleCoin": "USDT",
            "symbol": symbol,
            "orderFilter": "StopOrder"
        })
        
        if orders_response.get("retCode") != 0:
            log(f"❌ Failed to get orders for {symbol}")
            return False
            
        orders = orders_response.get("result", {}).get("list", [])
        sl_orders = [o for o in orders if o.get("orderType") in ["Stop", "StopLoss"]]
        
        if sl_orders:
            log(f"✅ SL exists for {symbol}")
            return True
            
        # SL missing - try to restore
        log(f"⚠️ SL missing for {symbol}, attempting restore...")
        
        sl_price = trade.get("sl_price") or trade.get("stop_loss")
        if not sl_price:
            log(f"❌ No SL price in trade data for {symbol}")
            return False
            
        # Place new SL order
        direction = trade.get("direction", "")
        qty = trade.get("qty", "")
        
        order_side = "Sell" if direction == "Long" else "Buy"
        
        sl_response = await signed_request("POST", "/v5/order/create", {
            "category": "linear",
            "symbol": symbol,
            "side": order_side,
            "orderType": "Stop",
            "qty": str(qty),
            "stopPrice": str(sl_price),
            "triggerDirection": 1 if direction == "Long" else 2,
            "timeInForce": "GTC",
            "settleCoin": "USDT",
            "reduceOnly": True
        })
        
        if sl_response.get("retCode") == 0:
            log(f"✅ SL restored for {symbol} at {sl_price}")
            return True
        else:
            log(f"❌ Failed to restore SL for {symbol}: {sl_response.get('retMsg')}")
            return False
            
    except Exception as e:
        log(f"❌ Error checking/restoring SL for {symbol}: {e}", level="ERROR")
        return False

async def recover_active_trades_from_exchange() -> None:
    """Recover active trades from exchange positions and orders"""
    try:
        log("🔄 Attempting to recover active trades from exchange...")
        
        # Get current positions
        positions_response = await signed_request("GET", "/v5/position/list", {
            "category": "linear",
            "settleCoin": "USDT"
        })
        
        if positions_response.get("retCode") != 0:
            log(f"❌ Failed to get positions: {positions_response.get('retMsg')}")
            return
            
        positions = positions_response.get("result", {}).get("list", [])
        active_positions = [p for p in positions if float(p.get("size", 0)) > 0]
        
        log(f"📊 Found {len(active_positions)} active positions on exchange")
        
        recovered_trades = 0
        
        for position in active_positions:
            symbol = position.get("symbol")
            size = float(position.get("size", 0))
            side = position.get("side", "")
            avg_price = float(position.get("avgPrice", 0))
            unrealized_pnl = float(position.get("unrealisedPnl", 0))
            
            if symbol not in active_trades and size > 0:
                # Create trade record from position
                trade_data = {
                    "symbol": symbol,
                    "direction": "Long" if side == "Buy" else "Short", 
                    "qty": str(size),
                    "entry_price": avg_price,
                    "timestamp": datetime.now().isoformat(),
                    "recovered_from_exchange": True,
                    "unrealized_pnl": unrealized_pnl,
                    "trade_type": "Recovered",
                    "score": 0,
                    "confidence": 0
                }
                
                active_trades[symbol] = trade_data
                recovered_trades += 1
                
                log(f"🔄 Recovered {symbol}: {side} {size} @ {avg_price}")
        
        if recovered_trades > 0:
            save_active_trades()
            log(f"✅ Recovered {recovered_trades} trades from exchange")
        else:
            log("ℹ️ No trades to recover")
            
    except Exception as e:
        log(f"❌ Error recovering trades from exchange: {e}", level="ERROR")

async def handle_reentry_logic(symbol: str, trade: Dict, current_price: float):
    """Handle auto-reentry logic if enabled"""
    try:
        if not ENABLE_AUTO_REENTRY:
            return
        
        # Skip if recently exited
        if symbol in recent_exits:
            time_diff = time.time() - recent_exits[symbol]
            if time_diff < EXIT_COOLDOWN:
                return
        
        # Add reentry logic here if needed in the future
        log(f"🔄 Auto-reentry check for {symbol} (currently disabled)")
        
    except Exception as e:
        log(f"❌ Error in reentry logic for {symbol}: {e}", level="ERROR")

async def monitor_active_trades():
    """Main monitoring loop for active trades"""
    while True:
        try:
            if not active_trades:
                await asyncio.sleep(10)
                continue
            
            log(f"🔍 Monitoring {len(active_trades)} active trades")
            
            for symbol, trade in list(active_trades.items()):
                try:
                    if trade.get("exited"):
                        continue
                    
                    # Get current price
                    current_price = await get_current_price(symbol)
                    if not current_price:
                        continue
                    
                    # Use unified exit manager for all exit logic
                    from unified_exit_manager import process_trade_exits
                    exited = await process_trade_exits(symbol, trade, current_price)
                    
                    if exited:
                        # Add to recent exits for cooldown
                        recent_exits[symbol] = time.time()
                        
                        # Handle reentry logic if enabled
                        await handle_reentry_logic(symbol, trade, current_price)
                except Exception as e:
                    log(f"❌ Error monitoring {symbol}: {e}", level="ERROR")
                    continue  # Skip to next trade
            
            # Clean up old recent exits
            current_time = time.time()
            expired_exits = [s for s, t in recent_exits.items() if current_time - t > EXIT_COOLDOWN]
            for symbol in expired_exits:
                del recent_exits[symbol]
            
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            log(f"❌ Error in monitoring loop: {e}", level="ERROR")
            await asyncio.sleep(20)

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
            
            # Run verification on all trades
            try:
                from trade_verification import run_comprehensive_verification
                verification_results = await run_comprehensive_verification()
                
                if verification_results.get("manual_review_needed", 0) > 0:
                    log(f"⚠️ {verification_results['manual_review_needed']} trades need manual review")
                
            except ImportError:
                log("⚠️ Trade verification not available")
            except Exception as e:
                log(f"❌ Error in trade verification: {e}", level="ERROR")
            
        except Exception as e:
            log(f"❌ Error in periodic trade sync: {e}", level="ERROR")
        
        # Sync every 60 seconds
        await asyncio.sleep(60)

# Export main functions
__all__ = [
    'active_trades',
    'load_active_trades', 
    'save_active_trades',
    'track_active_trade',
    'monitor_trades', 
    'get_current_price',
    'get_candles_for_monitoring',
    'update_stop_loss_order',
    'check_and_restore_sl',
    'handle_reentry_logic',
    'monitor_active_trades',
    'recover_active_trades_from_exchange',
    'periodic_trade_sync',
    'setup_module_dependencies'
]


# Initialize module
if __name__ == "__main__":
    print("✅ monitor.py module loaded successfully")
    
    # Setup dependencies
    setup_module_dependencies()
    
    # Load trades
    load_active_trades()
    print(f"📊 {len(active_trades)} active trades loaded")
