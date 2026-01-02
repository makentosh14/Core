#!/usr/bin/env python3
"""
FIXED monitor.py - Complete imports and dependency setup
Fixed: track_active_trade() signature to match trade_executor.py calls
Fixed: Added monitor_active_trades() task start
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
            active_trades.update(json.load(f))  # Use update to preserve reference
        log(f"✅ Loaded {len(active_trades)} active trades from file")
    except FileNotFoundError:
        log("📁 No active trades file found, starting fresh")
    except Exception as e:
        log(f"❌ Error loading active trades: {e}", level="ERROR")


def save_active_trades():
    """Save active trades to file"""
    try:
        with open("active_trades.json", "w") as f:
            json.dump(active_trades, f, indent=2, default=str)
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
                return float(tickers[0].get("lastPrice", 0))
        return None
        
    except Exception as e:
        log(f"❌ Error getting price for {symbol}: {e}", level="ERROR")
        return None


async def get_candles_for_monitoring(symbol: str, interval: str = "5", limit: int = 20):
    """Get candles for monitoring calculations"""
    try:
        result = await signed_request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        })
        
        if result.get("retCode") == 0:
            return result.get("result", {}).get("list", [])
        return []
        
    except Exception as e:
        log(f"❌ Error getting candles for {symbol}: {e}", level="ERROR")
        return []


async def update_stop_loss_order(symbol: str, trade: Dict, new_sl: float) -> bool:
    """Update stop loss order for a trade"""
    try:
        # Cancel existing SL order if present
        sl_order_id = trade.get("sl_order_id")
        if sl_order_id:
            cancel_response = await signed_request("POST", "/v5/order/cancel", {
                "category": "linear",
                "symbol": symbol,
                "orderId": sl_order_id
            })
            
            if cancel_response.get("retCode") != 0:
                log(f"⚠️ Could not cancel old SL order for {symbol}: {cancel_response.get('retMsg')}")
        
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
            trade["sl"] = new_sl  # Update current SL
            log(f"✅ Updated SL order for {symbol} to {new_sl}")
            save_active_trades()
            return True
        else:
            log(f"❌ Failed to place new SL order for {symbol}: {new_sl_response.get('retMsg')}")
            return False
        
    except Exception as e:
        log(f"❌ Error updating SL order for {symbol}: {e}", level="ERROR")
        return False


async def check_and_restore_sl(symbol: str, trade: Dict) -> bool:
    """Check if SL order exists and restore if missing"""
    try:
        # Get open orders for symbol
        orders_response = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear",
            "symbol": symbol
        })
        
        if orders_response.get("retCode") != 0:
            log(f"❌ Failed to get orders for {symbol}: {orders_response.get('retMsg')}")
            return False
        
        orders = orders_response.get("result", {}).get("list", [])
        
        # Check if SL order exists
        sl_exists = any(
            order.get("orderType") == "Stop" and order.get("reduceOnly") 
            for order in orders
        )
        
        if sl_exists:
            log(f"✅ SL order exists for {symbol}")
            return True
        
        # SL missing - restore it
        log(f"⚠️ SL order missing for {symbol} - restoring...")
        
        sl_price = trade.get("sl") or trade.get("original_sl")
        if not sl_price:
            log(f"❌ No SL price found for {symbol}")
            return False
        
        return await update_stop_loss_order(symbol, trade, sl_price)
        
    except Exception as e:
        log(f"❌ Error checking/restoring SL for {symbol}: {e}", level="ERROR")
        return False


# ============================================================
# FIXED: track_active_trade() with correct signature
# ============================================================
async def track_active_trade(
    symbol: str,
    trade_data: Dict[str, Any] = None,  # Made optional with default None
    trade_type: Optional[str] = None,
    initial_score: Optional[float] = None,
    entry_price: Optional[float] = None,
    direction: Optional[str] = None,
    # NEW parameters that trade_executor.py passes:
    trailing_pct: Optional[float] = None,
    tp1_target: Optional[float] = None,
    tp1_pct: Optional[float] = None,
    sl: Optional[float] = None,
    sl_order_id: Optional[str] = None,
    qty: Optional[float] = None,
    # Additional optional params
    score: Optional[float] = None,
    confidence: Optional[float] = None
) -> None:
    """
    Track an active trade - called when trade is executed
    
    Can be called in two ways:
    1. With trade_data dict: track_active_trade(symbol, trade_data_dict)
    2. With individual params: track_active_trade(symbol=x, direction=y, ...)
    """
    try:
        # If trade_data is None, create it from individual parameters
        if trade_data is None:
            trade_data = {}
        
        # Add all parameters to trade_data
        if trade_type:
            trade_data['trade_type'] = trade_type
        if initial_score is not None:
            trade_data['initial_score'] = initial_score
        if score is not None:
            trade_data['score'] = score
        if entry_price is not None:
            trade_data['entry_price'] = entry_price
        if direction:
            trade_data['direction'] = direction
        if trailing_pct is not None:
            trade_data['trailing_pct'] = trailing_pct
        if tp1_target is not None:
            trade_data['tp1_target'] = tp1_target
        if tp1_pct is not None:
            trade_data['tp1_pct'] = tp1_pct
        if sl is not None:
            trade_data['sl'] = sl
            trade_data['original_sl'] = sl  # Keep original for reference
        if sl_order_id:
            trade_data['sl_order_id'] = sl_order_id
        if qty is not None:
            trade_data['qty'] = qty
        if confidence is not None:
            trade_data['confidence'] = confidence
        
        # Add timestamp if not present
        if 'timestamp' not in trade_data:
            trade_data['timestamp'] = datetime.now().isoformat()
        
        # Add symbol to trade_data
        trade_data['symbol'] = symbol
        
        # Initialize tracking fields
        if 'exited' not in trade_data:
            trade_data['exited'] = False
        if 'tp1_hit' not in trade_data:
            trade_data['tp1_hit'] = False
        if 'trailing_active' not in trade_data:
            trade_data['trailing_active'] = False
        
        # Store in active_trades
        active_trades[symbol] = trade_data
        save_active_trades()
        
        log(f"📌 Now tracking {symbol}: {trade_data.get('direction', 'Unknown')} | "
            f"Entry: {trade_data.get('entry_price', 'N/A')} | "
            f"SL: {trade_data.get('sl', 'N/A')} | "
            f"TP1: {trade_data.get('tp1_target', 'N/A')} | "
            f"Type: {trade_data.get('trade_type', 'Unknown')}")
        
    except Exception as e:
        log(f"❌ Error tracking trade for {symbol}: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")


async def monitor_trades(score_data: Dict[str, Any]) -> None:
    """Monitor trades based on current scores - compatibility function"""
    try:
        for symbol, data in score_data.items():
            if symbol in active_trades and not active_trades[symbol].get("exited"):
                trade = active_trades[symbol]
                current_score = data.get("score", 0)
                
                # Update score history
                if "score_history" not in trade:
                    trade["score_history"] = []
                trade["score_history"].append(current_score)
                
                # Keep only last 10 scores
                if len(trade["score_history"]) > 10:
                    trade["score_history"] = trade["score_history"][-10:]
                    
    except Exception as e:
        log(f"❌ Error in monitor_trades: {e}", level="ERROR")


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
                    "confidence": 0,
                    "exited": False
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
    log("🔍 Starting monitor_active_trades() loop...")
    
    # Import unified exit manager once at start
    try:
        from unified_exit_manager import process_trade_exits
        exit_manager_available = True
    except ImportError:
        log("⚠️ unified_exit_manager not available, using basic monitoring")
        exit_manager_available = False
    
    while True:
        try:
            # Filter only non-exited trades
            active = {k: v for k, v in active_trades.items() if not v.get("exited", False)}
            
            if not active:
                await asyncio.sleep(10)
                continue
            
            log(f"🔍 Monitoring {len(active)} active trades")
            
            for symbol, trade in list(active.items()):
                try:
                    # Get current price
                    current_price = await get_current_price(symbol)
                    if not current_price:
                        log(f"⚠️ Could not get price for {symbol}")
                        continue
                    
                    # Check and restore SL if missing
                    await check_and_restore_sl(symbol, trade)
                    
                    # Use unified exit manager for all exit logic
                    if exit_manager_available:
                        exited = await process_trade_exits(symbol, trade, current_price)
                        
                        if exited:
                            # Add to recent exits for cooldown
                            recent_exits[symbol] = time.time()
                            
                            # Handle reentry logic if enabled
                            await handle_reentry_logic(symbol, trade, current_price)
                    else:
                        # Basic P&L logging if exit manager not available
                        entry_price = trade.get("entry_price", 0)
                        direction = trade.get("direction", "").lower()
                        
                        if entry_price and direction:
                            if direction == "long":
                                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                            else:
                                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                            
                            log(f"📊 {symbol}: Price={current_price:.4f} | P&L={pnl_pct:+.2f}%")
                            
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
            log(traceback.format_exc(), level="ERROR")
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
            
            active_count = sum(1 for t in active_trades.values() if not t.get("exited", False))
            log(f"🔄 Periodic sync: {active_count} active trades (total: {len(active_trades)})")
            
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


async def debug_stop_loss(symbol: str):
    """Debug function to check stop loss status"""
    try:
        if symbol not in active_trades:
            await send_telegram_message(f"❌ {symbol} not in active trades")
            return
            
        trade = active_trades[symbol]
        
        # Get open orders
        orders_response = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear",
            "symbol": symbol
        })
        
        orders = orders_response.get("result", {}).get("list", []) if orders_response.get("retCode") == 0 else []
        
        # Get position
        position_response = await signed_request("GET", "/v5/position/list", {
            "category": "linear",
            "symbol": symbol
        })
        
        positions = position_response.get("result", {}).get("list", []) if position_response.get("retCode") == 0 else []
        
        msg = f"🔍 <b>Debug Report for {symbol}</b>\n\n"
        msg += f"<b>Trade Data:</b>\n"
        msg += f"• Direction: {trade.get('direction')}\n"
        msg += f"• Entry: {trade.get('entry_price')}\n"
        msg += f"• SL: {trade.get('sl')}\n"
        msg += f"• Original SL: {trade.get('original_sl')}\n"
        msg += f"• SL Order ID: {trade.get('sl_order_id')}\n"
        msg += f"• Qty: {trade.get('qty')}\n"
        msg += f"• Exited: {trade.get('exited')}\n\n"
        
        msg += f"<b>Exchange Orders ({len(orders)}):</b>\n"
        for order in orders:
            msg += f"• {order.get('orderType')} {order.get('side')} @ {order.get('stopPrice', order.get('price'))} (ID: {order.get('orderId')[:8]}...)\n"
        
        msg += f"\n<b>Position:</b>\n"
        for pos in positions:
            if float(pos.get("size", 0)) > 0:
                msg += f"• {pos.get('side')} {pos.get('size')} @ {pos.get('avgPrice')}\n"
        
        await send_telegram_message(msg)
        
    except Exception as e:
        log(f"❌ Error in debug_stop_loss: {e}", level="ERROR")
        await send_telegram_message(f"❌ Error debugging {symbol}: {e}")


async def verify_trade_integrity():
    """Verify all trades have proper SL orders"""
    try:
        log("🔍 Starting trade integrity verification...")
        
        for symbol, trade in active_trades.items():
            if trade.get("exited"):
                continue
                
            await check_and_restore_sl(symbol, trade)
            await asyncio.sleep(0.5)  # Rate limiting
            
        log("✅ Trade integrity verification completed")
        await send_telegram_message("✅ Trade integrity verification completed")
        
    except Exception as e:
        log(f"❌ Error in trade integrity verification: {e}", level="ERROR")


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
    'setup_module_dependencies',
    'debug_stop_loss',
    'verify_trade_integrity'
]


# Initialize module
if __name__ == "__main__":
    print("✅ monitor.py module loaded successfully")
    
    # Setup dependencies
    setup_module_dependencies()
    
    # Load trades
    load_active_trades()
    print(f"📊 {len(active_trades)} active trades loaded")
