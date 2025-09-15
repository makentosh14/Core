# dca_manager.py - Dollar Cost Averaging strategy for handling fakeouts
# FIXED: Removes all artificial minimum order checks - if Bybit allowed the initial order, it should allow DCA

import asyncio
import json
import time
from datetime import datetime
from logger import log, write_log
from bybit_api import place_market_order, place_stop_loss_with_retry, signed_request
from symbol_info import round_qty
from error_handler import send_telegram_message
from config import DCA_FAST_BUFFER          # <-- NEW
from trade_verification import verify_position_and_orders, validate_dca_position_size # <-- NEW
from auto_exit_handler import auto_exit_past_sl

# SAFE DCA Configuration - Maximum 2 entries for ALL trade types
DCA_CONFIG = {
    "Scalp": {
        "trigger_drop_pct": 0.4,    # Trigger DCA at -0.4% drop
        "add_size_pct": 50,         # Add 50% of original position (safer)
        "max_adds": 2,              # 🔒 MAXIMUM 2 DCA adds
        "new_sl_adjustment": 0.6,   # New SL at 0.6% below average entry
        "new_tp_adjustment": 0.9    # New TP at 0.9% above average entry
    },
    "Intraday": {
        "trigger_drop_pct": 0.6,    # Trigger DCA at -0.6% drop
        "add_size_pct": 50,         # Add 50% of original position (safer)
        "max_adds": 2,              # 🔒 MAXIMUM 2 DCA adds
        "new_sl_adjustment": 0.8,   # New SL at 0.8% below average entry
        "new_tp_adjustment": 1.2    # New TP at 1.2% above average entry
    },
    "Swing": {
        "trigger_drop_pct": 1.0,    # Trigger DCA at -1.0% drop
        "add_size_pct": 50,         # Add 50% of original position (safer)
        "max_adds": 2,              # 🔒 MAXIMUM 2 DCA adds
        "new_sl_adjustment": 1.0,   # New SL at 1.0% below average entry
        "new_tp_adjustment": 2.0    # New TP at 2.0% above average entry
    }
}

# SAFETY CONSTANTS
MAX_DCA_COUNT_GLOBAL = 2           # 🔒 Global maximum DCAs per trade
MAX_DCA_PER_DAY = 10              # Maximum DCAs across all trades per day
DCA_COOLDOWN_SECONDS = 300        # 5 minutes between DCAs
MAX_POSITION_MULTIPLIER = 2.0     # Position can't grow more than 2x original

class DCAManager:
    def __init__(self):
        self.active_dca = {}
        self.dca_history = {}
        
    async def check_dca_opportunity(self, symbol, trade, current_price):
        """
        Check if a trade qualifies for DCA entry
        """
        try:
            # Skip if already exited or no entry price
            if trade.get("exited") or not await verify_position_and_orders(symbol, trade):
                return False
    
            if trade.get("exited") or not trade.get("entry_price"):
                return False
                
            # Get trade details
            direction = trade.get("direction", "").lower()
            entry_price = trade.get("entry_price")
            trade_type = trade.get("trade_type", "Intraday")
            
            # Get DCA config for this trade type
            dca_config = DCA_CONFIG.get(trade_type, DCA_CONFIG["Intraday"])
            
            # Check if we've already done maximum DCA adds
            dca_count = trade.get("dca_count", 0)

            # Check 1: Trade-specific limit
            if dca_count >= dca_config["max_adds"]:
                log(f"🚫 DCA blocked for {symbol}: Trade limit reached ({dca_count}/{dca_config['max_adds']})")
                return False

            # Check 2: Global safety limit (never exceed 2 DCAs)
            if dca_count >= MAX_DCA_COUNT_GLOBAL:
                log(f"🔒 DCA blocked for {symbol}: Global safety limit reached ({dca_count}/{MAX_DCA_COUNT_GLOBAL})")
                return False

            # Check 3: Position size multiplier limit
            original_qty = trade.get("original_qty") or trade.get("qty")
            current_qty = trade.get("qty")
            if current_qty >= original_qty * MAX_POSITION_MULTIPLIER:
                log(f"🔒 DCA blocked for {symbol}: Position size limit reached ({current_qty:.6f} >= {original_qty * MAX_POSITION_MULTIPLIER:.6f})")
                return False

            log(f"✅ DCA check passed for {symbol}: {dca_count}/{MAX_DCA_COUNT_GLOBAL} DCAs used")

            # --- NEW: block DCA only when price is BEYOND the SL by more than buffer
            sl_price = trade.get("original_sl")
            if sl_price:
                crossed = (
                    direction == "long"  and current_price <= sl_price * (1 - DCA_FAST_BUFFER / 100)
                ) or (
                    direction == "short" and current_price >= sl_price * (1 + DCA_FAST_BUFFER / 100)
                )
                if crossed:
                    log(f"🚫 DCA skipped for {symbol}: price >{DCA_FAST_BUFFER:.2f}% past SL - AUTO-EXITING")
        
                    # Auto-exit the trade instead of just skipping
                    exit_success = await auto_exit_past_sl(symbol, trade, current_price, DCA_FAST_BUFFER)
        
                    if exit_success:
                        log(f"✅ {symbol}: Trade auto-exited successfully")
                    else:
                        log(f"❌ {symbol}: Auto-exit failed, but DCA still blocked", level="ERROR")
        
                    return False
            
            # Calculate current drawdown
            if direction == "long":
                drawdown_pct = ((entry_price - current_price) / entry_price) * 100
            else:  # short
                drawdown_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Check if drawdown exceeds trigger threshold
            trigger_threshold = dca_config["trigger_drop_pct"] * (dca_count + 1)
            
            if drawdown_pct >= trigger_threshold:
                # Check cooldown (don't DCA too frequently)
                last_dca_time = trade.get("last_dca_time")
                if last_dca_time:
                    time_since_last = (datetime.utcnow() - datetime.fromisoformat(last_dca_time)).total_seconds()
                    if time_since_last < 300:  # 5 minute cooldown
                        return False
                
                log(f"💰 DCA opportunity for {symbol}: Drawdown {drawdown_pct:.2f}% exceeds threshold {trigger_threshold:.2f}%")
                return True
                
            return False
            
        except Exception as e:
            log(f"❌ Error checking DCA opportunity: {e}", level="ERROR")
            return False
    
    async def execute_dca_add(self, symbol, trade, current_price, account_balance):
        """
        Execute a DCA addition - adds exactly 100% of original position size
        NO MINIMUM ORDER VALUE CHECKS - if Bybit allowed the initial order, it should allow DCA
        """
        try:
            # Get trade details
            direction = trade.get("direction", "").lower()
            original_qty = trade.get("original_qty") or trade.get("qty")
            current_qty = trade.get("qty")
            entry_price = trade.get("entry_price")
            trade_type = trade.get("trade_type", "Intraday")
            
            # Get DCA config
            dca_config = DCA_CONFIG.get(trade_type, DCA_CONFIG["Intraday"])
            
            # Calculate DCA size - EXACTLY the percentage of original position
            add_size = original_qty * (dca_config["add_size_pct"] / 100)
            add_size = round_qty(symbol, add_size)
            
            # Calculate order value for logging
            order_value = add_size * current_price
            
            log(f"📊 DCA Calculation for {symbol}:")
            log(f"  Original Position: {original_qty:.8f}")
            log(f"  DCA Percentage: {dca_config['add_size_pct']}%")
            log(f"  DCA Add Size: {add_size:.8f}")
            log(f"  DCA Order Value: ${order_value:.2f}")
            log(f"  ✅ No minimum order value checks - trusting Bybit's validation")
            
            if add_size <= 0:
                log(f"⚠️ DCA size calculation resulted in zero for {symbol}", level="WARN")
                return None
            
            # Simple balance check - just ensure we have some balance
            if account_balance <= 1:  # Only check if we have at least $1
                log(f"⚠️ Insufficient account balance: ${account_balance:.2f}")
                return None
            
            # Execute the DCA market order directly - NO MINIMUM VALUE FILTERING
            side = "Buy" if direction == "long" else "Sell"
            
            log(f"📤 Executing DCA add for {symbol}: {side} {add_size:.8f} at ~${current_price:.6f}")
            log(f"💵 Order Value: ${order_value:.2f}")
            
            # Direct API call to Bybit - let Bybit decide if the order is valid
            result = await signed_request("POST", "/v5/order/create", {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(add_size),
                "timeInForce": "GTC"
            })
            
            if result.get("retCode") != 0:
                error_msg = result.get('retMsg', 'Unknown error')
                log(f"❌ DCA order failed: {error_msg}", level="ERROR")
                
                # Log the specific error for debugging
                if "minimum" in error_msg.lower():
                    log(f"💡 Bybit rejected for minimum order size - this means your initial order was different or there's a Bybit setting change")
                    log(f"💡 Initial order was accepted, so this might be a temporary Bybit issue")
                
                return None
            
            # Get actual execution price
            order_data = result.get("result", {})
            dca_price = float(order_data.get("avgPrice") or order_data.get("price") or current_price)
            
            # CRITICAL FIX: Wait for order settlement
            log("⏳ Waiting for DCA order settlement...")
            await asyncio.sleep(3)  # Give Bybit time to update position
            
            # CRITICAL FIX: Verify position with retries before calculating new values
            position_verified = False
            actual_total_qty = new_total_qty  # fallback
            actual_avg_price = None
            
            log(f"🔍 Verifying position after DCA for {symbol}...")
            try:
                validation_result = await validate_dca_position_size(symbol, trade)
    
                if validation_result:
                    log(f"✅ DCA position validation passed for {symbol}")
                else:
                    log(f"⚠️ DCA position validation failed for {symbol} - flagging for review", level="WARN")
                    trade["needs_manual_review"] = True
                    trade["dca_validation_failed"] = True
                    trade["validation_timestamp"] = datetime.utcnow().isoformat()
        
                    # Send alert about validation failure
                    await send_telegram_message(
                        f"⚠️ DCA Validation Failed\n"
                        f"Symbol: {symbol}\n"
                        f"Expected: {trade.get('qty')}\n"
                        f"Manual review required"
                    )
        
            except Exception as validation_error:
                log(f"❌ Error during DCA validation for {symbol}: {validation_error}", level="ERROR")
                # Don't fail the DCA, just log the validation error
                trade["validation_error"] = str(validation_error)
            
            for attempt in range(3):
                try:
                    pos_resp = await signed_request("GET", "/v5/position/list", {
                        "category": "linear",
                        "symbol": symbol
                    })
                    
                    if pos_resp.get("retCode") == 0:
                        positions = pos_resp.get("result", {}).get("list", [])
                        
                        for pos in positions:
                            position_size = abs(float(pos.get("size", 0)))
                            if position_size > 0:
                                actual_total_qty = position_size
                                actual_avg_price = float(pos.get("avgPrice", 0))
                                
                                log(f"✅ Position verified: {actual_total_qty} @ {actual_avg_price:.6f}")
                                position_verified = True
                                break
                    
                    if position_verified:
                        break
                        
                    log(f"⚠️ Position verification attempt {attempt + 1}/3 failed")
                    if attempt < 2:
                        await asyncio.sleep(2)
                        
                except Exception as e:
                    log(f"❌ Position verification attempt {attempt + 1}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)
            
            if not position_verified:
                log(f"❌ Could not verify position after DCA - using calculated values", level="WARN")
                # Still proceed but with calculated values
            
            # Calculate new average entry price (use verified or calculated)
            if actual_avg_price and actual_avg_price > 0:
                # Use actual average price from exchange
                new_avg_entry = actual_avg_price
                new_total_qty = actual_total_qty
                log(f"📊 Using verified values: {new_total_qty} @ {new_avg_entry:.6f}")
            else:
                # Fall back to calculated values
                total_cost = (entry_price * current_qty) + (dca_price * add_size)
                new_total_qty = current_qty + add_size
                new_avg_entry = total_cost / new_total_qty
                log(f"📊 Using calculated values: {new_total_qty} @ {new_avg_entry:.6f}")
            
            # Update trade data with verified/calculated values
            trade["entry_price"] = new_avg_entry
            trade["qty"] = new_total_qty
            trade["dca_count"] = trade.get("dca_count", 0) + 1
            trade["last_dca_time"] = datetime.utcnow().isoformat()
            trade["position_verified"] = position_verified

            from monitor import save_active_trades
            save_active_trades()
            log(f"💾 SAVED: Updated {symbol} qty from {current_qty} to {new_total_qty}")
            
            # Store original quantity if first DCA
            if "original_qty" not in trade:
                trade["original_qty"] = original_qty
            
            # Add DCA history
            if "dca_history" not in trade:
                trade["dca_history"] = []
            
            trade["dca_history"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "price": dca_price,
                "qty": add_size,
                "new_avg_entry": new_avg_entry,
                "order_value": order_value,
                "original_qty": original_qty,
                "add_percentage": dca_config["add_size_pct"]
            })
            
            # Calculate new SL and TP based on average entry
            if direction == "long":
                new_sl = new_avg_entry * (1 - dca_config["new_sl_adjustment"] / 100)
                new_tp = new_avg_entry * (1 + dca_config["new_tp_adjustment"] / 100)
            else:  # short
                new_sl = new_avg_entry * (1 + dca_config["new_sl_adjustment"] / 100)
                new_tp = new_avg_entry * (1 - dca_config["new_tp_adjustment"] / 100)

            log(f"📊 DCA Order Updates for {symbol}:")
            log(f"  New Avg Entry: ${new_avg_entry:.6f}")
            log(f"  New SL: ${new_sl:.6f}")
            log(f"  New TP1: ${new_tp:.6f}")
            log(f"  New Position Size: {new_total_qty:.8f}")

            # STEP 1: Cancel existing orders FIRST
            orders_cancelled = []
            
            # Cancel old SL order
            if trade.get("sl_order_id"):
                try:
                    cancel_sl_result = await signed_request("POST", "/v5/order/cancel", {
                        "category": "linear",
                        "symbol": symbol,
                        "orderId": trade["sl_order_id"]
                    })
        
                    if cancel_sl_result.get("retCode") == 0:
                        log(f"✅ Successfully cancelled existing SL order: {trade['sl_order_id']}")
                        orders_cancelled.append("SL")
                        trade["sl_order_id"] = None
                    else:
                        log(f"⚠️ Failed to cancel SL order: {cancel_sl_result.get('retMsg', 'Unknown error')}")
            
                except Exception as e:
                    log(f"❌ Error cancelling SL order: {e}", level="ERROR")

            # Cancel old TP1 order
            if trade.get("tp1_order_id"):
                try:
                    cancel_tp1_result = await signed_request("POST", "/v5/order/cancel", {
                        "category": "linear",
                        "symbol": symbol,
                        "orderId": trade["tp1_order_id"]
                    })
        
                    if cancel_tp1_result.get("retCode") == 0:
                        log(f"✅ Successfully cancelled existing TP1 order: {trade['tp1_order_id']}")
                        orders_cancelled.append("TP1")
                        trade["tp1_order_id"] = None
                    else:
                        log(f"⚠️ Failed to cancel TP1 order: {cancel_tp1_result.get('retMsg', 'Unknown error')}")
            
                except Exception as e:
                    log(f"❌ Error cancelling TP1 order: {e}", level="ERROR")

            # Log cancellation results
            if orders_cancelled:
                log(f"🗑️ Cancelled orders: {', '.join(orders_cancelled)}")
            else:
                log("⚠️ No orders were cancelled (might not have existed)")
            
            # Place new SL order
            sl_result = await place_stop_loss_with_retry(
                symbol=symbol,
                direction=direction,
                qty=new_total_qty,
                sl_price=new_sl
            )
            
            if sl_result.get("retCode") == 0:
                trade["sl_order_id"] = sl_result.get("result", {}).get("orderId")
                trade["original_sl"] = new_sl
            
            if trade.get("tp1_order_id"):
                try:
                    await signed_request("POST", "/v5/order/cancel", {
                        "category": "linear",
                        "symbol": symbol,
                        "orderId": trade["tp1_order_id"]
                    })
                except:
                    pass

            tp_qty = round_qty(symbol, new_total_qty * 0.5)

            # Place new TP1 limit order
            tp_side = "Sell" if direction == "long" else "Buy"
            tp_result = await signed_request("POST", "/v5/order/create", {
                "category": "linear",
                "symbol": symbol,
                "side": tp_side,
                "orderType": "Limit",
                "qty": str(round_qty(symbol, new_total_qty * 0.5)),  # TP1 for 50%
                "price": str(round(new_tp, 6)),
                "timeInForce": "GTC",
                "reduceOnly": True
            })

            if tp_result.get("retCode") == 0:
                trade["tp1_order_id"] = tp_result["result"]["orderId"]
                trade["tp1_target"] = new_tp

            update_success = update_trade_after_dca(
            trade, new_avg_entry, new_total_qty, new_sl, new_tp
            )
        
            if not update_success:
                log(f"⚠️ Warning: Trade update after DCA may have issues for {symbol}")
            else:
                log
            
            # Send notification
            await send_telegram_message(
                f"💰 <b>DCA Added</b> for <b>{symbol}</b>\n"
                f"Original Position: {original_qty:.8f}\n"
                f"Added: {add_size:.8f} ({dca_config['add_size_pct']}%)\n"
                f"Add Price: ${dca_price:.6f}\n"
                f"Order Value: ${order_value:.2f}\n"
                f"New Avg Entry: ${new_avg_entry:.6f}\n"
                f"New Total Size: {new_total_qty:.8f}\n"
                f"New SL: ${new_sl:.6f}\n"
                f"New TP: ${new_tp:.6f}\n"
                f"DCA Count: {trade['dca_count']}/{dca_config['max_adds']}"
            )
            
            log(f"✅ DCA executed for {symbol}: Added {add_size:.8f} ({dca_config['add_size_pct']}% of {original_qty:.8f}) at ${dca_price:.6f}")
            log(f"📈 New average entry: ${new_avg_entry:.6f}, Total size: {new_total_qty:.8f}")
            write_log(f"DCA_EXECUTED: {symbol} | Add: {add_size:.8f} @ ${dca_price:.6f} | New Avg: ${new_avg_entry:.6f} | Value: ${order_value:.2f}")
            
            return trade
            
        except Exception as e:
            log(f"❌ Error executing DCA: {e}", level="ERROR")
            import traceback
            log(traceback.format_exc(), level="ERROR")
            return None
    
    def get_dca_stats(self, symbol=None):
        """Get DCA statistics for reporting"""
        if symbol:
            return self.dca_history.get(symbol, {})
        return self.dca_history
    
    def get_dca_summary(self, symbol):
        """Get summary of DCA operations for a symbol"""
        try:
            if symbol not in self.dca_history:
                return None
                
            history = self.dca_history[symbol].get("dca_history", [])
            if not history:
                return None
                
            total_added = sum(dca["qty"] for dca in history)
            total_value = sum(dca["order_value"] for dca in history)
            avg_price = sum(dca["price"] * dca["qty"] for dca in history) / total_added if total_added > 0 else 0
            
            return {
                "total_dca_count": len(history),
                "total_qty_added": total_added,
                "total_value_added": total_value,
                "average_dca_price": avg_price,
                "latest_avg_entry": history[-1]["new_avg_entry"] if history else None
            }
            
        except Exception as e:
            log(f"❌ Error getting DCA summary: {e}", level="ERROR")
            return None

# Global DCA manager instance
dca_manager = DCAManager()
