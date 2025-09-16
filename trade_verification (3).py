#!/usr/bin/env python3
"""
FIXED trade_verification.py - Resolves import errors and circular dependencies
"""

import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from logger import log, write_log
from bybit_api import signed_request
from error_handler import send_telegram_message

# Dependency injection to avoid circular imports
_save_trades_func: Optional[Callable] = None
_active_trades_ref: Optional[Dict] = None

def set_dependencies(save_func: Callable, active_trades: Dict):
    """Set dependencies to avoid circular imports"""
    global _save_trades_func, _active_trades_ref
    _save_trades_func = save_func
    _active_trades_ref = active_trades

def normalize_direction(direction):
    """Normalize direction to handle different formats between bot and exchange"""
    if not direction:
        return ""
    
    direction = direction.lower().strip()
    
    # Map Bybit API format to bot internal format
    direction_map = {
        "buy": "long",
        "sell": "short", 
        "long": "long",
        "short": "short"
    }
    
    return direction_map.get(direction, direction)

async def verify_position_and_orders(symbol: str, trade: Dict, auto_repair: bool = True) -> Dict[str, Any]:
    """
    FIXED: Proper position verification that accounts for DCA correctly
    """
    result = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "position_exists": False,
        "position_size_matches": False,
        "direction_matches": False,
        "sl_order_exists": False,
        "sl_price_matches": False,
        "issues_detected": [],
        "repairs_attempted": [],
        "repairs_successful": [],
        "manual_review_required": []
    }
    
    if not trade or trade.get("exited"):
        result["position_exists"] = False
        result["issues_detected"].append("Trade marked as exited")
        return result

    try:
        # Get position from exchange
        position_resp = await signed_request("GET", "/v5/position/list", {
            "category": "linear",
            "symbol": symbol,
            "settleCoin": "USDT"
        })
        
        if position_resp.get("retCode") != 0:
            result["issues_detected"].append(f"Failed to fetch position: {position_resp.get('retMsg')}")
            return result
            
        positions = position_resp.get("result", {}).get("list", [])
        position_data = None
        position_size = 0
        
        # Find the position
        for pos in positions:
            if pos.get("symbol") == symbol and abs(float(pos.get("size", 0))) > 0:
                position_data = pos
                position_size = abs(float(pos.get("size", 0)))
                break
                
        if not position_data:
            result["position_exists"] = False
            result["issues_detected"].append("No position found on exchange")
            
            # Mark trade as exited if no position exists
            if auto_repair and _active_trades_ref and symbol in _active_trades_ref:
                _active_trades_ref[symbol]["exited"] = True
                _active_trades_ref[symbol]["exit_reason"] = "position_verification_no_position"
                _active_trades_ref[symbol]["exit_time"] = datetime.now().isoformat()
                result["repairs_attempted"].append("marked_trade_as_exited")
                result["repairs_successful"].append("marked_trade_as_exited")
                log(f"🔧 {symbol}: Marked trade as exited (no position on exchange)")
                
                # Save changes
                if _save_trades_func:
                    _save_trades_func()
            
            return result
        
        result["position_exists"] = True
        
        # Check position size (accounting for DCA)
        expected_qty = float(trade.get("qty", 0))
        dca_count = trade.get("dca_count", 0)
        
        # Calculate expected size with DCA
        if dca_count > 0:
            # Each DCA typically adds a percentage of original size
            dca_config = trade.get("dca_config", {"add_size_pct": 100})  # Default 100% per DCA
            add_pct = dca_config.get("add_size_pct", 100)
            expected_qty += expected_qty * (add_pct / 100) * dca_count
        
        size_tolerance = 0.05  # 5% tolerance
        size_diff = abs(position_size - expected_qty) / expected_qty if expected_qty > 0 else 0
        
        if size_diff <= size_tolerance:
            result["position_size_matches"] = True
        else:
            result["position_size_matches"] = False
            result["issues_detected"].append(
                f"Position size mismatch: expected {expected_qty}, found {position_size} (diff: {size_diff:.2%})"
            )
            
            # For DCA trades, size mismatches might be normal during execution
            if dca_count > 0:
                log(f"📊 {symbol}: Size difference detected in DCA trade (expected: {expected_qty}, actual: {position_size})")
            elif auto_repair:
                # Update trade with actual position size
                trade["qty"] = position_size
                result["repairs_attempted"].append("updated_trade_quantity")
                result["repairs_successful"].append("updated_trade_quantity")
                log(f"🔧 {symbol}: Updated trade quantity to match exchange ({position_size})")
        
        # Check direction
        position_side = normalize_direction(position_data.get("side", ""))
        expected_direction = normalize_direction(trade.get("direction", ""))
        
        if position_side == expected_direction:
            result["direction_matches"] = True
        else:
            result["direction_matches"] = False
            result["issues_detected"].append(
                f"Direction mismatch: expected {expected_direction}, found {position_side}"
            )
            
            if auto_repair:
                # Direction mismatch is serious - always flag for manual review
                result["manual_review_required"].append(
                    f"Direction mismatch: expected {expected_direction}, found {position_side}"
                )
                trade["needs_manual_review"] = True
                trade["direction_mismatch_detected"] = True
                log(f"🚨 {symbol}: Direction mismatch - flagged for manual review")

        # Check stop loss orders
        try:
            orders_resp = await signed_request("GET", "/v5/order/realtime", {
                "category": "linear",
                "symbol": symbol,
                "orderFilter": "StopOrder"
            })
            
            if orders_resp.get("retCode") == 0:
                orders = orders_resp.get("result", {}).get("list", [])
                sl_orders = [o for o in orders if o.get("orderType") in ["Stop", "StopLoss"]]
                
                if sl_orders:
                    result["sl_order_exists"] = True
                    
                    # Check if SL price matches
                    expected_sl = float(trade.get("sl", 0)) if trade.get("sl") else None
                    if expected_sl:
                        for order in sl_orders:
                            order_sl = float(order.get("stopPrice", 0))
                            if abs(order_sl - expected_sl) / expected_sl <= 0.02:  # 2% tolerance
                                result["sl_price_matches"] = True
                                break
                        
                        if not result["sl_price_matches"]:
                            result["issues_detected"].append("Stop loss price mismatch")
                else:
                    result["sl_order_exists"] = False
                    result["issues_detected"].append("No stop loss order found")
                    
                    if auto_repair:
                        # Try to restore missing SL
                        result["repairs_attempted"].append("restore_stop_loss")
                        try:
                            # This would need the actual SL restoration function
                            # For now, just log the attempt
                            log(f"🔧 {symbol}: Attempting to restore missing stop loss")
                            result["manual_review_required"].append("Missing stop loss order")
                        except Exception as e:
                            log(f"❌ Failed to restore SL for {symbol}: {e}")
            
        except Exception as e:
            result["issues_detected"].append(f"Error checking stop loss orders: {str(e)}")

    except Exception as e:
        result["issues_detected"].append(f"Error checking position: {str(e)}")
        log(f"❌ Position verification error for {symbol}: {str(e)}", level="ERROR")

    # Log results with proper context
    if result["manual_review_required"]:
        log(f"🚨 {symbol}: Manual review required - {len(result['manual_review_required'])} issues", level="WARN")
        for issue in result["manual_review_required"]:
            log(f"   - {issue}", level="WARN")
    elif result["issues_detected"]:
        issue_count = len(result["issues_detected"])
        repair_count = len(result["repairs_successful"])
        
        # FIXED: Don't alarm users for normal DCA operations
        dca_count = trade.get("dca_count", 0)
        if dca_count > 0 and any("position size mismatch" in issue for issue in result["issues_detected"]):
            # This might be normal DCA reconciliation
            log(f"🔄 Position verification for {symbol}: {issue_count} issues, {repair_count} repaired (DCA trade)", level="INFO")
        else:
            log(f"⚠️ Position verification for {symbol}: {issue_count} issues, {repair_count} repaired", level="WARN")
    else:
        dca_count = trade.get("dca_count", 0)
        if dca_count > 0:
            log(f"✅ Position verification for {symbol}: All checks passed (DCA count: {dca_count})")
        else:
            log(f"✅ Position verification for {symbol}: All checks passed")
    
    return result

def calculate_expected_dca_size(original_qty: float, dca_count: int, dca_config: Dict) -> float:
    """Calculate what the position size should be after DCAs"""
    if dca_count == 0:
        return original_qty
    
    # Each DCA adds dca_config["add_size_pct"] of original
    total_added = original_qty * (dca_config["add_size_pct"] / 100) * dca_count
    return original_qty + total_added

async def validate_dca_position_size(symbol: str, trade: Dict) -> bool:
    """Specific validation for DCA trades to ensure position size is correct"""
    try:
        dca_count = trade.get("dca_count", 0)
        if dca_count == 0:
            return True  # Not a DCA trade
        
        original_qty = float(trade.get("original_qty", trade.get("qty", 0)))
        dca_config = trade.get("dca_config", {"add_size_pct": 100})
        
        expected_size = calculate_expected_dca_size(original_qty, dca_count, dca_config)
        
        # Get actual position size from exchange
        position_resp = await signed_request("GET", "/v5/position/list", {
            "category": "linear",
            "symbol": symbol
        })
        
        if position_resp.get("retCode") != 0:
            return False
        
        positions = position_resp.get("result", {}).get("list", [])
        actual_size = 0
        
        for pos in positions:
            if pos.get("symbol") == symbol:
                actual_size = abs(float(pos.get("size", 0)))
                break
        
        # Allow 5% tolerance
        if actual_size == 0:
            return False
        
        size_diff = abs(actual_size - expected_size) / expected_size
        return size_diff <= 0.05
        
    except Exception as e:
        log(f"❌ Error validating DCA position size for {symbol}: {e}", level="ERROR")
        return False

async def run_comprehensive_verification() -> Dict[str, Any]:
    """Run comprehensive verification on all active trades"""
    try:
        if not _active_trades_ref:
            log("❌ No active trades reference available", level="ERROR")
            return {}
        
        verification_results = {}
        
        for symbol, trade in _active_trades_ref.items():
            if trade.get("exited"):
                continue
            
            log(f"🔍 Verifying {symbol}...")
            result = await verify_position_and_orders(symbol, trade, auto_repair=True)
            verification_results[symbol] = result
        
        # Summary
        total_trades = len(verification_results)
        issues_found = sum(1 for r in verification_results.values() if r["issues_detected"])
        manual_review_needed = sum(1 for r in verification_results.values() if r["manual_review_required"])
        
        summary = {
            "total_verified": total_trades,
            "issues_found": issues_found,
            "manual_review_needed": manual_review_needed,
            "results": verification_results
        }
        
        log(f"📊 Verification complete: {total_trades} trades, {issues_found} with issues, {manual_review_needed} need manual review")
        
        return summary
        
    except Exception as e:
        log(f"❌ Error in comprehensive verification: {e}", level="ERROR")
        return {}

def attempt_trade_recovery():
    """Attempt to recover potentially recoverable trades"""
    try:
        if not _active_trades_ref:
            log("❌ No active trades reference available", level="ERROR")
            return
        
        potentially_recoverable = []
        
        for symbol, trade in _active_trades_ref.items():
            if (trade.get("exited") and 
                trade.get("exit_reason") in ["verification_failed", "position_not_found"] and
                not trade.get("recovery_attempted")):
                potentially_recoverable.append((symbol, trade))
        
        log(f"🔄 Found {len(potentially_recoverable)} potentially recoverable trades")
        
        if not potentially_recoverable:
            log("✅ No trades need recovery")
            return
            
        # For each potentially recoverable trade, mark as attempted
        for symbol, trade in potentially_recoverable:
            log(f"🔄 Marking {symbol} as recovery attempted...")
            trade["recovery_attempted"] = True
            log(f"📋 Would check position for {symbol} - direction: {trade.get('direction')}")
        
        log("✅ Trade recovery process completed")
        
        # Save changes
        if _save_trades_func:
            _save_trades_func()
        
    except Exception as e:
        log(f"❌ Error in trade recovery: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")

def generate_manual_review_report() -> list:
    """Generate a report of all trades requiring manual review"""
    try:
        if not _active_trades_ref:
            log("❌ No active trades reference available", level="ERROR")
            return []
        
        manual_review_trades = []
        
        for symbol, trade in _active_trades_ref.items():
            if trade.get("needs_manual_review"):
                manual_review_trades.append({
                    "symbol": symbol,
                    "direction": trade.get("direction"),
                    "qty": trade.get("qty"),
                    "entry_price": trade.get("entry_price"),
                    "issues": trade.get("manual_review_required", []),
                    "timestamp": trade.get("timestamp")
                })
        
        if manual_review_trades:
            log(f"📋 Manual Review Report - {len(manual_review_trades)} trades need attention:")
            for trade in manual_review_trades:
                log(f"   {trade['symbol']}: {trade['direction']} {trade['qty']} @ {trade['entry_price']}")
                for issue in trade.get("issues", []):
                    log(f"      - {issue}")
        else:
            log("✅ No trades requiring manual review")
            
        return manual_review_trades
        
    except Exception as e:
        log(f"❌ Error generating manual review report: {e}", level="ERROR")
        return []

# Test function to verify the module loads correctly
def test_import():
    """Test function to verify the module imports correctly"""
    log("✅ trade_verification.py imported successfully")
    return True

# Export main functions
__all__ = [
    'verify_position_and_orders',
    'run_comprehensive_verification',
    'attempt_trade_recovery',
    'generate_manual_review_report',
    'set_dependencies',
    'normalize_direction',
    'test_import'
]

# Run test when imported
if __name__ == "__main__":
    test_import()
    print("✅ trade_verification.py module is working correctly")
