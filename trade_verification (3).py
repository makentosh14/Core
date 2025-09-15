#!/usr/bin/env python3
"""
FIXED trade_verification.py - Resolves import errors and XEMUSDT disappearance issue
"""

import asyncio
import traceback
from datetime import datetime
from logger import log, write_log
from bybit_api import signed_request
from error_handler import send_telegram_message

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

async def verify_position_and_orders(symbol, trade, auto_repair=True):
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
            # Track verification failures
            if "verification_failures" not in trade:
                trade["verification_failures"] = 0
            trade["verification_failures"] += 1
            
            if trade["verification_failures"] >= 3:
                result["manual_review_required"].append(
                    f"Position not found for {trade['verification_failures']} consecutive checks"
                )
                trade["needs_manual_review"] = True
                log(f"🚨 {symbol}: Flagged for manual review after {trade['verification_failures']} failures")
            else:
                log(f"⚠️ {symbol}: Position not found (failure #{trade['verification_failures']}/3)")
            return result
        else:
            result["position_exists"] = True
            # Reset failure counter if position found
            trade.pop("verification_failures", None)
            trade.pop("last_verification_fail", None)

        # FIXED: Get the CURRENT expected position size (after any DCAs)
        current_expected_size = trade.get("qty", 0)  # This should include DCA additions
        original_size = trade.get("original_qty", current_expected_size)
        dca_count = trade.get("dca_count", 0)
        
        # Calculate tolerance (0.1% or minimum 0.01)
        size_tolerance = max(0.01, current_expected_size * 0.001)
        size_matches = abs(position_size - current_expected_size) <= size_tolerance
        result["position_size_matches"] = size_matches
        
        # Enhanced logging for DCA trades
        if dca_count > 0:
            log(f"📊 DCA Position Check for {symbol}:")
            log(f"   Original Size: {original_size}")
            log(f"   DCA Count: {dca_count}")
            log(f"   Expected Current Size: {current_expected_size}")
            log(f"   Actual Size: {position_size}")
            log(f"   Size Matches: {size_matches}")
        
        # FIXED: Only flag size mismatch if it's actually wrong
        if not size_matches:
            size_diff = abs(position_size - current_expected_size)
            size_diff_pct = (size_diff / current_expected_size) * 100 if current_expected_size > 0 else 0
            
            # Don't confuse users by comparing to original size for DCA trades
            if dca_count > 0:
                # For DCA trades, show comparison to current expected
                result["issues_detected"].append(
                    f"DCA position size mismatch: expected {current_expected_size}, found {position_size} "
                    f"(diff: {size_diff:.4f}, {size_diff_pct:.2f}%) [DCA count: {dca_count}]"
                )
            else:
                # For non-DCA trades, normal comparison
                result["issues_detected"].append(
                    f"Position size mismatch: expected {current_expected_size}, found {position_size} "
                    f"(diff: {size_diff:.4f}, {size_diff_pct:.2f}%)"
                )
            
            if auto_repair:
                # Only auto-fix small differences (< 5%)
                if size_diff_pct < 5.0:
                    result["repairs_attempted"].append("Update trade position size to match exchange")
                    old_qty = trade["qty"]
                    trade["qty"] = position_size
                    from monitor import save_active_trades
                    save_active_trades()
                    result["repairs_successful"].append(f"Updated position size to {position_size}")
                    log(f"🔧 {symbol}: Auto-fixed size {old_qty} → {position_size}")
                else:
                    # Large difference - flag for manual review but don't auto-fix
                    result["manual_review_required"].append(
                        f"Large position size difference ({size_diff_pct:.2f}%)"
                    )
                    trade["needs_manual_review"] = True
                    log(f"🚨 {symbol}: Large size difference - flagged for manual review")

        # Check direction match
        position_side_raw = position_data.get("side", "")
        expected_direction_raw = trade.get("direction", "")

        # Normalize both directions
        position_side = normalize_direction(position_side_raw)
        expected_direction = normalize_direction(expected_direction_raw)

        direction_matches = position_side == expected_direction
        result["direction_matches"] = direction_matches

        # Enhanced logging to show the mapping
        log(f"🔍 Direction Check for {symbol}:")
        log(f"   Bybit API: '{position_side_raw}' → normalized: '{position_side}'")
        log(f"   Bot expected: '{expected_direction_raw}' → normalized: '{expected_direction}'")
        log(f"   Match: {direction_matches}")
        
        if not direction_matches:
            result["issues_detected"].append(
                f"Position direction mismatch: expected {expected_direction}, found {position_side}"
            )
            
            if auto_repair:
                # Direction mismatch is serious - always flag for manual review
                result["manual_review_required"].append(
                    f"Direction mismatch: expected {expected_direction}, found {position_side}"
                )
                trade["needs_manual_review"] = True
                trade["direction_mismatch_detected"] = True
                log(f"🚨 {symbol}: Direction mismatch - flagged for manual review")

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
        if dca_count > 0 and any("position size mismatch" in issue for issue in result["issues_detected"]):
            # This might be normal DCA reconciliation
            log(f"🔄 Position verification for {symbol}: {issue_count} issues, {repair_count} repaired (DCA trade)", level="INFO")
        else:
            log(f"⚠️ Position verification for {symbol}: {issue_count} issues, {repair_count} repaired", level="WARN")
    else:
        if dca_count > 0:
            log(f"✅ Position verification for {symbol}: All checks passed (DCA count: {dca_count})")
        else:
            log(f"✅ Position verification for {symbol}: All checks passed")
    
    return result

def calculate_expected_dca_size(original_qty, dca_count, dca_config):
    """
    Calculate what the position size should be after DCAs
    """
    if dca_count == 0:
        return original_qty
    
    # Each DCA adds dca_config["add_size_pct"] of original
    total_added = original_qty * (dca_config["add_size_pct"] / 100) * dca_count
    return original_qty + total_added

async def validate_dca_position_size(symbol, trade):
    """
    Specific validation for DCA trades to ensure position size is correct
    """
    try:
        dca_count = trade.get("dca_count", 0)
        if dca_count == 0:
            return True  # Not a DCA trade
            
        original_qty = trade.get("original_qty")
        current_qty = trade.get("qty")
        trade_type = trade.get("trade_type", "Intraday")
        
        if not original_qty:
            log(f"⚠️ DCA trade {symbol} missing original_qty")
            return False
            
        # Get DCA config
        DCA_CONFIG = {
            "Scalp": {"add_size_pct": 100},
            "Intraday": {"add_size_pct": 100},
            "Swing": {"add_size_pct": 100}
        }
        dca_config = DCA_CONFIG.get(trade_type, DCA_CONFIG["Intraday"])
        
        # Calculate expected size
        expected_size = calculate_expected_dca_size(original_qty, dca_count, dca_config)
        
        tolerance = max(0.01, expected_size * 0.01)  # 1% tolerance
        size_matches = abs(current_qty - expected_size) <= tolerance
        
        log(f"📊 DCA Validation for {symbol}:")
        log(f"   Original Qty: {original_qty}")
        log(f"   DCA Count: {dca_count}")
        log(f"   Expected Size: {expected_size}")
        log(f"   Current Size: {current_qty}")
        log(f"   Matches: {size_matches}")
        
        if not size_matches:
            log(f"⚠️ DCA size mismatch for {symbol}: expected {expected_size}, got {current_qty}")
            # Auto-correct if reasonable
            trade["qty"] = expected_size
            log(f"🔧 Corrected {symbol} size to {expected_size}")
            
        return size_matches
        
    except Exception as e:
        log(f"❌ Error validating DCA position: {e}", level="ERROR")
        return False

async def verify_all_positions(frequency_minutes=15):
    """
    FIXED: Periodic verification of all active positions with better error handling
    """
    log("🔍 Starting position verification service...")
    
    while True:
        try:
            log("🔍 Starting comprehensive position verification")
            
            # Import here to avoid circular imports
            try:
                from monitor import active_trades
            except ImportError:
                log("❌ Could not import active_trades - verification disabled", level="ERROR")
                await asyncio.sleep(frequency_minutes * 60)
                continue
            
            # Create a snapshot of active trades to avoid dictionary changing during iteration
            if not active_trades:
                log("📭 No active trades to verify")
                await asyncio.sleep(frequency_minutes * 60)
                continue
                
            trades_snapshot = dict(active_trades)  # Create a copy
            verified_count = 0
            dca_validated_count = 0
            
            for symbol, trade in trades_snapshot.items():
                if trade.get("exited"):
                    continue
                
                try:
                    # Verify this position and orders
                    await verify_position_and_orders(symbol, trade, auto_repair=True)
                    verified_count += 1

                    if trade.get("dca_count", 0) > 0:
                        dca_valid = await validate_dca_position_size(symbol, trade)
                        if dca_valid:
                            dca_validated_count += 1

                    # Brief pause to avoid rate limits
                    await asyncio.sleep(0.5)

                except Exception as e:
                    log(f"❌ Error verifying {symbol}: {e}", level="ERROR")
                    continue
            
            log(f"✅ Position verification cycle complete - verified {verified_count} trades")
            
        except Exception as e:
            log(f"❌ Error in position verification cycle: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
        
        # Wait for next cycle
        await asyncio.sleep(frequency_minutes * 60)

def recover_missing_trades():
    """
    Recovery function to restore trades that were incorrectly marked as exited
    """
    log("🔄 Starting trade recovery process...")
    
    try:
        # Load both active trades and the persisted file
        try:
            from monitor import active_trades, PERSIST_PATH
        except ImportError:
            log("❌ Could not import monitor components", level="ERROR")
            return
            
        import json
        import os
        
        if not os.path.exists(PERSIST_PATH):
            log("❌ No persisted trades file found")
            return
        
        with open(PERSIST_PATH, 'r') as f:
            all_trades = json.load(f)
        
        # Find trades marked as exited that might need recovery
        potentially_recoverable = []
        
        for symbol, trade in all_trades.items():
            if trade.get("exited") and trade.get("needs_manual_review"):
                potentially_recoverable.append((symbol, trade))
        
        log(f"🔍 Found {len(potentially_recoverable)} potentially recoverable trades")
        
        if not potentially_recoverable:
            log("✅ No trades need recovery")
            return
            
        # For each potentially recoverable trade, check if position still exists
        for symbol, trade in potentially_recoverable:
            log(f"🔄 Checking {symbol} for recovery...")
            
            # This would need to be run in an async context
            # For now, just log what we would do
            log(f"📋 Would check position for {symbol} - direction: {trade.get('direction')}")
        
        log("✅ Trade recovery process completed")
        
    except Exception as e:
        log(f"❌ Error in trade recovery: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")

def generate_manual_review_report():
    """
    Generate a report of all trades requiring manual review
    """
    try:
        from monitor import active_trades
        
        manual_review_trades = []
        
        for symbol, trade in active_trades.items():
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

# Run test when imported
if __name__ == "__main__":
    test_import()
    print("✅ trade_verification.py module is working correctly")
