#!/usr/bin/env python3
"""
check_dca_limits.py - Monitor DCA usage and enforce limits
"""

import asyncio
from datetime import datetime, timedelta

try:
    from monitor import active_trades
    from dca_manager import DCA_CONFIG, MAX_DCA_COUNT_GLOBAL
    from logger import log
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

def check_current_dca_status():
    """Check current DCA status across all trades"""
    print("🔍 DCA STATUS CHECK")
    print("=" * 40)
    
    total_dcas_used = 0
    trades_with_dca = 0
    max_limit_reached = 0
    
    print(f"📊 Active Trades DCA Status:")
    
    for symbol, trade in active_trades.items():
        if trade.get("exited"):
            continue
            
        dca_count = trade.get("dca_count", 0)
        original_qty = trade.get("original_qty") or trade.get("qty", 0)
        current_qty = trade.get("qty", 0)
        trade_type = trade.get("trade_type", "Intraday")
        
        # Get max for this trade type
        max_for_type = DCA_CONFIG.get(trade_type, {}).get("max_adds", 2)
        
        # Calculate position multiplier
        if original_qty > 0:
            position_multiplier = current_qty / original_qty
        else:
            position_multiplier = 1.0
        
        if dca_count > 0:
            trades_with_dca += 1
            total_dcas_used += dca_count
            
        if dca_count >= MAX_DCA_COUNT_GLOBAL:
            max_limit_reached += 1
            
        status = "🔒 MAX" if dca_count >= MAX_DCA_COUNT_GLOBAL else f"{dca_count}/{MAX_DCA_COUNT_GLOBAL}"
        
        print(f"  {symbol}: {status} DCAs | Size: {original_qty:.4f} → {current_qty:.4f} ({position_multiplier:.1f}x) | Type: {trade_type}")
    
    print(f"\n📈 SUMMARY:")
    print(f"  Total active trades: {len([t for t in active_trades.values() if not t.get('exited')])}")
    print(f"  Trades with DCAs: {trades_with_dca}")
    print(f"  Total DCAs used: {total_dcas_used}")
    print(f"  Trades at max limit: {max_limit_reached}")
    print(f"  Global max per trade: {MAX_DCA_COUNT_GLOBAL}")
    
    return {
        "total_dcas": total_dcas_used,
        "trades_with_dca": trades_with_dca,
        "max_limit_reached": max_limit_reached
    }

def validate_dca_limits():
    """Validate that no trades exceed DCA limits"""
    print(f"\n🔒 VALIDATING DCA LIMITS:")
    
    violations = []
    
    for symbol, trade in active_trades.items():
        if trade.get("exited"):
            continue
            
        dca_count = trade.get("dca_count", 0)
        
        # Check if exceeds global limit
        if dca_count > MAX_DCA_COUNT_GLOBAL:
            violations.append({
                "symbol": symbol,
                "dca_count": dca_count,
                "violation": f"Exceeds global limit ({dca_count} > {MAX_DCA_COUNT_GLOBAL})"
            })
    
    if violations:
        print(f"  🚨 VIOLATIONS FOUND:")
        for violation in violations:
            print(f"    {violation['symbol']}: {violation['violation']}")
    else:
        print(f"  ✅ All trades within DCA limits")
    
    return violations

def show_dca_configuration():
    """Show current DCA configuration"""
    print(f"\n⚙️ DCA CONFIGURATION:")
    print(f"  Global max DCAs per trade: {MAX_DCA_COUNT_GLOBAL}")
    
    for trade_type, config in DCA_CONFIG.items():
        print(f"  {trade_type}:")
        print(f"    Max adds: {config['max_adds']}")
        print(f"    Add size: {config['add_size_pct']}% of original")
        print(f"    Trigger: -{config['trigger_drop_pct']}% drop")

def fix_dca_violations():
    """Fix any DCA violations by capping at maximum"""
    print(f"\n🔧 FIXING DCA VIOLATIONS:")
    
    fixes_made = 0
    
    for symbol, trade in active_trades.items():
        if trade.get("exited"):
            continue
            
        dca_count = trade.get("dca_count", 0)
        
        # Cap DCA count at global maximum
        if dca_count > MAX_DCA_COUNT_GLOBAL:
            print(f"  🔧 Fixing {symbol}: capping DCA count from {dca_count} to {MAX_DCA_COUNT_GLOBAL}")
            trade["dca_count"] = MAX_DCA_COUNT_GLOBAL
            trade["dca_capped"] = True
            trade["dca_cap_reason"] = f"Exceeded global limit ({dca_count} > {MAX_DCA_COUNT_GLOBAL})"
            fixes_made += 1
    
    if fixes_made > 0:
        print(f"  ✅ Fixed {fixes_made} DCA violations")
        
        # Save the fixes
        try:
            from monitor import save_active_trades
            save_active_trades()
            print(f"  💾 Saved fixes to active_trades.json")
        except Exception as e:
            print(f"  ❌ Error saving fixes: {e}")
    else:
        print(f"  ✅ No violations to fix")
    
    return fixes_made

async def main():
    """Run all DCA limit checks"""
    print("🔍 COMPREHENSIVE DCA LIMIT CHECK")
    print("=" * 50)
    
    # Show configuration
    show_dca_configuration()
    
    # Check current status
    status = check_current_dca_status()
    
    # Validate limits
    violations = validate_dca_limits()
    
    # Fix violations if any
    if violations:
        fix_dca_violations()
    
    print(f"\n✅ DCA LIMIT CHECK COMPLETED")
    
    # Recommendations
    if status["max_limit_reached"] > 0:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  - {status['max_limit_reached']} trades have reached maximum DCAs")
        print(f"  - Consider closing some positions to free up capital")
        print(f"  - Monitor these trades closely for exit opportunities")

if __name__ == "__main__":
    asyncio.run(main())
