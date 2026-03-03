# quick_reset_failed.py
import asyncio

async def reset():
    from trade_lock_manager import trade_lock_manager
    
    # Show current state
    blocked = {s: c for s, c in trade_lock_manager.failed_attempts.items() if c >= 3}
    print(f"Currently blocked symbols: {blocked}")
    
    # Reset failed attempts only (keeps real positions intact)
    trade_lock_manager.failed_attempts.clear()
    trade_lock_manager.failed_attempt_times = {}  # safe even if attr doesn't exist yet
    print("✅ Failed attempts cleared")
    
    # Re-sync with exchange to restore real position locks
    await trade_lock_manager.sync_with_exchange()
    print(f"✅ Re-synced. Confirmed trades: {trade_lock_manager.confirmed_trades}")

asyncio.run(reset())
