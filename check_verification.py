import asyncio
import json
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import bot modules - adjust if needed
try:
    from bybit_api import signed_request
    from monitor import active_trades, save_active_trades
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this from your bot directory")
    sys.exit(1)

async def verify_positions():
    """Get actual positions and fix ghost trades"""
    logger.info("🔍 Starting trade verification...")
    
    # 1. Count active trades in memory
    active_count = sum(1 for t in active_trades.values() if not t.get("exited"))
    logger.info(f"📊 Active trades in memory: {active_count}")
    
    # 2. Get actual positions from Bybit
    try:
        # FIX: Use settleCoin=USDT since we need to provide either symbol or settleCoin
        response = await signed_request("GET", "/v5/position/list", {
            "category": "linear", 
            "settleCoin": "USDT"  # Added this parameter
        })
        
        if response.get("retCode") != 0:
            logger.error(f"API error: {response.get('retMsg')}")
            return
            
        positions = response.get("result", {}).get("list", [])
        active_positions = []
        
        for pos in positions:
            symbol = pos.get("symbol")
            size = float(pos.get("size", "0"))
            if abs(size) > 0:
                active_positions.append(symbol)
                
        logger.info(f"📊 Actual positions on Bybit: {len(active_positions)}")
        logger.info(f"Active positions: {active_positions}")
        
        # 3. Fix ghost trades
        ghost_count = 0
        
        for symbol, trade in list(active_trades.items()):
            if not trade.get("exited") and symbol not in active_positions:
                logger.info(f"🔍 Ghost trade detected: {symbol} - marking as exited")
                trade["exited"] = True
                ghost_count += 1
        
        if ghost_count > 0:
            logger.info(f"🧹 Fixed {ghost_count} ghost trades")
            save_active_trades()
        else:
            logger.info("✅ No ghost trades found")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(verify_positions())
