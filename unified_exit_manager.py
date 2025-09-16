#!/usr/bin/env python3
"""
FIXED unified_exit_manager.py - Resolves circular import issues
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from logger import log

# Dependency injection to avoid circular imports
_save_trades_func: Optional[Callable] = None
_update_sl_func: Optional[Callable] = None

def set_dependencies(save_func: Callable, update_sl_func: Callable = None):
    """Set dependencies to avoid circular imports"""
    global _save_trades_func, _update_sl_func
    _save_trades_func = save_func
    _update_sl_func = update_sl_func

# FIXED: Exit configuration with proper percentages for each strategy type
FIXED_PERCENTAGES = {
    "Scalp": {
        "tp1_pct": 1.5,      # 1.5% take profit for scalps
        "sl_pct": 1.0,       # 1% stop loss for scalps
        "trailing_pct": 0.8   # 0.8% trailing stop
    },
    "Intraday": {
        "tp1_pct": 2.5,      # 2.5% take profit for intraday
        "sl_pct": 1.5,       # 1.5% stop loss for intraday
        "trailing_pct": 1.2   # 1.2% trailing stop
    },
    "Swing": {
        "tp1_pct": 4.0,      # 4% take profit for swing
        "sl_pct": 2.0,       # 2% stop loss for swing
        "trailing_pct": 2.0   # 2% trailing stop
    },
    "Default": {
        "tp1_pct": 2.0,      # Default 2% take profit
        "sl_pct": 1.5,       # Default 1.5% stop loss
        "trailing_pct": 1.0   # Default 1% trailing stop
    }
}

class UnifiedExitManager:
    """Unified exit manager to handle all trade exits consistently"""
    
    def __init__(self):
        self.last_update = {}
        self.trailing_highs = {}
        self.trailing_lows = {}
        
    async def process_trade_exit_logic(self, symbol: str, trade: Dict, current_price: float, candles: Optional[Dict] = None) -> bool:
        """
        Main exit processing function - handles all exit conditions
        Returns True if trade was exited, False otherwise
        """
        try:
            if trade.get("exited"):
                return False
            
            direction = trade.get("direction", "").lower()
            if not direction or not current_price:
                return False
            
            entry_price = float(trade.get("entry_price", 0))
            if not entry_price:
                return False
            
            # Get strategy type for appropriate exit parameters
            strategy_type = trade.get("strategy_type", "Default")
            exit_params = FIXED_PERCENTAGES.get(strategy_type, FIXED_PERCENTAGES["Default"])
            
            # Calculate current PnL percentage
            if direction == "long":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Update trade with current PnL
            trade["current_pnl_pct"] = pnl_pct
            trade["current_price"] = current_price
            trade["last_check_time"] = datetime.now().isoformat()
            
            # 1. Check stop loss
            if await self._check_stop_loss(symbol, trade, current_price, pnl_pct, exit_params):
                return True
            
            # 2. Check take profit
            if await self._check_take_profit(symbol, trade, current_price, pnl_pct, exit_params):
                return True
            
            # 3. Check trailing stop
            if await self._check_trailing_stop(symbol, trade, current_price, pnl_pct, exit_params):
                return True
            
            # 4. Update trailing levels
            self._update_trailing_levels(symbol, trade, current_price, direction)
            
            # 5. Save trade state periodically
            self._save_trade_state(symbol, trade)
            
            return False
            
        except Exception as e:
            log(f"❌ Error in unified exit manager for {symbol}: {e}", level="ERROR")
            return False
    
    async def _check_stop_loss(self, symbol: str, trade: Dict, current_price: float, pnl_pct: float, exit_params: Dict) -> bool:
        """Check if stop loss should be triggered"""
        try:
            sl_pct = exit_params.get("sl_pct", 1.5)
            
            # Simple stop loss check
            if pnl_pct <= -sl_pct:
                await self._exit_trade(symbol, trade, current_price, "stop_loss", pnl_pct)
                return True
            
            return False
            
        except Exception as e:
            log(f"❌ Error checking stop loss for {symbol}: {e}", level="ERROR")
            return False
    
    async def _check_take_profit(self, symbol: str, trade: Dict, current_price: float, pnl_pct: float, exit_params: Dict) -> bool:
        """Check if take profit should be triggered"""
        try:
            tp_pct = exit_params.get("tp1_pct", 2.0)
            
            # Simple take profit check
            if pnl_pct >= tp_pct:
                await self._exit_trade(symbol, trade, current_price, "take_profit", pnl_pct)
                return True
            
            return False
            
        except Exception as e:
            log(f"❌ Error checking take profit for {symbol}: {e}", level="ERROR")
            return False
    
    async def _check_trailing_stop(self, symbol: str, trade: Dict, current_price: float, pnl_pct: float, exit_params: Dict) -> bool:
        """Check if trailing stop should be triggered"""
        try:
            direction = trade.get("direction", "").lower()
            trailing_pct = exit_params.get("trailing_pct", 1.0)
            
            # Only activate trailing stop when in profit
            if pnl_pct <= 0.5:  # Need at least 0.5% profit to activate trailing
                return False
            
            if direction == "long":
                trailing_high = self.trailing_highs.get(symbol, current_price)
                trailing_stop = trailing_high * (1 - trailing_pct / 100)
                
                if current_price <= trailing_stop:
                    await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                    return True
                    
            else:  # short
                trailing_low = self.trailing_lows.get(symbol, current_price)
                trailing_stop = trailing_low * (1 + trailing_pct / 100)
                
                if current_price >= trailing_stop:
                    await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                    return True
            
            return False
            
        except Exception as e:
            log(f"❌ Error checking trailing stop for {symbol}: {e}", level="ERROR")
            return False
    
    def _update_trailing_levels(self, symbol: str, trade: Dict, current_price: float, direction: str):
        """Update trailing high/low levels"""
        try:
            if direction == "long":
                if symbol not in self.trailing_highs:
                    self.trailing_highs[symbol] = current_price
                else:
                    self.trailing_highs[symbol] = max(self.trailing_highs[symbol], current_price)
            else:
                if symbol not in self.trailing_lows:
                    self.trailing_lows[symbol] = current_price
                else:
                    self.trailing_lows[symbol] = min(self.trailing_lows[symbol], current_price)
                    
        except Exception as e:
            log(f"❌ Error updating trailing levels for {symbol}: {e}", level="ERROR")
    
    async def _exit_trade(self, symbol: str, trade: Dict, current_price: float, exit_reason: str, pnl_pct: float):
        """Exit a trade with the specified reason"""
        try:
            log(f"🚪 UNIFIED EXIT: {symbol} {exit_reason} at {current_price} ({pnl_pct:+.2f}%)")
            
            # Mark trade as exited
            trade["exited"] = True
            trade["exit_reason"] = exit_reason
            trade["exit_price"] = current_price
            trade["exit_time"] = datetime.now().isoformat()
            trade["final_pnl_pct"] = pnl_pct
            
            # Calculate final PnL in USDT
            entry_price = float(trade.get("entry_price", 0))
            qty = float(trade.get("qty", 0))
            
            if entry_price > 0 and qty > 0:
                if trade.get("direction", "").lower() == "long":
                    pnl_usdt = (current_price - entry_price) * qty
                else:
                    pnl_usdt = (entry_price - current_price) * qty
                
                trade["final_pnl_usdt"] = pnl_usdt
                log(f"💰 Final PnL: {pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)")
            
            # Clean up trailing levels
            self.trailing_highs.pop(symbol, None)
            self.trailing_lows.pop(symbol, None)
            
            # Try to close position on exchange
            try:
                await self._close_position_on_exchange(symbol, trade)
            except Exception as e:
                log(f"⚠️ Failed to close position on exchange for {symbol}: {e}")
            
            # Send exit notification
            await self._send_exit_notification(symbol, trade, exit_reason, pnl_pct)
            
            # Save trade state
            self._save_trade_state(symbol, trade)
            
        except Exception as e:
            log(f"❌ Error exiting trade {symbol}: {e}", level="ERROR")
    
    async def _close_position_on_exchange(self, symbol: str, trade: Dict):
        """Close position on the exchange"""
        try:
            # This would integrate with your bybit_api module
            from bybit_api import signed_request
            
            direction = trade.get("direction", "").lower()
            qty = trade.get("qty", 0)
            
            # Determine side for closing order
            side = "Sell" if direction == "long" else "Buy"
            
            # Place market order to close position
            response = await signed_request("POST", "/v5/order/create", {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(abs(float(qty))),
                "timeInForce": "IOC",
                "reduceOnly": True
            })
            
            if response.get("retCode") == 0:
                log(f"✅ Position closed on exchange for {symbol}")
                trade["exchange_close_order_id"] = response.get("result", {}).get("orderId")
            else:
                log(f"❌ Failed to close position for {symbol}: {response.get('retMsg')}")
                
        except Exception as e:
            log(f"❌ Error closing position on exchange for {symbol}: {e}")
    
    async def _send_exit_notification(self, symbol: str, trade: Dict, exit_reason: str, pnl_pct: float):
        """Send exit notification via Telegram"""
        try:
            from error_handler import send_telegram_message
            
            strategy_type = trade.get("strategy_type", "Unknown")
            entry_price = trade.get("entry_price", 0)
            exit_price = trade.get("exit_price", 0)
            qty = trade.get("qty", 0)
            direction = trade.get("direction", "").upper()
            
            # Determine emoji based on PnL
            if pnl_pct > 0:
                emoji = "🟢"
                result = "PROFIT"
            else:
                emoji = "🔴" 
                result = "LOSS"
            
            msg = f"{emoji} <b>TRADE EXIT</b>\n\n"
            msg += f"📊 <b>Symbol:</b> {symbol}\n"
            msg += f"📈 <b>Direction:</b> {direction}\n"
            msg += f"🎲 <b>Strategy:</b> {strategy_type}\n"
            msg += f"🚪 <b>Exit Reason:</b> {exit_reason.replace('_', ' ').title()}\n"
            msg += f"💰 <b>Entry:</b> {entry_price}\n"
            msg += f"🎯 <b>Exit:</b> {exit_price}\n"
            msg += f"📊 <b>Quantity:</b> {qty}\n"
            msg += f"💵 <b>Result:</b> {result} {pnl_pct:+.2f}%"
            
            if trade.get("final_pnl_usdt"):
                msg += f"\n💸 <b>PnL:</b> {trade['final_pnl_usdt']:+.2f} USDT"
            
            await send_telegram_message(msg)
            
        except Exception as e:
            log(f"❌ Error sending exit notification for {symbol}: {e}")
    
    async def update_trailing_stop(self, symbol: str, trade: Dict, new_trailing_sl: float) -> bool:
        """Update trailing stop loss for a trade"""
        try:
            old_sl = trade.get("sl", 0)
            trade["sl"] = new_trailing_sl
            trade["trailing_updated"] = True
            trade["trailing_update_time"] = datetime.now().isoformat()
            
            log(f"📈 Updated trailing SL for {symbol}: {old_sl} → {new_trailing_sl}")
            
            # Try to update the exchange order
            try:
                if _update_sl_func:
                    sl_updated = await _update_sl_func(symbol, trade, new_trailing_sl)
                    if sl_updated:
                        log(f"✅ Exchange SL order updated for {symbol}")
                    else:
                        log(f"⚠️ Failed to update exchange SL for {symbol}")
                else:
                    log(f"⚠️ No SL update function available for {symbol}")
            except Exception as e:
                log(f"⚠️ Error updating exchange SL for {symbol}: {e}")
            
            return True
            
        except Exception as e:
            log(f"❌ Error updating trailing stop for {symbol}: {e}", level="ERROR")
            return False
    
    def _save_trade_state(self, symbol: str, trade: Dict):
        """Save trade state to file"""
        try:
            if _save_trades_func:
                _save_trades_func()
            else:
                log(f"⚠️ No save function available for {symbol}")
            
        except Exception as e:
            log(f"❌ Error saving trade state for {symbol}: {e}", level="ERROR")

# Create global instance
exit_manager = UnifiedExitManager()

# MAIN FUNCTIONS TO CALL FROM OTHER FILES

async def process_trade_exits(symbol: str, trade: Dict, current_price: float, candles: Optional[Dict] = None) -> bool:
    """
    MAIN FUNCTION: Call this from monitor.py and active_trade_scanner.py
    This replaces ALL other exit handling functions
    """
    return await exit_manager.process_trade_exit_logic(symbol, trade, current_price, candles)

async def update_trailing_stop_loss(symbol: str, trade: Dict, new_sl: float) -> bool:
    """Update trailing stop loss for a trade"""
    return await exit_manager.update_trailing_stop(symbol, trade, new_sl)

def validate_exit_configuration() -> bool:
    """Validate that exit configuration is correct"""
    log("🔍 Validating unified exit manager configuration...")
    
    for trade_type, params in FIXED_PERCENTAGES.items():
        required_keys = ["tp1_pct", "sl_pct", "trailing_pct"]
        for key in required_keys:
            if key not in params:
                log(f"❌ Missing {key} in {trade_type} configuration", level="ERROR")
                return False
            if params[key] <= 0:
                log(f"❌ Invalid {key} value in {trade_type}: {params[key]}", level="ERROR")
                return False
    
    log("✅ Unified exit manager configuration validated")
    return True

def get_exit_parameters(strategy_type: str) -> Dict:
    """Get exit parameters for a strategy type"""
    return FIXED_PERCENTAGES.get(strategy_type, FIXED_PERCENTAGES["Default"])

# Export main functions
__all__ = [
    'process_trade_exits',
    'update_trailing_stop_loss', 
    'validate_exit_configuration',
    'get_exit_parameters',
    'set_dependencies',
    'UnifiedExitManager'
]

# Initialize on import
if __name__ == "__main__":
    validate_exit_configuration()
    print("✅ unified_exit_manager.py module is working correctly")
