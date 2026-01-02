#!/usr/bin/env python3
"""
unified_exit_manager.py - SIMPLIFIED VERSION
Strategy: TP1 → Breakeven → Trailing Stop → Let Winners Run

This is the optimal approach:
1. Set initial SL
2. Wait for TP1 level
3. When TP1 hit: Move SL to breakeven
4. Activate trailing stop
5. Let profits run until trailing catches the exit
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from logger import log

# Dependency injection to avoid circular imports
_save_trades_func: Optional[Callable] = None
_update_sl_func: Optional[Callable] = None


def set_dependencies(save_func: Callable = None, update_sl_func: Callable = None):
    """Set dependencies to avoid circular imports"""
    global _save_trades_func, _update_sl_func
    _save_trades_func = save_func
    _update_sl_func = update_sl_func
    log("✅ Unified exit manager dependencies set")


# SIMPLIFIED EXIT CONFIGURATION
# Only what we need: SL, TP1 trigger, and Trailing
EXIT_CONFIG = {
    "Scalp": {
        "sl_pct": 0.8,           # Initial stop loss: -0.8%
        "tp1_trigger_pct": 1.2,  # TP1 level to trigger breakeven + trailing: +1.2%
        "trailing_pct": 0.5,     # Trailing stop distance: 0.5%
        "breakeven_buffer": 0.1  # Buffer above entry for breakeven: +0.1%
    },
    "Intraday": {
        "sl_pct": 1.0,           # Initial stop loss: -1.0%
        "tp1_trigger_pct": 2.0,  # TP1 level: +2.0%
        "trailing_pct": 0.8,     # Trailing stop: 0.8%
        "breakeven_buffer": 0.1  # Breakeven buffer: +0.1%
    },
    "Swing": {
        "sl_pct": 1.5,           # Initial stop loss: -1.5%
        "tp1_trigger_pct": 3.5,  # TP1 level: +3.5%
        "trailing_pct": 1.2,     # Trailing stop: 1.2%
        "breakeven_buffer": 0.15 # Breakeven buffer: +0.15%
    },
    "CoreScalp": {
        "sl_pct": 0.8,
        "tp1_trigger_pct": 1.2,
        "trailing_pct": 0.5,
        "breakeven_buffer": 0.1
    },
    "CoreIntraday": {
        "sl_pct": 1.0,
        "tp1_trigger_pct": 2.0,
        "trailing_pct": 0.8,
        "breakeven_buffer": 0.1
    },
    "CoreSwing": {
        "sl_pct": 1.5,
        "tp1_trigger_pct": 3.5,
        "trailing_pct": 1.2,
        "breakeven_buffer": 0.15
    },
    "Default": {
        "sl_pct": 1.0,
        "tp1_trigger_pct": 2.0,
        "trailing_pct": 0.8,
        "breakeven_buffer": 0.1
    }
}


class UnifiedExitManager:
    """
    Simplified exit manager using the optimal strategy:
    Entry → TP1 → Breakeven → Trailing → Exit
    """
    
    def __init__(self):
        # Track highest price (for longs) or lowest price (for shorts) after TP1
        self.trailing_highs: Dict[str, float] = {}
        self.trailing_lows: Dict[str, float] = {}
        
        # Track which trades have hit TP1 (activates breakeven + trailing)
        self.tp1_activated: Dict[str, bool] = {}
        
        # Track breakeven SL prices
        self.breakeven_prices: Dict[str, float] = {}
        
        # Throttle saves
        self.last_save: Dict[str, float] = {}
    
    def get_config(self, trade: Dict) -> Dict:
        """Get exit config for trade type"""
        trade_type = (
            trade.get("strategy_type") or 
            trade.get("trade_type") or 
            trade.get("type") or 
            "Default"
        )
        # Normalize - remove "Core" prefix for lookup if not found
        if trade_type not in EXIT_CONFIG:
            trade_type = trade_type.replace("Core", "")
        if trade_type not in EXIT_CONFIG:
            trade_type = "Default"
        
        return EXIT_CONFIG[trade_type]
    
    async def process_exit(
        self, 
        symbol: str, 
        trade: Dict, 
        current_price: float
    ) -> bool:
        """
        Main exit processing - SIMPLIFIED LOGIC
        
        Flow:
        1. Check hard SL (initial or breakeven)
        2. Check if TP1 hit → activate breakeven + trailing
        3. If trailing active, check trailing stop
        4. Update trailing high/low
        
        Returns True if trade exited, False otherwise
        """
        try:
            # Skip if already exited
            if trade.get("exited"):
                return False
            
            # Validate inputs
            direction = trade.get("direction", "").lower()
            entry_price = float(trade.get("entry_price", 0))
            
            if not direction or not entry_price or not current_price:
                return False
            
            # Get config for this trade type
            config = self.get_config(trade)
            
            # Calculate current PnL
            if direction == "long":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Update trade state
            trade["current_pnl_pct"] = round(pnl_pct, 2)
            trade["current_price"] = current_price
            trade["last_check"] = datetime.now().isoformat()
            
            # === STEP 1: Check Stop Loss ===
            if await self._check_stop_loss(symbol, trade, current_price, pnl_pct, config, direction, entry_price):
                return True
            
            # === STEP 2: Check TP1 Trigger (activates breakeven + trailing) ===
            self._check_tp1_trigger(symbol, trade, current_price, pnl_pct, config, direction, entry_price)
            
            # === STEP 3: Check Trailing Stop (only if TP1 activated) ===
            if self.tp1_activated.get(symbol, False):
                if await self._check_trailing_stop(symbol, trade, current_price, pnl_pct, config, direction):
                    return True
                
                # Update trailing high/low
                self._update_trailing_level(symbol, current_price, direction)
            
            # Save state
            self._save_state(symbol, trade)
            
            return False
            
        except Exception as e:
            log(f"❌ Exit manager error for {symbol}: {e}", level="ERROR")
            import traceback
            log(traceback.format_exc(), level="ERROR")
            return False
    
    async def _check_stop_loss(
        self, 
        symbol: str, 
        trade: Dict, 
        current_price: float, 
        pnl_pct: float, 
        config: Dict,
        direction: str,
        entry_price: float
    ) -> bool:
        """Check if stop loss hit (either initial SL or breakeven SL)"""
        
        # If TP1 activated, use breakeven SL
        if self.tp1_activated.get(symbol, False):
            breakeven_sl = self.breakeven_prices.get(symbol)
            if breakeven_sl:
                if direction == "long" and current_price <= breakeven_sl:
                    await self._exit_trade(symbol, trade, current_price, "breakeven_stop", pnl_pct)
                    return True
                elif direction == "short" and current_price >= breakeven_sl:
                    await self._exit_trade(symbol, trade, current_price, "breakeven_stop", pnl_pct)
                    return True
        
        # Check initial SL (only if TP1 not yet activated)
        else:
            sl_pct = config["sl_pct"]
            if pnl_pct <= -sl_pct:
                await self._exit_trade(symbol, trade, current_price, "stop_loss", pnl_pct)
                return True
        
        return False
    
    def _check_tp1_trigger(
        self, 
        symbol: str, 
        trade: Dict, 
        current_price: float, 
        pnl_pct: float, 
        config: Dict,
        direction: str,
        entry_price: float
    ):
        """Check if TP1 level reached - activates breakeven + trailing"""
        
        # Skip if already activated
        if self.tp1_activated.get(symbol, False):
            return
        
        tp1_pct = config["tp1_trigger_pct"]
        
        # Check if TP1 reached
        if pnl_pct >= tp1_pct:
            self.tp1_activated[symbol] = True
            trade["tp1_hit"] = True
            trade["tp1_hit_time"] = datetime.now().isoformat()
            trade["tp1_hit_price"] = current_price
            
            # Calculate breakeven price (entry + small buffer for fees)
            buffer_pct = config["breakeven_buffer"]
            if direction == "long":
                breakeven_price = entry_price * (1 + buffer_pct / 100)
            else:
                breakeven_price = entry_price * (1 - buffer_pct / 100)
            
            self.breakeven_prices[symbol] = breakeven_price
            trade["breakeven_sl"] = breakeven_price
            
            # Initialize trailing from current price
            if direction == "long":
                self.trailing_highs[symbol] = current_price
            else:
                self.trailing_lows[symbol] = current_price
            
            log(f"🎯 {symbol}: TP1 HIT at {pnl_pct:.2f}%!")
            log(f"   → SL moved to breakeven: {breakeven_price:.4f}")
            log(f"   → Trailing stop ACTIVATED")
            
            # Try to update SL on exchange
            asyncio.create_task(self._update_exchange_sl(symbol, trade, breakeven_price))
    
    async def _check_trailing_stop(
        self, 
        symbol: str, 
        trade: Dict, 
        current_price: float, 
        pnl_pct: float, 
        config: Dict,
        direction: str
    ) -> bool:
        """Check if trailing stop triggered"""
        
        trailing_pct = config["trailing_pct"]
        
        if direction == "long":
            trailing_high = self.trailing_highs.get(symbol, current_price)
            trailing_stop = trailing_high * (1 - trailing_pct / 100)
            
            if current_price <= trailing_stop:
                log(f"📉 {symbol}: Trailing stop hit! High: {trailing_high:.4f}, Stop: {trailing_stop:.4f}, Current: {current_price:.4f}")
                await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                return True
        else:
            trailing_low = self.trailing_lows.get(symbol, current_price)
            trailing_stop = trailing_low * (1 + trailing_pct / 100)
            
            if current_price >= trailing_stop:
                log(f"📈 {symbol}: Trailing stop hit! Low: {trailing_low:.4f}, Stop: {trailing_stop:.4f}, Current: {current_price:.4f}")
                await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                return True
        
        return False
    
    def _update_trailing_level(self, symbol: str, current_price: float, direction: str):
        """Update trailing high/low as price moves in our favor"""
        
        if direction == "long":
            old_high = self.trailing_highs.get(symbol, current_price)
            if current_price > old_high:
                self.trailing_highs[symbol] = current_price
                log(f"📈 {symbol}: New trailing high: {old_high:.4f} → {current_price:.4f}")
        else:
            old_low = self.trailing_lows.get(symbol, current_price)
            if current_price < old_low:
                self.trailing_lows[symbol] = current_price
                log(f"📉 {symbol}: New trailing low: {old_low:.4f} → {current_price:.4f}")
    
    async def _exit_trade(
        self, 
        symbol: str, 
        trade: Dict, 
        current_price: float, 
        reason: str, 
        pnl_pct: float
    ):
        """Execute trade exit"""
        
        log(f"🚪 EXIT: {symbol} | Reason: {reason} | PnL: {pnl_pct:+.2f}%")
        
        # Mark trade as exited
        trade["exited"] = True
        trade["exit_reason"] = reason
        trade["exit_price"] = current_price
        trade["exit_time"] = datetime.now().isoformat()
        trade["final_pnl_pct"] = round(pnl_pct, 2)
        
        # Calculate USDT P&L
        entry_price = float(trade.get("entry_price", 0))
        qty = float(trade.get("qty", 0))
        direction = trade.get("direction", "").lower()
        
        if entry_price > 0 and qty > 0:
            if direction == "long":
                pnl_usdt = (current_price - entry_price) * qty
            else:
                pnl_usdt = (entry_price - current_price) * qty
            
            trade["final_pnl_usdt"] = round(pnl_usdt, 2)
            log(f"💰 P&L: {pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)")
        
        # Cleanup tracking
        self._cleanup(symbol)
        
        # Close on exchange
        await self._close_on_exchange(symbol, trade)
        
        # Send notification
        await self._send_notification(symbol, trade, reason, pnl_pct)
        
        # Save
        if _save_trades_func:
            _save_trades_func()
    
    def _cleanup(self, symbol: str):
        """Clean up tracking data for symbol"""
        self.trailing_highs.pop(symbol, None)
        self.trailing_lows.pop(symbol, None)
        self.tp1_activated.pop(symbol, None)
        self.breakeven_prices.pop(symbol, None)
        self.last_save.pop(symbol, None)
    
    async def _close_on_exchange(self, symbol: str, trade: Dict):
        """Close position on Bybit"""
        try:
            from bybit_api import signed_request
            
            direction = trade.get("direction", "").lower()
            qty = trade.get("qty", 0)
            
            if not qty:
                return
            
            side = "Sell" if direction == "long" else "Buy"
            
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
                log(f"✅ {symbol}: Position closed on exchange")
            else:
                log(f"❌ {symbol}: Failed to close - {response.get('retMsg')}", level="ERROR")
                
        except Exception as e:
            log(f"❌ {symbol}: Exchange close error - {e}", level="ERROR")
    
    async def _update_exchange_sl(self, symbol: str, trade: Dict, new_sl: float):
        """Update SL on exchange"""
        try:
            if _update_sl_func:
                await _update_sl_func(symbol, trade, new_sl)
                log(f"✅ {symbol}: Exchange SL updated to {new_sl:.4f}")
        except Exception as e:
            log(f"⚠️ {symbol}: Failed to update exchange SL - {e}", level="WARN")
    
    async def _send_notification(self, symbol: str, trade: Dict, reason: str, pnl_pct: float):
        """Send Telegram notification"""
        try:
            from error_handler import send_telegram_message
            
            # Emoji based on result
            if pnl_pct >= 2.0:
                emoji = "🟢🎉"
            elif pnl_pct > 0:
                emoji = "🟢"
            elif pnl_pct > -0.5:
                emoji = "🟡"
            else:
                emoji = "🔴"
            
            msg = f"{emoji} <b>TRADE EXIT</b>\n\n"
            msg += f"📊 Symbol: <b>{symbol}</b>\n"
            msg += f"📈 Direction: <b>{trade.get('direction', '').upper()}</b>\n"
            msg += f"🚪 Reason: <b>{reason.replace('_', ' ').title()}</b>\n"
            msg += f"💰 Entry: {trade.get('entry_price')}\n"
            msg += f"🎯 Exit: {trade.get('exit_price')}\n"
            msg += f"💵 PnL: <b>{pnl_pct:+.2f}%</b>"
            
            if trade.get("final_pnl_usdt"):
                msg += f" ({trade['final_pnl_usdt']:+.2f} USDT)"
            
            await send_telegram_message(msg)
            
        except Exception as e:
            log(f"⚠️ Notification error: {e}", level="WARN")
    
    def _save_state(self, symbol: str, trade: Dict):
        """Save trade state (throttled)"""
        now = time.time()
        if now - self.last_save.get(symbol, 0) < 5:
            return
        self.last_save[symbol] = now
        
        if _save_trades_func:
            _save_trades_func()
    
    def get_status(self, symbol: str) -> Dict:
        """Get current status for a symbol"""
        return {
            "tp1_activated": self.tp1_activated.get(symbol, False),
            "breakeven_sl": self.breakeven_prices.get(symbol),
            "trailing_high": self.trailing_highs.get(symbol),
            "trailing_low": self.trailing_lows.get(symbol)
        }
    
    def reset_symbol(self, symbol: str):
        """Reset tracking for a symbol (manual close)"""
        self._cleanup(symbol)
        log(f"🔄 {symbol}: Tracking reset")


# Global instance
exit_manager = UnifiedExitManager()


# === PUBLIC API ===

async def process_trade_exits(symbol: str, trade: Dict, current_price: float) -> bool:
    """
    MAIN FUNCTION - Call this from monitor.py
    
    Returns True if trade was exited
    """
    return await exit_manager.process_exit(symbol, trade, current_price)


def get_exit_status(symbol: str) -> Dict:
    """Get exit status for a symbol"""
    return exit_manager.get_status(symbol)


def reset_symbol_tracking(symbol: str):
    """Reset tracking for a symbol"""
    exit_manager.reset_symbol(symbol)


def get_exit_config(trade_type: str) -> Dict:
    """Get exit config for a trade type"""
    if trade_type not in EXIT_CONFIG:
        trade_type = trade_type.replace("Core", "")
    return EXIT_CONFIG.get(trade_type, EXIT_CONFIG["Default"])


# Exports
__all__ = [
    'process_trade_exits',
    'get_exit_status',
    'reset_symbol_tracking',
    'get_exit_config',
    'set_dependencies',
    'EXIT_CONFIG'
]


if __name__ == "__main__":
    print("✅ Simplified unified_exit_manager.py loaded")
    print("\n📊 Exit Configuration:")
    print("=" * 50)
    for trade_type, config in EXIT_CONFIG.items():
        if trade_type.startswith("Core"):
            continue  # Skip Core variants (same as base)
        print(f"\n{trade_type}:")
        print(f"  SL: -{config['sl_pct']}%")
        print(f"  TP1 Trigger: +{config['tp1_trigger_pct']}%")
        print(f"  Trailing: {config['trailing_pct']}%")
        print(f"  Breakeven Buffer: +{config['breakeven_buffer']}%")
