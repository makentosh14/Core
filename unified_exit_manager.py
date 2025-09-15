# unified_exit_manager.py
# Single source of truth for ALL TP and trailing SL logic
# This replaces all scattered logic across multiple files

import asyncio
import traceback
from datetime import datetime
from logger import log, write_log
from bybit_api import place_market_order, update_stop_loss_order
from error_handler import send_telegram_message
from activity_logger import log_trade_to_file

# SINGLE SOURCE OF TRUTH - Fixed percentages
FIXED_PERCENTAGES = {
    "Scalp": {
        "tp1_pct": 1.2,      # +1.2% take profit
        "sl_pct": 0.8,       # -0.8% stop loss
        "trailing_pct": 0.4  # 0.4% trailing stop
    },
    "Intraday": {
        "tp1_pct": 2.0,      # +2.0% take profit
        "sl_pct": 1.0,       # -1.0% stop loss
        "trailing_pct": 1.0  # 1.0% trailing stop
    },
    "Swing": {
        "tp1_pct": 3.5,      # +3.5% take profit
        "sl_pct": 1.5,       # -1.5% stop loss
        "trailing_pct": 1.5  # 1.5% trailing stop
    }
}

# Global state to prevent double execution
_processing_symbols = set()

class UnifiedExitManager:
    """Single manager for all exit logic - prevents double execution"""
    
    def __init__(self):
        self.processing_symbols = set()
    
    async def process_trade_exit_logic(self, symbol, trade, current_price, candles=None):
        """
        MAIN FUNCTION: Process all exit logic for a trade
        This is the ONLY function that should be called from monitor.py and scanner.py
        """
        # Prevent double execution
        if symbol in self.processing_symbols:
            return False
            
        try:
            self.processing_symbols.add(symbol)
            
            # Skip if trade is already exited
            if trade.get("exited"):
                return False
            
            direction = trade.get("direction", "").lower()
            entry_price = trade.get("entry_price")
            
            if not entry_price or not direction:
                return False
            
            # Step 1: Check TP1 hit (only if not already hit)
            if not trade.get("tp1_hit"):
                tp1_hit = self._check_tp1_hit(trade, current_price, candles)
                if tp1_hit:
                    success = await self._handle_tp1_hit(symbol, trade, current_price)
                    if success:
                        # Save the trade state after TP1
                        self._save_trade_state(symbol, trade)
                        return True
            
            # Step 2: Handle trailing stop (only after TP1)
            elif trade.get("tp1_hit") and not trade.get("exited"):
                # Check if trailing SL hit
                trailing_hit = self._check_trailing_sl_hit(trade, current_price, direction)
                if trailing_hit:
                    success = await self._handle_trailing_sl_exit(symbol, trade, current_price)
                    if success:
                        self._save_trade_state(symbol, trade)
                        return True
                
                # Update trailing stop if not hit
                else:
                    updated = await self._update_trailing_stop(symbol, trade, current_price, candles)
                    if updated:
                        self._save_trade_state(symbol, trade)
                        return True
            
            return False
            
        except Exception as e:
            log(f"❌ Error in unified exit manager for {symbol}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            return False
        finally:
            # Always remove from processing set
            self.processing_symbols.discard(symbol)
    
    def _check_tp1_hit(self, trade, current_price, candles):
        """Check if TP1 has been hit"""
        direction = trade.get("direction", "").lower()
        entry_price = trade.get("entry_price")
        trade_type = trade.get("trade_type", "Intraday")
        
        # Get TP1 percentage from fixed values
        tp1_pct = FIXED_PERCENTAGES.get(trade_type, FIXED_PERCENTAGES["Intraday"])["tp1_pct"]
        
        # Calculate TP1 level
        if direction == "long":
            tp1_level = entry_price * (1 + tp1_pct / 100)
            price_hit = current_price >= tp1_level
        else:  # short
            tp1_level = entry_price * (1 - tp1_pct / 100)
            price_hit = current_price <= tp1_level
        
        # Check current price first
        if price_hit:
            log(f"🎯 TP1 hit detected for {trade.get('symbol')} at {current_price} (target: {tp1_level})")
            return True
        
        # Check candle wicks if available
        if candles and len(candles) >= 1:
            last_candle = candles[-1]
            
            if direction == "long" and float(last_candle["high"]) >= tp1_level:
                log(f"🎯 TP1 hit via wick for {trade.get('symbol')}: high {last_candle['high']} >= {tp1_level}")
                return True
            elif direction == "short" and float(last_candle["low"]) <= tp1_level:
                log(f"🎯 TP1 hit via wick for {trade.get('symbol')}: low {last_candle['low']} <= {tp1_level}")
                return True
        
        return False
    
    async def _handle_tp1_hit(self, symbol, trade, current_price):
        """Handle TP1 hit - Exit 50% and activate trailing"""
        try:
            log(f"🎯 Processing TP1 hit for {symbol} at {current_price}")
            
            # Mark TP1 as hit FIRST
            trade["tp1_hit"] = True
            trade["tp1_price_actual"] = current_price
            trade["tp1_hit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get trade details
            direction = trade.get("direction", "").lower()
            entry_price = trade.get("entry_price")
            total_qty = trade.get("qty", 0)
            
            # Calculate 50% exit quantity
            exit_qty = total_qty * 0.5
            remaining_qty = total_qty - exit_qty
            
            # Execute 50% exit
            exit_side = "Sell" if direction == "long" else "Buy"
            
            log(f"💰 Executing 50% TP1 exit: {exit_qty} {symbol}")
            
            exit_order = await place_market_order(
                symbol=symbol,
                side=exit_side,
                qty=exit_qty,
                reduce_only=True
            )
            
            if exit_order:
                log(f"✅ TP1 50% exit successful for {symbol}")
                
                # Update trade quantities
                trade["qty"] = remaining_qty
                trade["tp1_exit_qty"] = exit_qty
                
                # Move SL to breakeven for remaining 50%
                sl_updated = await update_stop_loss_order(symbol, trade, entry_price)
                
                if sl_updated:
                    log(f"🛡️ SL moved to breakeven for {symbol}: {entry_price}")
                
                # Initialize trailing stop at breakeven
                trade["trailing_active"] = True
                trade["trailing_sl"] = entry_price
                
                # Calculate profit for the 50% exit
                if direction == "long":
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                else:
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                
                # Send notification
                await send_telegram_message(
                    f"🎯 <b>TP1 Hit</b> - <b>{symbol}</b>\n"
                    f"💰 Exit Price: {current_price:.6f}\n"
                    f"📦 50% Position Exited: {exit_qty}\n"
                    f"📍 50% Remaining: {remaining_qty}\n"
                    f"📈 Profit on Exit: +{profit_pct:.2f}%\n"
                    f"🛡️ SL Moved to Breakeven: {entry_price:.6f}\n"
                    f"📈 Trailing Stop Activated"
                )
                
                return True
            else:
                log(f"❌ Failed to execute TP1 exit for {symbol}", level="ERROR")
                return False
                
        except Exception as e:
            log(f"❌ Error handling TP1 hit for {symbol}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            return False
    
    def _check_trailing_sl_hit(self, trade, current_price, direction):
        """Check if trailing SL has been hit"""
        trailing_sl = trade.get("trailing_sl")
        if not trailing_sl:
            return False
        
        # Add small buffer to prevent false triggers
        buffer = 0.001  # 0.1%
        
        if direction == "long":
            return current_price <= trailing_sl * (1 - buffer)
        else:  # short
            return current_price >= trailing_sl * (1 + buffer)
    
    async def _handle_trailing_sl_exit(self, symbol, trade, current_price):
        """Handle trailing SL exit - Close remaining position"""
        try:
            log(f"💔 Processing trailing SL exit for {symbol} at {current_price}")
            
            direction = trade.get("direction", "").lower()
            entry_price = trade.get("entry_price")
            remaining_qty = trade.get("qty", 0)
            
            # Calculate final profit
            if direction == "long":
                profit_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Execute remaining position exit
            exit_side = "Sell" if direction == "long" else "Buy"
            
            exit_order = await place_market_order(
                symbol=symbol,
                side=exit_side,
                qty=remaining_qty,
                reduce_only=True
            )
            
            if exit_order:
                log(f"✅ Trailing SL exit executed for {symbol}: {remaining_qty} @ {current_price}")
                
                # Mark trade as exited
                trade["exited"] = True
                trade["exit_price"] = current_price
                trade["exit_reason"] = "Trailing_SL_Hit"
                trade["profit_pct"] = profit_pct
                trade["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Send notification
                await send_telegram_message(
                    f"💔 <b>Trailing SL Hit</b> - <b>{symbol}</b>\n"
                    f"💰 Exit Price: {current_price:.6f}\n"
                    f"📦 Position Closed: {remaining_qty}\n"
                    f"📈 Total Profit: +{profit_pct:.2f}%\n"
                    f"🎯 Strategy: 50% TP1 + 50% Trailing"
                )
                
                # Log to activity file
                log_trade_to_file(
                    symbol=symbol,
                    direction=direction,
                    entry=entry_price,
                    sl=trade.get("trailing_sl"),
                    tp1=trade.get("tp1_price_actual"),
                    tp2=None,
                    result="trailing_sl_win" if profit_pct > 0 else "trailing_sl_loss",
                    score=trade.get("score", 0),
                    trade_type=trade.get("trade_type", "Unknown"),
                    confidence=trade.get("confidence", 0)
                )
                
                return True
            else:
                log(f"❌ Failed to execute trailing SL exit for {symbol}", level="ERROR")
                return False
                
        except Exception as e:
            log(f"❌ Error handling trailing SL exit for {symbol}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            return False
    
    async def _update_trailing_stop(self, symbol, trade, current_price, candles):
        """Update trailing stop if conditions are met"""
        try:
            direction = trade.get("direction", "").lower()
            entry_price = trade.get("entry_price")
            current_trailing_sl = trade.get("trailing_sl")
            trade_type = trade.get("trade_type", "Intraday")
            
            # Get trailing percentage from fixed values
            trailing_pct = FIXED_PERCENTAGES.get(trade_type, FIXED_PERCENTAGES["Intraday"])["trailing_pct"]
            
            # Calculate new trailing SL
            if direction == "long":
                new_trailing_sl = current_price * (1 - trailing_pct / 100)
                # Only update if new SL is higher (better for long)
                if current_trailing_sl and new_trailing_sl <= current_trailing_sl:
                    return False
            else:  # short
                new_trailing_sl = current_price * (1 + trailing_pct / 100)
                # Only update if new SL is lower (better for short)
                if current_trailing_sl and new_trailing_sl >= current_trailing_sl:
                    return False
            
            # Round to appropriate precision
            new_trailing_sl = round(new_trailing_sl, 6)
            
            # Update the trade
            old_sl = trade.get("trailing_sl")
            trade["trailing_sl"] = new_trailing_sl
            trade["modified"] = True
            
            log(f"📈 Trailing SL updated for {symbol}: {old_sl} → {new_trailing_sl}")
            
            # Try to update the exchange order
            try:
                sl_updated = await update_stop_loss_order(symbol, trade, new_trailing_sl)
                if sl_updated:
                    log(f"✅ Exchange SL order updated for {symbol}")
                else:
                    log(f"⚠️ Failed to update exchange SL for {symbol}")
            except Exception as e:
                log(f"⚠️ Error updating exchange SL for {symbol}: {e}")
            
            return True
            
        except Exception as e:
            log(f"❌ Error updating trailing stop for {symbol}: {e}", level="ERROR")
            return False
    
    def _save_trade_state(self, symbol, trade):
        """Save trade state to file"""
        try:
            # Import here to avoid circular imports
            from monitor import save_active_trades
            save_active_trades()
            
        except Exception as e:
            log(f"❌ Error saving trade state for {symbol}: {e}", level="ERROR")

# Create global instance
exit_manager = UnifiedExitManager()

# MAIN FUNCTIONS TO CALL FROM OTHER FILES

async def process_trade_exits(symbol, trade, current_price, candles=None):
    """
    MAIN FUNCTION: Call this from monitor.py and active_trade_scanner.py
    This replaces ALL other exit handling functions
    """
    return await exit_manager.process_trade_exit_logic(symbol, trade, current_price, candles)

def validate_exit_configuration():
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

# Initialize on import
if __name__ == "__main__":
    validate_exit_configuration()
