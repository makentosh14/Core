# enhanced_dca_protection.py
import asyncio
import time
from datetime import datetime, timedelta
from config import FAST_DROP_PROTECTION, ENABLE_FAST_DROP_PROTECTION
from logger import log

class EnhancedDCAProtection:
    def __init__(self):
        self.price_history = {}
        self.fast_drop_detected = {}
        self.sl_paused_until = {}
        
    async def detect_fast_drop(self, symbol, current_price):
        """Detect if we're in a fast drop scenario"""
        if not ENABLE_FAST_DROP_PROTECTION:
            return False
            
        try:
            now = datetime.utcnow()
            
            # Initialize price history for symbol
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            # Add current price with timestamp
            self.price_history[symbol].append({
                'price': current_price,
                'timestamp': now
            })
            
            # Keep only monitoring window of data
            cutoff_time = now - timedelta(minutes=FAST_DROP_PROTECTION["monitoring_window"])
            self.price_history[symbol] = [
                p for p in self.price_history[symbol] 
                if p['timestamp'] > cutoff_time
            ]
            
            # Need minimum data points
            if len(self.price_history[symbol]) < FAST_DROP_PROTECTION["min_data_points"]:
                return False
            
            # Calculate price change velocity
            prices = [p['price'] for p in self.price_history[symbol]]
            time_span = (now - self.price_history[symbol][0]['timestamp']).total_seconds() / 60
            
            if time_span < 1:  # Less than 1 minute of data
                return False
                
            price_change_pct = abs(prices[-1] - prices[0]) / prices[0] * 100
            velocity = price_change_pct / time_span  # % change per minute
            
            # Fast drop detected if velocity exceeds threshold
            is_fast_drop = velocity > FAST_DROP_PROTECTION["velocity_threshold"]
            
            if is_fast_drop:
                self.fast_drop_detected[symbol] = now
                pause_until = now + timedelta(seconds=FAST_DROP_PROTECTION["pause_duration"])
                self.sl_paused_until[symbol] = pause_until
                log(f"⚡ Fast drop detected for {symbol}: {velocity:.2f}%/min - SL paused")
                return True
                
            return False
            
        except Exception as e:
            log(f"❌ Error detecting fast drop for {symbol}: {e}", level="ERROR")
            return False
    
    async def should_pause_stop_loss(self, symbol, trade, current_price):
        """Determine if we should pause normal stop loss execution"""
        if not ENABLE_FAST_DROP_PROTECTION:
            return False
            
        try:
            now = datetime.utcnow()
            
            # Check if SL is currently paused due to fast drop
            if symbol in self.sl_paused_until:
                if now < self.sl_paused_until[symbol]:
                    return True
                else:
                    # Pause period expired, remove it
                    del self.sl_paused_until[symbol]
            
            # Check if we're near DCA trigger point
            entry_price = trade.get("entry_price")
            direction = trade.get("direction", "").lower()
            
            if not entry_price:
                return False
            
            # Calculate current drawdown
            if direction == "long":
                drawdown_pct = ((entry_price - current_price) / entry_price) * 100
            else:
                drawdown_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Get DCA config
            trade_type = trade.get("trade_type", "Intraday")
            dca_config = {
                "Scalp": {"trigger_drop_pct": 0.4},
                "Intraday": {"trigger_drop_pct": 0.6},
                "Swing": {"trigger_drop_pct": 1.3}
            }
            
            trigger_threshold = dca_config.get(trade_type, dca_config["Intraday"])["trigger_drop_pct"]
            
            # Pause SL if we're within buffer distance of DCA trigger
            if drawdown_pct >= (trigger_threshold - FAST_DROP_PROTECTION["dca_buffer"]):
                log(f"🛡️ SL paused for {symbol} - Near DCA trigger: {drawdown_pct:.2f}% >= {trigger_threshold - FAST_DROP_PROTECTION['dca_buffer']:.2f}%")
                return True
                
            return False
            
        except Exception as e:
            log(f"❌ Error checking SL pause for {symbol}: {e}", level="ERROR")
            return False

# Global protection manager instance
protection_manager = EnhancedDCAProtection()
