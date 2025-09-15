# pre_trade_validator.py - Final validation before placing trade

from typing import Dict, Optional, Tuple
from logger import log
import asyncio
from bybit_api import signed_request

class PreTradeValidator:
    """Final validation checks before placing a trade"""
    
    async def final_validation(self, symbol: str, direction: str, entry_price: float,
                             sl_price: float, tp_price: float) -> Tuple[bool, str]:
        """
        Perform final checks before trade execution
        
        Returns:
            Tuple of (is_valid, reason)
        """
        
        # 1. Check current market price hasn't moved too much
        price_valid = await self._validate_current_price(symbol, entry_price, direction)
        if not price_valid[0]:
            return price_valid
            
        # 2. Check spread isn't too wide
        spread_valid = await self._validate_spread(symbol, entry_price)
        if not spread_valid[0]:
            return spread_valid
            
        # 3. Validate SL/TP levels are still valid
        levels_valid = self._validate_price_levels(entry_price, sl_price, tp_price, direction)
        if not levels_valid[0]:
            return levels_valid
            
        # 4. Check for sudden volatility spike
        volatility_valid = await self._check_volatility_spike(symbol)
        if not volatility_valid[0]:
            return volatility_valid
            
        return True, "All pre-trade validations passed"
    
    async def _validate_current_price(self, symbol: str, intended_entry: float, 
                                    direction: str) -> Tuple[bool, str]:
        """Check if current price is still close to intended entry"""
        
        try:
            # Get current market data
            ticker_resp = await signed_request("GET", "/v5/market/tickers", {
                "category": "linear",
                "symbol": symbol
            })
            
            if ticker_resp.get("retCode") != 0:
                return False, "Failed to get current price"
                
            data = ticker_resp.get("result", {}).get("list", [{}])[0]
            current_price = float(data.get("lastPrice", 0))
            
            if current_price <= 0:
                return False, "Invalid current price"
                
            # Calculate price movement
            price_move_pct = abs((current_price - intended_entry) / intended_entry) * 100
            
            # Allow maximum 0.3% movement
            if price_move_pct > 0.3:
                return False, f"Price moved {price_move_pct:.2f}% from intended entry"
                
            # Check direction hasn't reversed
            if direction == "long" and current_price < intended_entry * 0.997:
                return False, "Price moving against long entry"
            elif direction == "short" and current_price > intended_entry * 1.003:
                return False, "Price moving against short entry"
                
            return True, "Price validation passed"
            
        except Exception as e:
            log(f"❌ Error validating current price: {e}", level="ERROR")
            return False, f"Price validation error: {str(e)}"
    
    async def _validate_spread(self, symbol: str, entry_price: float) -> Tuple[bool, str]:
        """Check if bid-ask spread is reasonable"""
        
        try:
            # Get orderbook data
            orderbook_resp = await signed_request("GET", "/v5/market/orderbook", {
                "category": "linear",
                "symbol": symbol,
                "limit": 1
            })
            
            if orderbook_resp.get("retCode") != 0:
                return True, "Could not check spread"  # Don't block trade
                
            result = orderbook_resp.get("result", {})
            bids = result.get("b", [])
            asks = result.get("a", [])
            
            if not bids or not asks:
                return True, "No orderbook data"
                
            bid = float(bids[0][0])
            ask = float(asks[0][0])
            
            spread = ask - bid
            spread_pct = (spread / entry_price) * 100
            
            # Maximum allowed spread
            max_spread_pct = 0.1  # 0.1%
            
            if spread_pct > max_spread_pct:
                return False, f"Spread too wide: {spread_pct:.3f}%"
                
            return True, f"Spread acceptable: {spread_pct:.3f}%"
            
        except Exception as e:
            log(f"⚠️ Error checking spread: {e}", level="WARN")
            return True, "Spread check skipped"  # Don't block trade
    
    def _validate_price_levels(self, entry: float, sl: float, tp: float, 
                              direction: str) -> Tuple[bool, str]:
        """Validate SL and TP levels make sense"""
        
        if direction == "long":
            # For long: SL < Entry < TP
            if not (sl < entry < tp):
                return False, f"Invalid long levels: SL={sl}, Entry={entry}, TP={tp}"
                
            # Check minimum risk/reward
            risk = entry - sl
            reward = tp - entry
            
            if risk <= 0:
                return False, "Invalid risk calculation"
                
            rr_ratio = reward / risk
            
            if rr_ratio < 1.0:  # Minimum 1:1 RR
                return False, f"Risk/Reward too low: {rr_ratio:.2f}"
                
        else:  # Short
            # For short: TP < Entry < SL
            if not (tp < entry < sl):
                return False, f"Invalid short levels: TP={tp}, Entry={entry}, SL={sl}"
                
            # Check minimum risk/reward
            risk = sl - entry
            reward = entry - tp
            
            if risk <= 0:
                return False, "Invalid risk calculation"
                
            rr_ratio = reward / risk
            
            if rr_ratio < 1.0:  # Minimum 1:1 RR
                return False, f"Risk/Reward too low: {rr_ratio:.2f}"
                
        return True, "Price levels valid"
    
    async def _check_volatility_spike(self, symbol: str) -> Tuple[bool, str]:
        """Check for sudden volatility that might invalidate setup"""
        
        try:
            # Get recent klines
            klines_resp = await signed_request("GET", "/v5/market/kline", {
                "category": "linear",
                "symbol": symbol,
                "interval": "1",
                "limit": 5
            })
            
            if klines_resp.get("retCode") != 0:
                return True, "Could not check volatility"
                
            klines = klines_resp.get("result", {}).get("list", [])
            
            if len(klines) < 3:
                return True, "Insufficient data"
                
            # Check last 3 candles for huge moves
            for kline in klines[:3]:
                high = float(kline[2])
                low = float(kline[3])
                
                candle_range_pct = ((high - low) / low) * 100
                
                # If any candle has >1% range, consider it high volatility
                if candle_range_pct > 1.0:
                    return False, f"High volatility detected: {candle_range_pct:.2f}% candle"
                    
            return True, "Volatility acceptable"
            
        except Exception as e:
            log(f"⚠️ Error checking volatility: {e}", level="WARN")
            return True, "Volatility check skipped"

# Global instance
pre_trade_validator = PreTradeValidator()
