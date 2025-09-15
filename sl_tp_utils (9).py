import asyncio
import traceback
from datetime import datetime
import numpy as np
from logger import log, write_log
from symbol_info import get_precision, round_qty
from error_handler import send_telegram_message, send_error_to_telegram

# FIXED PERCENTAGES AS REQUESTED
FIXED_SL_TP = {
    "Scalp": {
        "sl_pct": 0.8,       # -0.8% stop loss
        "tp1_pct": 1.2,      # +1.2% take profit
        "trailing_pct": 0.4  # 0.4% trailing stop
    },
    "Intraday": {
        "sl_pct": 1.0,       # -1.0% stop loss
        "tp1_pct": 2.0,      # +2.0% take profit
        "trailing_pct": 1.0  # 1.0% trailing stop
    },
    "Swing": {
        "sl_pct": 1.5,       # -1.5% stop loss
        "tp1_pct": 3.5,      # +3.5% take profit
        "trailing_pct": 1.5  # 1.5% trailing stop
    }
}

# Remove activation thresholds - trailing starts immediately after TP1
TRAILING_ACTIVATION_THRESHOLDS = {
    "Scalp": 0.0,      # Immediate activation after TP1
    "Intraday": 0.0,   # Immediate activation after TP1
    "Swing": 0.0       # Immediate activation after TP1
}

# Other constants remain the same
MIN_SL_PERCENTAGE = 0.5       # Minimum SL distance
MIN_SL_ATR_FACTOR = 1.0       # Minimum ATR factor
MAX_SL_PERCENTAGE = 8.0       # Maximum SL distance
MAX_SL_ATR_FACTOR = 3.2       # Maximum ATR factor

# Exit tranches configuration - updated for letting winners run
EXIT_TRANCHES = {
    "Scalp": [0.20, 0.30, 0.50],     # 20% at TP1, 30% at TP2, 50% rides trend
    "Intraday": [0.20, 0.30, 0.50],  # 20% at TP1, 30% at TP2, 50% rides trend
    "Swing": [0.15, 0.25, 0.60],     # 15% at TP1, 25% at TP2, 60% rides trend
}

# Market regime adjustments
REGIME_ADJUSTMENTS = {
    "trending": {"sl": 1.0, "tp": 1.0, "trailing": 1.0},
    "ranging": {"sl": 1.0, "tp": 1.0, "trailing": 1.0},    # No adjustments
    "volatile": {"sl": 1.0, "tp": 1.0, "trailing": 1.0},   # No adjustments
}

def calculate_atr(candles, period=14):
    """
    Calculate the Average True Range (ATR) from candles
    
    Args:
        candles: List of candle dictionaries with 'high', 'low', 'close' keys
        period: ATR calculation period
        
    Returns:
        float: ATR value or None if not enough candles
    """
    if len(candles) < period + 1:
        return None

    try:
        highs = np.array([float(c['high']) for c in candles[-(period+1):]])
        lows = np.array([float(c['low']) for c in candles[-(period+1):]])
        closes = np.array([float(c['close']) for c in candles[-(period+1):]])

        # Calculate True Range series
        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            tr_list.append(tr)

        # Calculate ATR as simple average of true ranges
        atr = np.mean(tr_list)
        return round(atr, 6)
    except Exception as e:
        log(f"❌ Error calculating ATR: {e}", level="ERROR")
        return None

def detect_volatility_regime(candles, lookback=20):
    """
    Detect volatility regime based on ATR changes
    
    Args:
        candles: List of candle dictionaries
        lookback: Period for volatility comparison
        
    Returns:
        str: Volatility regime ('low', 'normal', 'high')
    """
    if len(candles) < lookback * 2:
        return "normal"  # Default if not enough data
    
    try:
        # Calculate recent ATR
        recent_atr = calculate_atr(candles[-lookback:], period=lookback//2)
        
        # Calculate prior ATR (earlier period)
        prior_atr = calculate_atr(candles[-(lookback*2):-lookback], period=lookback//2)
        
        if not recent_atr or not prior_atr or prior_atr == 0:
            return "normal"
        
        # Calculate volatility ratio
        vol_ratio = recent_atr / prior_atr
        
        # Categorize volatility
        if vol_ratio < 0.8:
            return "low"
        elif vol_ratio > 1.3:
            return "high"
        else:
            return "normal"
    except Exception as e:
        log(f"❌ Error detecting volatility regime: {e}", level="ERROR")
        return "normal"

def detect_price_momentum(candles, lookback=5):
    """
    Detect if price is showing strong momentum based on recent candles
    
    Args:
        candles: List of candle dictionaries
        lookback: Number of recent candles to analyze
        
    Returns:
        Tuple of (has_momentum: bool, direction: str, strength: float)
    """
    if len(candles) < lookback + 5:
        return False, None, 0.0
    
    try:
        # Get recent candles for analysis
        recent = candles[-lookback:]
        
        # Calculate consecutive up/down candles
        consecutive_up = 0
        consecutive_down = 0
        
        for i in range(len(recent)):
            if float(recent[i]['close']) > float(recent[i]['open']):
                consecutive_up += 1
                consecutive_down = 0
            elif float(recent[i]['close']) < float(recent[i]['open']):
                consecutive_down += 1
                consecutive_up = 0
        
        # Calculate volume increase
        recent_vol = sum(float(c['volume']) for c in recent) / len(recent)
        prev_vol = sum(float(c['volume']) for c in candles[-(lookback+5):-lookback]) / len(recent)
        vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1.0
        
        # Calculate price momentum
        price_change = (float(recent[-1]['close']) - float(recent[0]['open'])) / float(recent[0]['open']) * 100
        
        # Determine momentum direction and strength
        direction = "up" if price_change > 0 else "down"
        
        # Strength is based on consecutive candles, volume increase, and price change
        strength = 0.0
        if consecutive_up >= 3 or consecutive_down >= 3:
            strength += 0.5
        if vol_ratio > 1.5:
            strength += 0.3
        if abs(price_change) > 1.0:
            strength += 0.2
        
        has_momentum = strength >= 0.7  # 70% of criteria met
        
        return has_momentum, direction, strength
    
    except Exception as e:
        log(f"❌ Error detecting price momentum: {e}", level="ERROR")
        return False, None, 0.0

def calculate_dynamic_sl_tp(candles_by_tf, entry_price, trade_type, direction, score, confidence, regime="trending", strategy="core_strategy"):
    """
    Calculate FIXED SL/TP percentages as requested
    Returns exactly what you specified for Scalp and Intraday trades
    """

    # Normalize trade type
    trade_type = str(trade_type).strip().title()
    if trade_type not in ["Scalp", "Intraday", "Swing"]:
        log(f"⚠️ Invalid trade type '{trade_type}', defaulting to Intraday", level="WARN")
        trade_type = "Intraday"
        
    # Get the fixed percentages for this trade type
    fixed_params = FIXED_SL_TP.get(trade_type, FIXED_SL_TP["Intraday"])
    
    # Use the fixed percentages directly - no adjustments
    sl_pct = fixed_params["sl_pct"]
    tp_pct = fixed_params["tp1_pct"]
    trailing_pct = fixed_params["trailing_pct"]
    
    # Calculate actual prices
    if direction.lower() == "long":
        sl_price = entry_price * (1 - sl_pct/100)
        tp1_price = entry_price * (1 + tp_pct/100)
        tp2_price = entry_price * (1 + tp_pct/100 * 1.8)  # TP2 is 1.8x TP1
        tp3_price = entry_price * (1 + tp_pct/100 * 2.5)  # TP3 is 2.5x TP1
    else:  # short
        sl_price = entry_price * (1 + sl_pct/100)
        tp1_price = entry_price * (1 - tp_pct/100)
        tp2_price = entry_price * (1 - tp_pct/100 * 1.8)
        tp3_price = entry_price * (1 - tp_pct/100 * 2.5)
    
    # Calculate TP2 and TP3 percentages
    tp2_pct = tp_pct * 1.8
    tp3_pct = tp_pct * 2.5
    
    # Log the fixed calculation
    rr_ratio = tp_pct / sl_pct
    log(f"📊 FIXED SL/TP for {direction} {trade_type}:")
    log(f"  Entry: {entry_price} | SL: {sl_price} ({sl_pct}%) | TP1: {tp1_price} ({tp_pct}%) | RR: {rr_ratio:.1f}")
    log(f"  Trailing: {trailing_pct}% (activates immediately after TP1)")
    
    return (
        round(sl_price, 6),
        round(tp1_price, 6),
        round(sl_pct, 2),
        round(trailing_pct, 2),
        round(tp_pct, 2),
        round(tp2_price, 6),
        round(tp2_pct, 2),
        round(tp3_price, 6),
        round(tp3_pct, 2)
    )

def calculate_exit_tranches(symbol, total_qty, trade_type="Intraday", volatility="normal", momentum=False):
    """
    Calculate position size for each exit tranche - optimized for letting winners run
    
    Args:
        symbol: Trading symbol for precision lookup
        total_qty: Total position size
        trade_type: "Scalp", "Intraday", or "Swing"
        volatility: Volatility level ("low", "normal", "high")
        momentum: Flag for momentum presence
        
    Returns:
        List of quantities for each tranche
    """
    if total_qty <= 0:
        return []
    
    # Get symbol precision for rounding
    precision = get_precision(symbol)
    min_qty = 0.001  # Fallback min quantity
    
    # Get base distribution by trade type - optimized for letting winners run
    distribution = EXIT_TRANCHES.get(trade_type, [0.20, 0.30, 0.50])
    
    # No adjustments for volatility or momentum with fixed percentages
    # Keep the distribution as is to let winners run
    
    # Normalize distribution to sum to 1.0
    total = sum(distribution)
    if total > 0:
        distribution = [d / total for d in distribution]
    
    # Calculate raw tranches
    raw_tranches = [total_qty * dist for dist in distribution]
    
    # Round to valid quantities
    valid_tranches = []
    running_total = 0
    
    for i, qty in enumerate(raw_tranches):
        if i == len(raw_tranches) - 1:
            # Make sure final tranche captures any rounding errors
            final_qty = round_qty(symbol, total_qty - running_total)
            if final_qty >= min_qty:
                valid_tranches.append(final_qty)
        else:
            rounded_qty = round_qty(symbol, qty)
            if rounded_qty >= min_qty:
                valid_tranches.append(rounded_qty)
                running_total += rounded_qty
    
    # Ensure we have at least one valid tranche
    if not valid_tranches:
        valid_tranches = [total_qty]
    
    log(f"📊 Exit tranches for {symbol} ({trade_type}): {valid_tranches}")
    return valid_tranches

async def validate_sl_placement(symbol, direction, sl_price, market_type="linear"):
    """
    Validate that a stop loss price is on the correct side of current market price
    
    Args:
        symbol: Trading symbol
        direction: "long" or "short"
        sl_price: Proposed stop loss price
        market_type: Exchange market type
        
    Returns:
        float: Corrected SL price or original if already valid
    """
    from bybit_api import signed_request
    
    try:
        # Get current market price
        ticker_resp = await signed_request("GET", "/v5/market/tickers", 
                                           {"category": market_type, "symbol": symbol})
        
        # Extract mark price, last price, and bid/ask if available
        result_data = ticker_resp.get("result", {}).get("list", [{}])[0]
        mark_price = float(result_data.get("markPrice", 0))
        last_price = float(result_data.get("lastPrice", 0))
        
        # Use mark price if available, otherwise fall back to last price
        current_price = mark_price if mark_price > 0 else last_price
        
        if current_price <= 0:
            log(f"⚠️ Invalid market price for {symbol}", level="WARN")
            return sl_price  # Return original if we can't validate
        
        # Add safety buffer (1.0% buffer for more reliable stop placement)
        buffer_pct = 0.01  # 1.0%
        
        # For long positions, SL must be below current price
        if direction.lower() == "long":
            if sl_price >= current_price:
                new_sl = round(current_price * (1 - buffer_pct), 6)
                log(f"⚠️ Fixed long SL from {sl_price} to {new_sl} (below {current_price})", level="WARN")
                return new_sl
                
        # For short positions, SL must be above current price
        elif direction.lower() == "short":
            if sl_price <= current_price:
                new_sl = round(current_price * (1 + buffer_pct), 6)
                log(f"⚠️ Fixed short SL from {sl_price} to {new_sl} (above {current_price})", level="WARN")
                return new_sl
        
        # If already valid, return original SL price
        return sl_price
        
    except Exception as e:
        log(f"❌ Error validating SL placement: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return sl_price  # Return original if validation fails

def calculate_smart_trailing_stop(symbol, entry_price, current_price, direction, candles, base_trail_pct=0.5):
    """
    Calculate trailing stop with FIXED percentages - no adaptive changes
    
    Args:
        symbol: Trading symbol
        entry_price: Entry price
        current_price: Current market price
        direction: 'long' or 'short'
        candles: Recent candles for analysis
        base_trail_pct: Base trailing percentage (will use fixed values instead)
        
    Returns:
        float: Calculated stop loss price
    """
    try:
        # No trailing until we're in profit
        if direction.lower() == "long":
            if current_price <= entry_price:
                return None
        else:  # short
            if current_price >= entry_price:
                return None
        
        # Use the fixed trailing percentage - no adjustments
        # The base_trail_pct passed in should already be the fixed value
        trailing_pct = base_trail_pct
        
        # Calculate actual SL price
        if direction.lower() == "long":
            sl_price = current_price * (1 - (trailing_pct / 100))
        else:  # short
            sl_price = current_price * (1 + (trailing_pct / 100))
        
        return round(sl_price, 6)
        
    except Exception as e:
        log(f"❌ Error calculating trailing stop: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

def should_trail_stop(symbol, entry_price, current_price, direction, candles=None, trailing_pct=0.5):
    """
    Determine if trailing stop should be activated - IMMEDIATE activation after TP1
    
    Args:
        symbol: Trading symbol
        entry_price: Trade entry price
        current_price: Current market price
        direction: 'long' or 'short'
        candles: Recent candles for analysis
        trailing_pct: Base trailing percentage
        
    Returns:
        float or None: New stop loss price if trailing should activate, None otherwise
    """
    try:
        # NO activation threshold - trailing starts immediately after TP1
        # This function is called after TP1 is hit, so we can trail immediately
        
        # Just use the fixed trailing calculation
        return calculate_smart_trailing_stop(
            symbol=symbol,
            entry_price=entry_price,
            current_price=current_price,
            direction=direction,
            candles=candles,
            base_trail_pct=trailing_pct  # Use the fixed trailing percentage
        )
    
    except Exception as e:
        log(f"❌ Error in trailing stop logic: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

def calculate_breakeven_after_move(entry_price, direction, move_pct=1.0):
    """
    Calculate breakeven price after a certain percentage move in favor
    
    Args:
        entry_price: Trade entry price
        direction: "long" or "short"
        move_pct: Percentage move required before setting breakeven
        
    Returns:
        float: Breakeven price (usually entry, but can include buffer)
    """
    try:
        # Include a small buffer (0.1%) to account for fees and spread
        buffer = entry_price * 0.001
        
        # Calculate breakeven with buffer
        if direction.lower() == "long":
            return entry_price + buffer
        else:
            return entry_price - buffer
            
    except Exception as e:
        log(f"❌ Error calculating breakeven: {e}", level="ERROR")
        return entry_price  # Return exact entry price if calculation fails

def adjust_profit_protection(symbol, entry_price, current_price, direction, trade_type="Intraday"):
    """
    Adjust stop loss based on profit milestones - DISABLED for fixed SL/TP system
    """
    # Return None to indicate no adjustment needed
    # We're using fixed percentages, so no dynamic profit protection
    return None

def should_exit_by_time(trade, current_time=None, candles=None):
    """
    Check if trade should be exited based on time elapsed
    
    Args:
        trade: Trade data dictionary
        current_time: Current datetime (uses now if None)
        candles: Recent candles for momentum detection
        
    Returns:
        bool: True if trade should be exited
    """
    from datetime import datetime
    
    if not current_time:
        current_time = datetime.utcnow()
    
    try:
        # Get trade entry time
        entry_time_str = trade.get("timestamp")
        if not entry_time_str:
            return False
            
        entry_time = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
        trade_age_hours = (current_time - entry_time).total_seconds() / 3600
        
        trade_type = trade.get("trade_type", "Intraday")
        direction = trade.get("direction", "").lower()
        entry_price = trade.get("entry_price")
        
        # Don't exit if we're in profit and in momentum
        if candles:
            has_momentum, momentum_direction, _ = detect_price_momentum(candles)
            momentum_aligned = (direction == "long" and momentum_direction == "up") or \
                              (direction == "short" and momentum_direction == "down")
            
            # Check if in significant profit
            is_in_profit = False
            if entry_price and current_price:
                if direction == "long":
                    is_in_profit = current_price > entry_price * 1.02  # 2% profit
                else:
                    is_in_profit = current_price < entry_price * 0.98  # 2% profit
                    
            if has_momentum and momentum_aligned and is_in_profit:
                # Don't exit on time if in profitable momentum
                return False
        
        # Define max age based on trade type
        max_age = {
            "Scalp": 12,      # 12 hours for scalps
            "Intraday": 36,   # 36 hours for intraday
            "Swing": 120      # 120 hours (5 days) for swing trades
        }.get(trade_type, 36)
        
        # For scalps - check for progress
        if trade_type == "Scalp" and trade_age_hours > 4 and not trade.get("tp1_hit") and entry_price:
            # Check if price is making any progress
            if direction == "long":
                # For longs, exit if price is below entry after 4 hours
                if current_price < entry_price:
                    log(f"⏱ Time-based exit for {trade.get('symbol')}: Scalp not making progress after {trade_age_hours:.1f} hours")
                    return True
            else:  # short
                # For shorts, exit if price is above entry after 4 hours
                if current_price > entry_price:
                    log(f"⏱ Time-based exit for {trade.get('symbol')}: Scalp not making progress after {trade_age_hours:.1f} hours")
                    return True
        
        # If trade has hit TP1, give it more time
        if trade.get("tp1_hit"):
            max_age *= 1.5  # 50% more time after TP1 is hit
        
        # Exit any trade if max age exceeded
        if trade_age_hours > max_age:
            log(f"⏱ Time-based exit for {trade.get('symbol')}: Max age of {max_age} hours exceeded ({trade_age_hours:.1f} hours)")
            return True
            
    except Exception as e:
        log(f"❌ Error in time-based exit check: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
    
    return False

def evaluate_score_exit(symbol, scores, min_exit_cycles=3, trade_type="Intraday"):
    """
    Evaluate whether to exit based on score deterioration pattern
    DISABLED - using fixed SL for protection instead
    """
    # Return False to disable score-based exits
    # We're using fixed SL/TP system
    return False
