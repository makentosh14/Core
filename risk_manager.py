import asyncio
import json
import traceback
import numpy as np
from datetime import datetime, timedelta
from logger import log, write_log
from atr import calculate_atr

# Constants for risk models
DEFAULT_RISK_PER_TRADE = 0.02  # 2% per trade default
MAX_RISK_PER_TRADE = 0.05      # 5% maximum risk per trade
MIN_RISK_PER_TRADE = 0.005     # 0.5% minimum risk per trade
RISK_FREE_RATE = 0.04          # 4% annualized (for Kelly calculations)
MAX_DAILY_RISK = 0.10          # 10% maximum daily risk
MAX_DRAWDOWN_PAUSE = 1.0      # 15% drawdown triggers trading pause

# Volatility bands for adjusting position size
VOLATILITY_BANDS = {
    "very_low": 0.5,    # 50% of base risk
    "low": 0.75,        # 75% of base risk
    "normal": 1.0,      # 100% of base risk (baseline)
    "high": 0.7,        # 70% of base risk
    "very_high": 0.5,   # 50% of base risk
    "extreme": 0.3      # 30% of base risk
}

# Risk weights for different strategies
STRATEGY_RISK_WEIGHTS = {
    "core_strategy": 1.0,
    "mean_reversion": 0.85,
    "breakout_sniper": 0.7,
    "swing": 0.8,
    "scalp": 1.0,
    "intraday": 0.9
}

# Cache of strategy performance for risk allocation
strategy_performance = {}

# Cache for volatility calculations to avoid redundant computations
volatility_cache = {}

# Daily risk tracking
daily_risk = {
    "date": None,
    "used_risk": 0.0,
    "remaining_risk": MAX_DAILY_RISK,
    "trades": []
}

# Drawdown tracking
drawdown_tracking = {
    "peak_balance": 0.0,
    "current_balance": 0.0,
    "current_drawdown": 0.0,
    "max_drawdown": 0.0,
    "is_paused": False,
    "pause_timestamp": None
}

def reset_daily_risk():
    """Reset daily risk tracking for a new day"""
    today = datetime.now().date()
    if daily_risk["date"] != today:
        log(f"📆 Resetting daily risk for new day: {today}")
        daily_risk["date"] = today
        daily_risk["used_risk"] = 0.0
        daily_risk["remaining_risk"] = MAX_DAILY_RISK
        daily_risk["trades"] = []
        save_risk_state()

def save_risk_state():
    """Save risk management state to disk for persistence"""
    try:
        state = {
            "daily_risk": daily_risk,
            "drawdown_tracking": drawdown_tracking,
            "strategy_performance": strategy_performance,
            "volatility_cache": {k: v for k, v in volatility_cache.items() if isinstance(k, str)}
        }
        
        with open("risk_state.json", "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        log(f"❌ Failed to save risk state: {e}", level="ERROR")

def load_risk_state():
    """Load risk management state from disk"""
    global daily_risk, drawdown_tracking, strategy_performance, volatility_cache
    
    try:
        with open("risk_state.json", "r") as f:
            state = json.load(f)
            
        # Only restore if the date matches today
        today = datetime.now().date()
        if state["daily_risk"]["date"] == today.strftime("%Y-%m-%d"):
            daily_risk = state["daily_risk"]
        else:
            reset_daily_risk()
            
        drawdown_tracking = state["drawdown_tracking"]
        strategy_performance = state["strategy_performance"]
        volatility_cache = state["volatility_cache"]
        
        log(f"✅ Loaded risk state with {len(strategy_performance)} strategy records")
    except FileNotFoundError:
        log("⚠️ No risk state file found, starting fresh")
        reset_daily_risk()
    except Exception as e:
        log(f"❌ Failed to load risk state: {e}", level="ERROR")
        reset_daily_risk()

def calculate_market_volatility(candles, lookback=20):
    """
    Calculate market volatility based on ATR relative to price
    
    Args:
        candles: List of OHLCV candles
        lookback: Period for ATR calculation
    
    Returns:
        float: Normalized volatility value (0.0 - 5.0+)
        str: Volatility band category
    """
    if not candles or len(candles) < lookback + 10:
        return 1.0, "normal"  # Default to normal volatility
    
    # Calculate ATR
    atr = calculate_atr(candles, period=lookback)
    if not atr:
        return 1.0, "normal"
        
    # Get current price
    current_price = float(candles[-1]['close'])
    
    # Calculate ATR as percentage of price
    atr_pct = (atr / current_price) * 100
    
    # Get baseline ATR from recent history (last 100 candles)
    baseline_window = min(100, len(candles) - lookback)
    baseline_atrs = []
    
    for i in range(baseline_window):
        window = candles[-(lookback + i + 1):-(i + 1)]
        baseline_atr = calculate_atr(window, period=lookback)
        if baseline_atr:
            baseline_price = float(window[-1]['close'])
            baseline_atrs.append((baseline_atr / baseline_price) * 100)
    
    if not baseline_atrs:
        return 1.0, "normal"
        
    baseline_atr_pct = sum(baseline_atrs) / len(baseline_atrs)
    
    # Normalized volatility (1.0 means exactly average)
    normalized_volatility = atr_pct / baseline_atr_pct if baseline_atr_pct > 0 else 1.0
    
    # Categorize volatility
    if normalized_volatility < 0.6:
        band = "very_low"
    elif normalized_volatility < 0.8:
        band = "low"
    elif normalized_volatility < 1.2:
        band = "normal"
    elif normalized_volatility < 1.5:
        band = "high"
    elif normalized_volatility < 2.0:
        band = "very_high"
    else:
        band = "extreme"
    
    return normalized_volatility, band

async def calculate_symbol_volatility(symbol, candles_by_tf, timeframe='15'):
    """
    Calculate and cache volatility for a specific symbol
    
    Args:
        symbol: Trading symbol
        candles_by_tf: Dictionary of candles by timeframe
        timeframe: Timeframe to use for volatility calculation
    
    Returns:
        tuple: (normalized_volatility, volatility_band)
    """
    # Use cached value if less than 15 minutes old
    cache_key = f"{symbol}_{timeframe}"
    now = datetime.now()
    
    if cache_key in volatility_cache:
        timestamp, volatility, band = volatility_cache[cache_key]
        cache_age = (now - timestamp).total_seconds() / 60
        
        if cache_age < 15:  # Use cache if less than 15 minutes old
            return volatility, band
    
    # Calculate fresh volatility
    candles = candles_by_tf.get(timeframe, [])
    if not candles:
        return 1.0, "normal"
        
    volatility, band = calculate_market_volatility(candles)
    
    # Update cache
    volatility_cache[cache_key] = (now, volatility, band)
    
    log(f"🔄 Calculated volatility for {symbol} ({timeframe}m): {volatility:.2f} ({band})")
    return volatility, band

def calculate_strategy_risk_weight(strategy, win_rate=None, profit_factor=None):
    """
    Calculate risk weight for a strategy based on performance metrics
    
    Args:
        strategy: Strategy name
        win_rate: Win rate as decimal (0.0-1.0)
        profit_factor: Profit factor (avg_win/avg_loss)
    
    Returns:
        float: Risk weight multiplier (0.0-1.5)
    """
    # Use default if no metrics provided
    base_weight = STRATEGY_RISK_WEIGHTS.get(strategy, 0.8)
    
    if win_rate is None or profit_factor is None:
        # Try to get from cache
        if strategy in strategy_performance:
            win_rate = strategy_performance[strategy].get("win_rate", 0.5)
            profit_factor = strategy_performance[strategy].get("profit_factor", 1.0)
        else:
            return base_weight
    
    # Calculate dynamic weight based on performance
    # Better performance = higher weight
    performance_score = (win_rate * 0.6) + (min(profit_factor, 3.0) / 3.0 * 0.4)
    
    # Scale weight based on performance (0.5x - 1.5x of base weight)
    dynamic_weight = base_weight * (0.5 + performance_score)
    
    # Cap at 1.5x base weight
    return min(dynamic_weight, base_weight * 1.5)

def calculate_kelly_position_size(win_rate, profit_factor, risk_level=1.0):
    """
    Calculate position size using Kelly Criterion
    
    Args:
        win_rate: Historical win rate (0.0-1.0)
        profit_factor: Ratio of average win to average loss
        risk_level: Kelly fraction to use (0.0-1.0)
    
    Returns:
        float: Kelly percentage of capital to risk
    """
    # Sanity check inputs
    if win_rate <= 0 or win_rate >= 1 or profit_factor <= 0:
        return 0.02  # Default to 2% if inputs are invalid
    
    # Kelly formula: f* = (bp - q) / b
    # where p = win probability, q = loss probability (1-p), b = win/loss ratio
    win_loss_ratio = profit_factor  # Simplification
    loss_rate = 1 - win_rate
    
    kelly_pct = (win_loss_ratio * win_rate - loss_rate) / win_loss_ratio
    
    # Apply a fraction of Kelly for safety (typically 1/4 to 1/2)
    fractional_kelly = kelly_pct * risk_level
    
    # Cap the result between min and max risk
    return max(min(fractional_kelly, MAX_RISK_PER_TRADE), MIN_RISK_PER_TRADE)

def calculate_volatility_adjusted_position_size(symbol, base_risk, volatility_band, strategy, candles, confidence):
    """
    Calculate position size adjusted for volatility and strategy performance
    
    Args:
        symbol: Trading symbol
        base_risk: Base risk percentage
        volatility_band: Volatility category
        strategy: Trading strategy
        candles: Price candles
        confidence: Confidence score (0-100)
    
    Returns:
        float: Adjusted risk percentage
    """
    # Get volatility adjustment factor
    vol_factor = VOLATILITY_BANDS.get(volatility_band, 1.0)
    
    # Get strategy performance metrics if available
    if strategy in strategy_performance:
        win_rate = strategy_performance[strategy].get("win_rate", 0.5)
        profit_factor = strategy_performance[strategy].get("profit_factor", 1.0)
    else:
        # Use reasonable defaults if no data
        win_rate = 0.5
        profit_factor = 1.0
    
    # Calculate Kelly position size (theoretical optimal)
    kelly_size = calculate_kelly_position_size(win_rate, profit_factor, risk_level=0.3)
    
    # Calculate strategy risk weight
    strategy_weight = calculate_strategy_risk_weight(strategy, win_rate, profit_factor)
    
    # Confidence adjustment (0.7x - 1.3x)
    confidence_factor = 0.7 + (confidence / 100 * 0.6)
    
    # Calculate adjusted risk
    adjusted_risk = base_risk * vol_factor * strategy_weight * confidence_factor
    
    # Use Kelly as a reference point, but don't exceed it significantly
    kelly_cap = kelly_size * 1.5
    if adjusted_risk > kelly_cap:
        adjusted_risk = kelly_cap
        log(f"🔍 Position size for {symbol} capped by Kelly criterion ({kelly_size:.4f})")
    
    # Enforce min/max bounds
    adjusted_risk = max(min(adjusted_risk, MAX_RISK_PER_TRADE), MIN_RISK_PER_TRADE)
    
    log(f"📊 Position sizing for {symbol}: Base={base_risk:.4f}, Vol={vol_factor:.2f}, "
        f"Strategy={strategy_weight:.2f}, Confidence={confidence_factor:.2f}, Final={adjusted_risk:.4f}")
    
    return adjusted_risk

def update_strategy_performance(strategy, result, pnl):
    """
    Update performance metrics for a strategy
    
    Args:
        strategy: Strategy name
        result: 'win' or 'loss'
        pnl: Profit/loss amount or percentage
    """
    if strategy not in strategy_performance:
        strategy_performance[strategy] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.5,
            "total_profit": 0,
            "total_loss": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 1.0,
            "last_updated": datetime.now(),
        }
    
    stats = strategy_performance[strategy]
    stats["trades"] += 1
    
    if result == "win":
        stats["wins"] += 1
        stats["total_profit"] += pnl
    else:
        stats["losses"] += 1
        stats["total_loss"] += abs(pnl)
    
    # Update derived metrics
    if stats["trades"] > 0:
        stats["win_rate"] = stats["wins"] / stats["trades"]
    
    if stats["wins"] > 0:
        stats["avg_win"] = stats["total_profit"] / stats["wins"]
    
    if stats["losses"] > 0:
        stats["avg_loss"] = stats["total_loss"] / stats["losses"]
    
    if stats["avg_loss"] > 0:
        stats["profit_factor"] = stats["avg_win"] / stats["avg_loss"]
    
    stats["last_updated"] = datetime.now()
    save_risk_state()

def check_daily_risk_limit(risk_amount):
    """
    Check if a trade will exceed daily risk limits
    
    Args:
        risk_amount: Risk amount for the trade (decimal)
    
    Returns:
        bool: True if trade is allowed, False if it exceeds limits
    """
    reset_daily_risk()  # Ensure daily risk is reset if needed
    
    if daily_risk["used_risk"] + risk_amount > MAX_DAILY_RISK:
        log(f"⚠️ Trade blocked: Daily risk limit reached ({daily_risk['used_risk']:.2%} used, {risk_amount:.2%} requested, {MAX_DAILY_RISK:.2%} max)",
            level="WARN")
        return False
    
    return True

def register_trade_risk(symbol, risk_amount, strategy):
    """
    Register risk amount for a new trade
    
    Args:
        symbol: Trading symbol
        risk_amount: Risk amount for the trade (decimal)
        strategy: Strategy used for the trade
    """
    reset_daily_risk()  # Ensure daily risk is reset if needed
    
    # Add to daily risk tracking
    daily_risk["used_risk"] += risk_amount
    daily_risk["remaining_risk"] = MAX_DAILY_RISK - daily_risk["used_risk"]
    
    # Record trade
    daily_risk["trades"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "risk": risk_amount,
        "strategy": strategy
    })
    
    log(f"💹 Risk registered: {symbol} {risk_amount:.2%} ({strategy}) - Daily: {daily_risk['used_risk']:.2%}/{MAX_DAILY_RISK:.2%}")
    save_risk_state()

def update_account_balance(balance):
    """
    Update account balance tracking for drawdown calculations
    
    Args:
        balance: Current account balance
    """
    drawdown_tracking["current_balance"] = balance
    
    # Update peak balance if current balance is higher
    if balance > drawdown_tracking["peak_balance"]:
        drawdown_tracking["peak_balance"] = balance
    
    # Calculate current drawdown
    if drawdown_tracking["peak_balance"] > 0:
        drawdown = 1 - (balance / drawdown_tracking["peak_balance"])
        drawdown_tracking["current_drawdown"] = drawdown
        
        # Update max drawdown if current drawdown is higher
        if drawdown > drawdown_tracking["max_drawdown"]:
            drawdown_tracking["max_drawdown"] = drawdown
        
        # Check if we need to pause trading
        if drawdown >= MAX_DRAWDOWN_PAUSE and not drawdown_tracking["is_paused"]:
            drawdown_tracking["is_paused"] = True
            drawdown_tracking["pause_timestamp"] = datetime.now()
            log(f"🛑 Trading paused: Max drawdown threshold reached ({drawdown:.2%})", level="ALERT")
    
    save_risk_state()

def check_trading_allowed():
    """
    Check if trading is currently allowed based on drawdown limits
    
    Returns:
        bool: True if trading is allowed, False if paused
    """
    # If not paused, trading is allowed
    if not drawdown_tracking["is_paused"]:
        return True
    
    # Check if pause duration has expired (24 hours)
    if drawdown_tracking["pause_timestamp"]:
        pause_time = datetime.fromisoformat(drawdown_tracking["pause_timestamp"] 
                                           if isinstance(drawdown_tracking["pause_timestamp"], str) 
                                           else drawdown_tracking["pause_timestamp"].isoformat())
        now = datetime.now()
        
        # Resume trading after 24 hours
        if (now - pause_time).total_seconds() > 30 * 60:
            drawdown_tracking["is_paused"] = False
            log("✅ Trading resumed: Pause duration expired", level="INFO")
            save_risk_state()
            return True
    
    return False

async def calculate_position_size(symbol, candles_by_tf, account_balance, entry_price, stop_loss, 
                                  trade_type, strategy, confidence, market_type="linear"):
    """
    Master function to calculate position size based on all risk factors
    
    Args:
        symbol: Trading symbol
        candles_by_tf: Dictionary of candles by timeframe
        account_balance: Current account balance
        entry_price: Planned entry price
        stop_loss: Planned stop loss price
        trade_type: Trade type (Scalp, Intraday, Swing)
        strategy: Trading strategy
        confidence: Confidence score (0-100)
        market_type: Market type (linear/spot)
    
    Returns:
        tuple: (position_size, risk_amount, leverage)
    """
    # Verify trading is allowed
    if not check_trading_allowed():
        log(f"🛑 Position sizing blocked for {symbol}: Trading currently paused due to drawdown", level="WARN")
        return 0, 0, 1
    
    # Base risk by trade type
    base_risk_map = {"Scalp": 0.02, "Intraday": 0.015, "Swing": 0.01}
    base_risk = base_risk_map.get(trade_type, 0.015)
    
    # Calculate volatility
    timeframe_map = {"Scalp": "5", "Intraday": "15", "Swing": "60"}
    tf = timeframe_map.get(trade_type, "15")
    
    volatility, vol_band = await calculate_symbol_volatility(symbol, candles_by_tf, tf)
    
    # Get appropriate candles for strategy calculations
    candles = candles_by_tf.get(tf, [])
    if not candles:
        log(f"⚠️ No candles available for {symbol} on {tf}m timeframe", level="WARN")
        return 0, 0, 1
    
    # Calculate adjusted risk percentage
    adjusted_risk = calculate_volatility_adjusted_position_size(
        symbol, base_risk, vol_band, strategy, candles, confidence
    )
    
    # Check daily risk limit
    if not check_daily_risk_limit(adjusted_risk):
        return 0, 0, 1
    
    # Calculate dollar risk amount
    dollar_risk = account_balance * adjusted_risk
    
    # Default leverage based on market type
    leverage = 3 if market_type == "linear" else 1
    
    # Calculate position size based on stop loss distance
    if entry_price <= 0 or stop_loss <= 0 or entry_price == stop_loss:
        log(f"⚠️ Invalid price inputs for {symbol}: Entry={entry_price}, SL={stop_loss}", level="WARN")
        return 0, 0, leverage
    
    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss) / entry_price
    
    # Calculate position value and size
    position_value = dollar_risk / risk_per_unit
    position_size = position_value / entry_price
    
    # Apply leverage for futures
    if market_type == "linear":
        position_size *= leverage
    
    # Register trade risk
    register_trade_risk(symbol, adjusted_risk, strategy)
    
    return position_size, dollar_risk, leverage

async def update_risk_metrics():
    """Updated risk manager with less frequent balance calls"""
    while True:
        try:
            # Only fetch balance every 5 minutes instead of every update
            from bybit_api import get_futures_available_balance
            balance = await get_futures_available_balance(
                force_refresh=False,
                caller_name="risk_manager"
            )
            
            if balance > 0:
                update_account_balance(balance)
                
                # Log daily risk status (less frequently)
                reset_daily_risk()
                log(f"📊 Daily Risk: {daily_risk['used_risk']:.2%} used")
            
        except Exception as e:
            log(f"❌ Error updating risk metrics: {e}", level="ERROR")
        
        # Update every 15 minutes (increased from frequent updates)
        await asyncio.sleep(900)
