"""
Position Manager - Handles position sizing, execution and risk management
"""
import asyncio
import json
import traceback
import time
from datetime import datetime
from logger import log, write_log
from symbol_info import get_precision, round_qty
from bybit_api import place_market_order, get_futures_available_balance, signed_request
from error_handler import send_telegram_message, send_error_to_telegram

# Import the new advanced risk manager
from risk_manager import (
    calculate_position_size,
    update_strategy_performance,
    check_trading_allowed,
    reset_daily_risk,
    load_risk_state
)

# Import the improved SL/TP utilities
from sl_tp_utils import (
    calculate_dynamic_sl_tp,
    calculate_exit_tranches,
    validate_sl_placement
)

# Cache the account balance to avoid excessive API calls
_cached_balance = None
_balance_timestamp = 0
_balance_cache_ttl = 30  # 30 seconds TTL for balance cache

# Execution states for retry logic
EXECUTION_STATES = {}

async def get_account_balance():
    """FIXED: Updated position manager balance function with None protection"""
    try:
        # Use optimized balance with caller identification
        from bybit_api import get_futures_available_balance
        usdt_balance = await get_futures_available_balance(
            force_refresh=False, 
            caller_name="position_manager"
        )
        
        # ========== CRITICAL FIX: Handle None balance ==========
        if usdt_balance is None:
            log(f"⚠️ Balance API returned None - using fallback balance", level="WARN")
            return 1000.0  # Fallback balance for testing
        
        # Convert to float to be safe
        usdt_balance = float(usdt_balance)
        
        if usdt_balance > 0:
            return usdt_balance
        else:
            log(f"⚠️ Invalid balance returned: {usdt_balance}, using fallback", level="WARN")
            return 1000.0  # Fallback balance
            
    except Exception as e:
        log(f"❌ Failed to get wallet balance: {e}, using fallback", level="ERROR")
        return 1000.0  # Fallback balance

async def calculate_quantity(symbol, price, sl_price, account_balance, 
                            candles_by_tf, trade_type, strategy, confidence,
                            risk_pct=None, market_type="linear"):
    """
    Calculate position size based on account balance, risk, and SL distance
    
    Args:
        symbol: Trading symbol 
        price: Entry price
        sl_price: Stop loss price
        account_balance: Account balance
        candles_by_tf: Candles by timeframe for volatility calculation
        trade_type: Trade type (Scalp, Intraday, Swing)
        strategy: Strategy name
        confidence: Confidence percentage (0-100)
        risk_pct: Risk percentage override (if None, calculated dynamically)
        market_type: Market type (linear/spot)
        
    Returns:
        float: Calculated position size
    """

    try:
        # CRITICAL FIX: Add comprehensive null checks for SUNUSDT error
        if price is None:
            log(f"❌ Entry price is None for {symbol}", level="ERROR")
            return 0
        if sl_price is None:
            log(f"❌ Stop loss price is None for {symbol}", level="ERROR")
            return 0
        if account_balance is None:
            log(f"❌ Account balance is None for {symbol}", level="ERROR")
            return 0
    
        # Convert to float and validate
        try:
            price = float(price)
            sl_price = float(sl_price)
            account_balance = float(account_balance)
        except (ValueError, TypeError) as e:
            log(f"❌ Invalid numeric values for {symbol}: {e}", level="ERROR")
            return 0
    
        if price <= 0 or sl_price <= 0 or account_balance <= 0:
            log(f"❌ Invalid values for {symbol}: price={price}, sl={sl_price}, balance={account_balance}", level="ERROR")
            return 0

    except Exception as e:
        log(f"❌ Error: {e}", level="ERROR")
        return 0
                                
    try:
        # Use dynamic risk calculation if not provided
        if risk_pct is None:
            position_size, risk_amount, leverage = await calculate_position_size(
                symbol=symbol,
                candles_by_tf=candles_by_tf,
                account_balance=account_balance,
                entry_price=price,
                stop_loss=sl_price,
                trade_type=trade_type,
                strategy=strategy,
                confidence=confidence,
                market_type=market_type
            )
        else:
            # Calculate position size with fixed risk
            # In calculate_quantity function, find this section and replace:

            # Calculate position size with fixed risk
            risk_amount = account_balance * (risk_pct / 100)
            
            # FIXED: Add validation for risk_amount
            if risk_amount is None or risk_amount <= 0:
                log(f"❌ Invalid risk amount for {symbol}: {risk_amount}", level="ERROR")
                return 0

            # Calculate risk per unit
            if price <= 0 or sl_price <= 0 or price == sl_price:
                log(f"❌ Invalid prices for position sizing: Entry={price}, SL={sl_price}", level="ERROR")
                return 0

            risk_per_unit = abs(price - sl_price) / price

            # FIXED: Add validation for risk_per_unit
            if risk_per_unit is None or risk_per_unit <= 0:
                log(f"❌ Invalid risk per unit for {symbol}: {risk_per_unit}", level="ERROR")
                return 0

            # Default leverage
            leverage = 3 if market_type == "linear" else 1

            # Calculate position value and size
            position_value = risk_amount / risk_per_unit

            # FIXED: Add validation for position_value
            if position_value is None or position_value <= 0:
                log(f"❌ Invalid position value for {symbol}: {position_value}", level="ERROR")
                return 0

            # FIXED: Validate price before division
            if price is None or price <= 0:
                log(f"❌ Invalid price for position sizing: {price}", level="ERROR")
                return 0

            # NOW SAFE: Calculate position size
            position_size = position_value / price

            # FIXED: Validate position_size result
            if position_size is None or position_size <= 0:
                log(f"❌ Invalid position size calculated for {symbol}: {position_size}", level="ERROR")
                return 0

            # Apply leverage for futures with validation
            if market_type == "linear":
                if leverage is None or leverage <= 0:
                    leverage = 3
                    log(f"⚠️ Invalid leverage for {symbol}, using default: {leverage}")
    
                position_size = position_size * leverage
    
                # Final validation after leverage
                if position_size is None or position_size <= 0:
                    log(f"❌ Position size invalid after leverage for {symbol}: {position_size}", level="ERROR")
                    return 0

            precision = get_precision(symbol)
            position_size = round(position_size, precision)
        
        # Log detailed calculation
        log(f"📊 Position sizing for {symbol}:")
        log(f"  Price: {price}, SL: {sl_price}, Distance: {abs(price - sl_price) / price:.2%}")
        log(f"  Risk: ${risk_amount:.2f}, Leverage: {leverage}x")
        log(f"  Final Size: {position_size} units")
        
        return position_size
        
    except Exception as e:
        log(f"❌ Error calculating position size: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return 0

async def execute_twap_entry(symbol, direction, qty, category="linear", slices=3, delay_sec=2):
    """
    Execute a TWAP (Time-Weighted Average Price) entry to reduce market impact
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        qty: Total quantity to execute
        category: Market category
        slices: Number of slices to divide the order
        delay_sec: Delay between slices
        
    Returns:
        float or None: Average entry price if successful, None otherwise
    """
    try:
        # Calculate slice size
        slice_qty = round_qty(symbol, qty / slices)
        
        # Minimum quantity check
        if slice_qty <= 0:
            log(f"⚠️ TWAP slice quantity too small ({slice_qty}), executing as market order", level="WARN")
            return await execute_market_entry(symbol, direction, qty, category)
        
        side = "Buy" if direction.lower() == "long" else "Sell"
        entries = []
        
        log(f"🔄 Starting TWAP execution for {symbol} {side} {qty} in {slices} slices...")
        
        # Execute slices with delay
        for i in range(slices):
            # For the last slice, use remaining quantity
            if i == slices - 1:
                remaining_qty = qty - sum(entries[j]["qty"] for j in range(i) if j < len(entries))
                current_qty = round_qty(symbol, remaining_qty)
            else:
                current_qty = slice_qty
            
            if current_qty <= 0:
                continue
                
            log(f"📤 TWAP Slice {i+1}/{slices}: {current_qty} {side}")
            
            try:
                result = await place_market_order(
                    symbol=symbol,
                    side=side,
                    qty=str(current_qty),
                    market_type=category
                )
                
                if result.get("retCode") == 0:
                    order_data = result.get("result", {})
                    price = float(order_data.get("avgPrice") or order_data.get("price") or 0)
                    
                    if price > 0:
                        entries.append({"price": price, "qty": current_qty})
                        log(f"✅ TWAP Slice {i+1} executed at {price}")
                    else:
                        log(f"⚠️ TWAP Slice {i+1} missing price data", level="WARN")
                else:
                    log(f"❌ TWAP Slice {i+1} failed: {result.get('retMsg')}", level="ERROR")
            except Exception as e:
                log(f"❌ Error in TWAP Slice {i+1}: {e}", level="ERROR")
            
            # Delay between slices (except for the last one)
            if i < slices - 1:
                await asyncio.sleep(delay_sec)
        
        # Calculate average entry price
        if entries:
            total_value = sum(e["price"] * e["qty"] for e in entries)
            total_qty = sum(e["qty"] for e in entries)
            
            if total_qty > 0:
                avg_entry = round(total_value / total_qty, 6)
                log(f"✅ Final TWAP Entry Price: {avg_entry}")
                return avg_entry
        
        log(f"❌ TWAP execution failed for {symbol} - no valid entries", level="ERROR")
        return None
        
    except Exception as e:
        log(f"❌ Error in TWAP execution: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

async def execute_market_entry(symbol, direction, qty, category="linear"):
    """
    Execute a simple market order entry
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        qty: Quantity to execute
        category: Market category
        
    Returns:
        float or None: Entry price if successful, None otherwise
    """
    try:
        side = "Buy" if direction.lower() == "long" else "Sell"
        
        log(f"📤 Sending market order: {side} {qty} {symbol}")
        
        result = await place_market_order(
            symbol=symbol,
            side=side,
            qty=str(qty),
            market_type=category
        )
        
        if result.get("retCode") == 0:
            order_data = result.get("result", {})
            price = float(order_data.get("avgPrice") or order_data.get("price") or 0)
            
            if price > 0:
                log(f"✅ Market order executed at {price}")
                return price
            else:
                log(f"⚠️ Market order missing price data", level="WARN")
                
                # Try to get the price from get order API
                order_id = order_data.get("orderId")
                if order_id:
                    await asyncio.sleep(1)  # Brief delay to allow order to be processed
                    order_details = await signed_request("GET", "/v5/order/realtime", {
                        "category": category,
                        "symbol": symbol,
                        "orderId": order_id
                    })
                    
                    if order_details.get("retCode") == 0:
                        orders = order_details.get("result", {}).get("list", [])
                        if orders:
                            price = float(orders[0].get("avgPrice") or orders[0].get("price") or 0)
                            if price > 0:
                                log(f"✅ Retrieved order price from details: {price}")
                                return price
                
                # Fallback to current market price
                ticker_resp = await signed_request("GET", "/v5/market/tickers", {
                    "category": category, 
                    "symbol": symbol
                })
                
                if ticker_resp.get("retCode") == 0:
                    price = float(ticker_resp.get("result", {}).get("list", [{}])[0].get("lastPrice", 0))
                    if price > 0:
                        log(f"⚠️ Using market price as fallback: {price}", level="WARN")
                        return price
                
                log(f"❌ Failed to determine execution price", level="ERROR")
                return None
        else:
            log(f"❌ Market order failed: {result.get('retMsg')}", level="ERROR")
            return None
            
    except Exception as e:
        log(f"❌ Error in market entry: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

async def set_position_leverage(symbol, leverage, category="linear"):
    """
    Set leverage for a position
    
    Args:
        symbol: Trading symbol
        leverage: Leverage value (1-25)
        category: Market category
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result = await signed_request("POST", "/v5/position/set-leverage", {
            "category": category,
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        })
        
        ret_code = result.get("retCode")
        if ret_code == 0:
            log(f"✅ Set leverage for {symbol} to {leverage}x")
            return True
        elif ret_code == 110043:
            log(f"ℹ️ Leverage already set to {leverage}x for {symbol}")
            return True
        else:
            log(f"⚠️ Failed to set leverage: {result.get('retMsg')}", level="WARN")
            return False
            
    except Exception as e:
        log(f"❌ Error setting leverage: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return False

async def place_stop_loss_order(symbol, direction, qty, sl_price, market_type="linear"):
    """
    Place a stop loss order for a position
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        qty: Position size
        sl_price: Stop loss price
        market_type: Market type
        
    Returns:
        str or None: Order ID if successful, None otherwise
    """
    from bybit_api import place_stop_loss
    
    try:
        # Validate the SL price is on the correct side of the market
        sl_price = await validate_sl_placement(symbol, direction, sl_price, market_type)
        
        log(f"🛡️ Placing SL order for {symbol}: {direction} at {sl_price}")
        
        result = await place_stop_loss(
            symbol=symbol,
            direction=direction,
            qty=qty,
            sl_price=sl_price,
            market_type=market_type
        )
        
        if result.get("retCode") == 0:
            order_id = result.get("result", {}).get("orderId")
            log(f"✅ SL order placed: {order_id}")
            return order_id
        else:
            log(f"❌ Failed to place SL order: {result.get('retMsg')}", level="ERROR")
            return None
            
    except Exception as e:
        log(f"❌ Error placing SL order: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

async def place_take_profit_order(symbol, direction, qty, tp_price, market_type="linear"):
    """
    Place a take profit limit order
    
    Args:
        symbol: Trading symbol
        direction: 'long' or 'short'
        qty: Position size
        tp_price: Take profit price
        market_type: Market type
        
    Returns:
        str or None: Order ID if successful, None otherwise
    """
    try:
        side = "Sell" if direction.lower() == "long" else "Buy"
        
        log(f"💰 Placing TP order for {symbol}: {side} at {tp_price}")
        
        result = await signed_request("POST", "/v5/order/create", {
            "category": market_type,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(tp_price),
            "timeInForce": "GTC",
            "reduceOnly": True
        })
        
        if result.get("retCode") == 0:
            order_id = result.get("result", {}).get("orderId")
            log(f"✅ TP order placed: {order_id}")
            return order_id
        else:
            log(f"❌ Failed to place TP order: {result.get('retMsg')}", level="ERROR")
            return None
            
    except Exception as e:
        log(f"❌ Error placing TP order: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

async def cancel_all_orders(symbol, category="linear"):
    """
    Cancel all open orders for a symbol
    
    Args:
        symbol: Trading symbol
        category: Market category
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result = await signed_request("POST", "/v5/order/cancel-all", {
            "category": category,
            "symbol": symbol
        })
        
        if result.get("retCode") == 0:
            log(f"✅ Cancelled all orders for {symbol}")
            return True
        else:
            log(f"⚠️ Failed to cancel orders: {result.get('retMsg')}", level="WARN")
            return False
            
    except Exception as e:
        log(f"❌ Error cancelling orders: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return False

async def execute_trade(signal_data, use_twap=True):
    """
    Master function to execute a complete trade setup
    
    Args:
        signal_data: Dictionary containing trade setup details
        use_twap: Whether to use TWAP for entry (default True)
        
    Returns:
        dict or None: Trade details if executed successfully, None otherwise
    """
    # Start by loading risk state
    load_risk_state()
    
    # Reset daily risk tracking if needed
    reset_daily_risk()
    
    # Check if trading is allowed based on drawdown limits
    if not check_trading_allowed():
        log(f"🛑 Trading paused due to drawdown limits - trade blocked", level="WARN")
        await send_telegram_message("🛑 <b>Trade Blocked</b>: Trading paused due to drawdown limits")
        return None
    
    # Extract trade details
    symbol = signal_data["symbol"]
    category = signal_data.get("market_type", "linear")
    trade_type = signal_data.get("trade_type", "Intraday")
    direction = signal_data.get("direction", "Long").strip().lower()
    regime = signal_data.get("regime", "trending")
    score = signal_data.get("score", 0)
    confidence = signal_data.get("confidence", 60)
    entry_price = float(signal_data.get("price", 1.0))
    candles_by_tf = signal_data.get("candles", {})
    
    # Determine strategy type
    strategy = "core_strategy"
    if "mean_reversion" in signal_data.get("tf_scores", {}):
        strategy = "mean_reversion"
    elif "breakout_sniper" in signal_data.get("tf_scores", {}):
        strategy = "breakout_sniper"
    
    log(f"⚙️ Executing {direction.upper()} trade for {symbol} [{category.upper()}] as {trade_type} ({strategy})")
    
    # Execution state tracking
    exec_id = f"{symbol}_{int(time.time())}"
    EXECUTION_STATES[exec_id] = {"stage": "started", "success": False}
    
    try:
        # Step 1: Get account balance
        account_balance = await get_account_balance()
        if account_balance <= 0:
            log(f"❌ Invalid account balance: {account_balance} USDT", level="ERROR")
            await send_telegram_message(f"❌ <b>Execution Error</b>\nSymbol: <b>{symbol}</b>\nError: Invalid account balance.")
            return None
        
        # Step 2: Calculate SL/TP levels
        sl_tp_result = calculate_dynamic_sl_tp(
            candles_by_tf=candles_by_tf,
            entry_price=entry_price,
            trade_type=trade_type,
            direction=direction,
            score=score,
            confidence=confidence,
            regime=regime,
            strategy=strategy
        )
        
        if len(sl_tp_result) < 5:
            log(f"❌ Invalid SL/TP calculation result", level="ERROR")
            return None
            
        sl_price, tp1_price, sl_pct, trailing_pct, tp1_pct, tp2_price, tp2_pct, tp3_price, tp3_pct = sl_tp_result
        
        EXECUTION_STATES[exec_id]["stage"] = "sl_tp_calculated"
        
        # Step 3: Calculate position size
        qty = await calculate_quantity(
            symbol=symbol,
            price=entry_price,
            sl_price=sl_price,
            account_balance=account_balance,
            candles_by_tf=candles_by_tf,
            trade_type=trade_type,
            strategy=strategy,
            confidence=confidence,
            market_type=category
        )
        
        if qty <= 0:
            log(f"⚠️ Skipped {symbol}: Quantity too small or risk limit reached.")
            return None
        
        EXECUTION_STATES[exec_id]["stage"] = "position_sized"
        
        # Step 4: Set leverage (for futures)
        if category == "linear":
            leverage = 3  # Default leverage
            await set_position_leverage(symbol, leverage, category)
        else:
            leverage = 1  # No leverage for spot
        
        # Step 5: Cancel any existing orders
        await cancel_all_orders(symbol, category)
        
        EXECUTION_STATES[exec_id]["stage"] = "orders_cancelled"
        
        # Step 6: Execute entry
        if regime == "volatile" and use_twap and category == "linear":
            executed_price = await execute_twap_entry(
                symbol=symbol,
                direction=direction,
                qty=qty,
                category=category,
                slices=3,
                delay_sec=2
            )
        else:
            executed_price = await execute_market_entry(
                symbol=symbol,
                direction=direction,
                qty=qty,
                category=category
            )
        
        if not executed_price:
            log(f"❌ Entry execution failed for {symbol}", level="ERROR")
            EXECUTION_STATES[exec_id]["stage"] = "entry_failed"
            return None
        
        EXECUTION_STATES[exec_id]["stage"] = "entry_executed"
        
        # Step 7: Place stop loss order
        sl_order_id = await place_stop_loss_order(
            symbol=symbol,
            direction=direction,
            qty=qty,
            sl_price=sl_price,
            market_type=category
        )
        
        EXECUTION_STATES[exec_id]["stage"] = "sl_placed"
        
        # Step 8: Calculate exit tranches based on volatility and momentum
        volatility = "normal"
        if regime == "volatile":
            volatility = "high"
        elif regime == "ranging":
            volatility = "low"
            
        has_momentum = signal_data.get("momentum", False) or signal_data.get("pump_potential", False)
        
        exit_tranches = calculate_exit_tranches(
            symbol=symbol,
            total_qty=qty,
            trade_type=trade_type,
            volatility=volatility,
            momentum=has_momentum
        )
        
        # Step 9: Place TP order for first tranche
        tp1_qty = exit_tranches[0] if exit_tranches and len(exit_tranches) > 0 else round_qty(symbol, qty / 3)
        
        tp1_order_id = await place_take_profit_order(
            symbol=symbol,
            direction=direction,
            qty=tp1_qty,
            tp_price=tp1_price,
            market_type=category
        )
        
        EXECUTION_STATES[exec_id]["stage"] = "tp_placed"
        EXECUTION_STATES[exec_id]["success"] = True
        
        # Build complete trade details
        trade_details = {
            "symbol": symbol,
            "direction": direction,
            "entry": executed_price,
            "sl": sl_price,
            "tp1": tp1_price,
            "tp2": tp2_price,
            "tp3": tp3_price,
            "qty": qty,
            "type": trade_type,
            "strategy": strategy,
            "sl_pct": sl_pct,
            "tp1_pct": tp1_pct,
            "tp2_pct": tp2_pct,
            "tp3_pct": tp3_pct,
            "trailing_pct": trailing_pct,
            "sl_order_id": sl_order_id,
            "tp1_order_id": tp1_order_id,
            "exit_tranches": exit_tranches,
            "regime": regime,
            "leverage": leverage,
            "score": score,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Log trade execution details
        write_log(f"TRADE_EXECUTED: {json.dumps(trade_details, default=str)}")
        
        return trade_details
        
    except Exception as e:
        log(f"❌ Trade execution error for {symbol}: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        
        EXECUTION_STATES[exec_id]["stage"] = "error"
        EXECUTION_STATES[exec_id]["error"] = str(e)
        
        await send_telegram_message(
            f"❌ <b>Execution Error</b>\nSymbol: <b>{symbol}</b>\nError: {str(e)}"
        )
        
        return None

async def process_trade_result(trade_data, result_type, pnl_value=None):
    """
    Process trade result for tracking and performance updates
    
    Args:
        trade_data: Trade data dictionary
        result_type: Result type ('win', 'loss', 'partial')
        pnl_value: PnL value (percentage or absolute)
        
    Returns:
        bool: True if processed successfully
    """
    try:
        strategy = trade_data.get("strategy", "core_strategy")
        
        # Log to strategy performance tracking
        if result_type in ["win", "loss"]:
            update_strategy_performance(strategy, result_type, pnl_value or 0)
        
        # Log to trade history file
        from activity_logger import log_trade_to_file
        
        log_trade_to_file(
            symbol=trade_data.get("symbol"),
            direction=trade_data.get("direction"),
            entry=trade_data.get("entry"),
            sl=trade_data.get("sl"),
            tp1=trade_data.get("tp1"),
            tp2=trade_data.get("tp2"),
            result=result_type,
            score=trade_data.get("score", 0),
            trade_type=trade_data.get("type", "Unknown"),
            confidence=trade_data.get("confidence", 0)
        )
        
        return True
        
    except Exception as e:
        log(f"❌ Error processing trade result: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return False
