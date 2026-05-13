# trade_executor.py - PURE EXECUTION ONLY
# All strategies and logic are in main.py - this only executes what main.py decides

import asyncio
import traceback
import json
from datetime import datetime
from logger import log, write_log
from bybit_api import signed_request, place_market_order, place_stop_loss_with_retry
from error_handler import send_telegram_message, send_error_to_telegram
from config import DEFAULT_LEVERAGE
from symbol_utils import get_symbol_category
from symbol_info import round_qty
from activity_logger import log_trade_to_file

async def get_account_balance():
    """Get account balance from exchange"""
    try:
        result = await signed_request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        })
        
        if result.get("retCode") == 0:
            accounts = result.get("result", {}).get("list", [])
            for account in accounts:
                coins = account.get("coin", [])
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        balance = float(coin.get("walletBalance", 0))
                        log(f"💰 Account balance: {balance} USDT")
                        return balance
        
        log(f"❌ Failed to get account balance: {result.get('retMsg')}", level="ERROR")
    except Exception as e:
        log(f"❌ Error getting account balance: {e}", level="ERROR")
        return 0.0

async def get_symbol_price(symbol, category="linear"):
    """Get current symbol price from exchange"""
    try:
        resp = await signed_request("GET", "/v5/market/tickers", {
            "category": category,
            "symbol": symbol
        })
        if resp.get("retCode") == 0:
            return float(resp["result"]["list"][0]["lastPrice"])
    except Exception as e:
        log(f"❌ Error getting symbol price: {e}", level="ERROR")
    return 0

def calculate_dynamic_sl_tp(candles_by_tf, price, trade_type, direction, score, confidence, regime="trending", trend_context=None):
    """
    Calculate dynamic SL/TP - Compatibility function for main.py
    This is a simplified version - main.py should handle the full logic
    """
    try:
        # Import the enhanced function from sl_tp_utils
        from sl_tp_utils import calculate_dynamic_sl_tp as enhanced_calculate_dynamic_sl_tp
        
        return enhanced_calculate_dynamic_sl_tp(
            candles_by_tf=candles_by_tf,
            entry_price=price,
            trade_type=trade_type,
            direction=direction,
            score=score,
            confidence=confidence,
            regime=regime
        )
    except Exception as e:
        log(f"❌ Error in dynamic SL/TP calculation: {e}", level="ERROR")
        
        # Fallback calculation
        price = float(price)
        sl_pct = 0.008 if trade_type == "Scalp" else 0.015 if trade_type == "Intraday" else 0.02
        tp_pct = sl_pct * 1.5
        
        if direction.lower() == "long":
            sl_price = price * (1 - sl_pct)
            tp_price = price * (1 + tp_pct)
        else:
            sl_price = price * (1 + sl_pct)
            tp_price = price * (1 - tp_pct)
        
        return sl_price, tp_price, sl_pct, 0.005, tp_pct
        
    except Exception as e:
        log(f"❌ Error getting account balance: {e}", level="ERROR")
        return 0.0

async def execute_trade_if_valid(signal_data, max_risk=0.06):
    """
    WRAPPER FUNCTION - Converts main.py's dictionary call to individual parameters
    This maintains compatibility with existing main.py code
    
    Args:
        signal_data: Dictionary containing all trade parameters
        max_risk: Maximum risk percentage
        
    Returns:
        dict: Trade execution details or None if failed
    """
    try:
        # Extract parameters from signal_data dictionary
        symbol = signal_data.get("symbol")
        direction = signal_data.get("direction")
        if not direction:
            log(f"❌ No direction provided for {symbol}", level="ERROR")
            return None

        # Normalize direction
        if isinstance(direction, str):
            direction = direction.strip().lower()
            if direction not in ["long", "short"]:
                log(f"❌ Invalid direction '{direction}' for {symbol}", level="ERROR")
                return None
        else:
            log(f"❌ Direction is not string: {type(direction)}", level="ERROR")
            return None

        # Add debug logging
        log(f"🔍 DIRECTION DEBUG for {symbol}: '{direction}' (type: {type(direction)})")

        # Ensure it's a string and normalize
        if isinstance(direction, str):
            direction = direction.strip()
            log(f"🔍 Direction after strip: '{direction}'")
        else:
            log(f"❌ Direction is not a string: {type(direction)}", level="ERROR")
            return None
        strategy = signal_data.get("strategy", "core_strategy")
        score = signal_data.get("score", 0)
        confidence = signal_data.get("confidence", 60)
        regime = signal_data.get("regime", "trending")
        is_scalp_hunter = signal_data.get("is_scalp_hunter", False)

        from monitor import active_trades
        if symbol in active_trades and not active_trades[symbol].get("exited", False):
            log(f"🚫 Duplicate entry prevented for {symbol} (local active trade already open)")
            return None

        # Second layer – ask the exchange in case our file-state is stale
        # FIX: Pass the trade data from active_trades, not undefined 'trade_data'
        from trade_verification import verify_position_and_orders
        
        # Check if we have an active trade to verify
        if symbol in active_trades:
            trade_to_verify = active_trades[symbol]
            if await verify_position_and_orders(symbol, trade_to_verify):
                log(f"🚫 Duplicate entry prevented for {symbol} (exchange still reports open pos/order)")
                return None
        
        # Get account balance if not provided
        account_balance = signal_data.get("account_balance")
        if not account_balance:
            account_balance = await get_account_balance()
        
        # Call the actual execution function
        return await execute_trade_core(
            symbol=symbol,
            direction=direction, 
            signal_data=signal_data,
            strategy=strategy,
            score=score,
            confidence=confidence,
            regime=regime,
            account_balance=account_balance,
            risk_per_trade=max_risk  # Pass as decimal (0.06 for 6%)
        )
        
    except Exception as e:
        log(f"❌ Error in execute_trade_if_valid wrapper: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        return None

async def execute_trade_core(
    symbol, 
    direction, 
    signal_data, 
    strategy, 
    score, 
    confidence, 
    regime, 
    account_balance,
    risk_per_trade=1.0
):
    """
    PURE EXECUTION FUNCTION - Executes trades based on main.py decisions
    
    Args:
        symbol: Trading symbol (from main.py)
        direction: Trade direction (from main.py)
        signal_data: Signal data with all calculations (from main.py)
        strategy: Strategy name (from main.py)
        score: Signal score (from main.py)
        confidence: Signal confidence (from main.py)
        regime: Market regime (from main.py)
        account_balance: Account balance (from main.py)
        risk_per_trade: Risk percentage (from main.py)
        
    Returns:
        dict: Trade execution details or None if failed
    """
    try:
        log(f"🔄 EXECUTING TRADE: {direction} {symbol} | Strategy: {strategy}")
        
        # Get execution parameters from signal_data
        category = signal_data.get("market_type", get_symbol_category(symbol))
        trade_type = signal_data.get("trade_type", "Intraday")

        # Get entry price — try multiple keys
        entry_price = signal_data.get("entry_price") or signal_data.get("price")

        # If still missing, fetch live price
        if not entry_price or float(entry_price) == 0:
            entry_price = await get_symbol_price(symbol)
            log(f"⚠️ {symbol}: entry_price was missing, fetched live: {entry_price}")

        # Final guard — abort if no valid price
        if not entry_price or float(entry_price) == 0:
            log(f"❌ {symbol}: Cannot execute — no valid entry price", level="ERROR")
            return None

        entry_price = float(entry_price)
        
        # Get prices from signal_data or calculate them
        current_price = signal_data.get("price", 0)
        if current_price <= 0:
            current_price = await get_symbol_price(symbol, category)
            
        sl_price = signal_data.get("sl_price", 0)
        tp1_price = signal_data.get("tp1_price", 0)
        qty = signal_data.get("qty", 0)
        candles_by_tf = signal_data.get("candles", {})

        # Risk normalization — used by both scalp hunter and standard path
        if risk_per_trade > 1:
            risk_decimal = risk_per_trade / 100
        else:
            risk_decimal = risk_per_trade
        risk_decimal = min(risk_decimal, 0.03)  # hard cap at 3% per trade

        if signal_data.get("is_scalp_hunter") and signal_data.get("sl_price"):
            # SCALP HUNTER path — SL/TP are pre-computed by scalp_hunter.py,
            # but qty is NOT in signal_data. Previously this branch left qty=0,
            # silently breaking every scalp-hunter trade. Compute it here.
            sl_price = float(signal_data["sl_price"])
            tp1_price = float(signal_data["tp1_price"])
            sl_pct = float(signal_data.get("sl_pct", 0.005))
            tp1_pct = float(signal_data.get("tp1_pct", 0.009))
            trailing_pct = float(signal_data.get("trailing_pct", 0.5))

            price_diff = abs(current_price - sl_price)
            if price_diff <= 0 or current_price <= 0:
                log(f"❌ SCALP HUNTER {symbol}: invalid price_diff={price_diff} / price={current_price}", level="ERROR")
                return None

            risk_amount = account_balance * risk_decimal
            qty = round_qty(symbol, risk_amount / price_diff)

            # 25%-of-balance position cap also applies to scalp hunter
            max_position_value = account_balance * 0.25
            position_value = qty * current_price
            if position_value > max_position_value:
                log(f"⚠️ SCALP HUNTER {symbol}: capping position from ${position_value:.2f} to ${max_position_value:.2f}")
                qty = round_qty(symbol, max_position_value / current_price)

            if qty <= 0:
                log(f"❌ SCALP HUNTER {symbol}: qty rounded to {qty}, cannot execute", level="ERROR")
                return None

            log(f"✅ SCALP HUNTER: SL={sl_price:.6f} TP1={tp1_price:.6f} qty={qty} risk=${risk_amount:.2f}")
            sl_tp_result = (sl_price, tp1_price, sl_pct, trailing_pct, tp1_pct)
        else:
            # STANDARD path — calculate SL/TP and size from risk %
            sl_tp_result = calculate_dynamic_sl_tp(
                candles_by_tf=candles_by_tf,
                price=current_price,
                trade_type=trade_type,
                direction=direction,
                score=score,
                confidence=confidence,
                regime=regime
            )

            if not sl_tp_result or len(sl_tp_result) < 5:
                log(f"❌ Failed to calculate SL/TP for {symbol}", level="ERROR")
                return None

            sl_price, tp1_price, sl_pct, trailing_pct, tp1_pct = sl_tp_result[:5]

            price_diff = abs(current_price - sl_price)
            if price_diff <= 0:
                log(f"❌ {symbol}: SL distance is zero, refusing to trade", level="ERROR")
                return None

            risk_amount = account_balance * risk_decimal
            qty = round_qty(symbol, risk_amount / price_diff)

            max_position_value = account_balance * 0.25
            position_value = qty * current_price
            if position_value > max_position_value:
                log(f"⚠️ Position too large ({position_value:.2f}), capping at 25% of account ({max_position_value:.2f})")
                qty = round_qty(symbol, max_position_value / current_price)

            if qty <= 0:
                log(f"❌ {symbol}: qty rounded to {qty}, cannot execute", level="ERROR")
                return None

            log(f"📊 Position sizing: Risk={risk_decimal*100:.1f}%, Amount=${risk_amount:.2f}, Qty={qty}, Value=${qty*current_price:.2f}")
        
        log(f"📊 EXECUTING: {symbol} | Qty: {qty} | Entry: {current_price} | SL: {sl_price} | TP1: {tp1_price}")
        
        # Step 1: Set leverage if needed
        if category == "linear":
            lev = signal_data.get("leverage", DEFAULT_LEVERAGE) if signal_data.get("is_scalp_hunter") else DEFAULT_LEVERAGE
            await set_leverage(symbol, lev, category)
        
        # Step 2: Execute market order
        side = "Buy" if direction.lower() == "long" else "Sell"

        result = await place_market_order(
            symbol=symbol,
            side=side,
            qty=str(qty),
            market_type=category
        )

        if result.get("retCode") != 0:
            log(f"❌ Market order failed: {result.get('retMsg')}", level="ERROR")
            return None

        # --- FIX: Bybit V5 market order response does NOT include fill price.
        # Previously this code read `result.price` (often 0 / empty) and fell back
        # to the pre-trade `current_price` snapshot. On any slippage event, SL/TP
        # were anchored to the wrong price. Now we query the actual position to
        # get avgPrice and size from the exchange's books.
        order_id = result.get("result", {}).get("orderId", "")
        log(f"🚀 Market order submitted: orderId={order_id}, intended qty={qty}")

        # Give exchange a moment to settle the position
        await asyncio.sleep(0.5)

        avg_entry_price = 0.0
        executed_qty = 0.0
        for _attempt in range(3):
            try:
                pos_resp = await signed_request("GET", "/v5/position/list", {
                    "category": category,
                    "symbol": symbol,
                })
                if pos_resp.get("retCode") == 0:
                    for pos in pos_resp.get("result", {}).get("list", []):
                        if pos.get("symbol") != symbol:
                            continue
                        size = float(pos.get("size", 0) or 0)
                        if size > 0:
                            avg_entry_price = float(pos.get("avgPrice", 0) or 0)
                            executed_qty = size
                            break
                if avg_entry_price > 0 and executed_qty > 0:
                    break
            except Exception as e:
                log(f"⚠️ Position query attempt {_attempt+1} failed: {e}", level="WARN")
            await asyncio.sleep(0.5)

        if avg_entry_price <= 0 or executed_qty <= 0:
            # Position may have filled but we can't confirm. Fall back to snapshot
            # so downstream code has something usable, but flag prominently.
            log(f"⚠️ {symbol}: could not confirm fill via /v5/position/list — "
                f"falling back to snapshot price {current_price} and intended qty {qty}",
                level="WARN")
            avg_entry_price = current_price
            executed_qty = qty
            await send_error_to_telegram(
                f"⚠️ {symbol}: order submitted but fill unconfirmed. "
                f"Manual check advised (orderId={order_id})."
            )

        log(f"✅ Fill confirmed: qty={executed_qty} avgPrice={avg_entry_price}")

        # --- FIX: Re-anchor SL/TP to the actual fill price.
        # If slippage moved us, the pre-computed prices are off by the slip amount,
        # which can mean a tight scalp SL ends up the wrong side of the fill.
        if signal_data.get("is_scalp_hunter"):
            # Scalp hunter has fixed % SL/TP — recompute from real entry
            if direction.lower() == "long":
                sl_price = round(avg_entry_price * (1 - sl_pct / 100), 6)
                tp1_price = round(avg_entry_price * (1 + tp1_pct / 100), 6)
            else:
                sl_price = round(avg_entry_price * (1 + sl_pct / 100), 6)
                tp1_price = round(avg_entry_price * (1 - tp1_pct / 100), 6)
            log(f"🔄 Re-anchored scalp SL={sl_price} TP1={tp1_price} to actual fill")
        else:
            # Standard path uses fixed % from sl_tp_utils.FIXED_SL_TP
            if direction.lower() == "long":
                sl_price = round(avg_entry_price * (1 - sl_pct / 100), 6)
                tp1_price = round(avg_entry_price * (1 + tp1_pct / 100), 6)
            else:
                sl_price = round(avg_entry_price * (1 + sl_pct / 100), 6)
                tp1_price = round(avg_entry_price * (1 - tp1_pct / 100), 6)
            log(f"🔄 Re-anchored SL={sl_price} TP1={tp1_price} to actual fill")

        # Step 3: Place Stop Loss order (retries built-in via place_stop_loss_with_retry)
        sl_order_id = await place_stop_loss_order(
            symbol=symbol,
            direction=direction,
            qty=executed_qty,
            sl_price=sl_price,
            market_type=category
        )

        # --- FIX: NAKED POSITION GUARD ---
        # Previously: if SL placement failed, code just logged WARN and continued
        # with an unprotected live position. That window of "filled but no stop"
        # is where flash dumps cause >20% drawdowns. Now: close immediately.
        if not sl_order_id:
            log(f"🚨 SL PLACEMENT FAILED for {symbol} — closing position to avoid naked exposure", level="ERROR")
            try:
                close_side = "Sell" if direction.lower() == "long" else "Buy"
                close_result = await place_market_order(
                    symbol=symbol,
                    side=close_side,
                    qty=str(executed_qty),
                    market_type=category,
                    reduce_only=True,
                )
                if close_result.get("retCode") == 0:
                    log(f"✅ Emergency close succeeded for {symbol} after SL failure")
                else:
                    log(f"❌ EMERGENCY CLOSE ALSO FAILED for {symbol}: {close_result.get('retMsg')}",
                        level="ERROR")
            except Exception as e:
                log(f"❌ Exception during emergency close: {e}", level="ERROR")
            await send_error_to_telegram(
                f"🚨 {symbol}: SL placement failed after fill. "
                f"Attempted emergency reduce-only close. CHECK POSITIONS MANUALLY."
            )
            return None

        # Step 4: Place Take Profit (TP1) as a conditional market reduce-only.
        # --- FIX: Previously this was a plain Limit + reduceOnly which can be
        # skipped over by wicks on thin books, leaving the trade unfilled at TP1
        # and trailing never activating. Conditional Market w/ triggerPrice
        # guarantees execution once price touches the level.
        tp1_qty = round_qty(symbol, executed_qty * 0.5)
        if tp1_qty <= 0:
            log(f"⚠️ TP1 qty rounded to 0 for {symbol} — skipping partial TP, full position rides SL/trail", level="WARN")
            tp1_order_id = None
        else:
            tp1_order_id = await place_take_profit_order(
                symbol=symbol,
                direction=direction,
                qty=tp1_qty,
                tp_price=tp1_price,
                market_type=category
            )

        if signal_data.get("is_scalp_hunter"):
            tp1_partial_close = signal_data.get("tp1_partial_close", 0.5)

        if not tp1_order_id and tp1_qty > 0:
            log(f"⚠️ TP1 placement failed for {symbol} — position has SL but no TP1. Monitor will manage exits.", level="WARN")
        
        # Step 5: Register with monitor for trailing management
        try:
            from monitor import track_active_trade
            
            # Get trailing percentage from signal_data or sl_tp_result
            trailing_pct = signal_data.get("trailing_pct", 1.0)
            if 'trailing_pct' in locals() and len(sl_tp_result) >= 4:
                trailing_pct = sl_tp_result[3]
            
            # Get tp1_pct for monitor registration
            tp1_pct = signal_data.get("tp1_pct", 2.0) 
            if 'tp1_pct' in locals() and len(sl_tp_result) >= 5:
                tp1_pct = sl_tp_result[4]
            
            track_active_trade(
                symbol=symbol,
                trade_type=trade_type,
                initial_score=score,
                entry_price=avg_entry_price,
                direction=direction,
                trailing_pct=trailing_pct,
                tp1_target=tp1_price,
                tp1_pct=tp1_pct,
                sl=sl_price,
                sl_order_id=sl_order_id,
                qty=executed_qty
            )
            
            log(f"✅ Trade registered with monitor for trailing management")
            
        except Exception as e:
            log(f"⚠️ Monitor registration failed: {e}", level="WARN")
        
        # Calculate actual risk (for logging)
        actual_risk = calculate_actual_risk_percentage(avg_entry_price, sl_price, executed_qty, account_balance)
        
        # Prepare trade details
        trade_details = {
            "symbol": symbol,
            "direction": direction,
            "trade_type": trade_type,
            "entry_price": avg_entry_price,
            "qty": executed_qty,
            "sl_price": sl_price,
            "tp1_price": tp1_price,
            "sl_order_id": sl_order_id,
            "tp1_order_id": tp1_order_id,
            "strategy": strategy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "leverage": DEFAULT_LEVERAGE if category == "linear" else 1,
            "actual_risk_pct": actual_risk,
            "confidence": confidence,
            "score": score,
            "regime": regime,
            "market_type": category,
            "monitor_managed": True  # Flag for monitor
        }
        
        # Log execution
        write_log(f"TRADE_EXECUTED: {json.dumps(trade_details, default=str)}")
        
        # Send notification
        await send_telegram_message(
            f"✅ <b>Trade Executed</b>\n"
            f"Symbol: <b>{symbol}</b>\n"
            f"Direction: <b>{direction.upper()}</b>\n"
            f"Strategy: <b>{strategy}</b>\n"
            f"Entry: <b>{avg_entry_price}</b>\n"
            f"Quantity: <b>{executed_qty}</b>\n"
            f"SL: <b>{sl_price}</b>\n"
            f"TP1: <b>{tp1_price}</b> (50% exit)\n"
            f"Risk: <b>{actual_risk:.2f}%</b>\n"
            f"🔄 Monitor handles: TP1 → 50% exit → Trailing SL"
        )
        
        # Log to activity file
        log_trade_to_file(
            symbol=symbol,
            direction=direction,
            entry=avg_entry_price,
            sl=sl_price,
            tp1=tp1_price,
            tp2=None,
            result="executed",
            score=score,
            trade_type=trade_type,
            confidence=confidence
        )
        
        log(f"🎯 Trade execution complete for {symbol}")
        return trade_details
        
    except Exception as e:
        log(f"❌ Trade execution error: {e}", level="ERROR")
        log(traceback.format_exc(), level="ERROR")
        await send_error_to_telegram(f"Trade execution failed for {symbol}: {str(e)}")
        return None

async def set_leverage(symbol, leverage, market_type="linear"):
    """Set leverage for a symbol"""
    try:
        result = await signed_request("POST", "/v5/position/set-leverage", {
            "category": market_type,
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        })
        
        if result.get("retCode") == 0:
            log(f"✅ Leverage set to {leverage}x for {symbol}")
            return True
        elif result.get("retCode") == 110043:  # Leverage not modified
            log(f"ℹ️ Leverage already set to {leverage}x for {symbol}")
            return True
        else:
            log(f"⚠️ Failed to set leverage: {result.get('retMsg')}", level="WARN")
            return False
            
    except Exception as e:
        log(f"❌ Error setting leverage: {e}", level="ERROR")
        return False

async def place_stop_loss_order(symbol, direction, qty, sl_price, market_type="linear"):
    """Place stop loss order"""
    try:
        log(f"🛡️ Placing SL order for {symbol}: {direction} at {sl_price}")
        
        from bybit_api import place_stop_loss_with_retry
        
        result = await place_stop_loss_with_retry(
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
            log(f"❌ Failed to place SL: {result.get('retMsg')}", level="ERROR")
            return None
            
    except Exception as e:
        log(f"❌ Error placing SL: {e}", level="ERROR")
        return None

async def place_take_profit_order(symbol, direction, qty, tp_price, market_type="linear"):
    """
    Place TP1 as a CONDITIONAL MARKET ORDER (triggerPrice + reduce-only),
    not a plain Limit. Plain Limits can be skipped over by wicks on thin books,
    leaving the position with no partial exit and trailing never activated.

    Trigger direction:
      - Long position closes at TP via Sell triggered when price RISES to tp_price → triggerDirection=1
      - Short position closes at TP via Buy triggered when price FALLS to tp_price → triggerDirection=2
    """
    try:
        side = "Sell" if direction.lower() == "long" else "Buy"
        trigger_direction = 1 if direction.lower() == "long" else 2

        log(f"💰 Placing conditional TP1 for {symbol}: {side} triggered at {tp_price} (dir={trigger_direction})")

        payload = {
            "category": market_type,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "triggerPrice": str(tp_price),
            "triggerDirection": trigger_direction,
            "triggerBy": "LastPrice",
            "timeInForce": "GTC",
            "reduceOnly": True,
            "closeOnTrigger": False,  # partial — do not close entire position
            "orderFilter": "Stop",
        }

        result = await signed_request("POST", "/v5/order/create", payload)

        if result.get("retCode") == 0:
            order_id = result.get("result", {}).get("orderId")
            log(f"✅ TP1 conditional order placed: {order_id}")
            return order_id
        else:
            err = result.get("retMsg", "")
            log(f"❌ Failed to place TP1 conditional: {err}", level="ERROR")
            # If exchange rejected triggerDirection, retry with opposite direction
            # (some Bybit endpoints differ on convention). One retry only — no loop.
            if "triggerdirection" in err.lower() or "trigger price" in err.lower():
                payload["triggerDirection"] = 2 if trigger_direction == 1 else 1
                log(f"🔄 Retrying TP1 with flipped triggerDirection={payload['triggerDirection']}")
                retry = await signed_request("POST", "/v5/order/create", payload)
                if retry.get("retCode") == 0:
                    return retry.get("result", {}).get("orderId")
                log(f"❌ TP1 retry also failed: {retry.get('retMsg')}", level="ERROR")
            return None

    except Exception as e:
        log(f"❌ Error placing TP1: {e}", level="ERROR")
        return None

def calculate_actual_risk_percentage(entry_price, sl_price, qty, account_balance):
    """Calculate actual risk percentage"""
    try:
        risk_amount = abs(entry_price - sl_price) * qty
        risk_percentage = (risk_amount / account_balance) * 100
        return risk_percentage
    except Exception as e:
        log(f"❌ Error calculating risk: {e}", level="ERROR")
        return 0.0

# REMOVED FUNCTIONS (handled by main.py):
# - All strategy logic
# - Signal generation
# - Risk calculations
# - SL/TP calculations
# - Position sizing
# - Market analysis
# - Trailing stop placement (monitor handles this)

# ONLY HANDLES:
# - Market order execution
# - SL order placement
# - TP1 order placement
# - Monitor registration
# - Execution logging



