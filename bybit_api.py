# bybit_api.py - Complete Bybit API implementation with all required functions

import asyncio
import time
import traceback
import json
import hmac
import hashlib
import httpx
import os
from logger import log

# Import the global balance manager
try:
    from balance_manager import get_cached_balance
    BALANCE_MANAGER_AVAILABLE = True
    log("✅ Using optimized balance manager")
except ImportError:
    BALANCE_MANAGER_AVAILABLE = False
    log("⚠️ Balance manager not available, using legacy balance caching")

# Get API credentials from environment
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_API_URL = "https://api.bybit.com"

# Legacy balance cache (fallback if balance_manager is not available)
_legacy_balance_cache = {
    "balance": None,
    "timestamp": 0,
    "ttl": 60  # 60 seconds TTL
}

def create_signature(api_secret, sign_payload):
    """Create signature for API requests"""
    return hmac.new(
        api_secret.encode("utf-8"),
        sign_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

async def signed_request(method, endpoint, params, suppress_balance_logs=False):
    """
    Make signed request to Bybit API with optional log suppression for balance calls
    """
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        if method.upper() == "GET":
            query_string = "&".join(f"{k}={v}" for k, v in params.items()) if params else ""
            sign_payload = f"{timestamp}{BYBIT_API_KEY}{recv_window}{query_string}"
            full_url = f"{BYBIT_API_URL}{endpoint}?{query_string}" if query_string else f"{BYBIT_API_URL}{endpoint}"
            body = None
        else:
            body = json.dumps(params, separators=(",", ":")) if params else "{}"
            sign_payload = f"{timestamp}{BYBIT_API_KEY}{recv_window}{body}"
            full_url = f"{BYBIT_API_URL}{endpoint}"

        signature = create_signature(BYBIT_API_SECRET, sign_payload)

        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

        # Suppress verbose logging for balance calls
        is_balance_call = "/v5/account/wallet-balance" in endpoint
        
        if not (is_balance_call and suppress_balance_logs):
            log(f"🔗 {method} {full_url}")
            if params:
                safe_params = {k: v for k, v in params.items() if k not in ['apiKey', 'secret']}
                log(f"📦 Params: {json.dumps(safe_params)}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            if method.upper() == "GET":
                response = await client.get(full_url, headers=headers)
            elif method.upper() == "POST":
                response = await client.post(full_url, headers=headers, data=body)
            else:
                raise ValueError("Unsupported HTTP method")

        result = response.json()
        
        # Suppress response logging for balance calls to reduce spam
        if not (is_balance_call and suppress_balance_logs):
            log(f"📨 Response: {result}")
        elif is_balance_call:
            # Just log success/failure for balance calls
            if result.get("retCode") == 0:
                log(f"✅ Balance API call successful")
            else:
                log(f"❌ Balance API call failed: {result.get('retMsg')}")
        
        return result
        
    except Exception as e:
        log(f"❌ API Request Error: {str(e)}", level="ERROR")
        return {"retCode": -1, "retMsg": f"API Request Error: {str(e)}"}

async def get_futures_available_balance(force_refresh=False, caller_name="unknown"):
    """
    Get futures available balance with intelligent caching
    
    Args:
        force_refresh (bool): Force fresh API call
        caller_name (str): Identifier for the calling module
    
    Returns:
        float: Available balance in USDT
    """
    if BALANCE_MANAGER_AVAILABLE:
        # Use the optimized global balance manager
        return await get_cached_balance(force_refresh, caller_name)
    else:
        # Fallback to legacy caching
        return await _get_balance_legacy(force_refresh, caller_name)

async def _get_balance_legacy(force_refresh=False, caller_name="unknown"):
    """Legacy balance fetching with basic caching"""
    global _legacy_balance_cache
    
    current_time = time.time()
    cache = _legacy_balance_cache
    
    # Use cache if valid and not forcing refresh
    if (not force_refresh and 
        cache["balance"] is not None and 
        current_time - cache["timestamp"] < cache["ttl"]):
        log(f"💰 Using cached balance: {cache['balance']} USDT (caller: {caller_name})")
        return cache["balance"]
    
    try:
        log(f"🔄 Fetching fresh balance (caller: {caller_name})")
        
        result = await signed_request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED"
        }, suppress_balance_logs=True)
        
        if result.get("retCode") == 0:
            accounts = result.get("result", {}).get("list", [])
            for account in accounts:
                coins = account.get("coin", [])
                for coin in coins:
                    if coin.get("coin") == "USDT":
                        balance = float(coin.get("walletBalance", 0))
                        # Update cache
                        cache["balance"] = balance
                        cache["timestamp"] = current_time
                        log(f"💰 Fresh balance retrieved: {balance} USDT (caller: {caller_name})")
                        return balance
            
            log(f"❌ No USDT found in account")
            return cache["balance"] if cache["balance"] is not None else 0.0
            
        log(f"❌ Failed to get balance: {result.get('retMsg')}")
        return cache["balance"] if cache["balance"] is not None else 0.0
        
    except Exception as e:
        log(f"❌ Failed to fetch balance: {e}", level="ERROR")
        return cache["balance"] if cache["balance"] is not None else 0.0

async def place_market_order(symbol, side, qty, market_type="linear", reduce_only=False):
    """
    Place a market order
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        side: "Buy" or "Sell"
        qty: Quantity as string
        market_type: Market type ("linear" for futures)
    
    Returns:
        dict: API response
    """
    try:
        params = {
            "category": market_type,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC" # Immediate or Cancel for market orders
        }

        if reduce_only:
            params["reduceOnly"] = True
        
        log(f"🚀 Placing market order: {side} {qty} {symbol}")
        
        result = await signed_request("POST", "/v5/order/create", params)
        
        if result.get("retCode") == 0:
            log(f"✅ Market order placed successfully")
        else:
            log(f"❌ Market order failed: {result.get('retMsg')}", level="ERROR")
        
        return result
        
    except Exception as e:
        log(f"❌ Error placing market order: {e}", level="ERROR")
        return {"retCode": -1, "retMsg": f"Error placing market order: {str(e)}"}

async def cleanup_orphaned_stop_orders(symbol=None):
    """
    Clean up orphaned stop orders to prevent hitting the 10 order limit
    """
    try:
        # Get all stop orders
        params = {"category": "linear", "orderFilter": "Stop"}
        if symbol:
            params["symbol"] = symbol
            
        orders_resp = await signed_request("GET", "/v5/order/realtime", params)
        
        if orders_resp.get("retCode") == 0:
            orders = orders_resp.get("result", {}).get("list", [])
            log(f"🧹 Found {len(orders)} stop orders{'for ' + symbol if symbol else ''}")
            
            cancelled_count = 0
            for order in orders:
                order_symbol = order.get("symbol")
                order_id = order.get("orderId")
                
                try:
                    # Cancel the order
                    cancel_resp = await signed_request("POST", "/v5/order/cancel", {
                        "category": "linear",
                        "symbol": order_symbol,
                        "orderId": order_id
                    })
                    
                    if cancel_resp.get("retCode") == 0:
                        cancelled_count += 1
                        log(f"✅ Cancelled orphaned stop order: {order_id} for {order_symbol}")
                    else:
                        log(f"⚠️ Failed to cancel order {order_id}: {cancel_resp.get('retMsg')}")
                        
                    # Brief delay to avoid rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    log(f"❌ Error cancelling order {order_id}: {e}")
                    
            log(f"🧹 Cleanup complete: Cancelled {cancelled_count} orphaned stop orders")
            return cancelled_count
            
    except Exception as e:
        log(f"❌ Error in stop order cleanup: {e}", level="ERROR")
        return 0

async def cleanup_old_orders():
    """Clean up old/orphaned orders across all symbols"""
    try:
        # Get all open orders
        orders_resp = await signed_request("GET", "/v5/order/realtime", {
            "category": "linear",
            "orderFilter": "Stop"
        })
        
        if orders_resp.get("retCode") == 0:
            orders = orders_resp.get("result", {}).get("list", [])
            log(f"🧹 Found {len(orders)} stop orders to clean up")
            
            for order in orders:
                symbol = order.get("symbol")
                order_id = order.get("orderId")
                
                # Cancel orphaned orders
                await signed_request("POST", "/v5/order/cancel", {
                    "category": "linear",
                    "symbol": symbol,
                    "orderId": order_id
                })
                await asyncio.sleep(0.1)  # Rate limit protection
                
    except Exception as e:
        log(f"❌ Error in cleanup: {e}")

# === STOP LOSS FUNCTIONS ===
async def place_stop_loss(symbol, direction, qty, sl_price, market_type="linear"):
    """
    Enhanced stop loss placement with automatic cleanup and corrected trigger direction
    """
    # STEP 1: Clean up any existing stop orders for this symbol first
    log(f"🧹 Cleaning up existing stop orders for {symbol} before placing new SL")
    await cleanup_orphaned_stop_orders(symbol)
    
    # Brief pause after cleanup
    await asyncio.sleep(1)
    
    side = "Sell" if direction.lower() == "long" else "Buy"
    
    # STEP 2: CORRECTED trigger direction logic
    # For long positions: SL triggers when price FALLS below SL (triggerDirection = 2)
    # For short positions: SL triggers when price RISES above SL (triggerDirection = 1)
    trigger_direction = 2 if direction.lower() == "long" else 1
    
    # STEP 3: Validate the SL price is on correct side of market
    try:
        ticker_resp = await signed_request("GET", "/v5/market/tickers", {"category": market_type, "symbol": symbol})
        mark_price = float(ticker_resp.get("result", {}).get("list", [{}])[0].get("markPrice", 0))
        
        # Add safety buffer for SL placement
        buffer = 0.005  # 0.5% safety buffer
        
        if direction.lower() == "long" and sl_price >= mark_price:
            old_sl = sl_price
            sl_price = round(mark_price * (1 - buffer), 6)
            log(f"⚠️ Adjusted long SL from {old_sl} to {sl_price} (below mark price {mark_price})", level="WARN")
        elif direction.lower() == "short" and sl_price <= mark_price:
            old_sl = sl_price
            sl_price = round(mark_price * (1 + buffer), 6)
            log(f"⚠️ Adjusted short SL from {old_sl} to {sl_price} (above mark price {mark_price})", level="WARN")
            
    except Exception as e:
        log(f"❌ Failed to validate SL price: {e}", level="ERROR")
    
    # STEP 4: Construct the stop loss order payload
    sl_payload = {
        "category": market_type,
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "triggerPrice": str(sl_price),
        "triggerDirection": trigger_direction,  # FIXED: Correct trigger direction
        "triggerBy": "MarkPrice",
        "qty": str(qty),
        "reduceOnly": True,
        "timeInForce": "GTC",
        "orderFilter": "Stop"
    }
    
    log(f"🛡️ Placing SL order for {symbol}: {direction} | TriggerDir: {trigger_direction} | Price: {sl_price}")
    
    # STEP 5: Place the stop loss order
    result = await signed_request("POST", "/v5/order/create", sl_payload)
    
    if result.get("retCode") != 0:
        error_msg = result.get("retMsg", "Unknown error")
        log(f"❌ Failed to place SL order: {error_msg}", level="ERROR")
        
        # If still failing due to too many orders, do a broader cleanup
        if "10 working" in error_msg.lower():
            log("🧹 Performing broader stop order cleanup due to limit hit")
            await cleanup_orphaned_stop_orders()  # Clean all symbols
            await asyncio.sleep(2)
            
            # Retry after cleanup
            log("🔄 Retrying SL placement after cleanup")
            result = await signed_request("POST", "/v5/order/create", sl_payload)
            
        # If trigger direction error, try with LastPrice instead of MarkPrice
        elif "triggerdirection" in error_msg.lower():
            log("🔄 Retrying SL with LastPrice trigger")
            sl_payload["triggerBy"] = "LastPrice"
            result = await signed_request("POST", "/v5/order/create", sl_payload)
    
    return result

# --- STOP LOSS WITH RETRY FUNCTION ---
async def place_stop_loss_with_retry(symbol, direction, qty, sl_price, market_type="linear", max_attempts=3):
    """Enhanced stop loss placement with exponential backoff retries"""
    attempt = 0
    delay = 1  # Start with 1 second delay
    
    while attempt < max_attempts:
        try:
            result = await place_stop_loss(symbol, direction, qty, sl_price, market_type)
            
            if result.get("retCode") == 0:
                log(f"✅ SL order placed successfully for {symbol} on attempt {attempt+1}")
                return result
                
            # Handle specific error codes that might be temporary
            if result.get("retCode") in [10002, 10006, 10010]:  # Rate limit or temporary server issues
                log(f"⚠️ Temporary error placing SL for {symbol}: {result.get('retMsg')}", level="WARN")
                await asyncio.sleep(delay)
                attempt += 1
                delay *= 2  # Exponential backoff
                continue
            else:
                # For permanent errors, try a different approach
                return await fallback_stop_loss(symbol, direction, qty, sl_price, market_type)
                
        except Exception as e:
            log(f"❌ Exception in SL placement for {symbol}: {e}", level="ERROR")
            await asyncio.sleep(delay)
            attempt += 1
            delay *= 2
            
    # If we get here, all attempts failed
    from error_handler import send_telegram_message
    await send_telegram_message(f"⚠️ <b>Critical SL Failure</b> for {symbol} after {max_attempts} attempts")
    return {"retCode": -1, "retMsg": f"Failed after {max_attempts} attempts"}

# --- FALLBACK STOP LOSS FUNCTION ---
async def fallback_stop_loss(symbol, direction, qty, sl_price, market_type="linear"):
    """Alternative approach for placing stop loss when standard method fails"""
    
    # Try a conditional order approach as fallback
    side = "Sell" if direction.lower() == "long" else "Buy"
    
    # CRITICAL FIX: Correctly set trigger direction
    trigger_direction = 2 if direction.lower() == "long" else 1
    
    # First, try a StopLimit order 
    fallback_payload = {
        "category": market_type,
        "symbol": symbol,
        "side": side,
        "orderType": "Limit",  # Try limit instead of market
        "price": str(sl_price),  # Add limit price
        "triggerPrice": str(sl_price),
        "triggerDirection": trigger_direction,  # FIXED: Use correct trigger direction
        "triggerBy": "LastPrice",  # Try LastPrice as another alternative
        "qty": str(qty),
        "reduceOnly": True,
        "timeInForce": "GTC",
        "orderFilter": "StopLimit"  # Change to StopLimit order
    }
    
    log(f"🔄 Using fallback StopLimit SL for {symbol}: {fallback_payload}")
    result = await signed_request("POST", "/v5/order/create", fallback_payload)
    
    # If StopLimit also fails, try a conditional TP order as SL (last resort)
    if result.get("retCode") != 0:
        log(f"❌ StopLimit fallback failed: {result.get('retMsg')}", level="ERROR")
        
        # Get current price
        try:
            ticker_resp = await signed_request("GET", "/v5/market/tickers", {"category": market_type, "symbol": symbol})
            mark_price = float(ticker_resp.get("result", {}).get("list", [{}])[0].get("markPrice", 0))
            
            # For long positions: if SL < mark, use TakeProfit
            # For short positions: if SL > mark, use TakeProfit
            if (direction.lower() == "long" and sl_price < mark_price) or (direction.lower() == "short" and sl_price > mark_price):
                last_resort_payload = {
                    "category": market_type,
                    "symbol": symbol,
                    "side": side,
                    "orderType": "Market",
                    "triggerPrice": str(sl_price),
                    "qty": str(qty),
                    "reduceOnly": True,
                    "timeInForce": "GTC",
                    "orderFilter": "tpslOrder",
                    "orderIv": "0",
                    "tpslMode": "Partial",
                    "tpOrderType": "Market",
                    "slOrderType": "Market",
                    "tpTriggerBy": "LastPrice",
                    "slTriggerBy": "LastPrice"
                }
                
                if direction.lower() == "long":
                    last_resort_payload["takeProfit"] = str(sl_price)
                else:
                    last_resort_payload["stopLoss"] = str(sl_price)
                
                log(f"🆘 Last resort TP/SL approach for {symbol}: {last_resort_payload}")
                result = await signed_request("POST", "/v5/order/create", last_resort_payload)
        except Exception as e:
            log(f"❌ Error in last resort SL approach: {e}", level="ERROR")
    
    return result

async def check_order_exists(order_id, symbol, market_type="linear"):
    """
    Check if an order exists
    
    Args:
        order_id: Order ID to check
        symbol: Trading symbol
        market_type: Market type
    
    Returns:
        bool: True if order exists, False otherwise
    """
    try:
        result = await signed_request("GET", "/v5/order/realtime", {
            "category": market_type,
            "symbol": symbol,
            "orderId": order_id
        })
        
        if result.get("retCode") == 0:
            orders = result.get("result", {}).get("list", [])
            return len(orders) > 0
        
        return False
        
    except Exception as e:
        log(f"❌ Error checking order existence: {e}", level="ERROR")
        return False

async def update_stop_loss_order(symbol, order_id, new_sl_price, market_type="linear"):
    """
    Update an existing stop loss order with a new price
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        order_id: The order ID of the existing stop loss order
        new_sl_price: New stop loss price
        market_type: Market type ("linear" for futures)
    
    Returns:
        dict: API response
    """
    try:
        log(f"🔄 Updating SL order {order_id} for {symbol} to {new_sl_price}")
        
        params = {
            "category": market_type,
            "symbol": symbol,
            "orderId": order_id,
            "triggerPrice": str(new_sl_price)
        }
        
        result = await signed_request("POST", "/v5/order/amend", params)
        
        if result.get("retCode") == 0:
            log(f"✅ Stop loss updated successfully for {symbol}")
        else:
            error_msg = result.get("retMsg", "Unknown error")
            log(f"❌ Failed to update stop loss: {error_msg}", level="ERROR")
            
            # If update fails, try to cancel and replace
            log("🔄 Attempting to cancel and replace stop loss order")
            
            # First cancel the existing order
            cancel_result = await signed_request("POST", "/v5/order/cancel", {
                "category": market_type,
                "symbol": symbol,
                "orderId": order_id
            })
            
            if cancel_result.get("retCode") == 0:
                log("✅ Old stop loss cancelled, placing new one")
                
                # Need to determine direction and quantity from the original order
                # Get order details first
                order_info_result = await signed_request("GET", "/v5/order/history", {
                    "category": market_type,
                    "symbol": symbol,
                    "orderId": order_id
                })
                
                if order_info_result.get("retCode") == 0:
                    orders = order_info_result.get("result", {}).get("list", [])
                    if orders:
                        original_order = orders[0]
                        direction = "Long" if original_order.get("side") == "Sell" else "Short"
                        qty = original_order.get("qty")
                        
                        # Place new stop loss
                        new_sl_result = await place_stop_loss_with_retry(
                            symbol, direction, qty, new_sl_price, market_type
                        )
                        return new_sl_result
        
        return result
        
    except Exception as e:
        log(f"❌ Error updating stop loss order: {e}", level="ERROR")
        return {"retCode": -1, "retMsg": f"Error updating stop loss: {str(e)}"}

# === POSITIONS ===
async def get_positions(symbol: str | None = None, market_type: str = "linear"):
    """
    Fetch open positions from Bybit V5.

    Args:
        symbol: Optional symbol filter (e.g., "BTCUSDT")
        market_type: "linear" for USDT perpetuals

    Returns:
        List[dict]: [{"symbol": str, "side": "Buy"/"Sell", "size": float, "entryPrice": float, ...}, ...]
    """
    try:
        params = {"category": market_type}
        if symbol:
            params["symbol"] = symbol

        resp = await signed_request("GET", "/v5/position/list", params)

        if resp.get("retCode") == 0:
            items = (resp.get("result", {}) or {}).get("list", []) or []
            positions = []
            for p in items:
                # size field name can vary; fall back safely
                size = float(p.get("size") or p.get("qty") or 0)
                positions.append({
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),  # "Buy" or "Sell"
                    "size": size,
                    "entryPrice": float(p.get("avgPrice") or p.get("entryPrice") or 0),
                    "leverage": p.get("leverage"),
                    "positionValue": float(p.get("positionValue") or 0),
                })
            return positions

        log(f"❌ Failed to fetch positions: {resp.get('retMsg')}", level="ERROR")
        return []

    except Exception as e:
        log(f"❌ Error fetching positions: {e}", level="ERROR")
        return []


async def get_wallet_balance(force_refresh=False):
    """Get wallet balance - redirects to optimized version"""
    return await get_futures_available_balance(force_refresh, "get_wallet_balance")

async def signed_request_with_balance_optimization(method, endpoint, params):
    """Wrapper that automatically suppresses logs for balance calls"""
    suppress_logs = "/v5/account/wallet-balance" in endpoint
    return await signed_request(method, endpoint, params, suppress_logs)

