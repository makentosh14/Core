# balance_manager.py - Global balance caching system to reduce API calls

import asyncio
import time
import traceback
from datetime import datetime
from logger import log

# Global balance cache - shared across all modules
_global_balance_cache = {
    "balance": None,
    "timestamp": 0,
    "ttl": 60,  # 60 seconds TTL (instead of 30)
    "fetch_lock": asyncio.Lock(),  # Prevent concurrent API calls
    "last_log_time": 0,  # Track when we last logged cache usage
    "cache_hits": 0,
    "api_calls": 0
}

# Extended cache for different balance requirements
_balance_cache_extended = {
    "last_fetch_time": 0,
    "min_fetch_interval": 45,  # Minimum 45 seconds between API calls
    "force_refresh_interval": 300,  # Force refresh every 5 minutes
    "consecutive_cache_hits": 0,
    "max_consecutive_hits": 10  # Force refresh after 10 consecutive cache hits
}

async def get_cached_balance(force_refresh=False, caller_name="unknown"):
    """
    Get cached balance with smart refresh logic
    
    Args:
        force_refresh (bool): Force API call regardless of cache
        caller_name (str): Name of the calling module for logging
    
    Returns:
        float: Account balance in USDT
    """
    global _global_balance_cache, _balance_cache_extended
    
    current_time = time.time()
    cache = _global_balance_cache
    extended = _balance_cache_extended
    
    # Check if we can use cached value
    cache_age = current_time - cache["timestamp"]
    time_since_last_fetch = current_time - extended["last_fetch_time"]
    
    use_cache = (
        not force_refresh and
        cache["balance"] is not None and
        cache_age < cache["ttl"] and
        time_since_last_fetch >= extended["min_fetch_interval"] and
        extended["consecutive_cache_hits"] < extended["max_consecutive_hits"]
    )
    
    if use_cache:
        cache["cache_hits"] += 1
        extended["consecutive_cache_hits"] += 1
        
        # Log cache usage periodically (every 30 seconds max)
        if current_time - cache["last_log_time"] > 30:
            hit_ratio = cache["cache_hits"] / max(cache["api_calls"] + cache["cache_hits"], 1) * 100
            log(f"💰 Using cached balance: {cache['balance']} USDT (age: {cache_age:.0f}s, caller: {caller_name}, hit ratio: {hit_ratio:.1f}%)")
            cache["last_log_time"] = current_time
        
        return cache["balance"]
    
    # Use lock to prevent concurrent API calls
    async with cache["fetch_lock"]:
        # Double-check cache after acquiring lock (another call might have refreshed it)
        cache_age = current_time - cache["timestamp"]
        if (not force_refresh and 
            cache["balance"] is not None and 
            cache_age < cache["ttl"] and
            time_since_last_fetch < extended["min_fetch_interval"]):
            cache["cache_hits"] += 1
            return cache["balance"]
        
        # Force refresh after too many consecutive cache hits
        force_api_call = (
            force_refresh or
            cache["balance"] is None or
            cache_age >= cache["ttl"] or
            time_since_last_fetch >= extended["force_refresh_interval"] or
            extended["consecutive_cache_hits"] >= extended["max_consecutive_hits"]
        )
        
        if force_api_call:
            try:
                # Import here to avoid circular imports
                from bybit_api import signed_request
                
                log(f"🔄 Fetching fresh balance (caller: {caller_name}, reason: {'force' if force_refresh else 'expired'})")
                
                response = await signed_request("GET", "/v5/account/wallet-balance", {
                    "accountType": "UNIFIED"
                })
                
                cache["api_calls"] += 1
                extended["last_fetch_time"] = current_time
                extended["consecutive_cache_hits"] = 0  # Reset consecutive hits
                
                if response.get("retCode") != 0:
                    log(f"❌ Failed to fetch balance: {response.get('retMsg')}", level="ERROR")
                    # Return cached value if available, otherwise 0
                    return cache["balance"] if cache["balance"] is not None else 0.0
                
                # Parse balance from response
                balance = await _parse_balance_response(response)
                
                if balance > 0:
                    cache["balance"] = balance
                    cache["timestamp"] = current_time
                    log(f"💰 Fresh balance fetched: {balance} USDT (caller: {caller_name})")
                    return balance
                else:
                    log(f"⚠️ Invalid balance returned: {balance}", level="WARN")
                    # Return cached value if available
                    return cache["balance"] if cache["balance"] is not None else 0.0
                    
            except Exception as e:
                log(f"❌ Error fetching balance: {e}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")
                
                # Return cached value as fallback
                if cache["balance"] is not None:
                    log(f"💰 Using cached balance as fallback: {cache['balance']} USDT")
                    return cache["balance"]
                
                return 0.0
        
        # Should not reach here, but return cached value as safety
        return cache["balance"] if cache["balance"] is not None else 0.0

async def _parse_balance_response(response):
    """Parse balance from Bybit API response"""
    try:
        if "list" in response["result"] and len(response["result"]["list"]) > 0:
            account_info = response["result"]["list"][0]
            
            # Try different balance fields in order of preference
            balance_fields = [
                "totalAvailableBalance",  # Best for trading
                "totalMarginBalance", 
                "totalWalletBalance"
            ]
            
            for field in balance_fields:
                if field in account_info:
                    balance = float(account_info[field])
                    if balance > 0:
                        log(f"💰 Using {field}: {balance} USDT")
                        return balance
            
            log(f"❌ No valid balance field found in response")
            return 0.0
        
        log("❌ No account list found in balance response")
        return 0.0
        
    except Exception as e:
        log(f"❌ Error parsing balance response: {e}", level="ERROR")
        return 0.0

def get_balance_cache_stats():
    """Get statistics about balance cache usage"""
    cache = _global_balance_cache
    extended = _balance_cache_extended
    current_time = time.time()
    
    total_requests = cache["api_calls"] + cache["cache_hits"]
    hit_ratio = cache["cache_hits"] / max(total_requests, 1) * 100
    cache_age = current_time - cache["timestamp"]
    
    return {
        "cached_balance": cache["balance"],
        "cache_age_seconds": cache_age,
        "total_requests": total_requests,
        "api_calls": cache["api_calls"],
        "cache_hits": cache["cache_hits"],
        "hit_ratio_percent": hit_ratio,
        "consecutive_cache_hits": extended["consecutive_cache_hits"],
        "last_fetch_time": extended["last_fetch_time"]
    }

def reset_balance_cache():
    """Reset balance cache (useful for testing or after errors)"""
    global _global_balance_cache, _balance_cache_extended
    
    _global_balance_cache.update({
        "balance": None,
        "timestamp": 0,
        "cache_hits": 0,
        "api_calls": 0,
        "last_log_time": 0
    })
    
    _balance_cache_extended.update({
        "last_fetch_time": 0,
        "consecutive_cache_hits": 0
    })
    
    log("🔄 Balance cache reset")

# Convenience function for backward compatibility
async def get_futures_available_balance_cached(force_refresh=False, caller_name="legacy"):
    """Backward compatible wrapper for existing code"""
    return await get_cached_balance(force_refresh, caller_name)
