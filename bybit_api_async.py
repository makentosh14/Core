import time
import hmac
import hashlib
import aiohttp
import asyncio

# --- CONFIG ---
API_KEY = "HjOc2BXYebIqLyI2ug"
API_SECRET = "6mHzcGIxmeiRcVpiglOPDwVk4fFPFDF4SBRR"
BASE_URL = "https://api.bybit.com"  # For live trading; use "https://api-testnet.bybit.com" for testnet
RECV_WINDOW = 5000
# Global session for reuse
_global_session = None

async def get_or_create_session():
    """Get or create a global aiohttp session"""
    global _global_session
    if _global_session is None or _global_session.closed:
        _global_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'TradingBot/1.0'}
        )
    return _global_session

async def close_global_session():
    """Close the global session"""
    global _global_session
    if _global_session and not _global_session.closed:
        await _global_session.close()
        _global_session = None

# --- SIGNATURE FUNCTION ---
def generate_signature(params, api_secret):
    ordered_params = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(api_secret.encode(), ordered_params.encode(), hashlib.sha256).hexdigest()

# --- ASYNC REQUEST FUNCTION ---
async def signed_request(method, endpoint, params=None):
    if params is None:
        params = {}

    params["api_key"] = API_KEY
    params["timestamp"] = str(int(time.time() * 1000))
    params["recvWindow"] = RECV_WINDOW

    # Create signature
    params["sign"] = generate_signature(params, API_SECRET)

    url = BASE_URL + endpoint
    if use_global_session:
        # Use global session (recommended for multiple requests)
        session = await get_or_create_session()
        if method == "GET":
            async with session.get(url, params=params) as response:
                return await response.json()
        elif method == "POST":
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            async with session.post(url, data=params, headers=headers) as response:
                return await response.json()
        else:
            raise ValueError("Unsupported HTTP method")
    else:
        # Use temporary session (for single requests)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            if method == "GET":
                async with session.get(url, params=params) as response:
                    return await response.json()
            elif method == "POST":
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                async with session.post(url, data=params, headers=headers) as response:
                    return await response.json()
            else:
                raise ValueError("Unsupported HTTP method")

# --- EXAMPLE USAGE ---
class BybitAPIClient:
    """Context manager for Bybit API operations"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = await get_or_create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Don't close global session here, let it be managed globally
        pass
    
    async def request(self, method, endpoint, params=None):
        """Make request using this client's session"""
        return await signed_request(method, endpoint, params, use_global_session=True)

# --- EXAMPLE USAGE ---
async def main():
    try:
        # Example 1: Get Wallet Balance (v5 API)
        response = await signed_request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED",
             "settleCoin": "USDT"
        })
        print(response)

        # Example 2: Using context manager
        async with BybitAPIClient() as client:
            response = await client.request("GET", "/v5/account/wallet-balance", {
                "accountType": "UNIFIED",
                 "settleCoin": "USDT"
            })
            print("Via client:", response)
            
    finally:
        # Always close the global session when done
        await close_global_session()

if __name__ == "__main__":
    asyncio.run(main())
