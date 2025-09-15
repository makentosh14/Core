import time
import hmac
import hashlib
import aiohttp
import asyncio

# --- CONFIG ---
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
BASE_URL = "https://api.bybit.com"  # For live trading; use "https://api-testnet.bybit.com" for testnet
RECV_WINDOW = 5000

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
    async with aiohttp.ClientSession() as session:
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
async def main():
    # Example 1: Get Wallet Balance (v5 API)
    response = await signed_request("GET", "/v5/account/wallet-balance", {
        "accountType": "UNIFIED"
    })
    print(response)

    # Example 2: Place a Test Market Order (Optional)
    # response = await signed_request("POST", "/v5/order/create", {
    #     "category": "linear",
    #     "symbol": "BTCUSDT",
    #     "side": "Buy",
    #     "orderType": "Market",
    #     "qty": 0.001,
    #     "timeInForce": "GTC"
    # })
    # print(response)

if __name__ == "__main__":
    asyncio.run(main())
