import asyncio
from bybit_api import signed_request

async def main():
    symbol = "4USDT"

    resp = await signed_request(
        "GET",
        "/v5/market/instruments-info",
        {"category": "linear", "symbol": symbol}
    )

    print("Instrument info response:")
    print(resp)

asyncio.run(main())
