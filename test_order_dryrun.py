import asyncio
from bybit_api import signed_request

async def main():
    symbol = "4USDT"

    # 1) get mark price
    tick = await signed_request("GET", "/v5/market/tickers", {"category":"linear", "symbol":symbol})
    mp = float(tick["result"]["list"][0]["markPrice"])
    print("markPrice:", mp)

    # 2) get filters
    inst = await signed_request("GET", "/v5/market/instruments-info", {"category":"linear", "symbol":symbol})
    f = inst["result"]["list"][0]["lotSizeFilter"]
    min_qty = float(f["minOrderQty"])
    step = float(f["qtyStep"])
    min_notional = float(f["minNotionalValue"])
    print("minQty:", min_qty, "qtyStep:", step, "minNotional:", min_notional)

    # 3) compute the minimum qty that meets minNotional
    # qty must be integer because step=1
    min_qty_by_notional = int((min_notional / mp) + 0.999999)  # ceil
    qty = max(int(min_qty), min_qty_by_notional)
    notional = qty * mp
    print("computed qty:", qty, "computed notional:", notional)

asyncio.run(main())
