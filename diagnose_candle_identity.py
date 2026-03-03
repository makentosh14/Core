#!/usr/bin/env python3
"""
diagnose_candle_identity.py
Run this while the bot is live to find the shared-reference bug.
"""
import asyncio

async def diagnose():
    from websocket_candles import live_candles

    print("=== CANDLE IDENTITY CHECK ===\n")

    for symbol in list(live_candles.keys())[:5]:
        sym_data = live_candles[symbol]
        tfs = list(sym_data.keys())
        print(f"Symbol: {symbol}  |  TFs present: {tfs}")

        for tf in tfs:
            obj = sym_data[tf]
            print(f"  [{tf}] id={id(obj)}  len={len(obj)}  type={type(obj).__name__}")

        # Check if any two TFs share the SAME object in memory
        tf_ids = {tf: id(sym_data[tf]) for tf in tfs}
        for a in tfs:
            for b in tfs:
                if a < b and tf_ids[a] == tf_ids[b]:
                    print(f"  🚨 BUG CONFIRMED: TF '{a}' and TF '{b}' share the SAME object (id={tf_ids[a]})")
                else:
                    if a < b:
                        print(f"  ✅ TF '{a}' and TF '{b}' are separate objects")

        # Check first and last candle timestamps per TF
        for tf in tfs:
            candles = list(sym_data[tf])
            if candles:
                first_ts = candles[0].get('timestamp', candles[0].get('ts', '?'))
                last_ts  = candles[-1].get('timestamp', candles[-1].get('ts', '?'))
                print(f"  [{tf}] first_ts={first_ts}  last_ts={last_ts}")
        print()

asyncio.run(diagnose())
