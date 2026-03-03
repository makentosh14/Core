#!/usr/bin/env python3
"""
diagnose_rsi_bug.py
===================
Fetches candles directly via REST for 5m and 15m, then:
1. Prints first 3 and last 3 timestamps for each TF
2. Calculates RSI manually for each TF
3. Checks if live_candles has data and whether 5m == 15m

Run: python diagnose_rsi_bug.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Helpers ────────────────────────────────────────────────────────────────

def calc_rsi(candles, period=14):
    """Simple Wilder RSI — returns last RSI value or None."""
    closes = []
    for c in candles:
        try:
            closes.append(float(c.get("close") or c.get("c") or 0))
        except Exception:
            pass
    if len(closes) < period + 1:
        return None, None, None

    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))

    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period

    rs = avg_g / avg_l if avg_l != 0 else float("inf")
    rsi = 100 - 100 / (1 + rs)
    return round(rsi, 4), round(avg_g, 6), round(avg_l, 6)


def get_ts(c):
    return c.get("timestamp") or c.get("ts") or c.get("t") or "?"


# ── Main ───────────────────────────────────────────────────────────────────

async def main():
    print("=" * 65)
    print("RSI DATA-PIPELINE DIAGNOSTIC")
    print("=" * 65)

    # ── Step 1: REST fetch for a known symbol ───────────────────────────
    TEST_SYMBOL = "BTCUSDT"
    TFS = ["5", "15"]

    try:
        from bybit_api import signed_request
    except ImportError:
        print("❌ Cannot import bybit_api. Run from your bot directory.")
        return

    rest_candles = {}
    print(f"\n[1] Fetching {TEST_SYMBOL} candles via REST …")
    for tf in TFS:
        resp = await signed_request("GET", "/v5/market/kline", {
            "category": "linear",
            "symbol": TEST_SYMBOL,
            "interval": tf,
            "limit": "50",
        })
        if resp.get("retCode") != 0:
            print(f"  [{tf}m] ❌ API error: {resp.get('retMsg')}")
            rest_candles[tf] = []
            continue

        raw = resp.get("result", {}).get("list", []) or []
        candles = []
        for row in reversed(raw):           # Bybit: newest-first → reverse
            candles.append({
                "timestamp": int(row[0]),
                "open":  str(row[1]),
                "high":  str(row[2]),
                "low":   str(row[3]),
                "close": str(row[4]),
                "volume": str(row[5]),
            })
        rest_candles[tf] = candles
        print(f"  [{tf}m] fetched {len(candles)} candles")

    print()
    for tf in TFS:
        candles = rest_candles[tf]
        if not candles:
            print(f"[{tf}m] NO DATA")
            continue

        rsi, avg_g, avg_l = calc_rsi(candles)
        first3 = [get_ts(c) for c in candles[:3]]
        last3  = [get_ts(c) for c in candles[-3:]]

        print(f"[{tf}m]  len={len(candles)}")
        print(f"        first timestamps : {first3}")
        print(f"        last  timestamps : {last3}")
        print(f"        RSI(14)={rsi}  avg_gain={avg_g}  avg_loss={avg_l}")

    # ── Step 2: Are 5m and 15m timestamp arrays identical? ─────────────
    print()
    c5  = rest_candles.get("5",  [])
    c15 = rest_candles.get("15", [])

    if c5 and c15:
        ts5  = [get_ts(c) for c in c5]
        ts15 = [get_ts(c) for c in c15]

        if ts5 == ts15:
            print("🚨 CONFIRMED BUG: 5m and 15m REST candles have IDENTICAL timestamps!")
            print("   This means the REST fetch is using the SAME interval for both TFs.")
        else:
            print("✅ REST candles: 5m and 15m have DIFFERENT timestamps (REST fetch is correct)")
            # Count how many timestamps overlap
            overlap = set(ts5) & set(ts15)
            print(f"   Overlapping timestamps: {len(overlap)} "
                  f"(expected ~1/3 of 5m count for aligned candles)")
    else:
        print("⚠️  Could not compare — one or both TFs have no data")

    # ── Step 3: Check live_candles in memory ────────────────────────────
    print()
    print("[3] Checking live_candles in memory …")
    try:
        from websocket_candles import live_candles
        total_syms = len(live_candles)
        print(f"  live_candles has {total_syms} symbol(s)")

        if total_syms == 0:
            print("  ⚠️  live_candles is empty — websocket not running in this process.")
            print("     That is NORMAL for a standalone script.")
            print("     The REST check above (Steps 1-2) is the authoritative test.")
        else:
            for sym in list(live_candles.keys())[:3]:
                print(f"\n  {sym}:")
                tfs_present = list(live_candles[sym].keys())
                print(f"    TFs present: {tfs_present}")

                for tf in ["5", "15"]:
                    if tf not in live_candles[sym]:
                        print(f"    [{tf}m] NOT in live_candles")
                        continue

                    obj = live_candles[sym][tf]
                    clist = list(obj)
                    rsi, avg_g, avg_l = calc_rsi(clist)
                    print(f"    [{tf}m] id={id(obj)}  len={len(clist)}")
                    if clist:
                        print(f"           first_ts={get_ts(clist[0])}  last_ts={get_ts(clist[-1])}")
                    print(f"           RSI(14)={rsi}  avg_gain={avg_g}  avg_loss={avg_l}")

                # Identity check
                if "5" in live_candles[sym] and "15" in live_candles[sym]:
                    id5  = id(live_candles[sym]["5"])
                    id15 = id(live_candles[sym]["15"])
                    if id5 == id15:
                        print(f"    🚨 BUG: 5m and 15m point to THE SAME deque object (id={id5})")
                    else:
                        print(f"    ✅ 5m and 15m are separate deque objects")

    except ImportError as e:
        print(f"  ❌ Cannot import websocket_candles: {e}")

    # ── Step 4: Check interval extraction from topic strings ────────────
    print()
    print("[4] Sanity-check topic string parsing …")
    test_topics = [
        "kline.1.BTCUSDT",
        "kline.5.BTCUSDT",
        "kline.15.BTCUSDT",
        "kline.60.BTCUSDT",
    ]
    for topic in test_topics:
        parts = topic.split(".")
        interval_extracted = parts[1] if len(parts) > 1 else "PARSE_ERROR"
        symbol_extracted   = parts[-1] if parts else "PARSE_ERROR"
        ok = "✅" if interval_extracted in ["1","5","15","60"] else "❌"
        print(f"  {ok}  topic='{topic}'  → interval='{interval_extracted}'  symbol='{symbol_extracted}'")

    print()
    print("=" * 65)
    print("DIAGNOSIS COMPLETE")
    print("=" * 65)
    print()
    print("What to look for:")
    print("  • Step 2 → If 5m and 15m have IDENTICAL timestamps from REST,")
    print("    the bug is in how fetch_candles_rest is called (wrong interval param).")
    print("  • Step 3 → If id(5m) == id(15m), it's a shared-deque Python bug.")
    print("  • Step 3 → If RSI values are identical but ids differ,")
    print("    the same data was WRITTEN to both TF keys.")
    print("  • Step 4 → All topic parses should show ✅")


if __name__ == "__main__":
    asyncio.run(main())
