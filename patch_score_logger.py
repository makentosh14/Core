#!/usr/bin/env python3
"""
patch_score_logger.py
=====================
Monkey-patches score_symbol to print candle fingerprints for 5m vs 15m
EVERY TIME it is called inside the running bot.

Run INSTEAD of your normal bot start:
    python patch_score_logger.py

It imports and starts main.py exactly as normal, but intercepts
every score_symbol call to check whether 5m and 15m data are identical.

Stop with Ctrl+C when you see enough log output.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Patch must happen BEFORE main.py is imported ───────────────────────────
import score as _score_module

_original_score_symbol = _score_module.score_symbol

_call_count = 0

def _patched_score_symbol(symbol, candles_by_timeframe, market_context=None):
    global _call_count
    _call_count += 1

    # ── Fingerprint each TF ─────────────────────────────────────────────────
    fingerprints = {}
    lengths = {}
    first_ts = {}
    last_ts = {}

    for tf in ["1", "5", "15"]:
        candles = candles_by_timeframe.get(tf)
        if not candles:
            fingerprints[tf] = None
            lengths[tf] = 0
            continue
        try:
            clist = list(candles)
        except Exception:
            clist = []

        lengths[tf] = len(clist)

        def _get_ts(c):
            return c.get("timestamp") or c.get("ts") or c.get("t") or 0

        def _get_close(c):
            try:
                return float(c.get("close") or c.get("c") or 0)
            except Exception:
                return 0.0

        first_ts[tf] = _get_ts(clist[0])  if clist else 0
        last_ts[tf]  = _get_ts(clist[-1]) if clist else 0

        # Fingerprint = tuple of first 3 and last 3 close prices
        closes = [_get_close(c) for c in clist]
        head = tuple(closes[:3])
        tail = tuple(closes[-3:])
        fingerprints[tf] = (head, tail)

    # ── Compare 5m vs 15m ───────────────────────────────────────────────────
    fp5  = fingerprints.get("5")
    fp15 = fingerprints.get("15")

    identical = (fp5 is not None and fp15 is not None and fp5 == fp15)

    # Always print a one-liner; print WARNING block if identical
    tag = "🚨 IDENTICAL" if identical else "✅ OK"
    print(
        f"[SCORE#{_call_count:04d}] {tag} | {symbol} | "
        f"5m: len={lengths.get('5',0)} first_ts={first_ts.get('5',0)} last_ts={last_ts.get('5',0)} | "
        f"15m: len={lengths.get('15',0)} first_ts={first_ts.get('15',0)} last_ts={last_ts.get('15',0)}",
        flush=True,
    )

    if identical:
        print(f"  ⚠️  5m  fingerprint : {fp5}")
        print(f"  ⚠️  15m fingerprint : {fp15}")
        print(f"  ⚠️  5m  id in candles_by_timeframe : {id(candles_by_timeframe.get('5'))}")
        print(f"  ⚠️  15m id in candles_by_timeframe : {id(candles_by_timeframe.get('15'))}")
        # Also check object identity — are they literally the same list?
        obj5  = candles_by_timeframe.get("5")
        obj15 = candles_by_timeframe.get("15")
        if obj5 is obj15:
            print(f"  🔴 SAME OBJECT: 5m and 15m point to the SAME list/deque in memory!")
        else:
            print(f"  🟡 Different objects but SAME DATA — bug is in data writing, not reference.")
        print(flush=True)

    # Call the real function
    return _original_score_symbol(symbol, candles_by_timeframe, market_context)


# Patch it in both places it might be referenced
_score_module.score_symbol = _patched_score_symbol

# Also patch it in main's namespace if main has already imported it
try:
    import main as _main_module
    if hasattr(_main_module, "score_symbol"):
        _main_module.score_symbol = _patched_score_symbol
        print("[PATCH] Patched score_symbol in main module namespace", flush=True)
except Exception:
    pass  # main not imported yet, will pick up via score module

print("[PATCH] score_symbol is now instrumented — starting bot normally...\n", flush=True)

# ── Now start the bot exactly as main.py would ──────────────────────────────
# Import main and run its entry point
import main

if __name__ == "__main__":
    # Re-patch in main's namespace after import (in case main imported score_symbol directly)
    if hasattr(main, "score_symbol"):
        main.score_symbol = _patched_score_symbol
        print("[PATCH] Re-patched score_symbol in main after import", flush=True)

    # Find and run main's async entry point
    entry = None
    for name in ["main", "run", "start", "async_main", "bot_main"]:
        if hasattr(main, name) and asyncio.iscoroutinefunction(getattr(main, name)):
            entry = getattr(main, name)
            print(f"[PATCH] Running main.{name}()", flush=True)
            break

    if entry is None:
        print("[PATCH] Could not find async entry point in main.py", flush=True)
        print("[PATCH] Available coroutines:", flush=True)
        for name in dir(main):
            obj = getattr(main, name)
            if asyncio.iscoroutinefunction(obj):
                print(f"  - {name}", flush=True)
        sys.exit(1)

    try:
        asyncio.run(entry())
    except KeyboardInterrupt:
        print(f"\n[PATCH] Stopped after {_call_count} score_symbol calls.", flush=True)
