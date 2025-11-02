#!/usr/bin/env python3
"""
Scan Activity Smoke Test (no pytest needed)

What it checks:
- main.core_strategy_scan() actually processes symbols (increments scanned_count).
- score_symbol(...) is invoked (meaning the scan reached eval stage).
- Prevents real orders by stubbing execute_core_trade(...).
- Parses the summary log line to report how many were scanned.

Run:
    python3 test_scan_activity_runner.py

Exit code:
    0 = success (scanned >= EXPECT_MIN_SCANNED)
    1 = failure or exception
"""

import asyncio
import importlib
import re
import sys
import traceback
from time import time

EXPECT_MIN_SCANNED = 3  # set how many you expect to be scanned in this smoke test
SYMBOLS = [f"TEST{i}USDT" for i in range(1, 6)]  # 5 symbols

def make_candles(n: int):
    # Only length matters for this test
    return [{"t": i, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 1000} for i in range(n)]

class DummyLock:
    def __init__(self):
        self._locked = True
    def locked(self):
        return self._locked
    def release(self):
        self._locked = False

class DummyTLM:
    async def can_process_symbol(self, symbol):
        return True, ""
    async def acquire_trade_lock(self, symbol):
        return True
    def get_lock(self, symbol):
        return DummyLock()
    def release_trade_lock(self, symbol, success):
        pass

async def run():
    try:
        print("[SCAN-TEST] Importing main.py ...")
        main = importlib.import_module("main")
    except Exception:
        print("[SCAN-TEST][ERROR] Failed to import main.py:")
        traceback.print_exc()
        sys.exit(1)

    # ---- Prepare stub candle data for multiple symbols ----
    print(f"[SCAN-TEST] Preparing candles for {len(SYMBOLS)} symbols ...")
    # Satisfy your (relaxed) candles gate:
    # - 1m >= 12, 5m >= 12, 15m optional >= 8
    main.live_candles = {
        sym: {
            "1": make_candles(12),
            "5": make_candles(12),
            "15": make_candles(8),
        } for sym in SYMBOLS
    }

    # ---- Ensure gates aren't too strict for the scan stage ----
    # These do not force a trade; they just help the scan proceed to eval
    setattr(main, "MIN_SCALP_SCORE", 8.0)
    setattr(main, "EXIT_COOLDOWN", 0)
    setattr(main, "MAX_CORE_POSITIONS", 10)

    # ---- Pass-through candle structure function ----
    main.fix_live_candles_structure = lambda x: x

    # ---- Do not narrow candidates further in this test ----
    async def _filter_core_symbols(symbols):
        return symbols
    main.filter_core_symbols = _filter_core_symbols

    # ---- No active trades / exits ----
    main.active_trades = {}
    main.recent_exits = {}

    # ---- Deterministic helpers to reach scoring (still not forcing a trade) ----
    async def _calculate_core_score(sym, candles, ctx):
        # Enough to pass core_score gate (>= MIN_SCALP_SCORE)
        return 8.4
    main.calculate_core_score = _calculate_core_score

    def _determine_core_direction(candles, ctx):
        # Provide a direction so confidence can be computed
        return "LONG"
    main.determine_core_direction = _determine_core_direction

    def _calculate_confidence(score, tf_scores, ctx, trade_type):
        # Enough to pass the 60% confidence gate if your code checks it early
        return 65
    main.calculate_confidence = _calculate_confidence

    async def _validate_core_conditions(sym, candles, direction, ctx):
        # Let it pass this gate to reach confirmations
        return True
    main.validate_core_conditions = _validate_core_conditions

    def _determine_core_strategy_type(core_score, confidence, trend_strength):
        # Provide a valid type; we will still stub trading below
        return "CoreScalp"
    main.determine_core_strategy_type = _determine_core_strategy_type

    def _check_strategy_position_limits(strategy_type):
        return True
    main.check_strategy_position_limits = _check_strategy_position_limits

    async def _get_core_confirmations(sym, candles, direction, ctx):
        # Return 2 confirmations to pass final gate if needed
        return ["rsi", "volume"]
    main.get_core_confirmations = _get_core_confirmations

    # ---- Score function; its call count will also prove evaluation happened ----
    score_calls = {"count": 0}
    def _score_symbol(sym, candles, ctx):
        score_calls["count"] += 1
        # score, tf_scores, trade_type, indicator_scores, used_indicators
        return (9.0, {"1": 9, "5": 8.5, "15": 7.5}, "Scalp",
                {"rsi": 1, "macd": 1}, ["rsi", "macd"])
    main.score_symbol = _score_symbol

    # ---- Stub the trade lock mgr ----
    main.trade_lock_manager = DummyTLM()

    # ---- Block actual trade placement; we only test scan activity ----
    async def _execute_core_trade(**kwargs):
        # Simulate success without hitting any API
        return {"success": True, "orderId": "SIM"}
    main.execute_core_trade = _execute_core_trade
    async def _send_core_strategy_notification(*args, **kwargs):
        return None
    main.send_core_strategy_notification = _send_core_strategy_notification

    # ---- Capture logs to parse scanned count from summary line ----
    captured_logs = []
    original_log = getattr(main, "log", None)

    def capturing_log(msg, level="INFO"):
        # Keep your normal print/log side-effect if you want
        # but also capture into a list for parsing
        captured_logs.append((level, msg))
        if original_log:
            original_log(msg, level=level)
        else:
            # fallback print if main.log doesn't exist
            print(f"[{level}] {msg}")

    # Replace main.log with our capturing wrapper
    main.log = capturing_log

    trend_context = {"trend": "up", "trend_strength": 0.7}

    print("[SCAN-TEST] Running core_strategy_scan() ...")
    try:
        await main.core_strategy_scan(SYMBOLS, trend_context)
    except Exception:
        print("[SCAN-TEST][ERROR] core_strategy_scan raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    # ---- Parse scanned count from summary ----
    scanned = None
    quality = None
    pattern = re.compile(r"CORE STRATEGY SUMMARY:\s*(\d+)\s+scanned,\s*(\d+)\s+quality")
    for level, msg in captured_logs:
        m = pattern.search(msg)
        if m:
            scanned = int(m.group(1))
            quality = int(m.group(2))
            break

    print(f"[SCAN-TEST] score_symbol calls counted: {score_calls['count']}")
    if scanned is not None:
        print(f"[SCAN-TEST] Parsed summary → scanned={scanned}, quality={quality}")
    else:
        print("[SCAN-TEST][WARN] Could not find summary line in logs.")
        scanned = 0

    # ---- Decide pass/fail ----
    if scanned >= EXPECT_MIN_SCANNED and score_calls["count"] >= EXPECT_MIN_SCANNED:
        print("[SCAN-TEST][OK] Scanner is active ✅")
        sys.exit(0)
    else:
        print("[SCAN-TEST][FAIL] Scanner activity too low ❌")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[SCAN-TEST] Interrupted")
        sys.exit(130)
