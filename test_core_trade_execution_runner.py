#!/usr/bin/env python3
"""
Smoke-test runner (no pytest needed)

Goal:
- Prove that when a valid core signal appears, core_strategy_scan() calls execute_core_trade(...).

How it works:
- Imports your main.py as a module.
- Stubs out data/locks/validators so one symbol will pass all gates.
- Replaces execute_core_trade(...) with a stub and reports if it was called.

Run:
    python3 test_core_trade_execution_runner.py

Exit code:
    0 = success (trade execution path called)
    1 = failure (no call), or import/runtime error
"""

import asyncio
import importlib
import sys
import traceback

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
        print("[TEST] Importing main.py...")
        main = importlib.import_module("main")
    except Exception as e:
        print("[TEST][ERROR] Failed to import main.py:")
        traceback.print_exc()
        sys.exit(1)

    # ---- Test settings ----
    symbol = "TESTUSDT"
    print(f"[TEST] Preparing live candles for {symbol} ...")
    # Satisfy your relaxed gates: 1m>=12, 5m>=12, 15m>=8 (optional)
    main.live_candles = {
        symbol: {
            "1": make_candles(12),
            "5": make_candles(12),
            "15": make_candles(8),
        }
    }

    # Loosen gates a bit for a guaranteed pass (only for this test)
    setattr(main, "MIN_SCALP_SCORE", 8.0)
    setattr(main, "EXIT_COOLDOWN", 0)
    setattr(main, "MAX_CORE_POSITIONS", 10)

    # Pass-through the candle structure fixer
    main.fix_live_candles_structure = lambda x: x

    # Don’t narrow candidates in this test
    async def _filter_core_symbols(symbols):
        return symbols
    main.filter_core_symbols = _filter_core_symbols

    # No open trades / recent exits
    main.active_trades = {}
    main.recent_exits = {}

    # Deterministic helpers so our symbol clears all core gates
    async def _calculate_core_score(sym, candles, ctx):
        return 8.7  # >= MIN_SCALP_SCORE
    main.calculate_core_score = _calculate_core_score

    def _determine_core_direction(candles, ctx):
        return "LONG"
    main.determine_core_direction = _determine_core_direction

    def _calculate_confidence(score, tf_scores, ctx, trade_type):
        return 75  # >= 60
    main.calculate_confidence = _calculate_confidence

    async def _validate_core_conditions(sym, candles, direction, ctx):
        return True
    main.validate_core_conditions = _validate_core_conditions

    def _determine_core_strategy_type(core_score, confidence, trend_strength):
        return "CoreScalp"
    main.determine_core_strategy_type = _determine_core_strategy_type

    def _check_strategy_position_limits(strategy_type):
        return True
    main.check_strategy_position_limits = _check_strategy_position_limits

    async def _get_core_confirmations(sym, candles, direction, ctx):
        return ["rsi", "volume"]  # >= 2 confirmations
    main.get_core_confirmations = _get_core_confirmations

    # score_symbol tuple: (score, tf_scores, trade_type, indicator_scores, used_indicators)
    def _score_symbol(sym, candles, ctx):
        return (9.2, {"1": 9, "5": 8.5, "15": 7.5}, "Scalp",
                {"rsi": 1, "macd": 1}, ["rsi", "macd"])
    main.score_symbol = _score_symbol

    # Stub the trade-lock manager to always allow
    main.trade_lock_manager = DummyTLM()

    # Capture execute_core_trade calls
    calls = {"execute_core_trade": []}
    async def _execute_core_trade(**kwargs):
        print(f"[TEST] execute_core_trade called with: {kwargs}")
        calls["execute_core_trade"].append(kwargs)
        # Simulate success from downstream
        return {"success": True, "orderId": "TEST123"}
    main.execute_core_trade = _execute_core_trade

    # If execute_core_trade calls notifications, make them no-ops
    async def _send_core_strategy_notification(*args, **kwargs):
        return None
    main.send_core_strategy_notification = _send_core_strategy_notification

    trend_context = {"trend": "up", "trend_strength": 0.7}

    print("[TEST] Running core_strategy_scan() ...")
    try:
        await main.core_strategy_scan([symbol], trend_context)
    except Exception as e:
        print("[TEST][ERROR] core_strategy_scan raised an exception:")
        traceback.print_exc()
        sys.exit(1)

    if len(calls["execute_core_trade"]) >= 1:
        print("[TEST][OK] Trade execution path WAS called ✅")
        sys.exit(0)
    else:
        print("[TEST][FAIL] Trade execution path was NOT called ❌")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        sys.exit(130)
