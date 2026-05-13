#!/usr/bin/env python3
"""
Trade execution path integration test (mocked Bybit API).

Validates the Phase 1 patches end-to-end WITHOUT hitting the real exchange:

  * Fix #1  — Quality gates active in scan path (test runs through them
              by stubbing the gate helpers, but proves the path reaches
              execute_core_trade)
  * Fix #3  — Scalp Hunter qty computed (was 0)
  * Fix #4  — SL placement failure triggers emergency reduce-only close
  * Fix #5  — Real fill price queried from /v5/position/list, SL/TP
              re-anchored to actual avgPrice
  * Fix #6  — TP1 placed as conditional Market with triggerPrice
              (orderType=Market + triggerPrice in payload)

Run:
    python test_core_trade_execution_runner.py

Exit code 0 = all scenarios passed.
"""

import asyncio
import importlib
import sys
import traceback
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# MOCK CANDLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_candles(n: int, base_price: float = 100.0, bullish: bool = True) -> List[Dict[str, Any]]:
    """Build n OHLCV candles with the keys indicators expect.

    The data is realistic enough to not crash ATR/RSI/VWAP calculations.
    Direction is monotonic so the scoring pipeline can pick a side cleanly.
    """
    candles = []
    price = base_price
    for i in range(n):
        step = 0.10 if bullish else -0.10
        open_ = price
        close = price + step
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        candles.append({
            "timestamp": 1700000000 + i * 60,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0 + (i * 5),  # mild rising volume
        })
        price = close
    return candles


# ─────────────────────────────────────────────────────────────────────────────
# BYBIT API MOCK
# ─────────────────────────────────────────────────────────────────────────────

class BybitMock:
    """Stateful mock that records every API call and routes responses.

    Knobs (set before running a scenario):
        sl_should_fail        — order/create returns retCode != 0 for SL orders
        position_avg_price    — what /v5/position/list reports as fill price
        position_size         — what /v5/position/list reports as position size
        wallet_balance        — what /v5/account/wallet-balance reports
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.sl_should_fail = False
        self.position_avg_price = "100.50"
        self.position_size = "10.0"
        self.wallet_balance = "1000.0"
        self._order_seq = 0

    def _next_order_id(self) -> str:
        self._order_seq += 1
        return f"ORDER_{self._order_seq:04d}"

    def _classify_order(self, params: Dict[str, Any]) -> str:
        """Distinguish market / stop-loss / take-profit by payload shape."""
        if params.get("orderFilter") == "Stop" and params.get("triggerDirection") in (1, 2):
            # Conditional stop order — could be SL or TP depending on side
            if params.get("closeOnTrigger", False):
                return "sl"
            # SL has triggerDirection=2 for long (price falls), 1 for short
            # TP has triggerDirection=1 for long (price rises),  2 for short
            # We can't fully disambiguate without direction context; use reduceOnly+orderType
            if params.get("reduceOnly") and params.get("orderType") == "Market":
                # Could be either. Use stopPrice/triggerPrice naming convention:
                # SL uses stopPrice, TP uses triggerPrice — bybit_api.place_stop_loss uses triggerPrice though
                # Disambiguate by the value's relation to position avg (rough heuristic, but workable)
                return "stop_order"
        if params.get("orderType") == "Market" and not params.get("triggerPrice"):
            return "market"
        return "other"

    async def signed_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        record = {"method": method, "endpoint": endpoint, "params": dict(params)}
        self.calls.append(record)

        # /v5/account/wallet-balance
        if endpoint == "/v5/account/wallet-balance":
            return {
                "retCode": 0,
                "result": {
                    "list": [
                        {"coin": [{"coin": "USDT", "walletBalance": self.wallet_balance}]}
                    ]
                },
            }

        # /v5/market/tickers
        if endpoint == "/v5/market/tickers":
            return {
                "retCode": 0,
                "result": {
                    "list": [{
                        "symbol": params.get("symbol", "TESTUSDT"),
                        "lastPrice": self.position_avg_price,
                        "markPrice": self.position_avg_price,
                    }]
                },
            }

        # /v5/position/set-leverage
        if endpoint == "/v5/position/set-leverage":
            return {"retCode": 0, "result": {}}

        # /v5/position/list — post-fill confirmation
        if endpoint == "/v5/position/list":
            return {
                "retCode": 0,
                "result": {
                    "list": [{
                        "symbol": params.get("symbol", "TESTUSDT"),
                        "size": self.position_size,
                        "avgPrice": self.position_avg_price,
                        "side": "Buy",
                    }]
                },
            }

        # /v5/order/realtime — used by cleanup_orphaned_stop_orders
        if endpoint == "/v5/order/realtime":
            return {"retCode": 0, "result": {"list": []}}

        # /v5/order/cancel
        if endpoint == "/v5/order/cancel":
            return {"retCode": 0, "result": {}}

        # /v5/order/create — market, stop, or TP
        if endpoint == "/v5/order/create":
            kind = self._classify_order(params)
            # SL orders are placed with orderType=Market + triggerPrice + orderFilter=Stop
            # by bybit_api.place_stop_loss. If sl_should_fail is set, fail those.
            is_stop = params.get("orderFilter") == "Stop" or params.get("triggerPrice")
            if is_stop and self.sl_should_fail:
                # Check if it's an SL (price below market for long) or TP
                # The naked-position fix only triggers on SL failure, so we fail
                # the FIRST stop-style order placed (which is the SL by code order).
                # Subsequent stop orders (TP) succeed.
                self.sl_should_fail = False  # one-shot
                return {"retCode": 10001, "retMsg": "simulated SL failure"}
            return {"retCode": 0, "result": {"orderId": self._next_order_id()}}

        # /v5/market/kline — for any candle fetches in execution path
        if endpoint == "/v5/market/kline":
            return {"retCode": 0, "result": {"list": []}}

        return {"retCode": 0, "result": {}}

    # Recording helpers ------------------------------------------------------

    def calls_to(self, endpoint: str) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["endpoint"] == endpoint]

    def order_create_calls(self) -> List[Dict[str, Any]]:
        return [c for c in self.calls if c["endpoint"] == "/v5/order/create"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MODULE STUBS
# ─────────────────────────────────────────────────────────────────────────────

class DummyTLM:
    async def can_process_symbol(self, symbol):
        return True, ""
    async def acquire_trade_lock(self, symbol):
        return True
    def release_trade_lock(self, symbol, success):
        pass
    async def sync_with_exchange(self):
        pass
    async def cleanup_stale_locks(self):
        pass


def stub_main_module(main, symbol: str):
    """Stub gate helpers so signal generation reaches execute_core_trade."""
    main.live_candles = {
        symbol: {
            "1": make_candles(50),
            "5": make_candles(50),
            "15": make_candles(50),
        }
    }
    setattr(main, "MIN_SCALP_SCORE",    8.0)
    setattr(main, "MIN_INTRADAY_SCORE", 8.0)
    setattr(main, "MIN_SWING_SCORE",    8.0)
    setattr(main, "EXIT_COOLDOWN", 0)
    setattr(main, "SIGNAL_COOLDOWN_TIME", 0)
    setattr(main, "MAX_CORE_POSITIONS", 10)
    setattr(main, "MIN_CONFIRMATIONS", 1)
    setattr(main, "EARLY_CONFIDENCE_GATE", 0)

    main.fix_live_candles_structure = lambda x: x

    async def _filter_core_symbols(symbols):
        return symbols
    main.filter_core_symbols = _filter_core_symbols

    main.active_trades = {}
    main.recent_exits = {}
    main.signal_cooldown = {}

    # Bypass the expensive scoring chain — return a clean valid signal
    def _enhanced_score_symbol(sym, candles, ctx):
        return (12.0, {"1": 4.0, "5": 4.0, "15": 4.0}, "Scalp",
                {"rsi": 1.0, "macd": 1.0, "supertrend": 1.0}, ["rsi", "macd", "supertrend"])
    main.enhanced_score_symbol = _enhanced_score_symbol

    def _determine_core_direction(candles, ctx):
        return "Long"
    main.determine_core_direction = _determine_core_direction

    def _calculate_confidence(score, tf_scores, ctx, trade_type):
        return 80
    main.calculate_confidence = _calculate_confidence

    async def _validate_core_conditions(sym, candles, direction, ctx):
        return True
    main.validate_core_conditions = _validate_core_conditions

    def _determine_core_strategy_type(score, conf, strength):
        return "CoreScalp"
    main.determine_core_strategy_type = _determine_core_strategy_type

    def _check_strategy_position_limits(strategy_type):
        return True
    main.check_strategy_position_limits = _check_strategy_position_limits

    async def _get_core_confirmations(sym, candles, direction, ctx):
        return ["momentum_alignment", "volume_breakout", "strong_trend"]
    main.get_core_confirmations = _get_core_confirmations

    main.trade_lock_manager = DummyTLM()

    async def _is_duplicate_signal(*a, **kw):
        return False
    main.is_duplicate_signal = lambda *a, **kw: False

    async def _send_telegram_message(*a, **kw):
        return None
    main.send_telegram_message = _send_telegram_message
    main.send_core_strategy_notification = _send_telegram_message


def stub_execution_layer(bybit_mock: BybitMock):
    """Replace external side-effects in bybit_api / trade_executor / monitor."""
    import bybit_api
    import trade_executor
    import monitor
    import error_handler
    import telegram_bot
    import symbol_info
    import activity_logger

    # signed_request is imported by-name into trade_executor — patch both modules.
    bybit_api.signed_request = bybit_mock.signed_request
    trade_executor.signed_request = bybit_mock.signed_request

    # place_market_order — record + delegate to mock signed_request for retCode
    async def mock_place_market_order(symbol, side, qty, market_type="linear", reduce_only=False):
        params = {
            "category": market_type, "symbol": symbol, "side": side,
            "orderType": "Market", "qty": str(qty), "timeInForce": "IOC",
        }
        if reduce_only:
            params["reduceOnly"] = True
        return await bybit_mock.signed_request("POST", "/v5/order/create", params)

    bybit_api.place_market_order = mock_place_market_order
    trade_executor.place_market_order = mock_place_market_order

    # place_stop_loss_with_retry — delegate to signed_request mock so sl_should_fail works
    async def mock_place_sl_with_retry(symbol, direction, qty, sl_price, market_type="linear", max_attempts=3):
        side = "Sell" if direction.lower() == "long" else "Buy"
        trigger_direction = 2 if direction.lower() == "long" else 1
        payload = {
            "category": market_type, "symbol": symbol, "side": side,
            "orderType": "Market", "triggerPrice": str(sl_price),
            "triggerDirection": trigger_direction, "triggerBy": "MarkPrice",
            "qty": str(qty), "reduceOnly": True, "timeInForce": "GTC",
            "orderFilter": "Stop",
        }
        return await bybit_mock.signed_request("POST", "/v5/order/create", payload)

    bybit_api.place_stop_loss_with_retry = mock_place_sl_with_retry
    bybit_api.place_stop_loss = mock_place_sl_with_retry

    # Telegram and logging — silent no-ops
    async def _async_noop(*a, **kw):
        return None
    error_handler.send_telegram_message = _async_noop
    error_handler.send_error_to_telegram = _async_noop
    telegram_bot.send_telegram_message = _async_noop
    telegram_bot.send_error_to_telegram = _async_noop
    trade_executor.send_telegram_message = _async_noop
    trade_executor.send_error_to_telegram = _async_noop

    activity_logger.log_trade_to_file = lambda *a, **kw: None
    trade_executor.log_trade_to_file = lambda *a, **kw: None

    # symbol_info helpers — return inputs unchanged so qty math is predictable
    def _round_qty(symbol, qty):
        return round(float(qty), 3)
    symbol_info.round_qty = _round_qty
    trade_executor.round_qty = _round_qty
    symbol_info.get_precision = lambda symbol: 3

    async def _fetch_symbol_info():
        return None
    symbol_info.fetch_symbol_info = _fetch_symbol_info

    # Monitor — record track_active_trade calls, no file I/O
    monitor.track_active_trade_calls = []
    async def _track_active_trade(*args, **kwargs):
        monitor.track_active_trade_calls.append(kwargs)
    monitor.track_active_trade = _track_active_trade
    monitor.save_active_trades = lambda: None

    # trade_verification.verify_position_and_orders — return False (no existing pos)
    import trade_verification
    async def _verify_position_and_orders(symbol, trade):
        return False
    trade_verification.verify_position_and_orders = _verify_position_and_orders


# ─────────────────────────────────────────────────────────────────────────────
# TEST SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

class TestFailure(Exception):
    pass


def assert_(cond: bool, msg: str):
    if not cond:
        raise TestFailure(msg)


async def scenario_happy_path(main, mock: BybitMock) -> None:
    """Standard signal -> execute_trade_core completes successfully.

    Validates:
      - /v5/position/list was queried for actual fill (Fix #5)
      - SL placed (orderFilter=Stop, triggerPrice present)
      - TP1 placed as conditional Market (orderType=Market + triggerPrice + reduceOnly)
      - track_active_trade was registered
    """
    print("\n---- SCENARIO 1: Happy path -------------------------------")
    symbol = "TESTUSDT"
    mock.calls.clear()
    mock.sl_should_fail = False
    mock.position_avg_price = "100.50"  # 0.5% slippage vs. snapshot
    mock.position_size = "10.0"

    import monitor
    monitor.track_active_trade_calls = []

    trend_context = {"trend": "up", "trend_strength": 0.7,
                     "recommendations": {"primary_strategy": "go_long"}, "opportunity_score": 0.8}

    await main.core_strategy_scan([symbol], trend_context)

    # 1. /v5/position/list queried for real fill (Fix #5)
    pos_calls = mock.calls_to("/v5/position/list")
    assert_(len(pos_calls) >= 1, "expected /v5/position/list to be queried for fill confirmation (Fix #5)")
    print(f"  [PASS] Fill confirmation via /v5/position/list: {len(pos_calls)} call(s)")

    # 2. Market order placed
    order_calls = mock.order_create_calls()
    market_orders = [c for c in order_calls
                     if c["params"].get("orderType") == "Market"
                     and not c["params"].get("triggerPrice")]
    assert_(len(market_orders) >= 1, f"expected market order; got {[c['params'] for c in order_calls]}")
    print(f"  [PASS] Market order placed: side={market_orders[0]['params'].get('side')}, qty={market_orders[0]['params'].get('qty')}")

    # 3. SL order placed (Stop with triggerPrice)
    sl_orders = [c for c in order_calls
                 if c["params"].get("orderFilter") == "Stop"
                 and c["params"].get("triggerDirection") == 2]  # 2 = long SL
    assert_(len(sl_orders) >= 1, f"expected SL order with triggerDirection=2; calls={[c['params'] for c in order_calls]}")
    print(f"  [PASS] SL placed at trigger={sl_orders[0]['params'].get('triggerPrice')}")

    # 4. TP1 as conditional Market (Fix #6) - orderType=Market, triggerPrice present, triggerDirection=1 for long
    tp_orders = [c for c in order_calls
                 if c["params"].get("orderType") == "Market"
                 and c["params"].get("triggerPrice")
                 and c["params"].get("triggerDirection") == 1]
    assert_(len(tp_orders) >= 1, f"expected conditional Market TP1 (orderType=Market + triggerPrice + triggerDirection=1); got {[c['params'] for c in order_calls]}")
    tp_params = tp_orders[0]["params"]
    assert_(tp_params.get("reduceOnly") is True, "TP1 must be reduceOnly")
    print(f"  [PASS] TP1 conditional Market: trigger={tp_params.get('triggerPrice')}, reduceOnly={tp_params.get('reduceOnly')}")

    # 5. SL/TP re-anchored to actual fill price (Fix #5)
    # avgPrice is 100.50; for Scalp (0.8% SL, 1.2% TP), expect:
    #   SL ~= 100.50 * 0.992 = 99.696
    #   TP ~= 100.50 * 1.012 = 101.706
    sl_trigger = float(sl_orders[0]["params"].get("triggerPrice"))
    tp_trigger = float(tp_orders[0]["params"].get("triggerPrice"))
    assert_(99.5 < sl_trigger < 99.9, f"SL not anchored to avgPrice 100.50; expected ~99.70, got {sl_trigger}")
    assert_(101.5 < tp_trigger < 101.9, f"TP not anchored to avgPrice 100.50; expected ~101.70, got {tp_trigger}")
    print(f"  [PASS] SL/TP re-anchored to actual fill 100.50 (SL={sl_trigger}, TP={tp_trigger})")

    # 6. Monitor registration
    assert_(len(monitor.track_active_trade_calls) >= 1, "expected track_active_trade to be called")
    reg = monitor.track_active_trade_calls[0]
    assert_(reg.get("entry_price") == 100.5, f"monitor entry_price wrong: {reg.get('entry_price')}")
    assert_(reg.get("qty") == 10.0, f"monitor qty wrong: {reg.get('qty')}")
    print(f"  [PASS] Monitor registered: entry={reg.get('entry_price')}, qty={reg.get('qty')}, sl_order_id={reg.get('sl_order_id')}")


async def scenario_sl_failure_emergency_close(main, mock: BybitMock) -> None:
    """Fix #4: when SL placement fails, position must be closed immediately."""
    print("\n---- SCENARIO 2: SL placement fails -> emergency close ----")
    symbol = "TESTUSDT2"
    main.live_candles[symbol] = {
        "1": make_candles(50, base_price=50.0),
        "5": make_candles(50, base_price=50.0),
        "15": make_candles(50, base_price=50.0),
    }
    main.active_trades.clear()
    main.signal_cooldown.clear()
    mock.calls.clear()
    mock.sl_should_fail = True  # one-shot fail on first stop order (the SL)
    mock.position_avg_price = "50.25"
    mock.position_size = "20.0"

    import monitor
    monitor.track_active_trade_calls = []

    trend_context = {"trend": "up", "trend_strength": 0.7,
                     "recommendations": {"primary_strategy": "go_long"}, "opportunity_score": 0.8}

    await main.core_strategy_scan([symbol], trend_context)

    order_calls = mock.order_create_calls()

    # Expect: market order placed, SL attempted and failed, emergency close (Market reduce-only) placed
    market_orders = [c for c in order_calls
                     if c["params"].get("orderType") == "Market"
                     and not c["params"].get("triggerPrice")]
    assert_(len(market_orders) >= 2, f"expected entry market + emergency close market; got {len(market_orders)} market orders")
    print(f"  [PASS] {len(market_orders)} market orders placed (entry + emergency close)")

    # Emergency close should be reduce-only with opposite side of entry
    entry = market_orders[0]["params"]
    emergency = market_orders[1]["params"]
    assert_(emergency.get("reduceOnly") is True, "emergency close must be reduceOnly")
    assert_(entry.get("side") != emergency.get("side"),
            f"emergency close side ({emergency.get('side')}) must be opposite of entry ({entry.get('side')})")
    print(f"  [PASS] Emergency close: side={emergency.get('side')}, reduceOnly=True (entry was {entry.get('side')})")

    # Monitor must NOT have registered the trade (we bailed out)
    assert_(len(monitor.track_active_trade_calls) == 0,
            f"trade should not be registered after SL failure; got {len(monitor.track_active_trade_calls)} registrations")
    print(f"  [PASS] Monitor NOT registered (trade aborted, as designed)")


async def scenario_scalp_hunter_qty(main, mock: BybitMock) -> None:
    """Fix #3: Scalp Hunter path must compute qty > 0 from risk %.

    The standard path goes through enhanced_score_symbol etc. The scalp hunter
    path passes pre-computed SL/TP via signal_data with is_scalp_hunter=True
    and historically left qty unset → orders failed silently. Test by calling
    execute_trade_if_valid directly with a scalp-hunter signal.
    """
    print("\n---- SCENARIO 3: Scalp Hunter qty computation (Fix #3) ----")
    symbol = "SCALPUSDT"
    mock.calls.clear()
    mock.sl_should_fail = False
    mock.position_avg_price = "10.00"
    mock.position_size = "100.0"

    import monitor
    monitor.track_active_trade_calls = []

    from trade_executor import execute_trade_if_valid

    signal_data = {
        "symbol": symbol,
        "direction": "Long",
        "strategy": "ScalpHunter",
        "trade_type": "ScalpHunter",
        "score": 8.5,
        "confidence": 75,
        "regime": "scalp_hunter",
        "sl_price": 9.95,
        "tp1_price": 10.10,
        "sl_pct": 0.5,
        "tp1_pct": 1.0,
        "trailing_pct": 0.5,
        "tp1_partial_close": 0.5,
        "leverage": 5,
        "candles": {"1": make_candles(50, base_price=10.0)},
        "is_scalp_hunter": True,
        "price": 10.0,
        "entry_price": 10.0,
    }

    result = await execute_trade_if_valid(signal_data, max_risk=0.01)

    # Pre-fix: result would be None / silent failure due to qty=0
    assert_(result is not None, "scalp hunter trade returned None — qty=0 bug not fixed")

    order_calls = mock.order_create_calls()
    market_orders = [c for c in order_calls
                     if c["params"].get("orderType") == "Market"
                     and not c["params"].get("triggerPrice")]
    assert_(len(market_orders) >= 1, "expected market order for scalp hunter entry")

    qty = float(market_orders[0]["params"].get("qty", "0"))
    assert_(qty > 0, f"scalp hunter market order qty must be > 0; got {qty}")
    print(f"  [PASS] Scalp Hunter market order placed with qty={qty} (Fix #3 confirmed)")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_all() -> int:
    try:
        print("[SETUP] Importing main.py and execution modules...")
        main = importlib.import_module("main")
    except Exception:
        traceback.print_exc()
        return 1

    print("[SETUP] Patching Bybit API surface with mock...")
    mock = BybitMock()
    stub_execution_layer(mock)

    print("[SETUP] Stubbing main.py signal-gate helpers...")
    stub_main_module(main, "TESTUSDT")

    scenarios = [
        ("happy_path",                scenario_happy_path),
        ("sl_failure_emergency_close", scenario_sl_failure_emergency_close),
        ("scalp_hunter_qty",          scenario_scalp_hunter_qty),
    ]

    results = []
    for name, fn in scenarios:
        try:
            await fn(main, mock)
            results.append((name, "PASS", None))
        except TestFailure as e:
            results.append((name, "FAIL", str(e)))
            print(f"  [FAIL] {e}")
        except Exception as e:
            results.append((name, "ERROR", repr(e)))
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    for name, status, err in results:
        line = f"[{status:5}]  {name}"
        if err:
            line += f"  ({err})"
        print(line)

    failed = sum(1 for _, s, _ in results if s != "PASS")
    print("=" * 64)
    print(f"Total: {len(results)} | Passed: {len(results) - failed} | Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(run_all()))
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted")
        sys.exit(130)
