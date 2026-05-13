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
    Timestamps end at "now" so the staleness guard (Phase 3 Fix #3) accepts them.
    """
    import time as _time
    candles = []
    price = base_price
    # Anchor newest candle at now (ms); generate one bar per minute going backwards.
    now_ms = int(_time.time() * 1000)
    for i in range(n):
        # i = 0 is the oldest; i = n-1 is newest (== now).
        ts_ms = now_ms - (n - 1 - i) * 60_000
        step = 0.10 if bullish else -0.10
        open_ = price
        close = price + step
        high = max(open_, close) + 0.05
        low = min(open_, close) - 0.05
        candles.append({
            "timestamp": ts_ms,
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

        # /v5/order/amend — used by unified_exit_manager._update_exchange_sl
        if endpoint == "/v5/order/amend":
            return {"retCode": 0, "result": {"orderId": params.get("orderId", "")}}

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
# PHASE 2 EXIT-PATH SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

async def scenario_force_exit(main, mock: BybitMock) -> None:
    """Phase 2 Fix #8: process_trade_exits(force_exit=True) must immediately
    close the position regardless of PnL thresholds."""
    print("\n---- SCENARIO 4: force_exit kwarg actually works (Fix #8) -")
    from unified_exit_manager import process_trade_exits, exit_manager

    symbol = "FORCEUSDT"
    mock.calls.clear()
    exit_manager.reset_symbol(symbol)

    trade = {
        "symbol": symbol,
        "direction": "Long",
        "entry_price": 100.0,
        "qty": 10.0,
        "trade_type": "Scalp",
        "sl_order_id": "OLD_SL_ID",
        "tp1_order_id": "OLD_TP_ID",
        "exited": False,
    }

    # Current price barely above entry — pnl ~ +0.1%, well below any SL/TP trigger.
    exited = await process_trade_exits(symbol, trade, 100.1, force_exit=True)
    assert_(exited is True, "force_exit must return True (trade exited)")
    assert_(trade.get("exited") is True, "trade['exited'] must be True after force_exit")
    assert_(trade.get("exit_reason") == "force_exit", f"exit_reason wrong: {trade.get('exit_reason')}")
    print(f"  [PASS] Force-exit closed trade with pnl=+0.10%")


async def scenario_exit_cancels_residual_orders(main, mock: BybitMock) -> None:
    """Phase 2 Fix #2: on trade exit, the SL and TP1 conditional orders
    must be cancelled before/alongside the market close so stale orders
    don't accumulate on the exchange."""
    print("\n---- SCENARIO 5: exit cancels residual SL + TP1 orders ----")
    from unified_exit_manager import process_trade_exits, exit_manager

    symbol = "CANCELUSDT"
    mock.calls.clear()
    exit_manager.reset_symbol(symbol)

    # Mock returns size=0 post-close so the verification loop succeeds.
    mock.position_size = "0"

    trade = {
        "symbol": symbol,
        "direction": "Long",
        "entry_price": 100.0,
        "qty": 10.0,
        "trade_type": "Scalp",
        "sl_order_id": "SL_TO_CANCEL",
        "tp1_order_id": "TP_TO_CANCEL",
        "exited": False,
    }

    await process_trade_exits(symbol, trade, 100.1, force_exit=True)

    cancel_calls = mock.calls_to("/v5/order/cancel")
    cancelled_ids = {c["params"].get("orderId") for c in cancel_calls}
    assert_("SL_TO_CANCEL" in cancelled_ids,
            f"SL order not cancelled on exit; cancelled={cancelled_ids}")
    assert_("TP_TO_CANCEL" in cancelled_ids,
            f"TP1 order not cancelled on exit; cancelled={cancelled_ids}")
    print(f"  [PASS] Both SL_TO_CANCEL and TP_TO_CANCEL were cancelled on exit")

    # Restore default for subsequent scenarios.
    mock.position_size = "10.0"


async def scenario_tp1_via_fill_detection(main, mock: BybitMock) -> None:
    """Phase 2 Fix #1: TP1 must be detected from actual position size shrinkage
    (exchange's conditional Market filled), NOT only from PnL %.

    Setup: trade["original_qty"] = 10.0, mock /v5/position/list returns size 5.0
    (TP1 just filled, position halved). Current price is BELOW the PnL trigger
    but above 50% of it (the gate). Bot should detect TP1 via fill."""
    print("\n---- SCENARIO 6: TP1 detected via position-size shrink (Fix #1)")
    from unified_exit_manager import process_trade_exits, exit_manager

    symbol = "TP1FILLUSDT"
    exit_manager.reset_symbol(symbol)
    mock.calls.clear()
    mock.position_size = "5.0"        # exchange shows position halved

    trade = {
        "symbol": symbol,
        "direction": "Long",
        "entry_price": 100.0,
        "qty": 10.0,
        "original_qty": 10.0,
        "trade_type": "Scalp",
        "tp1_target": 101.2,           # TP1 trigger level
        "tp1_order_id": "TP1_FILLED",
        "sl_order_id": "SL_PRE_TP1",
        "exited": False,
    }

    # Current price = 100.7 (pnl=+0.7%, below 1.2% trigger but above gate=0.6%)
    exited = await process_trade_exits(symbol, trade, 100.7)
    assert_(exited is False, "tp1 fill detection should NOT exit the trade")
    assert_(trade.get("tp1_hit") is True, "tp1_hit flag must be set after fill detection")
    assert_(trade.get("breakeven_sl") is not None, "breakeven SL must be computed")
    # qty must be re-anchored to the actual remaining size
    assert_(float(trade.get("qty", 0)) == 5.0,
            f"trade['qty'] must be re-anchored to remaining size 5.0; got {trade.get('qty')}")

    # SL update must have been ATTEMPTED — either via amend or replacement.
    amend_calls = mock.calls_to("/v5/order/amend")
    sl_calls = [c for c in mock.order_create_calls()
                if c["params"].get("orderFilter") == "Stop"]
    assert_(len(amend_calls) + len(sl_calls) >= 1,
            "expected /v5/order/amend or new SL placement after TP1 detection")
    print(f"  [PASS] TP1 detected via fill; qty re-anchored 10.0 -> 5.0; SL moved to breakeven")

    mock.position_size = "10.0"


async def scenario_sl_update_via_amend(main, mock: BybitMock) -> None:
    """Phase 2 Fix #4: SL updates use /v5/order/amend (atomic), not
    cancel-then-place. No naked-position window during trailing pumps."""
    print("\n---- SCENARIO 7: SL update via /v5/order/amend (Fix #4) ----")
    from unified_exit_manager import exit_manager

    symbol = "AMENDUSDT"
    exit_manager.reset_symbol(symbol)
    mock.calls.clear()

    trade = {
        "symbol": symbol,
        "direction": "Long",
        "entry_price": 100.0,
        "qty": 10.0,
        "trade_type": "Scalp",
        "sl_order_id": "EXISTING_SL",
        "exited": False,
    }

    ok = await exit_manager._update_exchange_sl(symbol, trade, 100.5)
    assert_(ok is True, "_update_exchange_sl should succeed when amend works")

    amend_calls = mock.calls_to("/v5/order/amend")
    assert_(len(amend_calls) >= 1,
            f"expected /v5/order/amend call; got calls to: {[c['endpoint'] for c in mock.calls]}")
    assert_(amend_calls[0]["params"].get("orderId") == "EXISTING_SL",
            "amend must target the existing SL orderId")
    assert_(float(amend_calls[0]["params"].get("triggerPrice")) == 100.5,
            "amend must carry the new trigger price")

    # CRUCIAL: no cancel call should fire when amend succeeds — that was the
    # old naked-window bug. Verify no cancel happened on the SL order.
    cancel_calls = mock.calls_to("/v5/order/cancel")
    sl_cancels = [c for c in cancel_calls if c["params"].get("orderId") == "EXISTING_SL"]
    assert_(len(sl_cancels) == 0,
            "old cancel-then-place pattern must not be used when amend succeeds")
    print(f"  [PASS] SL atomically amended via /v5/order/amend (no cancel-then-place)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 WEBSOCKET / STALENESS SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

async def scenario_candle_dedup(main, mock: BybitMock) -> None:
    """Phase 3 Fix #1 (showstopper): intra-bar WS updates with the same
    timestamp must REPLACE the last deque entry, not append. Otherwise the
    deque fills with duplicates and every indicator computes on garbage.
    """
    print("\n---- SCENARIO 8: candle dedup on timestamp (Fix #1) -------")
    from collections import deque
    from websocket_candles import _append_or_replace_candle

    d = deque(maxlen=100)

    # Three intra-bar updates for the same 1m bar (same timestamp).
    _append_or_replace_candle(d, {"timestamp": 1_700_000_000_000, "close": "100.0"})
    _append_or_replace_candle(d, {"timestamp": 1_700_000_000_000, "close": "100.5"})
    _append_or_replace_candle(d, {"timestamp": 1_700_000_000_000, "close": "100.7"})
    assert_(len(d) == 1, f"intra-bar updates must collapse to 1 entry; got {len(d)}")
    assert_(d[-1]["close"] == "100.7",
            f"last update must win; got {d[-1]['close']}")

    # A new-timestamp message starts a new bar.
    _append_or_replace_candle(d, {"timestamp": 1_700_000_060_000, "close": "101.0"})
    assert_(len(d) == 2, f"new bar must append; got {len(d)}")

    # Another intra-bar update on the second bar
    _append_or_replace_candle(d, {"timestamp": 1_700_000_060_000, "close": "101.5"})
    assert_(len(d) == 2, f"second bar's update must replace, not append; got {len(d)}")
    assert_(d[-1]["close"] == "101.5", f"bar2 last update wrong: {d[-1]['close']}")

    print(f"  [PASS] dedup correct: 3 ticks + 2 ticks -> 2 distinct bars")


async def scenario_staleness_guard(main, mock: BybitMock) -> None:
    """Phase 3 Fix #3: candles_are_fresh() must reject stale candles.
    A 1m candle older than ~3 minutes should be considered stale; one
    timestamped 'now' must be considered fresh."""
    print("\n---- SCENARIO 9: candle staleness detector (Fix #3) -------")
    import time as _time
    from websocket_candles import candles_are_fresh

    now_ms = int(_time.time() * 1000)

    fresh = [{"timestamp": now_ms - 5_000}]            # 5 seconds old
    stale = [{"timestamp": now_ms - 10 * 60_000}]      # 10 minutes old
    ancient = [{"timestamp": now_ms - 60 * 60_000}]    # 1 hour old
    empty: List[Dict[str, Any]] = []
    bad_ts = [{"timestamp": 0}]

    assert_(candles_are_fresh(fresh, "1") is True, "5s-old 1m candle should be fresh")
    assert_(candles_are_fresh(stale, "1") is False, "10min-old 1m candle should be stale")
    assert_(candles_are_fresh(ancient, "1") is False, "1hr-old candle should be stale")
    assert_(candles_are_fresh(empty, "1") is False, "empty deque must be 'stale'")
    assert_(candles_are_fresh(bad_ts, "1") is False, "zero timestamp must be 'stale'")

    # Higher-interval candle: 1hr-old 1h candle is still fresh (max age 3hr)
    assert_(candles_are_fresh(ancient, "60") is True,
            "1hr-old 60m candle is well within the 3-bar grace window")
    print(f"  [PASS] staleness boundaries correct across intervals")


async def scenario_supervisor_restart(main, mock: BybitMock) -> None:
    """Phase 3 Fix #2: supervise() must restart a crashing coroutine
    transparently. Verify by counting executions of a flaky factory."""
    print("\n---- SCENARIO 10: task supervisor restart (Fix #2) --------")
    from task_supervisor import supervise

    state = {"runs": 0, "alerts": []}

    async def alert_fn(msg: str):
        state["alerts"].append(msg)

    async def flaky_coro():
        state["runs"] += 1
        # First two runs crash; third hangs (simulates a healthy long-lived task).
        if state["runs"] < 3:
            await asyncio.sleep(0.01)
            raise RuntimeError(f"simulated crash #{state['runs']}")
        # Healthy: wait until cancelled.
        await asyncio.Event().wait()

    task = asyncio.create_task(
        supervise(
            name="test_flaky",
            coro_factory=flaky_coro,
            base_backoff=0.05,
            max_backoff=0.2,
            alert_fn=alert_fn,
        )
    )

    # Give the supervisor time to crash twice and start the third (healthy) run.
    await asyncio.sleep(0.6)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert_(state["runs"] >= 3,
            f"supervisor must restart on crash; got only {state['runs']} runs")
    assert_(len(state["alerts"]) >= 2,
            f"supervisor must fire alerts on each crash; got {len(state['alerts'])}")
    print(f"  [PASS] supervisor restarted {state['runs']} times, fired {len(state['alerts'])} alerts")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 BACKTEST ENGINE SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

async def scenario_backtest_fees_and_slippage(main, mock: BybitMock) -> None:
    """Phase 5: fees + slippage must reduce returns vs a zero-cost baseline
    on the same synthetic dataset. If they don't, the cost model is broken."""
    print("\n---- SCENARIO 11: backtest fees+slippage drag (Phase 5) ----")
    from backtest_engine import BacktestConfig, BacktestEngine

    # Trivial trend-following scorer for deterministic results:
    # always trigger an Intraday Long when the last close is above the 20-bar
    # mean. Returns the 5-tuple shape the engine expects.
    def trivial_scorer(symbol, candles_by_tf, trend_context):
        bars = candles_by_tf.get("5") or candles_by_tf.get("1") or []
        if len(bars) < 21:
            return (0.0, {"5": 0.0}, "Intraday", {}, [])
        closes = [float(c["close"]) for c in bars[-20:]]
        cur = float(bars[-1]["close"])
        mean = sum(closes) / len(closes)
        if cur > mean * 1.001:
            return (20.0, {"5": 3.0, "1": 3.0}, "Intraday",
                    {"sig": 1.0}, ["sig"])
        return (0.0, {"5": 0.0}, "Intraday", {}, [])

    # Synthetic monotonic uptrend — 300 bars rising 0.5% per bar.
    base = 100.0
    candles = []
    for i in range(300):
        price = base * (1.0 + 0.005 * i)
        candles.append({
            "timestamp": 1_700_000_000_000 + i * 60_000,
            "open":  price * 0.999,
            "high":  price * 1.003,
            "low":   price * 0.997,
            "close": price,
            "volume": 1000.0 + i,
        })
    data = {"BTCUSDT": {"1": candles, "5": candles, "15": candles, "60": candles}}

    # Run A: zero costs.
    cfg_free = BacktestConfig(
        starting_balance=1000.0, risk_per_trade_pct=2.5,
        taker_fee_pct=0.0, slippage_bps=0.0,
        min_scalp_score=8.0, min_intraday_score=8.0, min_swing_score=8.0,
        warmup_bars=25,
    )
    engine_free = BacktestEngine(cfg_free, score_fn=trivial_scorer)
    m_free = engine_free.run(data, primary_tf="5", btc_symbol="BTCUSDT")

    # Run B: realistic costs.
    cfg_real = BacktestConfig(
        starting_balance=1000.0, risk_per_trade_pct=2.5,
        taker_fee_pct=0.055, slippage_bps=5.0,
        min_scalp_score=8.0, min_intraday_score=8.0, min_swing_score=8.0,
        warmup_bars=25,
    )
    engine_real = BacktestEngine(cfg_real, score_fn=trivial_scorer)
    m_real = engine_real.run(data, primary_tf="5", btc_symbol="BTCUSDT")

    assert_(m_free["total"] > 0, "expected at least one trade in zero-cost run")
    assert_(m_real["total"] > 0, "expected at least one trade in realistic-cost run")
    assert_(
        m_free["expectancy_pct"] > m_real["expectancy_pct"],
        f"fee/slippage drag missing: free={m_free['expectancy_pct']:.3f}% "
        f"real={m_real['expectancy_pct']:.3f}%",
    )
    drag = m_free["expectancy_pct"] - m_real["expectancy_pct"]
    print(f"  [PASS] free expectancy {m_free['expectancy_pct']:+.3f}% / "
          f"real {m_real['expectancy_pct']:+.3f}% / drag {drag:.3f}% per trade")


async def scenario_pagination_assembles_history(main, mock: BybitMock) -> None:
    """Phase 5 pagination: fetch_history_paginated must walk backwards via
    Bybit's `end` param and assemble distinct bars, no duplicates."""
    print("\n---- SCENARIO 13: pagination assembles history correctly ---")
    import bybit_api
    from backtest_engine import fetch_history_paginated

    # Build a deterministic "server" with 350 bars, 1 minute apart.
    base_ts = 1_700_000_000_000  # ms
    server_bars = [
        [base_ts + i * 60_000, str(100 + i * 0.5), str(101 + i * 0.5),
         str(99 + i * 0.5), str(100.5 + i * 0.5), str(1000), str(0)]
        for i in range(350)
    ]
    # Bybit returns newest-first, so reverse for the mock to mimic that.
    server_bars_newest_first = list(reversed(server_bars))

    call_count = {"n": 0}

    async def fake_signed_request(method, endpoint, params=None):
        params = params or {}
        if endpoint != "/v5/market/kline":
            return {"retCode": 0, "result": {"list": []}}
        call_count["n"] += 1
        limit = int(params.get("limit", 200))
        end_ms = params.get("end")
        # Apply the `end` filter: keep only bars whose timestamp <= end_ms.
        if end_ms is not None:
            cutoff = int(end_ms)
            bars = [b for b in server_bars_newest_first if int(b[0]) <= cutoff]
        else:
            bars = list(server_bars_newest_first)
        # Return the most recent `limit` from the filtered set.
        out = bars[:limit]
        return {"retCode": 0, "result": {"list": out}}

    # Patch signed_request module-globally so fetch_history_paginated picks it up.
    original = bybit_api.signed_request
    bybit_api.signed_request = fake_signed_request
    try:
        # Request 300 bars — should require 2 pages (200 + 100).
        result = await fetch_history_paginated(
            symbol="TESTUSDT", interval="1", total_bars=300,
            category="linear", rate_limit_sleep=0.0,
        )
    finally:
        bybit_api.signed_request = original

    assert_(len(result) == 300, f"expected 300 bars, got {len(result)}")
    assert_(call_count["n"] >= 2, f"expected >=2 paginated requests, got {call_count['n']}")

    # Verify ordering (oldest-first) and no duplicates.
    timestamps = [c["timestamp"] for c in result]
    assert_(timestamps == sorted(timestamps), "result not sorted ascending")
    assert_(len(set(timestamps)) == len(timestamps), "duplicate timestamps in result")

    # Verify the assembled span is contiguous (each bar = 60000 ms apart).
    deltas = {timestamps[i+1] - timestamps[i] for i in range(len(timestamps) - 1)}
    assert_(deltas == {60_000}, f"non-contiguous bars; gap set {deltas}")
    print(f"  [PASS] pagination: {len(result)} contiguous bars across {call_count['n']} requests")


async def scenario_engine_skips_empty_anchor(main, mock: BybitMock) -> None:
    """Bug A regression test: when the FIRST symbol's primary TF is empty
    (e.g. cached as [] from a failed fetch), the engine must still run on
    the remaining symbols rather than bailing immediately."""
    print("\n---- SCENARIO 14: engine handles missing primary TF for first sym ---")
    from backtest_engine import BacktestConfig, BacktestEngine

    def trivial_scorer(symbol, candles_by_tf, trend_context):
        bars = candles_by_tf.get("5", [])
        if len(bars) < 11:
            return (0.0, {"5": 0.0}, "Intraday", {}, [])
        closes = [float(c["close"]) for c in bars[-10:]]
        cur = float(bars[-1]["close"])
        if cur >= max(closes):
            return (15.0, {"5": 3.0, "1": 3.0}, "Intraday",
                    {"breakout": 1.0}, ["breakout"])
        return (0.0, {"5": 0.0}, "Intraday", {}, [])

    # Build an uptrend candle set, then make the first symbol's 5m an empty
    # list (simulating the bug scenario).
    import random
    random.seed(123)
    base = 50.0
    bars = []
    price = base
    for i in range(150):
        price *= 1.004 + random.uniform(-0.001, 0.001)
        bars.append({
            "timestamp": 1_700_000_000_000 + i * 60_000,
            "open": price * 0.999, "high": price * 1.004,
            "low": price * 0.997, "close": price, "volume": 1000.0 + i,
        })

    # BTCUSDT is first but has NO 5m data — simulating Bug A.
    # ETHUSDT has the full set.
    data = {
        "BTCUSDT": {"1": [], "5": [], "15": bars, "60": bars},
        "ETHUSDT": {"1": bars, "5": bars, "15": bars, "60": bars},
    }

    cfg = BacktestConfig(
        starting_balance=1000.0, risk_per_trade_pct=2.0,
        taker_fee_pct=0.055, slippage_bps=5.0,
        min_scalp_score=8.0, min_intraday_score=8.0, min_swing_score=8.0,
        warmup_bars=25,
    )
    engine = BacktestEngine(cfg, score_fn=trivial_scorer)
    m = engine.run(data, primary_tf="5", btc_symbol="BTCUSDT")

    # Before the fix this returned total=0 because BTCUSDT was the anchor.
    # After the fix, ETHUSDT's data drives the timeline.
    assert_(m["total"] > 0,
            f"engine must process other symbols when first symbol has empty primary TF; "
            f"got {m['total']} trades")
    print(f"  [PASS] processed {m['total']} trade(s) on ETHUSDT despite empty BTCUSDT 5m")


async def scenario_no_cache_empty_results(main, mock: BybitMock) -> None:
    """Bug B regression: load_or_fetch_history must NOT write a cache file
    when the fetch returned an empty list. Otherwise subsequent runs read
    the empty cache and skip the fetch forever."""
    print("\n---- SCENARIO 15: empty fetch results aren't cached --------")
    import bybit_api
    from backtest_engine import load_or_fetch_history, CACHE_DIR
    from pathlib import Path

    # Pick a name that won't collide with real caches.
    test_symbol = "EMPTYCACHETESTUSDT"
    test_interval = "1"
    test_limit = 500

    # Pre-clean any stale cache.
    cache_file = Path(CACHE_DIR) / f"{test_symbol}_linear_{test_interval}_{test_limit}.json"
    if cache_file.exists():
        cache_file.unlink()

    # Mock signed_request to always return retCode=0 with empty list.
    async def empty_signed_request(method, endpoint, params=None):
        if endpoint == "/v5/market/kline":
            return {"retCode": 0, "result": {"list": []}}
        return {"retCode": 0, "result": {}}

    original = bybit_api.signed_request
    bybit_api.signed_request = empty_signed_request
    try:
        result = await load_or_fetch_history(
            symbol=test_symbol, interval=test_interval,
            limit=test_limit, use_cache=False,
        )
    finally:
        bybit_api.signed_request = original

    assert_(result == [], f"expected empty result; got {len(result)} bars")
    assert_(not cache_file.exists(),
            f"empty fetch should NOT have written cache file {cache_file}")
    print(f"  [PASS] empty fetch result not persisted to cache")


async def scenario_backtest_uptrend_profits(main, mock: BybitMock) -> None:
    """Phase 5 sanity check: a monotonic uptrend with realistic costs and
    a trivial breakout scorer must produce positive expectancy and
    end with final balance > starting balance. If this fails, the engine
    isn't capturing trends correctly."""
    print("\n---- SCENARIO 12: backtest captures uptrend profit -------")
    from backtest_engine import BacktestConfig, BacktestEngine

    def trivial_scorer(symbol, candles_by_tf, trend_context):
        bars = candles_by_tf.get("5", [])
        if len(bars) < 11:
            return (0.0, {"5": 0.0}, "Intraday", {}, [])
        # Score Long when current close > 10-bar high (breakout).
        closes = [float(c["close"]) for c in bars[-10:]]
        cur = float(bars[-1]["close"])
        if cur >= max(closes):
            return (15.0, {"5": 3.0, "1": 3.0}, "Intraday",
                    {"breakout": 1.0}, ["breakout"])
        return (0.0, {"5": 0.0}, "Intraday", {}, [])

    # Strong uptrend with mild noise — designed so SL gets touched rarely
    # but TP1 (2%) gets hit on most trades.
    import random
    random.seed(42)
    base = 100.0
    candles = []
    price = base
    for i in range(300):
        price *= 1.005 + random.uniform(-0.002, 0.002)
        candles.append({
            "timestamp": 1_700_000_000_000 + i * 60_000,
            "open": price * (1 + random.uniform(-0.001, 0.001)),
            "high": price * (1 + random.uniform(0.001, 0.005)),
            "low":  price * (1 - random.uniform(0.001, 0.003)),
            "close": price,
            "volume": 1000.0 + i,
        })
    data = {"BTCUSDT": {"1": candles, "5": candles, "15": candles, "60": candles}}

    cfg = BacktestConfig(
        starting_balance=1000.0, risk_per_trade_pct=2.0,
        taker_fee_pct=0.055, slippage_bps=5.0,
        min_scalp_score=8.0, min_intraday_score=8.0, min_swing_score=8.0,
        warmup_bars=25,
    )
    engine = BacktestEngine(cfg, score_fn=trivial_scorer)
    m = engine.run(data, primary_tf="5", btc_symbol="BTCUSDT")

    assert_(m["total"] > 0, "expected trades in clear uptrend")
    assert_(m["expectancy_pct"] > 0,
            f"expected positive expectancy in clean uptrend; got {m['expectancy_pct']:+.3f}%")
    assert_(m["final_balance"] > m["starting_balance"],
            f"final balance should exceed start; {m['final_balance']} vs {m['starting_balance']}")
    assert_(m["win_rate"] > 0.5,
            f"win rate should be > 50% in clean uptrend; got {m['win_rate'] * 100:.1f}%")
    print(f"  [PASS] uptrend: {m['total']} trades, win_rate={m['win_rate'] * 100:.1f}%, "
          f"expectancy={m['expectancy_pct']:+.3f}%, total={m['total_pnl_pct']:+.2f}%")


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
        ("happy_path",                  scenario_happy_path),
        ("sl_failure_emergency_close",  scenario_sl_failure_emergency_close),
        ("scalp_hunter_qty",            scenario_scalp_hunter_qty),
        # Phase 2 exit-path scenarios
        ("force_exit",                  scenario_force_exit),
        ("exit_cancels_residual_orders", scenario_exit_cancels_residual_orders),
        ("tp1_via_fill_detection",      scenario_tp1_via_fill_detection),
        ("sl_update_via_amend",         scenario_sl_update_via_amend),
        # Phase 3 websocket / staleness scenarios
        ("candle_dedup",                scenario_candle_dedup),
        ("staleness_guard",             scenario_staleness_guard),
        ("supervisor_restart",          scenario_supervisor_restart),
        # Phase 5 backtest engine scenarios
        ("backtest_fees_and_slippage",   scenario_backtest_fees_and_slippage),
        ("backtest_uptrend_profits",     scenario_backtest_uptrend_profits),
        ("pagination_assembles_history", scenario_pagination_assembles_history),
        ("engine_skips_empty_anchor",    scenario_engine_skips_empty_anchor),
        ("no_cache_empty_results",       scenario_no_cache_empty_results),
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
