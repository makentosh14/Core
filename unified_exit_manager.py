#!/usr/bin/env python3
"""
unified_exit_manager.py  —  Phase 2 audit rewrite

Strategy: TP1 (exchange auto-fires) -> Breakeven -> Trailing -> Exit.

KEY ARCHITECTURAL DECISIONS (vs. prior version):

  * TP1 detection is now FILL-BASED (Phase 2 Fix #1).
    The exchange's conditional Market TP1 order auto-sells 50% when triggered.
    We detect this by polling /v5/position/list and noticing size dropped.
    PnL-based detection still acts as an early gate to limit the call rate.
    On detection we re-anchor trade["qty"] to the actual remaining size.

  * The bot no longer runs its OWN initial stop-loss check (Phase 2 Fix #5).
    The exchange-side conditional SL is the source of truth for the
    initial-SL trigger; running both fires twice on every stop, on different
    prices (Mark vs Last), and causes reduceOnly errors.
    Bot's role is now: TP1 detection -> breakeven move -> trailing.

  * SL updates use /v5/order/amend (Phase 2 Fix #4).
    Atomic. No cancel-then-place naked window. If amend fails (e.g. the
    underlying order doesn't exist anymore), fall back to PLACE-FIRST,
    THEN-CANCEL order — never the other way.

  * SL updates are AWAITED, not fire-and-forget (Phase 2 Fix #3).
    Per-symbol asyncio.Lock serializes mutations so we can't race-place
    two SLs on the same symbol.

  * On manual exit, sl_order_id AND tp1_order_id are cancelled (Phase 2 Fix #2).
    Otherwise stale conditional orders accumulate against Bybit's 10-order limit.

  * Close-verification: after _close_on_exchange we poll position size
    until it reaches 0 or we exhaust retries (Phase 2 Fix #13).

  * force_exit kwarg actually works now (Phase 2 Fix #8).

  * Trailing high/low and breakeven persisted to trade dict on every change
    (Phase 2 Fix #9). On restart, recovery picks up the true peak.

  * State transitions (TP1 hit, breakeven set, exit) force an immediate save,
    bypassing the 5-second throttle that previously dropped them on crash
    (Phase 2 Fix #10).

  * Breakeven buffer raised to 0.25% (Phase 2 Fix #6) to cover the
    Bybit taker round-trip (2 * 0.055% = 0.11%) plus slippage.

  * Emoji thresholds in the exit notification account for fees (Fix #14).

DEFERRED (Phase 2 Fix #7 — scalp trailing whip): The 0.5% scalp trail with
10-second poll cadence is structurally lagged. The proper fix is a websocket
subscription on position events; that's a bigger change than this PR. The
breakeven buffer bump (#6) mitigates the worst of the structural loss.
"""

import asyncio
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from logger import log

# Dependency injection kept for backwards compatibility with monitor.py
_save_trades_func: Optional[Callable] = None
_update_sl_func: Optional[Callable] = None  # unused after Fix #4 rewrite


def set_dependencies(save_func: Callable = None, update_sl_func: Callable = None):
    """Set dependencies to avoid circular imports."""
    global _save_trades_func, _update_sl_func
    _save_trades_func = save_func
    _update_sl_func = update_sl_func
    log("✅ Unified exit manager dependencies set")


# Fee-aware exit configuration. Bybit USDT linear taker fee = 0.055%.
# Round trip = 0.11%. Add 0.14% safety margin -> 0.25% breakeven buffer.
EXIT_CONFIG = {
    "Scalp":         {"sl_pct": 0.8, "tp1_trigger_pct": 1.2, "trailing_pct": 0.5, "breakeven_buffer": 0.25},
    "Intraday":      {"sl_pct": 1.0, "tp1_trigger_pct": 2.0, "trailing_pct": 0.8, "breakeven_buffer": 0.25},
    "Swing":         {"sl_pct": 1.5, "tp1_trigger_pct": 3.5, "trailing_pct": 1.2, "breakeven_buffer": 0.3},
    "CoreScalp":     {"sl_pct": 0.8, "tp1_trigger_pct": 1.2, "trailing_pct": 0.5, "breakeven_buffer": 0.25},
    "CoreIntraday":  {"sl_pct": 1.0, "tp1_trigger_pct": 2.0, "trailing_pct": 0.8, "breakeven_buffer": 0.25},
    "CoreSwing":     {"sl_pct": 1.5, "tp1_trigger_pct": 3.5, "trailing_pct": 1.2, "breakeven_buffer": 0.3},
    "Default":       {"sl_pct": 1.0, "tp1_trigger_pct": 2.0, "trailing_pct": 0.8, "breakeven_buffer": 0.25},
}

# Fee assumption used by notification thresholds. Two-sided taker.
ROUND_TRIP_FEE_PCT = 0.11


class UnifiedExitManager:
    """Exit pipeline: Entry -> TP1 (exchange auto) -> Breakeven -> Trailing -> Exit."""

    def __init__(self):
        self.trailing_highs: Dict[str, float] = {}
        self.trailing_lows: Dict[str, float] = {}
        self.tp1_activated: Dict[str, bool] = {}
        self.breakeven_prices: Dict[str, float] = {}
        self.last_save: Dict[str, float] = {}
        self.last_sl_update: Dict[str, float] = {}

        # Per-symbol locks serialize SL mutations so concurrent trailing
        # updates can't race-place two SLs (Phase 2 Fix #3).
        self._sl_locks: Dict[str, asyncio.Lock] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Lookups
    # ─────────────────────────────────────────────────────────────────────

    def get_config(self, trade: Dict) -> Dict:
        trade_type = (
            trade.get("strategy_type")
            or trade.get("trade_type")
            or trade.get("type")
            or "Default"
        )
        if trade_type not in EXIT_CONFIG:
            trade_type = trade_type.replace("Core", "")
        if trade_type not in EXIT_CONFIG:
            trade_type = "Default"
        return EXIT_CONFIG[trade_type]

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        lock = self._sl_locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._sl_locks[symbol] = lock
        return lock

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    async def process_exit(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        force_exit: bool = False,
    ) -> bool:
        """Returns True if the trade was exited this tick."""
        try:
            if trade.get("exited"):
                return False

            direction = trade.get("direction", "").lower()
            entry_price = float(trade.get("entry_price", 0))
            if not direction or not entry_price or not current_price:
                return False

            # Restore in-memory state from saved trade dict on restart (Fix #9).
            self._restore_state(symbol, trade, current_price, direction)

            config = self.get_config(trade)

            if direction == "long":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            trade["current_pnl_pct"] = round(pnl_pct, 2)
            trade["current_price"] = current_price
            trade["last_check"] = datetime.now().isoformat()

            # Force-exit short-circuits all logic (Fix #8).
            if force_exit:
                await self._exit_trade(symbol, trade, current_price, "force_exit", pnl_pct)
                return True

            # STEP 1: Breakeven SL check (only after TP1 has activated).
            # The initial-SL check is NOT here anymore (Fix #5): the
            # exchange's conditional SL handles it. Running both creates
            # double-fire / reduceOnly errors.
            if self.tp1_activated.get(symbol, False):
                if await self._check_breakeven_stop(symbol, trade, current_price, pnl_pct, direction):
                    return True

            # STEP 2: TP1 detection — fill-based first, PnL-based as gate (Fix #1).
            await self._check_tp1_trigger(symbol, trade, current_price, pnl_pct, config, direction, entry_price)

            # STEP 3: Trailing stop check (only if TP1 activated).
            if self.tp1_activated.get(symbol, False):
                if await self._check_trailing_stop(symbol, trade, current_price, pnl_pct, config, direction):
                    return True
                # STEP 4: Bump trailing high/low and push to exchange.
                await self._update_trailing_level(symbol, trade, current_price, direction, config)

            self._save_state(symbol, trade)
            return False

        except Exception as e:
            log(f"❌ Exit manager error for {symbol}: {e}", level="ERROR")
            log(traceback.format_exc(), level="ERROR")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Restart recovery — pull persisted state back into memory (Fix #9)
    # ─────────────────────────────────────────────────────────────────────

    def _restore_state(self, symbol: str, trade: Dict, current_price: float, direction: str):
        if trade.get("tp1_hit") and not self.tp1_activated.get(symbol, False):
            self.tp1_activated[symbol] = True

            if trade.get("breakeven_sl"):
                self.breakeven_prices[symbol] = float(trade["breakeven_sl"])
            else:
                config = self.get_config(trade)
                entry_price = float(trade.get("entry_price", 0))
                buffer_pct = config["breakeven_buffer"]
                if direction == "long":
                    self.breakeven_prices[symbol] = entry_price * (1 + buffer_pct / 100)
                else:
                    self.breakeven_prices[symbol] = entry_price * (1 - buffer_pct / 100)

            # Prefer the persisted true peak (Fix #9), falling back to tp1_hit_price
            # then current_price.
            if direction == "long":
                peak = trade.get("trailing_high") or trade.get("tp1_hit_price") or current_price
                self.trailing_highs[symbol] = max(float(peak), current_price)
            else:
                trough = trade.get("trailing_low") or trade.get("tp1_hit_price") or current_price
                self.trailing_lows[symbol] = min(float(trough), current_price)

            log(f"🔄 {symbol}: Restored TP1/trailing state from trade dict")

    # ─────────────────────────────────────────────────────────────────────
    # Breakeven SL check (post-TP1 only)
    # ─────────────────────────────────────────────────────────────────────

    async def _check_breakeven_stop(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        pnl_pct: float,
        direction: str,
    ) -> bool:
        breakeven_sl = self.breakeven_prices.get(symbol)
        if breakeven_sl is None:
            return False

        if direction == "long" and current_price <= breakeven_sl:
            await self._exit_trade(symbol, trade, current_price, "breakeven_stop", pnl_pct)
            return True
        if direction == "short" and current_price >= breakeven_sl:
            await self._exit_trade(symbol, trade, current_price, "breakeven_stop", pnl_pct)
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # TP1 detection — fill-based with PnL gate (Fix #1)
    # ─────────────────────────────────────────────────────────────────────

    async def _check_tp1_trigger(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        pnl_pct: float,
        config: Dict,
        direction: str,
        entry_price: float,
    ):
        if self.tp1_activated.get(symbol, False):
            return

        tp1_pct = config["tp1_trigger_pct"]

        # Easy path: PnL is currently at or above TP1 level.
        if pnl_pct >= tp1_pct:
            await self._activate_tp1(symbol, trade, current_price, config, direction, entry_price)
            return

        # Wick path: PnL is below TP1 now, but a transient spike could have
        # filled the conditional TP1 order. Only check fill if we're at
        # least halfway there — keeps API rate down.
        if pnl_pct >= tp1_pct * 0.5 and trade.get("tp1_order_id"):
            filled, remaining_size = await self._tp1_filled_on_exchange(symbol, trade)
            if filled:
                # Re-anchor qty to the actual remaining size (Fix #12).
                original_qty = float(trade.get("original_qty", trade.get("qty", 0)))
                if not trade.get("original_qty"):
                    trade["original_qty"] = original_qty
                trade["qty"] = remaining_size
                # Use the TP1 target as the fill price proxy.
                tp1_target = float(trade.get("tp1_target", current_price))
                await self._activate_tp1(symbol, trade, tp1_target, config, direction, entry_price)

    async def _tp1_filled_on_exchange(self, symbol: str, trade: Dict):
        """Return (filled: bool, remaining_position_size: float)."""
        try:
            from bybit_api import signed_request
            resp = await signed_request("GET", "/v5/position/list", {
                "category": "linear",
                "symbol": symbol,
            })
            if resp.get("retCode") != 0:
                return False, 0.0
            for pos in resp.get("result", {}).get("list", []):
                if pos.get("symbol") != symbol:
                    continue
                current_size = float(pos.get("size", 0) or 0)
                original_qty = float(trade.get("original_qty", trade.get("qty", 0)))
                if original_qty <= 0:
                    return False, current_size
                # TP1 closes 50%; allow 30% margin for rounding/precision steps.
                if current_size < original_qty * 0.7:
                    return True, current_size
                return False, current_size
            return False, 0.0
        except Exception as e:
            log(f"⚠️ {symbol}: _tp1_filled_on_exchange error: {e}", level="WARN")
            return False, 0.0

    async def _activate_tp1(
        self,
        symbol: str,
        trade: Dict,
        fill_price: float,
        config: Dict,
        direction: str,
        entry_price: float,
    ):
        self.tp1_activated[symbol] = True
        trade["tp1_hit"] = True
        trade["tp1_hit_time"] = datetime.now().isoformat()
        trade["tp1_hit_price"] = fill_price

        buffer_pct = config["breakeven_buffer"]
        if direction == "long":
            breakeven_price = entry_price * (1 + buffer_pct / 100)
        else:
            breakeven_price = entry_price * (1 - buffer_pct / 100)
        breakeven_price = round(breakeven_price, 6)

        self.breakeven_prices[symbol] = breakeven_price
        trade["breakeven_sl"] = breakeven_price

        # Initialize trailing reference from the fill price.
        if direction == "long":
            self.trailing_highs[symbol] = fill_price
            trade["trailing_high"] = fill_price
        else:
            self.trailing_lows[symbol] = fill_price
            trade["trailing_low"] = fill_price

        log(f"🎯 {symbol}: TP1 HIT at fill ~{fill_price:.6f}!")
        log(f"   → SL moved to breakeven: {breakeven_price:.6f}  (buffer {buffer_pct:.2f}%)")
        log(f"   → Trailing stop ACTIVATED from {fill_price:.6f}")

        # Force-save the transition immediately (Fix #10), then push SL.
        self._force_save()
        await self._update_exchange_sl(symbol, trade, breakeven_price)

    # ─────────────────────────────────────────────────────────────────────
    # Trailing stop
    # ─────────────────────────────────────────────────────────────────────

    async def _check_trailing_stop(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        pnl_pct: float,
        config: Dict,
        direction: str,
    ) -> bool:
        trailing_pct = config["trailing_pct"]
        if direction == "long":
            trailing_high = self.trailing_highs.get(symbol, current_price)
            trailing_stop = trailing_high * (1 - trailing_pct / 100)
            if current_price <= trailing_stop:
                log(f"📉 {symbol}: Trailing stop hit | high={trailing_high:.6f} stop={trailing_stop:.6f} px={current_price:.6f}")
                await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                return True
        else:
            trailing_low = self.trailing_lows.get(symbol, current_price)
            trailing_stop = trailing_low * (1 + trailing_pct / 100)
            if current_price >= trailing_stop:
                log(f"📈 {symbol}: Trailing stop hit | low={trailing_low:.6f} stop={trailing_stop:.6f} px={current_price:.6f}")
                await self._exit_trade(symbol, trade, current_price, "trailing_stop", pnl_pct)
                return True
        return False

    async def _update_trailing_level(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        direction: str,
        config: Dict,
    ):
        trailing_pct = config["trailing_pct"]

        if direction == "long":
            old_high = self.trailing_highs.get(symbol, current_price)
            if current_price > old_high:
                self.trailing_highs[symbol] = current_price
                trade["trailing_high"] = current_price  # persist true peak (Fix #9)
                new_trailing_sl = round(current_price * (1 - trailing_pct / 100), 6)
                trade["trailing_sl"] = new_trailing_sl
                log(f"📈 {symbol}: New trailing high: {old_high:.6f} → {current_price:.6f} | new SL {new_trailing_sl:.6f}")
                if self._should_push_sl(symbol):
                    await self._update_exchange_sl(symbol, trade, new_trailing_sl)
        else:
            old_low = self.trailing_lows.get(symbol, current_price)
            if current_price < old_low:
                self.trailing_lows[symbol] = current_price
                trade["trailing_low"] = current_price  # persist true trough (Fix #9)
                new_trailing_sl = round(current_price * (1 + trailing_pct / 100), 6)
                trade["trailing_sl"] = new_trailing_sl
                log(f"📉 {symbol}: New trailing low: {old_low:.6f} → {current_price:.6f} | new SL {new_trailing_sl:.6f}")
                if self._should_push_sl(symbol):
                    await self._update_exchange_sl(symbol, trade, new_trailing_sl)

    def _should_push_sl(self, symbol: str) -> bool:
        """Throttle exchange SL updates to once per 10s per symbol."""
        now = time.time()
        if now - self.last_sl_update.get(symbol, 0) >= 10:
            self.last_sl_update[symbol] = now
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # Atomic SL update via /v5/order/amend  (Phase 2 Fix #4)
    # ─────────────────────────────────────────────────────────────────────

    async def _update_exchange_sl(self, symbol: str, trade: Dict, new_sl: float) -> bool:
        """
        Atomically move the SL on the exchange. Tries /v5/order/amend first
        (single atomic call). If that fails, falls back to PLACE-FIRST then
        CANCEL-OLD — never the inverse, to eliminate the naked-position window.
        Returns True on success, False on failure (caller can decide to retry).
        """
        async with self._get_lock(symbol):
            try:
                from bybit_api import signed_request, place_stop_loss_with_retry

                direction = trade.get("direction", "").lower()
                sl_order_id = trade.get("sl_order_id")

                if not direction:
                    log(f"⚠️ {symbol}: cannot update SL — missing direction")
                    return False

                # Path 1: amend the existing SL conditional order in place.
                if sl_order_id:
                    amend_resp = await signed_request("POST", "/v5/order/amend", {
                        "category": "linear",
                        "symbol": symbol,
                        "orderId": sl_order_id,
                        "triggerPrice": str(new_sl),
                    })
                    if amend_resp.get("retCode") == 0:
                        trade["sl"] = new_sl
                        log(f"✅ {symbol}: SL amended to {new_sl:.6f} (orderId={sl_order_id})")
                        self._force_save()
                        return True
                    # Amend failed (order doesn't exist, was filled, etc.) — fall through.
                    log(f"⚠️ {symbol}: SL amend failed ({amend_resp.get('retMsg')}); placing fresh SL")

                # Path 2: place-first, cancel-old. Order matters: never cancel
                # before the new SL is confirmed in place.
                actual_size = await self._actual_position_size(symbol)
                qty = actual_size if actual_size > 0 else float(trade.get("qty", 0) or 0)
                if qty <= 0:
                    log(f"⚠️ {symbol}: cannot place new SL — position size is 0")
                    return False

                place_resp = await place_stop_loss_with_retry(
                    symbol=symbol,
                    direction=direction,
                    qty=qty,
                    sl_price=new_sl,
                )
                if place_resp.get("retCode") != 0:
                    log(f"❌ {symbol}: SL replacement placement failed: {place_resp.get('retMsg')}", level="ERROR")
                    return False

                new_id = place_resp.get("result", {}).get("orderId")
                # New SL is now in place — safe to cancel the old one.
                if sl_order_id and sl_order_id != new_id:
                    cancel_resp = await signed_request("POST", "/v5/order/cancel", {
                        "category": "linear",
                        "symbol": symbol,
                        "orderId": sl_order_id,
                    })
                    if cancel_resp.get("retCode") != 0:
                        # Old order may already be filled/cancelled. Non-fatal.
                        log(f"⚠️ {symbol}: could not cancel old SL ({sl_order_id}): {cancel_resp.get('retMsg')}")

                trade["sl_order_id"] = new_id
                trade["sl"] = new_sl
                log(f"✅ {symbol}: SL replaced; new orderId={new_id} @ {new_sl:.6f}")
                self._force_save()
                return True

            except Exception as e:
                log(f"❌ {symbol}: _update_exchange_sl error: {e}", level="ERROR")
                log(traceback.format_exc(), level="ERROR")
                return False

    async def _actual_position_size(self, symbol: str) -> float:
        try:
            from bybit_api import signed_request
            resp = await signed_request("GET", "/v5/position/list", {
                "category": "linear",
                "symbol": symbol,
            })
            if resp.get("retCode") != 0:
                return 0.0
            for pos in resp.get("result", {}).get("list", []):
                if pos.get("symbol") == symbol:
                    return float(pos.get("size", 0) or 0)
            return 0.0
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Exit pipeline
    # ─────────────────────────────────────────────────────────────────────

    async def _exit_trade(
        self,
        symbol: str,
        trade: Dict,
        current_price: float,
        reason: str,
        pnl_pct: float,
    ):
        log(f"🚪 EXIT: {symbol} | reason={reason} | pnl={pnl_pct:+.2f}%")

        trade["exited"] = True
        trade["exit_reason"] = reason
        trade["exit_price"] = current_price
        trade["exit_time"] = datetime.now().isoformat()
        trade["final_pnl_pct"] = round(pnl_pct, 2)

        entry_price = float(trade.get("entry_price", 0))
        qty = float(trade.get("qty", 0))
        direction = trade.get("direction", "").lower()
        if entry_price > 0 and qty > 0:
            if direction == "long":
                pnl_usdt = (current_price - entry_price) * qty
            else:
                pnl_usdt = (entry_price - current_price) * qty
            trade["final_pnl_usdt"] = round(pnl_usdt, 2)
            log(f"💰 P&L: {pnl_usdt:+.2f} USDT ({pnl_pct:+.2f}%)")

        self._cleanup(symbol)

        # Cancel any remaining conditional orders BEFORE closing position (Fix #2).
        # Otherwise stale SL/TP orders accumulate against Bybit's 10-order limit.
        await self._cancel_remaining_orders(symbol, trade)

        # Close position; verify it actually closed (Fix #13).
        await self._close_on_exchange(symbol, trade)

        await self._send_notification(symbol, trade, reason, pnl_pct)
        self._force_save()

    async def _cancel_remaining_orders(self, symbol: str, trade: Dict):
        """Cancel SL and TP1 conditional orders so they don't linger after exit (Fix #2)."""
        try:
            from bybit_api import signed_request
            for key in ("sl_order_id", "tp1_order_id"):
                oid = trade.get(key)
                if not oid:
                    continue
                resp = await signed_request("POST", "/v5/order/cancel", {
                    "category": "linear",
                    "symbol": symbol,
                    "orderId": oid,
                })
                if resp.get("retCode") == 0:
                    log(f"🧹 {symbol}: cancelled {key}={oid}")
                else:
                    # Non-fatal: order may have already filled.
                    log(f"⚠️ {symbol}: could not cancel {key}={oid}: {resp.get('retMsg')}")
        except Exception as e:
            log(f"⚠️ {symbol}: error cancelling residual orders: {e}", level="WARN")

    async def _close_on_exchange(self, symbol: str, trade: Dict):
        """Market reduceOnly close, then VERIFY position size == 0 (Fix #13)."""
        try:
            from bybit_api import signed_request

            direction = trade.get("direction", "").lower()
            qty = trade.get("qty", 0)
            if not qty:
                return

            side = "Sell" if direction == "long" else "Buy"
            resp = await signed_request("POST", "/v5/order/create", {
                "category": "linear",
                "symbol": symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(abs(float(qty))),
                "timeInForce": "IOC",
                "reduceOnly": True,
            })

            if resp.get("retCode") != 0:
                log(f"❌ {symbol}: market close rejected: {resp.get('retMsg')}", level="ERROR")
                await self._alert_stuck_position(symbol, trade, reason="market close rejected")
                return

            # Verify position actually went to 0. Three quick polls.
            for attempt in range(3):
                await asyncio.sleep(0.5)
                remaining = await self._actual_position_size(symbol)
                if remaining == 0:
                    log(f"✅ {symbol}: close confirmed (size=0)")
                    return
                log(f"⏳ {symbol}: post-close size still {remaining}, retry {attempt+1}/3")

            # Position still non-zero after retries — alert loudly.
            stuck = await self._actual_position_size(symbol)
            await self._alert_stuck_position(symbol, trade, reason=f"size {stuck} after close")

        except Exception as e:
            log(f"❌ {symbol}: exchange close error: {e}", level="ERROR")

    async def _alert_stuck_position(self, symbol: str, trade: Dict, reason: str):
        try:
            from error_handler import send_error_to_telegram
            msg = (
                f"🚨 {symbol}: position close incomplete — {reason}.\n"
                f"Direction: {trade.get('direction')} | qty: {trade.get('qty')}\n"
                f"Manual intervention may be required."
            )
            await send_error_to_telegram(msg)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────

    def _cleanup(self, symbol: str):
        self.trailing_highs.pop(symbol, None)
        self.trailing_lows.pop(symbol, None)
        self.tp1_activated.pop(symbol, None)
        self.breakeven_prices.pop(symbol, None)
        self.last_save.pop(symbol, None)
        self.last_sl_update.pop(symbol, None)
        self._sl_locks.pop(symbol, None)

    def _save_state(self, symbol: str, trade: Dict):
        """Throttled save for routine progress updates."""
        now = time.time()
        if now - self.last_save.get(symbol, 0) < 5:
            return
        self.last_save[symbol] = now
        if _save_trades_func:
            _save_trades_func()

    def _force_save(self):
        """Bypass throttle. Used after state transitions (Fix #10)."""
        if _save_trades_func:
            try:
                _save_trades_func()
            except Exception as e:
                log(f"⚠️ force-save failed: {e}", level="WARN")

    # ─────────────────────────────────────────────────────────────────────
    # Notification
    # ─────────────────────────────────────────────────────────────────────

    async def _send_notification(self, symbol: str, trade: Dict, reason: str, pnl_pct: float):
        try:
            from error_handler import send_telegram_message
            # Net of round-trip fee (Fix #14): a "green" exit needs pnl > 2x fees.
            if pnl_pct >= ROUND_TRIP_FEE_PCT + 2.0:
                emoji = "🟢🎉"
            elif pnl_pct > ROUND_TRIP_FEE_PCT:
                emoji = "🟢"
            elif pnl_pct > -ROUND_TRIP_FEE_PCT:
                emoji = "🟡"
            else:
                emoji = "🔴"

            msg  = f"{emoji} <b>TRADE EXIT</b>\n\n"
            msg += f"📊 Symbol: <b>{symbol}</b>\n"
            msg += f"📈 Direction: <b>{trade.get('direction', '').upper()}</b>\n"
            msg += f"🚪 Reason: <b>{reason.replace('_', ' ').title()}</b>\n"
            msg += f"💰 Entry: {trade.get('entry_price')}\n"
            msg += f"🎯 Exit: {trade.get('exit_price')}\n"
            msg += f"💵 PnL (gross): <b>{pnl_pct:+.2f}%</b>"
            if trade.get("final_pnl_usdt"):
                msg += f" ({trade['final_pnl_usdt']:+.2f} USDT)"
            await send_telegram_message(msg)
        except Exception as e:
            log(f"⚠️ Notification error: {e}", level="WARN")

    # ─────────────────────────────────────────────────────────────────────
    # Status / reset
    # ─────────────────────────────────────────────────────────────────────

    def get_status(self, symbol: str) -> Dict:
        return {
            "tp1_activated": self.tp1_activated.get(symbol, False),
            "breakeven_sl": self.breakeven_prices.get(symbol),
            "trailing_high": self.trailing_highs.get(symbol),
            "trailing_low": self.trailing_lows.get(symbol),
        }

    def reset_symbol(self, symbol: str):
        self._cleanup(symbol)
        log(f"🔄 {symbol}: tracking reset")


# Module-level singleton.
exit_manager = UnifiedExitManager()


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC API  —  unchanged signature for backwards compat (Fix #8 adds kwarg)
# ─────────────────────────────────────────────────────────────────────────

async def process_trade_exits(
    symbol: str,
    trade: Dict,
    current_price: float,
    force_exit: bool = False,
) -> bool:
    return await exit_manager.process_exit(symbol, trade, current_price, force_exit=force_exit)


def get_exit_status(symbol: str) -> Dict:
    return exit_manager.get_status(symbol)


def reset_symbol_tracking(symbol: str):
    exit_manager.reset_symbol(symbol)


def get_exit_config(trade_type: str) -> Dict:
    if trade_type not in EXIT_CONFIG:
        trade_type = trade_type.replace("Core", "")
    return EXIT_CONFIG.get(trade_type, EXIT_CONFIG["Default"])


__all__ = [
    "process_trade_exits",
    "get_exit_status",
    "reset_symbol_tracking",
    "get_exit_config",
    "set_dependencies",
    "EXIT_CONFIG",
]


if __name__ == "__main__":
    print("✅ unified_exit_manager.py (Phase 2 audit rewrite) loaded")
    print("\n📊 Exit Configuration:")
    print("=" * 50)
    for trade_type, config in EXIT_CONFIG.items():
        if trade_type.startswith("Core"):
            continue
        print(f"\n{trade_type}:")
        print(f"  SL:               -{config['sl_pct']}%")
        print(f"  TP1 Trigger:      +{config['tp1_trigger_pct']}%")
        print(f"  Trailing:          {config['trailing_pct']}%")
        print(f"  Breakeven Buffer: +{config['breakeven_buffer']}%  (covers {ROUND_TRIP_FEE_PCT}% round-trip fee)")
