"""
backtest_engine.py — Phase 5 historical replay framework.

Components:

  * BacktestConfig — knobs: starting balance, fees, slippage, risk, leverage,
    score gates, exit params per trade type.

  * SimulatedExchange — bar-by-bar broker simulator. Mirrors the production
    exit chain from unified_exit_manager: SL → TP1 (50% partial) → breakeven
    move → trailing → exit. Applies adverse slippage and Bybit taker fees
    at every fill. Maintains an equity curve for drawdown calc.

  * BacktestEngine — iterates historical candles bar-by-bar, calls the
    production scorer to decide entries, defers exit logic to the exchange.
    Lookahead-safe: at iteration time `t`, only candles with close_time <= t
    are visible (primary TF can show open bar; higher TFs only closed bars).

  * compute_metrics() — win rate, profit factor, per-trade Sharpe, max
    drawdown, expectancy, avg winner / loser, exit-reason distribution.

  * load_or_fetch_history() — minimal data loader that uses the live
    fetch_candles_rest with JSON disk caching. Capped at 200 bars per
    request (Bybit V5 limit); no pagination yet.

USAGE:

    cfg = BacktestConfig(starting_balance=1000.0, risk_per_trade_pct=2.5)
    engine = BacktestEngine(cfg)
    candles = {"BTCUSDT": {"1": [...], "5": [...], "15": [...], "60": [...]}}
    metrics = engine.run(candles, primary_tf="5", btc_symbol="BTCUSDT")
    print(metrics)

LIMITATIONS (honest about what this is and isn't):
  * Single-process synchronous run; the production scorer is sync-blocking
    and gets re-invoked every bar. Expect ~10ms/bar/symbol; 1000 bars × 10
    symbols ≈ 100 seconds.
  * 200-bar Bybit API limit means by default you get ~16h of 5m data per
    fetch. Override `limit` or add pagination for longer windows.
  * Slippage model is uniform (constant bps adverse on every fill).
    Real slippage is volume- and book-depth-dependent.
  * Exit chain mirrors unified_exit_manager but doesn't simulate the
    cancel-then-place SL update latency (a real production edge case).
  * Trade type detection trusts the scorer; doesn't simulate the late-bar
    arrival latency of the actual bot.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    starting_balance: float = 1000.0
    risk_per_trade_pct: float = 2.5
    taker_fee_pct: float = 0.055          # one-sided percent (Bybit USDT-M)
    slippage_bps: float = 5.0             # 5 bps = 0.05% adverse on each fill
    leverage: int = 5                     # informational; doesn't change PnL
    max_concurrent_positions: int = 3
    max_position_value_pct_of_balance: float = 25.0

    # Score gates — should match main.py
    min_scalp_score: float = 10.5
    min_intraday_score: float = 12.0
    min_swing_score: float = 15.5

    # Exit params per trade type — should match unified_exit_manager EXIT_CONFIG
    exit_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Scalp":    {"sl_pct": 0.8, "tp1_pct": 1.2, "trailing_pct": 0.5, "breakeven_buffer": 0.25},
        "Intraday": {"sl_pct": 1.0, "tp1_pct": 2.0, "trailing_pct": 0.8, "breakeven_buffer": 0.25},
        "Swing":    {"sl_pct": 1.5, "tp1_pct": 3.5, "trailing_pct": 1.2, "breakeven_buffer": 0.30},
    })

    # Warmup bars to skip at start of backtest so indicators have history.
    warmup_bars: int = 50

    # If True, only enter positions on direction picked by score.determine_direction.
    # If False, accept the trade_type from scorer but assume "Long" (debug only).
    require_clean_direction: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimPosition:
    symbol: str
    direction: str            # "long" / "short"
    trade_type: str           # "Scalp" / "Intraday" / "Swing"
    entry_price: float        # post-slippage
    qty: float                # remaining (drops to qty/2 after TP1)
    original_qty: float
    sl: float
    tp1: float
    sl_pct: float
    tp1_pct: float
    trailing_pct: float
    breakeven_buffer_pct: float
    entry_bar_idx: int
    entry_time_ms: int
    fees_paid: float = 0.0
    realized_pnl: float = 0.0
    tp1_hit: bool = False
    breakeven_sl: Optional[float] = None
    trailing_high: Optional[float] = None
    trailing_low: Optional[float] = None


@dataclass
class SimTrade:
    symbol: str
    direction: str
    trade_type: str
    entry_time_ms: int
    exit_time_ms: int
    entry_price: float        # post-slip
    exit_price: float         # post-slip
    qty: float                # original qty (full size)
    fees_paid: float
    pnl_pct: float            # NET, includes fees, on original notional
    pnl_usd: float
    exit_reason: str
    bars_held: int
    tp1_hit: bool


# ─────────────────────────────────────────────────────────────────────────────
# Simulated exchange
# ─────────────────────────────────────────────────────────────────────────────

INTERVAL_SECONDS = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800, "60": 3600, "240": 14400}


class SimulatedExchange:
    """Stateful broker simulator. Mirrors unified_exit_manager flow."""

    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.balance = cfg.starting_balance
        self.positions: Dict[str, SimPosition] = {}
        self.closed_trades: List[SimTrade] = []
        self.equity_curve: List[float] = [cfg.starting_balance]

    # ─── helpers ──────────────────────────────────────────────────────

    def _apply_slippage(self, price: float, direction: str, side: str) -> float:
        slip = price * (self.cfg.slippage_bps / 10_000.0)
        if side == "open":
            return price + slip if direction == "long" else price - slip
        return price - slip if direction == "long" else price + slip

    # ─── open ─────────────────────────────────────────────────────────

    def open(
        self,
        symbol: str,
        direction: str,
        trade_type: str,
        mark_price: float,
        bar_idx: int,
        time_ms: int,
    ) -> Optional[SimPosition]:
        if symbol in self.positions:
            return None
        if len(self.positions) >= self.cfg.max_concurrent_positions:
            return None

        params = self.cfg.exit_params.get(
            trade_type, self.cfg.exit_params["Intraday"]
        )
        entry_price = self._apply_slippage(mark_price, direction, "open")

        if direction == "long":
            sl = entry_price * (1 - params["sl_pct"] / 100)
            tp1 = entry_price * (1 + params["tp1_pct"] / 100)
        else:
            sl = entry_price * (1 + params["sl_pct"] / 100)
            tp1 = entry_price * (1 - params["tp1_pct"] / 100)

        # Size by risk %.
        risk_amount = self.balance * (self.cfg.risk_per_trade_pct / 100)
        risk_per_unit = abs(entry_price - sl)
        if risk_per_unit <= 0:
            return None
        qty = risk_amount / risk_per_unit

        # 25% balance cap.
        max_pos_value = self.balance * (self.cfg.max_position_value_pct_of_balance / 100)
        if qty * entry_price > max_pos_value:
            qty = max_pos_value / entry_price
        if qty <= 0:
            return None

        # Entry fee.
        entry_fee = qty * entry_price * (self.cfg.taker_fee_pct / 100)

        pos = SimPosition(
            symbol=symbol, direction=direction, trade_type=trade_type,
            entry_price=entry_price, qty=qty, original_qty=qty,
            sl=sl, tp1=tp1,
            sl_pct=params["sl_pct"], tp1_pct=params["tp1_pct"],
            trailing_pct=params["trailing_pct"],
            breakeven_buffer_pct=params["breakeven_buffer"],
            entry_bar_idx=bar_idx, entry_time_ms=time_ms,
            fees_paid=entry_fee,
        )
        self.positions[symbol] = pos
        return pos

    # ─── step (bar-by-bar) ────────────────────────────────────────────

    def step(self, symbol: str, candle: dict, bar_idx: int, time_ms: int) -> Optional[SimTrade]:
        """Process one bar against the symbol's open position.
        Returns the SimTrade if the position closed this bar."""
        pos = self.positions.get(symbol)
        if pos is None:
            return None

        high = float(candle["high"])
        low = float(candle["low"])

        # Order within a bar: worst case for the position.
        if pos.direction == "long":
            # 1. Initial SL fires before TP1 (if TP1 not yet hit).
            if not pos.tp1_hit and low <= pos.sl:
                return self._close(symbol, pos.sl, "stop_loss", bar_idx, time_ms)
            # 2. TP1 hit?
            if not pos.tp1_hit and high >= pos.tp1:
                self._activate_tp1(pos, pos.tp1)
                # In the same bar after TP1, breakeven might already be reached on the down-leg.
                if low <= pos.breakeven_sl:
                    return self._close(symbol, pos.breakeven_sl, "breakeven_stop", bar_idx, time_ms)
            # 3. Trailing once TP1 active.
            if pos.tp1_hit:
                # Update trailing high using THIS bar's high.
                if pos.trailing_high is None or high > pos.trailing_high:
                    pos.trailing_high = high
                trailing_sl = pos.trailing_high * (1 - pos.trailing_pct / 100)
                effective_sl = max(pos.breakeven_sl, trailing_sl)
                if low <= effective_sl:
                    reason = "trailing_stop" if effective_sl > pos.breakeven_sl else "breakeven_stop"
                    return self._close(symbol, effective_sl, reason, bar_idx, time_ms)
        else:  # short
            if not pos.tp1_hit and high >= pos.sl:
                return self._close(symbol, pos.sl, "stop_loss", bar_idx, time_ms)
            if not pos.tp1_hit and low <= pos.tp1:
                self._activate_tp1(pos, pos.tp1)
                if high >= pos.breakeven_sl:
                    return self._close(symbol, pos.breakeven_sl, "breakeven_stop", bar_idx, time_ms)
            if pos.tp1_hit:
                if pos.trailing_low is None or low < pos.trailing_low:
                    pos.trailing_low = low
                trailing_sl = pos.trailing_low * (1 + pos.trailing_pct / 100)
                effective_sl = min(pos.breakeven_sl, trailing_sl)
                if high >= effective_sl:
                    reason = "trailing_stop" if effective_sl < pos.breakeven_sl else "breakeven_stop"
                    return self._close(symbol, effective_sl, reason, bar_idx, time_ms)

        return None

    # ─── TP1 partial close ────────────────────────────────────────────

    def _activate_tp1(self, pos: SimPosition, fill_price: float) -> None:
        pos.tp1_hit = True
        partial_qty = pos.qty * 0.5

        # Slippage on the partial fill.
        partial_fill_price = self._apply_slippage(fill_price, pos.direction, "close")
        partial_fee = partial_qty * partial_fill_price * (self.cfg.taker_fee_pct / 100)
        pos.fees_paid += partial_fee

        if pos.direction == "long":
            partial_pnl = (partial_fill_price - pos.entry_price) * partial_qty
        else:
            partial_pnl = (pos.entry_price - partial_fill_price) * partial_qty

        net_partial = partial_pnl - partial_fee
        pos.realized_pnl += net_partial
        self.balance += net_partial

        pos.qty -= partial_qty

        bb = pos.breakeven_buffer_pct
        if pos.direction == "long":
            pos.breakeven_sl = pos.entry_price * (1 + bb / 100)
            pos.trailing_high = fill_price
        else:
            pos.breakeven_sl = pos.entry_price * (1 - bb / 100)
            pos.trailing_low = fill_price

    # ─── close ────────────────────────────────────────────────────────

    def _close(self, symbol: str, raw_price: float, reason: str, bar_idx: int, time_ms: int) -> SimTrade:
        pos = self.positions.pop(symbol)
        exit_price = self._apply_slippage(raw_price, pos.direction, "close")
        exit_fee = pos.qty * exit_price * (self.cfg.taker_fee_pct / 100)
        pos.fees_paid += exit_fee

        if pos.direction == "long":
            tail_pnl = (exit_price - pos.entry_price) * pos.qty
        else:
            tail_pnl = (pos.entry_price - exit_price) * pos.qty

        net_tail = tail_pnl - exit_fee
        pos.realized_pnl += net_tail
        self.balance += net_tail

        original_notional = pos.original_qty * pos.entry_price
        net_pct = (pos.realized_pnl / original_notional) * 100 if original_notional > 0 else 0.0

        trade = SimTrade(
            symbol=symbol,
            direction=pos.direction,
            trade_type=pos.trade_type,
            entry_time_ms=pos.entry_time_ms,
            exit_time_ms=time_ms,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.original_qty,
            fees_paid=pos.fees_paid,
            pnl_pct=net_pct,
            pnl_usd=pos.realized_pnl,
            exit_reason=reason,
            bars_held=bar_idx - pos.entry_bar_idx,
            tp1_hit=pos.tp1_hit,
        )
        self.closed_trades.append(trade)
        self.equity_curve.append(self.balance)
        return trade


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    trades: List[SimTrade],
    starting_balance: float,
    equity_curve: List[float],
) -> Dict[str, Any]:
    if not trades:
        return {
            "total": 0, "wins": 0, "losses": 0, "breakeven": 0,
            "win_rate": 0.0, "profit_factor": 0.0, "sharpe_per_trade": 0.0,
            "max_drawdown_pct": 0.0, "expectancy_pct": 0.0,
            "avg_winner_pct": 0.0, "avg_loser_pct": 0.0,
            "starting_balance": starting_balance,
            "final_balance": starting_balance,
            "total_pnl_pct": 0.0,
            "exit_reasons": {},
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd < 0]
    breakeven = [t for t in trades if t.pnl_usd == 0]

    total_gain = sum(t.pnl_usd for t in wins)
    total_loss = abs(sum(t.pnl_usd for t in losses))

    win_rate = len(wins) / len(trades)
    if total_loss > 0:
        profit_factor = total_gain / total_loss
    elif total_gain > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_winner_pct = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
    avg_loser_pct = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
    expectancy_pct = sum(t.pnl_pct for t in trades) / len(trades)

    pnls = [t.pnl_pct for t in trades]
    mean = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    # Guard against degenerate cases where all trades have nearly the same
    # pnl (e.g., all hit the same SL %): std → 0 makes the ratio explode.
    # Below 1e-6 we report 0.0 — there's no meaningful dispersion to divide by.
    sharpe_per_trade = mean / std if std > 1e-6 else 0.0

    peak = starting_balance
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    final_balance = equity_curve[-1] if equity_curve else starting_balance
    total_pnl_pct = (final_balance - starting_balance) / starting_balance * 100 if starting_balance > 0 else 0.0

    # Exit-reason histogram
    exit_counts: Dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(breakeven),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "sharpe_per_trade": round(sharpe_per_trade, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "expectancy_pct": round(expectancy_pct, 3),
        "avg_winner_pct": round(avg_winner_pct, 2),
        "avg_loser_pct": round(avg_loser_pct, 2),
        "starting_balance": round(starting_balance, 2),
        "final_balance": round(final_balance, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "exit_reasons": exit_counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """Bar-by-bar replay over a multi-symbol, multi-timeframe candle set."""

    def __init__(
        self,
        config: BacktestConfig,
        score_fn: Optional[Callable] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.exchange = SimulatedExchange(config)
        self.verbose = verbose

        if score_fn is None:
            # Default: production scorer with all quality gates.
            from score import enhanced_score_symbol
            score_fn = enhanced_score_symbol
        self.score_fn = score_fn

    # ─── trend context (synthetic from BTC) ───────────────────────────

    @staticmethod
    def _build_trend_context(btc_view: Dict[str, List]) -> Dict[str, Any]:
        btc_1h = btc_view.get("60", [])
        if not btc_1h or len(btc_1h) < 21:
            return {
                "trend": "neutral", "btc_trend": "neutral",
                "strength": 0.5, "trend_strength": 0.5, "regime": "trending",
                "recommendations": {"primary_strategy": ""}, "opportunity_score": 0.5,
            }
        closes = [float(c["close"]) for c in btc_1h[-50:]]
        ema9 = sum(closes[-9:]) / 9
        ema21 = sum(closes[-21:]) / 21
        if ema9 > ema21 * 1.005:
            btc_trend = "bullish"
        elif ema9 < ema21 * 0.995:
            btc_trend = "bearish"
        else:
            btc_trend = "neutral"
        return {
            "trend": btc_trend,
            "btc_trend": btc_trend,
            "strength": 0.6, "trend_strength": 0.6,
            "regime": "trending",
            "recommendations": {
                "primary_strategy": "go_long" if btc_trend == "bullish"
                                    else "go_short" if btc_trend == "bearish" else ""
            },
            "opportunity_score": 0.6,
        }

    # ─── view construction (lookahead-safe) ───────────────────────────

    @staticmethod
    def _build_view(
        by_tf: Dict[str, List[dict]],
        current_time_ms: int,
        primary_tf: str,
        window: int = 100,
    ) -> Dict[str, List[dict]]:
        """Build the candles-by-tf dict the scorer expects.

        Lookahead safety: for the primary TF the current bar is visible
        (we treat its close as 'current price'). For higher TFs only
        CLOSED bars are visible — open_time + interval_seconds <= now.
        """
        view: Dict[str, List[dict]] = {}
        for tf, candles in by_tf.items():
            if not candles:
                continue
            tf_sec = INTERVAL_SECONDS.get(tf, 60)
            if tf == primary_tf:
                visible = [c for c in candles if int(c["timestamp"]) <= current_time_ms]
            else:
                visible = [
                    c for c in candles
                    if int(c["timestamp"]) + tf_sec * 1000 <= current_time_ms
                ]
            if visible:
                view[tf] = visible[-window:]
        return view

    # ─── run ──────────────────────────────────────────────────────────

    def run(
        self,
        candles_by_symbol_tf: Dict[str, Dict[str, List[dict]]],
        primary_tf: str = "5",
        btc_symbol: str = "BTCUSDT",
    ) -> Dict[str, Any]:
        symbols = list(candles_by_symbol_tf.keys())
        if not symbols:
            return compute_metrics([], self.config.starting_balance, [self.config.starting_balance])

        # Use the first symbol's primary TF as the master timeline.
        # All symbols should be aligned in time; bars at the same timestamp
        # are processed together.
        sample = candles_by_symbol_tf[symbols[0]].get(primary_tf, [])
        if not sample:
            return compute_metrics([], self.config.starting_balance, [self.config.starting_balance])

        n_bars = len(sample)
        warmup = self.config.warmup_bars

        for bar_idx in range(warmup, n_bars):
            current_time_ms = int(sample[bar_idx]["timestamp"])

            btc_view = self._build_view(
                candles_by_symbol_tf.get(btc_symbol, {}), current_time_ms, primary_tf
            )
            trend_context = self._build_trend_context(btc_view)

            for symbol in symbols:
                by_tf = candles_by_symbol_tf[symbol]

                # 1. Step open position (if any) on the current bar.
                if symbol in self.exchange.positions:
                    candle = self._candle_at(by_tf.get(primary_tf, []), current_time_ms)
                    if candle:
                        self.exchange.step(symbol, candle, bar_idx, current_time_ms)
                    continue

                # 2. No position — try to open one based on scoring.
                view = self._build_view(by_tf, current_time_ms, primary_tf)
                # Need at least primary and one other TF for the scorer.
                if not view or len(view) < 2:
                    continue

                try:
                    score, tf_scores, trade_type, _ind, _used = self.score_fn(
                        symbol, view, trend_context
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"[{symbol}@{current_time_ms}] scorer error: {e}")
                    continue

                if score <= 0:
                    continue

                gate = {
                    "Scalp": self.config.min_scalp_score,
                    "Intraday": self.config.min_intraday_score,
                    "Swing": self.config.min_swing_score,
                }.get(trade_type, self.config.min_intraday_score)
                if score < gate:
                    continue

                if self.config.require_clean_direction:
                    from score import determine_direction
                    direction = determine_direction(tf_scores)
                    if direction is None:
                        continue
                    direction = direction.lower()
                else:
                    direction = "long"

                # Entry at the close of the current primary-TF bar.
                primary_candles = view.get(primary_tf, [])
                if not primary_candles:
                    continue
                entry_mark = float(primary_candles[-1]["close"])

                pos = self.exchange.open(
                    symbol, direction, trade_type, entry_mark, bar_idx, current_time_ms
                )
                if pos and self.verbose:
                    print(
                        f"[{symbol}@{current_time_ms}] OPEN {direction.upper()} {trade_type} "
                        f"@ {entry_mark:.4f} qty={pos.qty:.4f} score={score:.2f}"
                    )

        # Force-close remaining positions at the last primary-TF bar.
        for symbol in list(self.exchange.positions.keys()):
            primary_candles = candles_by_symbol_tf[symbol].get(primary_tf, [])
            if not primary_candles:
                continue
            last = primary_candles[-1]
            self.exchange._close(
                symbol, float(last["close"]), "end_of_backtest",
                n_bars - 1, int(last["timestamp"]),
            )

        return compute_metrics(
            self.exchange.closed_trades,
            self.config.starting_balance,
            self.exchange.equity_curve,
        )

    @staticmethod
    def _candle_at(candles: List[dict], time_ms: int) -> Optional[dict]:
        """Return the candle at the given timestamp, or None."""
        # Most-likely the last bar; check from end.
        for c in reversed(candles):
            ts = int(c["timestamp"])
            if ts == time_ms:
                return c
            if ts < time_ms:
                return None
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data loader with disk cache
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / ".backtest_cache"


async def load_or_fetch_history(
    symbol: str,
    interval: str,
    limit: int = 200,
    category: str = "linear",
    use_cache: bool = True,
) -> List[dict]:
    """Load historical candles. Tries disk cache first, falls back to REST."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol}_{category}_{interval}_{limit}.json"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass  # fall through to REST

    from websocket_candles import fetch_candles_rest
    candles = await fetch_candles_rest(symbol, interval=interval, limit=limit, category=category)

    try:
        with open(cache_file, "w") as f:
            json.dump(candles, f)
    except Exception:
        pass

    return candles


async def load_history_for_symbols(
    symbols: List[str],
    intervals: List[str],
    limit: int = 200,
    category: str = "linear",
) -> Dict[str, Dict[str, List[dict]]]:
    """Fetch (cached) history for all (symbol, interval) pairs."""
    out: Dict[str, Dict[str, List[dict]]] = {}
    for sym in symbols:
        out[sym] = {}
        for tf in intervals:
            try:
                candles = await load_or_fetch_history(sym, tf, limit=limit, category=category)
                if candles:
                    out[sym][tf] = candles
            except Exception as e:
                print(f"⚠️ {sym} {tf}m fetch failed: {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_report(metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST RESULTS")
    lines.append("=" * 60)
    lines.append(f"  Trades:                {metrics['total']}  ({metrics['wins']}W / {metrics['losses']}L / {metrics['breakeven']}BE)")
    lines.append(f"  Win rate:              {metrics['win_rate'] * 100:.1f}%")
    lines.append(f"  Profit factor:         {metrics['profit_factor']}")
    lines.append(f"  Sharpe (per trade):    {metrics['sharpe_per_trade']}")
    lines.append(f"  Max drawdown:          {metrics['max_drawdown_pct']:.2f}%")
    lines.append(f"  Expectancy / trade:    {metrics['expectancy_pct']:+.3f}%")
    lines.append(f"  Avg winner:            {metrics['avg_winner_pct']:+.2f}%")
    lines.append(f"  Avg loser:             {metrics['avg_loser_pct']:+.2f}%")
    lines.append("-" * 60)
    lines.append(f"  Starting balance:      ${metrics['starting_balance']:.2f}")
    lines.append(f"  Final balance:         ${metrics['final_balance']:.2f}")
    lines.append(f"  Total return:          {metrics['total_pnl_pct']:+.2f}%")
    lines.append("-" * 60)
    lines.append("  Exit reasons:")
    for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
        lines.append(f"    {reason:20s}  {count}")
    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
    "BacktestConfig",
    "SimPosition",
    "SimTrade",
    "SimulatedExchange",
    "BacktestEngine",
    "compute_metrics",
    "load_or_fetch_history",
    "load_history_for_symbols",
    "format_metrics_report",
]
