"""
runner_analyzer.py — Phase 4 missed-runner detector.

For each large directional move in a candle history, classify what the bot's
actual trade record did:

  CAUGHT          — bot entered in the right direction near the start, captured >= 30% of the move
  PARTIAL_CAPTURE — right direction & timing but exited too early (< 30% capture)
  LATE_ENTRY      — entered after the move was > 50% complete
  WRONG_DIRECTION — bot took a trade in the opposite direction during the move
  MISSED          — no trade overlapped this runner; diagnose why

The MISSED diagnosis re-scores the start-of-move bar with the actual scorer to
recover what the bot "saw" at that moment: raw score, the gate it needed to
clear, what direction it would have picked, and any explicit rejection reason
from anti-chase / 4h-veto / strong-indicator-gate.

USAGE:

    from backtest_engine import BacktestEngine, BacktestConfig, load_history_for_symbols
    from runner_analyzer import analyze_runner_capture, format_runner_report

    candles = await load_history_for_symbols([...], [...], limit=...)
    cfg = BacktestConfig(...)
    engine = BacktestEngine(cfg)
    backtest_metrics = engine.run(candles, primary_tf="5")

    report = analyze_runner_capture(
        candles_by_symbol_tf=candles,
        closed_trades=engine.exchange.closed_trades,
        score_fn=engine.score_fn,
        config=cfg,
        primary_tf="5",
        min_move_pct=3.0,
        window_bars=20,
    )
    print(format_runner_report(report))
    report.to_csv("runner_capture.csv")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunnerEvent:
    symbol: str
    direction: str            # "pump" or "dump"
    start_time_ms: int
    end_time_ms: int
    start_price: float
    end_price: float
    move_pct: float           # signed: positive for pump, negative for dump
    duration_bars: int

    # Bot interaction
    classification: str = "missed"   # CAUGHT / PARTIAL_CAPTURE / LATE_ENTRY / WRONG_DIRECTION / MISSED
    capture_pct: float = 0.0         # percent of the runner the bot captured (0-100)
    trade_entry_time_ms: Optional[int] = None
    trade_exit_time_ms: Optional[int] = None
    trade_entry_price: Optional[float] = None
    trade_exit_price: Optional[float] = None
    trade_pnl_pct: Optional[float] = None
    trade_exit_reason: Optional[str] = None

    # For MISSED runners: what the scorer saw at runner start
    miss_reason: Optional[str] = None       # short human-readable cause
    score_at_start: Optional[float] = None
    direction_at_start: Optional[str] = None
    trade_type_at_start: Optional[str] = None
    gate_at_start: Optional[float] = None

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunnerReport:
    runners: List[RunnerEvent] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.runners:
            out[r.classification] = out.get(r.classification, 0) + 1
        return out

    def avg_capture_when_caught(self) -> float:
        caught = [r for r in self.runners if r.classification in ("CAUGHT", "PARTIAL_CAPTURE")]
        if not caught:
            return 0.0
        return sum(r.capture_pct for r in caught) / len(caught)

    def miss_reasons(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.runners:
            if r.classification == "MISSED" and r.miss_reason:
                out[r.miss_reason] = out.get(r.miss_reason, 0) + 1
        return out

    def by_symbol(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for r in self.runners:
            bucket = out.setdefault(r.symbol, {})
            bucket[r.classification] = bucket.get(r.classification, 0) + 1
        return out

    def to_csv(self, path: str) -> None:
        import csv
        if not self.runners:
            with open(path, "w", newline="") as f:
                f.write("(no runners detected)\n")
            return
        fields = list(self.runners[0].to_row().keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.runners:
                w.writerow(r.to_row())


# ─────────────────────────────────────────────────────────────────────────────
# Runner detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_runners(
    candles: List[Dict[str, Any]],
    min_move_pct: float = 3.0,
    window_bars: int = 20,
    cooldown_bars: int = 6,
) -> List[Dict[str, Any]]:
    """Sliding-window scan for >= min_move_pct moves within window_bars.

    Same semantics as indicator_backscan.detect_moves but tunable. Returns
    a list of dicts (NOT RunnerEvent — those need symbol context attached
    later).

    A "pump" is a low->high move; "dump" is high->low. We use the bar at
    the START_IDX low/high as the reference price so that even a multi-bar
    consolidation that ends in a sharp move is counted from the breakout
    start, not from N bars before.
    """
    if not candles or len(candles) < window_bars + 1:
        return []

    out: List[Dict[str, Any]] = []
    cooldown_until = -1

    for i in range(len(candles) - window_bars):
        if i < cooldown_until:
            continue

        start_price = float(candles[i].get("close", 0))
        if start_price <= 0:
            continue

        # Look for the MAX subsequent close within window_bars (pump check),
        # and the MIN (dump check). Use highs / lows for intra-bar extremes.
        max_high = start_price
        max_high_idx = i
        min_low = start_price
        min_low_idx = i
        for j in range(i + 1, min(i + window_bars + 1, len(candles))):
            h = float(candles[j].get("high", candles[j].get("close", 0)))
            lo = float(candles[j].get("low", candles[j].get("close", 0)))
            if h > max_high:
                max_high = h
                max_high_idx = j
            if 0 < lo < min_low:
                min_low = lo
                min_low_idx = j

        pump_pct = ((max_high - start_price) / start_price) * 100 if start_price > 0 else 0
        dump_pct = ((min_low - start_price) / start_price) * 100 if start_price > 0 else 0

        if pump_pct >= min_move_pct:
            out.append({
                "direction": "pump",
                "start_idx": i,
                "end_idx": max_high_idx,
                "start_time_ms": int(candles[i].get("timestamp", 0)),
                "end_time_ms": int(candles[max_high_idx].get("timestamp", 0)),
                "start_price": start_price,
                "end_price": max_high,
                "move_pct": round(pump_pct, 2),
                "duration_bars": max_high_idx - i,
            })
            cooldown_until = max_high_idx + cooldown_bars
        elif dump_pct <= -min_move_pct:
            out.append({
                "direction": "dump",
                "start_idx": i,
                "end_idx": min_low_idx,
                "start_time_ms": int(candles[i].get("timestamp", 0)),
                "end_time_ms": int(candles[min_low_idx].get("timestamp", 0)),
                "start_price": start_price,
                "end_price": min_low,
                "move_pct": round(dump_pct, 2),
                "duration_bars": min_low_idx - i,
            })
            cooldown_until = min_low_idx + cooldown_bars

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Bot-response classification
# ─────────────────────────────────────────────────────────────────────────────

def _find_trade_for_runner(
    closed_trades: List[Any],
    symbol: str,
    runner: Dict[str, Any],
    entry_tolerance_bars: int,
    bar_ms: int,
) -> Optional[Any]:
    """Find a closed trade on this symbol that opened near the runner start.

    Tolerance: trade.entry_time_ms must be within
        [runner.start - entry_tolerance_bars*bar_ms,
         runner.end_time_ms]
    """
    start = runner["start_time_ms"] - entry_tolerance_bars * bar_ms
    end = runner["end_time_ms"]
    candidates = [
        t for t in closed_trades
        if t.symbol == symbol
        and start <= t.entry_time_ms <= end
    ]
    if not candidates:
        return None
    # Pick the trade whose entry is closest to the runner start.
    return min(candidates, key=lambda t: abs(t.entry_time_ms - runner["start_time_ms"]))


def _capture_pct(trade: Any, runner: Dict[str, Any]) -> float:
    """How much of the runner's move did the trade actually capture?
    100% means trade entered at runner.start_price and exited at runner.end_price.
    Negative if trade went the wrong direction."""
    move_range = runner["end_price"] - runner["start_price"]
    if abs(move_range) < 1e-9:
        return 0.0

    if runner["direction"] == "pump":
        # Long-aligned: exit_price - entry_price should be positive of the move
        if trade.direction.lower() != "long":
            return -100.0  # wrong direction
        trade_move = trade.exit_price - trade.entry_price
    else:  # dump
        if trade.direction.lower() != "short":
            return -100.0
        trade_move = trade.entry_price - trade.exit_price

    return (trade_move / abs(move_range)) * 100.0


def classify_trade_vs_runner(
    trade: Any,
    runner: Dict[str, Any],
    bar_ms: int,
) -> str:
    """Given an overlapping trade and a runner, classify the relationship."""
    # Wrong direction?
    if runner["direction"] == "pump" and trade.direction.lower() != "long":
        return "WRONG_DIRECTION"
    if runner["direction"] == "dump" and trade.direction.lower() != "short":
        return "WRONG_DIRECTION"

    # Where in the runner did we enter?
    runner_dur_ms = runner["end_time_ms"] - runner["start_time_ms"]
    if runner_dur_ms <= 0:
        runner_dur_ms = bar_ms
    progress_at_entry = (trade.entry_time_ms - runner["start_time_ms"]) / runner_dur_ms

    if progress_at_entry > 0.5:
        return "LATE_ENTRY"

    capture = _capture_pct(trade, runner)
    if capture >= 30.0:
        return "CAUGHT"
    return "PARTIAL_CAPTURE"


# ─────────────────────────────────────────────────────────────────────────────
# MISSED diagnosis (re-score the start-of-runner bar)
# ─────────────────────────────────────────────────────────────────────────────

INTERVAL_SECONDS_LOCAL = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800, "60": 3600, "240": 14400}


def _build_view_at_time(
    by_tf: Dict[str, List[Dict[str, Any]]],
    current_time_ms: int,
    primary_tf: str,
    window: int = 100,
) -> Dict[str, List[Dict[str, Any]]]:
    """Same lookahead-safe view used by BacktestEngine, copied here so
    runner_analyzer is self-contained."""
    view: Dict[str, List[Dict[str, Any]]] = {}
    for tf, candles in by_tf.items():
        if not candles:
            continue
        tf_sec = INTERVAL_SECONDS_LOCAL.get(tf, 60)
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


def _diagnose_missed(
    runner: Dict[str, Any],
    symbol: str,
    by_tf: Dict[str, List[Dict[str, Any]]],
    score_fn: Callable,
    trend_context: Dict[str, Any],
    config: Any,
    primary_tf: str,
) -> Dict[str, Any]:
    """For a missed runner, probe the scorer at the start bar and capture
    why the bot didn't enter."""
    view = _build_view_at_time(by_tf, runner["start_time_ms"], primary_tf)

    if len(view) < 2:
        return {
            "miss_reason": "insufficient_data",
            "score_at_start": None,
            "direction_at_start": None,
            "trade_type_at_start": None,
            "gate_at_start": None,
        }

    try:
        score, tf_scores, trade_type, _ind, _used = score_fn(symbol, view, trend_context)
    except Exception as e:
        return {
            "miss_reason": f"scorer_error: {type(e).__name__}",
            "score_at_start": None,
            "direction_at_start": None,
            "trade_type_at_start": None,
            "gate_at_start": None,
        }

    gate = {
        "Scalp": getattr(config, "min_scalp_score", 10.5),
        "Intraday": getattr(config, "min_intraday_score", 12.0),
        "Swing": getattr(config, "min_swing_score", 15.5),
    }.get(trade_type, getattr(config, "min_intraday_score", 12.0))

    # Apply per-symbol adjustment if config has one
    sym_adj_map = getattr(config, "symbol_score_adjustment", {}) or {}
    adjusted_score = score + sym_adj_map.get(symbol, 0.0)

    # Determine why the score wouldn't have produced an entry.
    direction = None
    try:
        from score import determine_direction
        direction = determine_direction(tf_scores)
    except Exception:
        pass

    if score <= 0:
        reason = "score_zero_or_negative_quality_gate"
    elif adjusted_score < gate:
        reason = f"score_below_{trade_type.lower()}_gate"
    elif direction is None:
        reason = "no_clean_direction"
    else:
        # Score passed the gate and direction was found, but trade didn't fire.
        # Could be anti-chase, position limit, cooldown, etc.
        reason = "passed_gate_but_other_filter_blocked"

    return {
        "miss_reason": reason,
        "score_at_start": round(float(score), 3),
        "direction_at_start": direction,
        "trade_type_at_start": trade_type,
        "gate_at_start": round(float(gate), 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_runner_capture(
    candles_by_symbol_tf: Dict[str, Dict[str, List[Dict[str, Any]]]],
    closed_trades: List[Any],
    score_fn: Optional[Callable] = None,
    config: Optional[Any] = None,
    primary_tf: str = "5",
    btc_symbol: str = "BTCUSDT",
    min_move_pct: float = 3.0,
    window_bars: int = 20,
    entry_tolerance_bars: int = 3,
    cooldown_bars: int = 6,
) -> RunnerReport:
    """Run runner-capture analysis across all symbols.

    Args:
        candles_by_symbol_tf: same shape as backtest input
        closed_trades: list of SimTrade objects from BacktestEngine
        score_fn: scoring function (only used for diagnosing MISSED runners).
                  If None, MISSED runners just get a generic "no_score_probe"
                  reason without diagnostic detail.
        config: BacktestConfig (used for gate values + per-symbol adjustment)
        primary_tf: timeframe to detect runners on (and to probe scorer with)

    Returns: RunnerReport with one RunnerEvent per detected runner.
    """
    report = RunnerReport()
    report.config_snapshot = {
        "min_move_pct": min_move_pct,
        "window_bars": window_bars,
        "primary_tf": primary_tf,
        "entry_tolerance_bars": entry_tolerance_bars,
    }

    bar_ms = INTERVAL_SECONDS_LOCAL.get(primary_tf, 60) * 1000

    for symbol, by_tf in candles_by_symbol_tf.items():
        primary_candles = by_tf.get(primary_tf, [])
        if not primary_candles or len(primary_candles) < window_bars + 1:
            continue

        # Detect raw runners
        raw_runners = detect_runners(
            primary_candles, min_move_pct=min_move_pct,
            window_bars=window_bars, cooldown_bars=cooldown_bars,
        )

        for runner in raw_runners:
            event = RunnerEvent(
                symbol=symbol,
                direction=runner["direction"],
                start_time_ms=runner["start_time_ms"],
                end_time_ms=runner["end_time_ms"],
                start_price=runner["start_price"],
                end_price=runner["end_price"],
                move_pct=runner["move_pct"],
                duration_bars=runner["duration_bars"],
            )

            # Did any bot trade overlap this runner?
            trade = _find_trade_for_runner(
                closed_trades, symbol, runner,
                entry_tolerance_bars=entry_tolerance_bars,
                bar_ms=bar_ms,
            )

            if trade is not None:
                event.classification = classify_trade_vs_runner(trade, runner, bar_ms)
                event.capture_pct = round(_capture_pct(trade, runner), 1)
                event.trade_entry_time_ms = trade.entry_time_ms
                event.trade_exit_time_ms = trade.exit_time_ms
                event.trade_entry_price = trade.entry_price
                event.trade_exit_price = trade.exit_price
                event.trade_pnl_pct = trade.pnl_pct
                event.trade_exit_reason = trade.exit_reason
            else:
                event.classification = "MISSED"
                if score_fn is not None:
                    # Diagnose why. Build a trend_context probe (synthetic
                    # from BTC if available, else neutral).
                    btc_view = _build_view_at_time(
                        candles_by_symbol_tf.get(btc_symbol, {}),
                        runner["start_time_ms"], primary_tf,
                    )
                    trend_context = _synthetic_trend_context(btc_view)

                    diagnosis = _diagnose_missed(
                        runner, symbol, by_tf, score_fn, trend_context, config, primary_tf
                    )
                    event.miss_reason = diagnosis["miss_reason"]
                    event.score_at_start = diagnosis["score_at_start"]
                    event.direction_at_start = diagnosis["direction_at_start"]
                    event.trade_type_at_start = diagnosis["trade_type_at_start"]
                    event.gate_at_start = diagnosis["gate_at_start"]

            report.runners.append(event)

    return report


def _synthetic_trend_context(btc_view: Dict[str, List]) -> Dict[str, Any]:
    """Same as BacktestEngine._build_trend_context — kept in sync."""
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
        "trend": btc_trend, "btc_trend": btc_trend,
        "strength": 0.6, "trend_strength": 0.6, "regime": "trending",
        "recommendations": {"primary_strategy":
            "go_long" if btc_trend == "bullish"
            else "go_short" if btc_trend == "bearish" else ""
        },
        "opportunity_score": 0.6,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def format_runner_report(report: RunnerReport) -> str:
    if not report.runners:
        return "No runners detected in the data."

    lines = []
    lines.append("=" * 70)
    lines.append("RUNNER CAPTURE REPORT")
    lines.append("=" * 70)
    lines.append(f"  Detection config: {report.config_snapshot}")
    lines.append(f"  Total runners detected: {len(report.runners)}")

    counts = report.counts()
    total = len(report.runners)
    lines.append("-" * 70)
    lines.append("  By classification:")
    for cls in ("CAUGHT", "PARTIAL_CAPTURE", "LATE_ENTRY", "WRONG_DIRECTION", "MISSED"):
        c = counts.get(cls, 0)
        pct = c / total * 100 if total else 0
        lines.append(f"    {cls:18s}  {c:>4}  ({pct:>5.1f}%)")

    avg_cap = report.avg_capture_when_caught()
    lines.append("-" * 70)
    lines.append(f"  Avg capture % when bot took the trade: {avg_cap:.1f}%")

    miss_reasons = report.miss_reasons()
    if miss_reasons:
        lines.append("-" * 70)
        lines.append("  Why missed runners were missed:")
        for reason, count in sorted(miss_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {reason:48s}  {count}")

    by_sym = report.by_symbol()
    if by_sym:
        lines.append("-" * 70)
        lines.append(f"  By symbol:")
        lines.append(
            f"    {'symbol':<14}{'total':>7}{'CAUGHT':>8}"
            f"{'PARTIAL':>9}{'LATE':>7}{'WRONG':>7}{'MISSED':>9}"
        )
        for sym, breakdown in sorted(by_sym.items(),
                                      key=lambda kv: -sum(kv[1].values())):
            t = sum(breakdown.values())
            lines.append(
                f"    {sym:<14}{t:>7}"
                f"{breakdown.get('CAUGHT', 0):>8}"
                f"{breakdown.get('PARTIAL_CAPTURE', 0):>9}"
                f"{breakdown.get('LATE_ENTRY', 0):>7}"
                f"{breakdown.get('WRONG_DIRECTION', 0):>7}"
                f"{breakdown.get('MISSED', 0):>9}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)


__all__ = [
    "RunnerEvent",
    "RunnerReport",
    "detect_runners",
    "classify_trade_vs_runner",
    "analyze_runner_capture",
    "format_runner_report",
]
