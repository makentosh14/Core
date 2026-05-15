#!/usr/bin/env python3
"""
run_runner_analysis.py — Phase 4 CLI runner.

Loads cached historical data, runs the backtest to produce the trade ledger,
then runs the runner-capture analyzer on the same candle data.

Outputs:
  * Summary report to stdout
  * `runner_capture.csv` with one row per detected runner
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    days_to_bars,
    load_or_fetch_history,
)
from runner_analyzer import analyze_runner_capture, format_runner_report


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
DEFAULT_INTERVALS = ["5", "15", "60"]


async def main():
    p = argparse.ArgumentParser(description="Runner-capture analysis on backtest data.")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--intervals", default=",".join(DEFAULT_INTERVALS))
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--primary-tf", default="5")
    p.add_argument("--min-move-pct", type=float, default=3.0,
                   help="Minimum move magnitude to be considered a runner (default: 3.0%%)")
    p.add_argument("--window-bars", type=int, default=20,
                   help="Max bars within which the move must complete (default: 20 = 100min on 5m)")
    p.add_argument("--output", default="runner_capture.csv")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]

    print(f"Loading {args.days}d of history for {len(symbols)} symbols, "
          f"{len(intervals)} TFs ...")
    t0 = time.time()
    candles: dict = {}
    for sym in symbols:
        candles[sym] = {}
        for tf in intervals:
            try:
                c = await load_or_fetch_history(sym, tf, limit=days_to_bars(args.days, tf))
                if c:
                    candles[sym][tf] = c
            except Exception as e:
                print(f"  [WARN] {sym} {tf}m: {e}")
    print(f"  data load: {time.time() - t0:.1f}s")

    print(f"\nRunning backtest on {args.primary_tf}m primary timeline ...")
    cfg = BacktestConfig()
    engine = BacktestEngine(cfg)
    t1 = time.time()
    bt_metrics = engine.run(candles, primary_tf=args.primary_tf)
    print(f"  backtest: {time.time() - t1:.1f}s  ({bt_metrics['total']} trades)")

    print(f"\nAnalyzing runner capture ...")
    t2 = time.time()
    report = analyze_runner_capture(
        candles_by_symbol_tf=candles,
        closed_trades=engine.exchange.closed_trades,
        score_fn=engine.score_fn,
        config=cfg,
        primary_tf=args.primary_tf,
        min_move_pct=args.min_move_pct,
        window_bars=args.window_bars,
    )
    print(f"  analysis: {time.time() - t2:.1f}s")

    print()
    print(format_runner_report(report))

    report.to_csv(args.output)
    print(f"\nWrote per-runner detail to: {args.output}")

    return 0 if report.runners else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
