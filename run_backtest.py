#!/usr/bin/env python3
"""
run_backtest.py — Phase 5 CLI runner for the backtest engine.

Default behavior: fetch 200 bars of 1m/5m/15m/60m history for a small basket
of liquid linear-perp symbols, run the production scorer over them, and
print the metrics report.

Usage:
    python run_backtest.py
    python run_backtest.py --symbols BTCUSDT,ETHUSDT --bars 200
    python run_backtest.py --symbols BTCUSDT --bars 200 --no-cache --verbose

Bybit's V5 kline endpoint caps at 200 bars per request. On 5m that's ~16h
of data; on 1m it's ~3h. For longer windows we'd need pagination — out of
scope for this first-pass tool.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    format_metrics_report,
    load_history_for_symbols,
)


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_INTERVALS = ["1", "5", "15", "60"]
DEFAULT_BARS = 200


async def main():
    parser = argparse.ArgumentParser(
        description="Run backtest on cached/live Bybit historical data."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbol list (default: BTCUSDT,ETHUSDT,SOLUSDT)",
    )
    parser.add_argument(
        "--intervals",
        default=",".join(DEFAULT_INTERVALS),
        help="Comma-separated TFs (default: 1,5,15,60)",
    )
    parser.add_argument(
        "--bars", type=int, default=DEFAULT_BARS,
        help="Bars per (symbol, interval) — max 200 (default: 200)",
    )
    parser.add_argument(
        "--primary-tf", default="5",
        help="Primary timeline for bar iteration (default: 5m)",
    )
    parser.add_argument(
        "--balance", type=float, default=1000.0,
        help="Starting balance (default: 1000)",
    )
    parser.add_argument(
        "--risk", type=float, default=2.5,
        help="Risk %% per trade (default: 2.5)",
    )
    parser.add_argument(
        "--fee", type=float, default=0.055,
        help="Taker fee %% one-sided (default: 0.055 = Bybit USDT-M)",
    )
    parser.add_argument(
        "--slippage", type=float, default=5.0,
        help="Slippage bps adverse on every fill (default: 5)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every trade open/close",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Skip disk cache, fetch fresh",
    )
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    intervals = [s.strip() for s in args.intervals.split(",") if s.strip()]

    print(f"Loading history for {len(symbols)} symbols × {len(intervals)} TFs "
          f"× {args.bars} bars each...")
    t0 = time.time()
    candles = await load_history_for_symbols(symbols, intervals, limit=args.bars)
    print(f"  done in {time.time() - t0:.1f}s")

    # Sanity-check what we loaded.
    for sym in symbols:
        tfs = candles.get(sym, {})
        counts = {tf: len(tfs.get(tf, [])) for tf in intervals}
        print(f"  {sym}: {counts}")

    config = BacktestConfig(
        starting_balance=args.balance,
        risk_per_trade_pct=args.risk,
        taker_fee_pct=args.fee,
        slippage_bps=args.slippage,
    )
    engine = BacktestEngine(config, verbose=args.verbose)

    print(f"\nRunning backtest on {args.primary_tf}m primary timeline...")
    t0 = time.time()
    metrics = engine.run(candles, primary_tf=args.primary_tf)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")

    print()
    print(format_metrics_report(metrics))

    return 0 if metrics["total"] > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
