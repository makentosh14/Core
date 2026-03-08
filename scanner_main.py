#!/usr/bin/env python3
"""
scanner_main.py - Research Scanner Main Entry Point
=====================================================
Crypto Market Research Scanner — discovers pre-pump/pre-dump patterns.

ARCHITECTURE:
  1. Fetch top N symbols by 24h volume from Bybit
  2. For each symbol, fetch 4 timeframes (5m/15m/1h/4h) with full warmup
  3. Calculate 30+ indicators per timeframe
  4. Build MTF snapshots with market regime + setup type classification
  5. Store ALL events to research dataset (parquet)
  6. Separately log only signals to signal_log.csv
  7. Record signal patterns to pattern_database.json
  8. Optionally run analysis after collection

WORKFLOW MODES:
  Collect:         python scanner_main.py --loop
  Label outcomes:  python outcome_labeler.py
  Analyze:         python pattern_analyzer.py
  Full pipeline:   python scanner_main.py --loop --analyze-every 12

USAGE:
    python scanner_main.py                          # Single scan
    python scanner_main.py --top 50                 # Top 50 symbols
    python scanner_main.py --symbols BTCUSDT,ETHUSDT
    python scanner_main.py --loop                   # Continuous loop
    python scanner_main.py --loop --analyze-every 12
    python scanner_main.py --analyze                # Analyze only
"""

import asyncio
import aiohttp
import argparse
import time
import sys
import os
from typing import List, Dict, Optional
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BYBIT_API, API_TIMEOUT, TIMEFRAMES, WARMUP_CANDLES,
    TOP_SYMBOLS, SCAN_INTERVAL, DATA_DIR,
)
from indicators import Candle, analyze_all_indicators
from mtf_analyzer import build_snapshot, MTFSnapshot
from data_fetcher import (
    fetch_top_symbols_by_volume,
    fetch_batch_symbols,
)
from data_storage import (
    ensure_data_dir,
    save_research_events,
    save_snapshots_csv,
    save_signal_log,
    load_pattern_db,
    save_pattern_db,
    record_pattern,
    analyze_historical_patterns,
    generate_scan_summary,
)


# ============================================================
# WARMUP VALIDATION
# ============================================================

# Minimum candles required per timeframe for indicators to be valid.
# Based on the slowest indicator: EMA200 = 200 bars.
# We add the analysis window on top.
#
# FIXED: This now aligns EXACTLY with WARMUP_CANDLES in config.py
# and with the fetch logic in data_fetcher.py.
# Previously: validation used candles + 50 (too loose).
# Now:        validation uses WARMUP_CANDLES + candles (correct).

def _min_required(tf_key: str) -> int:
    """Return the minimum number of candles required for tf_key to be valid."""
    candles = TIMEFRAMES[tf_key]["candles"]
    # WARMUP_CANDLES already covers EMA200 + safety margin
    return WARMUP_CANDLES + candles


# ============================================================
# CORE SYMBOL PROCESSOR
# ============================================================

def process_symbol(
    symbol:         str,
    candles_by_tf:  Dict[str, List[Candle]],
) -> Optional[MTFSnapshot]:
    """
    Process a single symbol:
    1. Strict warmup validation (aligned with fetch logic)
    2. Calculate all indicators per TF
    3. Build research snapshot with market regime + setup type

    Returns None if any TF has insufficient data.
    """
    # ── Strict validation ──
    # This is the FIXED version. We now check against WARMUP_CANDLES + tf_candles,
    # which exactly matches what fetch_symbol_all_timeframes() fetches.
    for tf_key, tf_config in TIMEFRAMES.items():
        candles  = candles_by_tf.get(tf_key, [])
        required = _min_required(tf_key)
        if len(candles) < required:
            # Skip symbol: not enough data for this timeframe
            return None

    # ── Calculate indicators per TF ──
    ind_5m  = analyze_all_indicators(candles_by_tf["5"],   prefix="5m")
    ind_15m = analyze_all_indicators(candles_by_tf["15"],  prefix="15m")
    ind_1h  = analyze_all_indicators(candles_by_tf["60"],  prefix="1h")
    ind_4h  = analyze_all_indicators(candles_by_tf["240"], prefix="4h")

    # ── Current price and timestamp from 5m (fastest TF) ──
    last_candle = candles_by_tf["5"][-1]
    price       = last_candle.close
    timestamp   = last_candle.timestamp

    # ── Build synchronized research snapshot ──
    snapshot = build_snapshot(
        symbol    = symbol,
        timestamp = timestamp,
        price     = price,
        ind_5m    = ind_5m,
        ind_15m   = ind_15m,
        ind_1h    = ind_1h,
        ind_4h    = ind_4h,
    )

    return snapshot


# ============================================================
# SCAN CYCLE
# ============================================================

_cycle_counter = 0   # global cycle counter for event_id generation


async def run_scan_cycle(
    symbols:    Optional[List[str]] = None,
    top_n:      int                 = TOP_SYMBOLS,
    pattern_db: Optional[List]      = None,
) -> List[MTFSnapshot]:
    """Execute one full scan cycle. Returns list of MTFSnapshots."""
    global _cycle_counter
    _cycle_counter += 1
    cycle      = _cycle_counter
    scan_start = time.time()
    now        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("=" * 72)
    print(f"  RESEARCH MARKET SCANNER  |  Cycle #{cycle}")
    print(f"  Started: {now}")
    print(f"  Timeframes: 5m/15m/1h/4h  |  Warmup: {WARMUP_CANDLES} bars")
    print("=" * 72)

    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # ── Step 1: Symbols ──
        if not symbols:
            print(f"\n  [1/5] Fetching top {top_n} symbols by 24h volume ...")
            tickers = await fetch_top_symbols_by_volume(session, top_n=top_n)
            symbols = [t["symbol"] for t in tickers]
            if tickers:
                top3 = ", ".join(
                    f"{t['symbol']}(${t['turnover24h']/1e6:.0f}M)"
                    for t in tickers[:3]
                )
                print(f"        Found {len(symbols)} symbols. Top 3: {top3}")
        else:
            print(f"\n  [1/5] Using {len(symbols)} provided symbols")

        if not symbols:
            print("  ERROR: No symbols found. Check API connection.")
            return []

        # ── Step 2: Fetch candles ──
        print(f"\n  [2/5] Fetching candle data ({len(symbols)} symbols × 4 TFs) ...")
        print(f"        Each TF: {WARMUP_CANDLES} warmup + window bars")
        all_data = await fetch_batch_symbols(session, symbols)
        print(f"        Fetched: {len(all_data)} symbols")

    # ── Step 3: Process indicators ──
    print(f"\n  [3/5] Computing indicators ...")
    snapshots = []
    skipped   = 0

    for i, (symbol, candles_by_tf) in enumerate(all_data.items()):
        snap = process_symbol(symbol, candles_by_tf)
        if snap:
            snapshots.append(snap)
        else:
            skipped += 1

        if (i + 1) % 25 == 0 or i == len(all_data) - 1:
            print(f"        [{i + 1}/{len(all_data)}] snaps={len(snapshots)} skipped={skipped}")

    print(f"        Done: {len(snapshots)} snapshots, {skipped} skipped (insufficient data)")

    # ── Step 4: Store data ──
    print(f"\n  [4/5] Storing data to {DATA_DIR}/ ...")

    # Primary research storage (ALL snapshots — for base rate analysis)
    save_research_events(snapshots, scan_cycle=cycle, signals_only=False)

    # Legacy CSV (signals only)
    save_signal_log(snapshots)

    # Pattern database (signals with key indicator snapshot)
    if pattern_db is not None:
        sig_snaps = [s for s in snapshots if s.direction]
        for snap in sig_snaps:
            record_pattern(snap, outcome=None, pattern_db=pattern_db)
        print(f"        +{len(sig_snaps)} patterns added to pattern DB")

    # ── Step 5: Summary ──
    print(f"\n  [5/5] Generating summary ...")
    pattern_analysis = None
    if pattern_db:
        pattern_analysis = analyze_historical_patterns(pattern_db)

    elapsed = time.time() - scan_start
    summary = generate_scan_summary(snapshots, pattern_analysis)
    display_results(snapshots, summary, elapsed, cycle)

    return snapshots


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    snapshots: List[MTFSnapshot],
    summary:   Dict,
    elapsed:   float,
    cycle:     int,
):
    """Print formatted scan results."""
    signals = [s for s in snapshots if s.direction]
    pumps   = sorted([s for s in signals if s.direction == "pump"],
                     key=lambda x: (x.tier, x.confidence), reverse=True)
    dumps   = sorted([s for s in signals if s.direction == "dump"],
                     key=lambda x: (x.tier, x.confidence), reverse=True)

    tier_emoji = {3: "🎯", 2: "⚡", 1: "📊"}
    tier_name  = {3: "SNIPER", 2: "ELITE", 1: "STD"}

    print("\n" + "=" * 72)
    print(f"  CYCLE #{cycle}  |  {len(snapshots)} symbols  |  {len(signals)} signals  |  {elapsed:.1f}s")
    print("=" * 72)

    # Market overview
    bull_dom = sum(1 for s in snapshots if s.mtf_signals.get("mtf_moderate_bull", False))
    bear_dom = sum(1 for s in snapshots if s.mtf_signals.get("mtf_moderate_bear", False))

    # Regime distribution
    regimes: Dict[str, int] = {}
    for s in snapshots:
        regimes[s.market_regime] = regimes.get(s.market_regime, 0) + 1
    top_regime = max(regimes, key=regimes.get) if regimes else "unknown"

    print(f"\n  Market: Bullish={bull_dom} | Bearish={bear_dom} | Dominant regime={top_regime}")

    def _print_signals(sigs, emoji_dir):
        print(f"  {'Symbol':<14} {'Type':<22} {'Tier':<8} {'Conf':<7} {'Regime':<18} {'Setup':<18} {'Price'}")
        print(f"  {'─' * 100}")
        for s in sigs[:15]:
            e  = tier_emoji.get(s.tier, "")
            tn = tier_name.get(s.tier, "")
            print(f"  {e} {s.symbol:<12} {s.signal_type:<22} {tn:<8} "
                  f"{s.confidence:>5.1f}%  {s.market_regime:<18} {s.setup_type:<18} "
                  f"${s.price:<12.6g}")

    if pumps:
        print(f"\n  🚀 PUMP SIGNALS ({len(pumps)})")
        _print_signals(pumps, "pump")

    if dumps:
        print(f"\n  🔻 DUMP SIGNALS ({len(dumps)})")
        _print_signals(dumps, "dump")

    if not signals:
        print("\n  No active signals in this cycle.")

    # Pattern analysis summary
    pa = summary.get("pattern_analysis")
    if pa and pa.get("evaluated", 0) >= 5:
        print(f"\n  🧠 Pattern DB: {pa['total_patterns']} total | {pa['evaluated']} evaluated")
        by_type = pa.get("by_signal_type", {})
        for st, s in sorted(by_type.items(), key=lambda x: x[1].get("win_rate", 0), reverse=True)[:5]:
            if s.get("evaluated", 0) >= 5:
                print(f"     {st:<25}: {s['win_rate']:.1f}% WR ({s['evaluated']} samples)")

        best = pa.get("best_indicators", {})
        if best:
            print(f"\n  📊 Best indicators by win rate (top 5):")
            for name, d in list(best.items())[:5]:
                print(f"     {name:<40}: {d['win_rate']:.1f}% WR  n={d['sample_size']}")

    print(f"\n  Data: {DATA_DIR}/  |  Cycle={cycle}  |  {elapsed:.1f}s")
    print("=" * 72)


# ============================================================
# CONTINUOUS LOOP
# ============================================================

async def run_continuous(
    symbols:        Optional[List[str]] = None,
    top_n:          int                 = TOP_SYMBOLS,
    interval:       int                 = SCAN_INTERVAL,
    analyze_every:  int                 = 0,
):
    """Run scanner in continuous loop."""
    pattern_db = load_pattern_db()
    cycle      = 0

    print(f"\n  Research Scanner — Continuous Mode")
    print(f"  Interval: {interval}s ({interval // 60} min)")
    print(f"  Patterns loaded: {len(pattern_db)}")
    if analyze_every:
        print(f"  Auto-analyze every {analyze_every} cycles")
    print(f"  Press Ctrl+C to stop\n")

    while True:
        try:
            cycle += 1
            await run_scan_cycle(symbols=symbols, top_n=top_n, pattern_db=pattern_db)
            save_pattern_db(pattern_db)

            # Auto-analyze if requested
            if analyze_every and cycle % analyze_every == 0:
                print(f"\n  Auto-running pattern analysis (cycle {cycle}) ...")
                try:
                    from pattern_analyzer import run_analysis
                    run_analysis(min_samples=5)
                except Exception as e:
                    print(f"  [ANALYZE] Error: {e}")

            # Also run outcome labeler periodically
            if cycle % 6 == 0:   # every 30 min
                print(f"\n  Auto-running outcome labeler ...")
                try:
                    from outcome_labeler import label_events
                    await label_events(rerun=False)
                except Exception as e:
                    print(f"  [LABELER] Error: {e}")

            next_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"\n  Next scan in {interval}s (now={next_time})")
            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n  Scanner stopped. {cycle} cycles. {len(pattern_db)} patterns saved.")
            save_pattern_db(pattern_db)
            break
        except Exception as e:
            print(f"  [LOOP] Error: {e}. Retrying in 30s ...")
            await asyncio.sleep(30)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Research Scanner — pre-pump/dump pattern data collector"
    )
    parser.add_argument("--top",          type=int, default=TOP_SYMBOLS,
                        help="Number of top symbols to scan")
    parser.add_argument("--symbols",      type=str, default=None,
                        help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)")
    parser.add_argument("--loop",         action="store_true",
                        help="Run continuously")
    parser.add_argument("--interval",     type=int, default=SCAN_INTERVAL,
                        help="Seconds between scans (loop mode)")
    parser.add_argument("--analyze",      action="store_true",
                        help="Run pattern analysis only (no scan)")
    parser.add_argument("--label",        action="store_true",
                        help="Run outcome labeler only")
    parser.add_argument("--analyze-every",type=int, default=0,
                        help="Auto-analyze every N cycles (loop mode)")
    args = parser.parse_args()

    ensure_data_dir()

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Analyze-only mode
    if args.analyze:
        from pattern_analyzer import run_analysis
        run_analysis()
        return

    # Label-only mode
    if args.label:
        asyncio.run(
            __import__("outcome_labeler").label_events()
        )
        return

    # Loop mode
    if args.loop:
        asyncio.run(run_continuous(
            symbols       = symbols,
            top_n         = args.top,
            interval      = args.interval,
            analyze_every = args.analyze_every,
        ))
        return

    # Single scan
    pattern_db = load_pattern_db()
    asyncio.run(run_scan_cycle(symbols=symbols, top_n=args.top, pattern_db=pattern_db))
    save_pattern_db(pattern_db)


if __name__ == "__main__":
    main()
