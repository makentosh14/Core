#!/usr/bin/env python3
"""
scanner_main.py - Advanced Multi-Timeframe Crypto Market Scanner
================================================================
Professional-grade full-market scanner for pump/dump detection.

ARCHITECTURE:
  1. Fetch top N symbols by 24h volume from Bybit
  2. For each symbol, fetch 4 timeframes (5m/15m/1h/4h)
  3. Calculate 30+ indicators per timeframe
  4. Build synchronized MTF snapshots (evaluated every 5 min)
  5. Detect v1 scores + v3 tiered combos (sniper/elite/std)
  6. Store all data to CSV/JSON for dataset analysis
  7. Run historical pattern analysis to discover best combos
  8. Print live signal alerts sorted by tier and confidence

USAGE:
    python scanner_main.py                          # Full scan, top 100 by volume
    python scanner_main.py --top 50                 # Top 50 symbols
    python scanner_main.py --symbols BTCUSDT,ETHUSDT
    python scanner_main.py --loop                   # Continuous scanning
    python scanner_main.py --analyze                # Analyze stored patterns only
"""

import asyncio
import aiohttp
import argparse
import time
import sys
import os
from typing import List, Dict, Optional
from datetime import datetime, timezone

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BYBIT_API, API_TIMEOUT, TIMEFRAMES, WARMUP_CANDLES,
    TOP_SYMBOLS, SCAN_INTERVAL, DATA_DIR,
)
from indicators import Candle, analyze_all_indicators
from mtf_analyzer import build_snapshot, MTFSnapshot
from data_fetcher import (
    fetch_top_symbols_by_volume,
    fetch_symbol_all_timeframes,
    fetch_batch_symbols,
)
from data_storage import (
    ensure_data_dir,
    save_snapshots_csv,
    save_signal_log,
    load_pattern_db,
    save_pattern_db,
    record_pattern,
    analyze_historical_patterns,
    generate_scan_summary,
)


# ============================================================
# CORE SCAN LOGIC
# ============================================================

def process_symbol(
    symbol: str,
    candles_by_tf: Dict[str, List[Candle]],
) -> Optional[MTFSnapshot]:
    """
    Process a single symbol: calculate indicators on all timeframes,
    build synchronized MTF snapshot.
    """
    # Validate we have enough data
    for tf_key, tf_config in TIMEFRAMES.items():
        candles = candles_by_tf.get(tf_key, [])
        min_needed = tf_config["candles"] + 50  # minimum for indicators
        if len(candles) < min_needed:
            return None

    # Calculate indicators per timeframe
    ind_5m = analyze_all_indicators(candles_by_tf["5"], prefix="5m")
    ind_15m = analyze_all_indicators(candles_by_tf["15"], prefix="15m")
    ind_1h = analyze_all_indicators(candles_by_tf["60"], prefix="1h")
    ind_4h = analyze_all_indicators(candles_by_tf["240"], prefix="4h")

    # Get current price and timestamp from 5m (fastest TF)
    last_candle = candles_by_tf["5"][-1]
    price = last_candle.close
    timestamp = last_candle.timestamp

    # Build synchronized snapshot
    snapshot = build_snapshot(
        symbol=symbol,
        timestamp=timestamp,
        price=price,
        ind_5m=ind_5m,
        ind_15m=ind_15m,
        ind_1h=ind_1h,
        ind_4h=ind_4h,
    )

    return snapshot


# ============================================================
# SCAN CYCLE
# ============================================================

async def run_scan_cycle(
    symbols: Optional[List[str]] = None,
    top_n: int = TOP_SYMBOLS,
    pattern_db: Optional[List] = None,
) -> List[MTFSnapshot]:
    """
    Execute one full scan cycle across all symbols.
    Returns list of MTFSnapshots.
    """
    scan_start = time.time()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("=" * 72)
    print(f"  ADVANCED MULTI-TIMEFRAME MARKET SCANNER")
    print(f"  Scan started: {now}")
    print(f"  Timeframes: 5m (15 candles) | 15m (12) | 1h (7) | 4h (5)")
    print(f"  Indicators: 30+ per TF | Signal tiers: Sniper/Elite/Std")
    print("=" * 72)

    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        # Step 1: Get symbols
        if not symbols:
            print(f"\n  [1/5] Fetching top {top_n} symbols by 24h volume...")
            tickers = await fetch_top_symbols_by_volume(session, top_n=top_n)
            symbols = [t["symbol"] for t in tickers]
            if tickers:
                top3 = ", ".join(f"{t['symbol']}(${t['turnover24h']/1e6:.0f}M)"
                                 for t in tickers[:3])
                print(f"        Found {len(symbols)} symbols. Top 3: {top3}")
        else:
            print(f"\n  [1/5] Using {len(symbols)} provided symbols")

        if not symbols:
            print("  ERROR: No symbols found. Check API connection.")
            return []

        # Step 2: Fetch all timeframe data
        print(f"\n  [2/5] Fetching candle data for {len(symbols)} symbols × 4 timeframes...")
        all_data = await fetch_batch_symbols(session, symbols)
        print(f"        Successfully fetched {len(all_data)} symbols")

    # Step 3: Process indicators and build snapshots
    print(f"\n  [3/5] Computing 30+ indicators per timeframe...")
    snapshots = []
    skipped = 0

    for i, (symbol, candles_by_tf) in enumerate(all_data.items()):
        snap = process_symbol(symbol, candles_by_tf)
        if snap:
            snapshots.append(snap)
        else:
            skipped += 1

        if (i + 1) % 20 == 0 or i == len(all_data) - 1:
            print(f"        [{i + 1}/{len(all_data)}] processed...")

    print(f"        Computed: {len(snapshots)} snapshots ({skipped} skipped)")

    # Step 4: Store data
    print(f"\n  [4/5] Storing data to {DATA_DIR}/...")
    save_snapshots_csv(snapshots)
    save_signal_log(snapshots)

    # Record patterns for learning
    if pattern_db is not None:
        for snap in snapshots:
            if snap.direction:
                record_pattern(snap, pattern_db=pattern_db)

    # Step 5: Generate summary and display results
    print(f"\n  [5/5] Analyzing results...")

    pattern_analysis = None
    if pattern_db:
        pattern_analysis = analyze_historical_patterns(pattern_db)
        save_pattern_db(pattern_db)

    summary = generate_scan_summary(snapshots, pattern_analysis)

    # Display results
    elapsed = time.time() - scan_start
    display_results(snapshots, summary, elapsed)

    return snapshots


def display_results(
    snapshots: List[MTFSnapshot],
    summary: Dict,
    elapsed: float,
):
    """Print formatted scan results to console."""

    signals = [s for s in snapshots if s.direction]
    pumps = sorted([s for s in signals if s.direction == "pump"],
                   key=lambda x: (x.tier, x.confidence), reverse=True)
    dumps = sorted([s for s in signals if s.direction == "dump"],
                   key=lambda x: (x.tier, x.confidence), reverse=True)

    tier_emoji = {3: "🎯", 2: "⚡", 1: "📊", 0: ""}
    tier_name = {3: "SNIPER", 2: "ELITE", 1: "STD", 0: ""}

    print("\n" + "=" * 72)
    print(f"  SCAN RESULTS  |  {len(snapshots)} symbols  |  {len(signals)} signals  |  {elapsed:.1f}s")
    print("=" * 72)

    # Pump signals
    if pumps:
        print(f"\n  🚀 PUMP SIGNALS ({len(pumps)})")
        print(f"  {'Symbol':<14} {'Type':<22} {'Tier':<10} {'Conf':<8} {'Score':<8} {'Price':<14}")
        print(f"  {'─' * 80}")
        for s in pumps[:15]:
            emoji = tier_emoji.get(s.tier, "")
            tn = tier_name.get(s.tier, "")
            print(f"  {emoji} {s.symbol:<12} {s.signal_type:<22} {tn:<8} "
                  f"{s.confidence:>5.1f}%  {s.net_score:>+6.1f}  ${s.price:<12.6g}")

    # Dump signals
    if dumps:
        print(f"\n  🔻 DUMP SIGNALS ({len(dumps)})")
        print(f"  {'Symbol':<14} {'Type':<22} {'Tier':<10} {'Conf':<8} {'Score':<8} {'Price':<14}")
        print(f"  {'─' * 80}")
        for s in dumps[:15]:
            emoji = tier_emoji.get(s.tier, "")
            tn = tier_name.get(s.tier, "")
            print(f"  {emoji} {s.symbol:<12} {s.signal_type:<22} {tn:<8} "
                  f"{s.confidence:>5.1f}%  {s.net_score:>+6.1f}  ${s.price:<12.6g}")

    if not signals:
        print("\n  No active signals detected in this scan cycle.")

    # Market overview
    bull_dom = sum(1 for s in snapshots if s.mtf_signals.get("mtf_moderate_bull", False))
    bear_dom = sum(1 for s in snapshots if s.mtf_signals.get("mtf_moderate_bear", False))
    neutral = len(snapshots) - bull_dom - bear_dom

    print(f"\n  📈 Market Overview:")
    print(f"     Bullish: {bull_dom} | Bearish: {bear_dom} | Neutral: {neutral}")
    print(f"     Snipers: {summary.get('signal_breakdown', {}).get('snipers', 0)} | "
          f"Elite: {summary.get('signal_breakdown', {}).get('elite', 0)} | "
          f"Standard: {summary.get('signal_breakdown', {}).get('standard', 0)}")

    # Pattern analysis summary (if available)
    pa = summary.get("pattern_analysis")
    if pa and pa.get("total_patterns", 0) > 10:
        print(f"\n  🧠 Pattern Database: {pa['total_patterns']} patterns stored")
        by_type = pa.get("by_signal_type", {})
        for sig_type, stats in sorted(by_type.items()):
            wr = stats.get("win_rate")
            if wr is not None and stats.get("evaluated", 0) >= 5:
                print(f"     {sig_type}: {wr:.1f}% win rate ({stats['evaluated']} evaluated)")

        best = pa.get("best_indicators", {})
        if best:
            print(f"\n  📊 Best Indicators (by win rate):")
            for name, stats in list(best.items())[:5]:
                print(f"     {name}: {stats['win_rate']:.1f}% "
                      f"({stats['win_count']}W/{stats['loss_count']}L)")

    print(f"\n  ⏱  Scan completed in {elapsed:.1f}s")
    print(f"  💾 Data saved to: {DATA_DIR}/")
    print("=" * 72)


# ============================================================
# CONTINUOUS LOOP MODE
# ============================================================

async def run_continuous(
    symbols: Optional[List[str]] = None,
    top_n: int = TOP_SYMBOLS,
    interval: int = SCAN_INTERVAL,
):
    """Run scanner in continuous loop mode."""
    pattern_db = load_pattern_db()
    cycle = 0

    print(f"\n  Starting continuous scanning mode")
    print(f"  Interval: {interval}s ({interval // 60}min)")
    print(f"  Pattern DB: {len(pattern_db)} existing patterns loaded")
    print(f"  Press Ctrl+C to stop\n")

    while True:
        try:
            cycle += 1
            print(f"\n{'━' * 72}")
            print(f"  CYCLE #{cycle}")
            print(f"{'━' * 72}")

            await run_scan_cycle(symbols=symbols, top_n=top_n, pattern_db=pattern_db)

            # Save patterns after each cycle
            save_pattern_db(pattern_db)

            # Wait for next cycle
            next_scan = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(f"\n  Next scan in {interval}s (at ~{next_scan} + {interval}s)")
            await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\n  Scanner stopped by user after {cycle} cycles.")
            print(f"  Pattern DB: {len(pattern_db)} patterns saved.")
            save_pattern_db(pattern_db)
            break
        except Exception as e:
            print(f"\n  ERROR in cycle {cycle}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Retrying in 30s...")
            await asyncio.sleep(30)


# ============================================================
# PATTERN ANALYSIS MODE
# ============================================================

def run_analysis_only():
    """Load and analyze stored patterns without scanning."""
    ensure_data_dir()
    pattern_db = load_pattern_db()

    if not pattern_db:
        print("  No patterns in database. Run a scan first.")
        return

    print(f"\n  Analyzing {len(pattern_db)} stored patterns...")
    analysis = analyze_historical_patterns(pattern_db)

    print(f"\n{'=' * 60}")
    print(f"  PATTERN ANALYSIS RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total patterns: {analysis['total_patterns']}")

    print(f"\n  By Signal Type:")
    for sig_type, stats in analysis.get("by_signal_type", {}).items():
        wr = stats.get("win_rate")
        wr_str = f"{wr:.1f}%" if wr is not None else "N/A"
        print(f"    {sig_type:<25} total={stats['total']:<5} "
              f"evaluated={stats.get('evaluated', 0):<5} "
              f"win_rate={wr_str}")

    print(f"\n  By Tier:")
    for tier, stats in sorted(analysis.get("by_tier", {}).items()):
        wr = stats.get("win_rate")
        wr_str = f"{wr:.1f}%" if wr is not None else "N/A"
        tier_name = {3: "Sniper", 2: "Elite", 1: "Standard", 0: "None"}.get(tier, f"T{tier}")
        print(f"    {tier_name:<12} total={stats['total']:<5} "
              f"evaluated={stats.get('evaluated', 0):<5} "
              f"win_rate={wr_str}")

    print(f"\n  Best Indicators:")
    for name, stats in list(analysis.get("best_indicators", {}).items())[:10]:
        print(f"    {name:<30} win_rate={stats['win_rate']:.1f}% "
              f"({stats['win_count']}W/{stats['loss_count']}L, "
              f"n={stats['total_occurrences']})")

    # Save analysis
    import json
    fp = os.path.join(DATA_DIR, "pattern_analysis.json")
    with open(fp, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\n  Analysis saved to: {fp}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Multi-Timeframe Crypto Market Scanner"
    )
    parser.add_argument(
        "--top", type=int, default=TOP_SYMBOLS,
        help=f"Number of top symbols to scan (default: {TOP_SYMBOLS})"
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbol list (e.g., BTCUSDT,ETHUSDT)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run in continuous scanning mode"
    )
    parser.add_argument(
        "--interval", type=int, default=SCAN_INTERVAL,
        help=f"Seconds between scan cycles in loop mode (default: {SCAN_INTERVAL})"
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="Analyze stored patterns only (no scanning)"
    )

    args = parser.parse_args()

    ensure_data_dir()

    if args.analyze:
        run_analysis_only()
        return

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]

    if args.loop:
        asyncio.run(run_continuous(
            symbols=symbols,
            top_n=args.top,
            interval=args.interval,
        ))
    else:
        asyncio.run(run_scan_cycle(
            symbols=symbols,
            top_n=args.top,
        ))


if __name__ == "__main__":
    main()
