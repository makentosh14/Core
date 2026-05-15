#!/usr/bin/env python3
"""
build_labeled_dataset.py — Phase 7 Turn 1 CLI.

Builds a labeled training dataset for replacing score.py:WEIGHTS with a
properly-trained classifier.

Output: labeled_dataset.csv with ~80 feature columns + label + metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from labeled_dataset_builder import (
    DatasetBuilderConfig,
    build_dataset,
    write_dataset_csv,
    summarize_dataset,
)


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


async def main():
    p = argparse.ArgumentParser(description="Build labeled training dataset for scoring model.")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--feature-tfs", default="5,15,60",
                   help="Comma-separated TFs for feature extraction (default: 5,15,60)")
    p.add_argument("--primary-tf", default="5",
                   help="Primary TF for the decision timeline (default: 5)")
    p.add_argument("--min-move-pct", type=float, default=3.0,
                   help="Pump/dump threshold for label=1 (default: 3.0%%)")
    p.add_argument("--future-bars", type=int, default=4,
                   help="Bars ahead to check for the move (default: 4 = 20min on 5m)")
    p.add_argument("--negative-sample-rate", type=float, default=0.10,
                   help="Keep this fraction of negatives (default: 0.10)")
    p.add_argument("--output", default="labeled_dataset.csv")
    args = p.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    feature_tfs = tuple(s.strip() for s in args.feature_tfs.split(",") if s.strip())

    cfg = DatasetBuilderConfig(
        symbols=symbols,
        days=args.days,
        feature_tfs=feature_tfs,
        primary_tf=args.primary_tf,
        min_move_pct=args.min_move_pct,
        future_bars=args.future_bars,
        negative_sample_rate=args.negative_sample_rate,
    )

    print(f"Building labeled dataset:")
    print(f"  symbols:               {symbols}")
    print(f"  days:                  {args.days}")
    print(f"  feature TFs:           {feature_tfs}")
    print(f"  primary TF:            {args.primary_tf}")
    print(f"  label threshold:       {args.min_move_pct}% within {args.future_bars} bars")
    print(f"  negative sample rate:  {args.negative_sample_rate * 100:.0f}%")
    print()

    t0 = time.time()
    samples = await build_dataset(cfg, verbose=True)
    print(f"\n  data + features: {time.time() - t0:.1f}s")

    n_rows, n_pos = write_dataset_csv(samples, args.output)
    print(f"\nWrote {n_rows} rows ({n_pos} positives) to {args.output}\n")
    print(summarize_dataset(samples))

    return 0 if n_rows > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n[interrupted]")
        sys.exit(130)
