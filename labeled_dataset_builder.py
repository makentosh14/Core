"""
labeled_dataset_builder.py — Phase 7 Turn 1.

Build a properly-labeled training dataset for the scoring model.

Why this exists:
  The current `score.py:WEIGHTS` were hand-tuned (or derived from
  indicator_backscan output) using a PURE POSITIVE-CLASS dataset — only
  3%+ pump/dump events were recorded. As a result the weights had high
  recall on pumps but no precision against false positives. The Phase 4
  runner analyzer confirmed this empirically: 13 of 14 ≥3% moves on
  30-day data were MISSED, because the scorer didn't fire at start-of-move
  bars (it requires "high confluence" that mostly only appears after
  the move is well underway).

What this module does:
  Walks historical candles and emits ONE LABELED SAMPLE PER BAR (with
  stratified subsampling of negatives). Features per sample come from
  indicator_backscan's analyze_tf() applied to 5m, 15m, and 60m TFs.
  Label = 1 if a >= MIN_MOVE_PCT move happens in the next FUTURE_BARS,
  0 otherwise.

  The resulting CSV can be loaded into scikit-learn / LightGBM / etc.
  to train a real classifier whose output replaces score.py:WEIGHTS.

  CRITICAL: the lookahead label uses ONLY future candles relative to
  the feature-extraction bar. Indicators at bar T see candles [0..T].
  Label at bar T uses candles [T+1 .. T+FUTURE_BARS]. No leakage.

USAGE:
  python build_labeled_dataset.py \
      --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT \
      --days 90 \
      --output labeled_dataset.csv
"""

from __future__ import annotations

import asyncio
import csv
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Label settings — calibrated to match the runner analyzer's detection
# window. The smoke test with future_bars=4 produced ZERO positives across
# 30 days × 5 symbols × 8640 bars because 3% moves rarely complete in
# 20 minutes; they typically take 1-2 hours. Defaults bumped to:
#   * future_bars = 20 (= 100 min on 5m) — matches runner_analyzer.detect_runners
#   * min_move_pct = 2.0 — catches moves that the bot's 1.0% Intraday SL
#     can profit from, even if they don't reach the 3% "runner" definition.
#     Users wanting only big-mover labels can override to 3.0 on the CLI.
MIN_MOVE_PCT = 2.0
FUTURE_BARS = 20
FEATURE_TFS = ("5", "15", "60")
PRIMARY_TF = "5"

# How many bars of history each TF needs before we trust analyze_tf output.
MIN_HISTORY = 60


# Indicator helpers come from indicator_backscan — same feature extraction
# as the original backscan, so a model trained here is directly comparable
# to (and replaceable for) the existing score.py:WEIGHTS.


def _dict_candle_to_indicator_backscan_candle(c: Dict[str, Any]):
    """Convert a dict-style candle (used everywhere else) to the
    dataclass instance that indicator_backscan.analyze_tf expects."""
    from indicator_backscan import Candle
    return Candle(
        timestamp=int(c["timestamp"]),
        open=float(c["open"]),
        high=float(c["high"]),
        low=float(c["low"]),
        close=float(c["close"]),
        volume=float(c["volume"]),
    )


def _candles_for_analysis(
    raw_candles: List[Dict[str, Any]],
    up_to_idx: int,
    history_bars: int = 200,
) -> List:
    """Return the most recent `history_bars` candles up to AND INCLUDING
    raw_candles[up_to_idx], converted to indicator_backscan Candle dataclass."""
    if up_to_idx < 0 or up_to_idx >= len(raw_candles):
        return []
    start = max(0, up_to_idx - history_bars + 1)
    end = up_to_idx + 1
    return [_dict_candle_to_indicator_backscan_candle(c) for c in raw_candles[start:end]]


def _label_at(
    primary_candles: List[Dict[str, Any]],
    bar_idx: int,
    future_bars: int = FUTURE_BARS,
    min_move_pct: float = MIN_MOVE_PCT,
) -> Tuple[int, float, str]:
    """Compute the label for the bar at `bar_idx`. The label is 1 iff,
    within the next `future_bars` bars after bar_idx, price moves by at
    least `min_move_pct` in either direction.

    Returns (label, max_signed_pct_move_in_window, "pump"/"dump"/"none").

    No lookahead bias: this function reads ONLY candles strictly after
    bar_idx, never the features-extraction bar itself.
    """
    if bar_idx + future_bars >= len(primary_candles):
        return (0, 0.0, "none")  # not enough future history to label

    anchor_price = float(primary_candles[bar_idx]["close"])
    if anchor_price <= 0:
        return (0, 0.0, "none")

    max_high = anchor_price
    min_low = anchor_price
    for j in range(bar_idx + 1, bar_idx + 1 + future_bars):
        h = float(primary_candles[j].get("high", primary_candles[j].get("close", 0)))
        lo = float(primary_candles[j].get("low", primary_candles[j].get("close", 0)))
        if h > max_high:
            max_high = h
        if 0 < lo < min_low:
            min_low = lo

    pump_pct = (max_high - anchor_price) / anchor_price * 100
    dump_pct = (min_low - anchor_price) / anchor_price * 100

    if pump_pct >= min_move_pct and pump_pct >= abs(dump_pct):
        return (1, round(pump_pct, 3), "pump")
    if dump_pct <= -min_move_pct and abs(dump_pct) > pump_pct:
        return (1, round(dump_pct, 3), "dump")
    # Return the LARGER magnitude direction for diagnostic, even if below threshold.
    if abs(dump_pct) > pump_pct:
        return (0, round(dump_pct, 3), "none")
    return (0, round(pump_pct, 3), "none")


def _build_features_at(
    primary_candles: List[Dict[str, Any]],
    by_tf_candles: Dict[str, List[Dict[str, Any]]],
    primary_idx: int,
    primary_tf: str,
    feature_tfs: Tuple[str, ...],
    history_bars: int = 200,
) -> Optional[Dict[str, Any]]:
    """Extract a flat feature dict from analyze_tf called on each MTF.

    Lookahead-safe: each TF's candles are filtered to those whose close-time
    is <= the primary bar's timestamp. (The primary TF includes the current
    bar — its close is "current price" from the bot's POV.)
    """
    from indicator_backscan import analyze_tf

    # Local interval map — keep this module self-contained.
    interval_sec = {"1": 60, "3": 180, "5": 300, "15": 900,
                    "30": 1800, "60": 3600, "240": 14400}

    primary_ts = int(primary_candles[primary_idx]["timestamp"])

    features: Dict[str, Any] = {}
    for tf in feature_tfs:
        if tf not in by_tf_candles:
            continue
        candles_for_tf = by_tf_candles[tf]
        if tf == primary_tf:
            visible = [c for c in candles_for_tf if int(c["timestamp"]) <= primary_ts]
        else:
            tf_sec = interval_sec.get(tf, 60)
            visible = [c for c in candles_for_tf
                       if int(c["timestamp"]) + tf_sec * 1000 <= primary_ts]

        if len(visible) < MIN_HISTORY:
            continue

        # Convert to indicator_backscan Candle dataclass for analyze_tf
        bs_candles = [_dict_candle_to_indicator_backscan_candle(c)
                      for c in visible[-history_bars:]]

        tf_features = analyze_tf(bs_candles, prefix=f"{tf}m")
        features.update(tf_features)

    return features if features else None


@dataclass
class DatasetBuilderConfig:
    symbols: List[str] = field(default_factory=list)
    days: int = 90
    feature_tfs: Tuple[str, ...] = FEATURE_TFS
    primary_tf: str = PRIMARY_TF
    min_move_pct: float = MIN_MOVE_PCT
    future_bars: int = FUTURE_BARS
    # Stratified subsampling: keep this fraction of negative samples.
    negative_sample_rate: float = 0.10
    # Warmup bars on primary TF before sampling starts (need indicator history)
    warmup_bars: int = MIN_HISTORY


@dataclass
class LabeledSample:
    symbol: str
    timestamp_ms: int
    bar_idx: int
    label: int
    move_pct_following: float
    move_direction: str
    features: Dict[str, Any]


async def build_dataset(
    config: DatasetBuilderConfig,
    verbose: bool = True,
) -> List[LabeledSample]:
    """Build a labeled dataset across the specified symbols.

    Uses the same disk-cache and pagination as backtest_engine for data load.
    """
    from backtest_engine import load_or_fetch_history, days_to_bars
    import random

    # Load candles for all symbols × feature TFs.
    candles_by_symbol_tf: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for sym in config.symbols:
        if verbose:
            print(f"  [data] {sym} ...", end="", flush=True)
        candles_by_symbol_tf[sym] = {}
        for tf in config.feature_tfs:
            bars_needed = days_to_bars(config.days, tf)
            try:
                c = await load_or_fetch_history(sym, tf, limit=bars_needed)
                if c:
                    candles_by_symbol_tf[sym][tf] = c
            except Exception as e:
                if verbose:
                    print(f"\n    [WARN] {sym} {tf}m: {e}", end="")
        if verbose:
            counts = {tf: len(candles_by_symbol_tf[sym].get(tf, [])) for tf in config.feature_tfs}
            print(f" {counts}")

    # For each symbol, walk the primary-TF series and emit labeled samples.
    samples: List[LabeledSample] = []
    rng = random.Random(42)  # deterministic stratified subsampling

    for sym, by_tf in candles_by_symbol_tf.items():
        primary = by_tf.get(config.primary_tf, [])
        if not primary or len(primary) < config.warmup_bars + config.future_bars + 1:
            if verbose:
                print(f"  [{sym}] skipped — insufficient primary-TF history")
            continue

        n_processed = 0
        n_positives = 0
        n_negatives_kept = 0
        n_negatives_skipped = 0

        for idx in range(config.warmup_bars, len(primary) - config.future_bars):
            label, move_pct, direction = _label_at(
                primary, idx,
                future_bars=config.future_bars,
                min_move_pct=config.min_move_pct,
            )

            # Subsample negatives so the dataset isn't completely imbalanced.
            if label == 0 and rng.random() > config.negative_sample_rate:
                n_negatives_skipped += 1
                continue

            features = _build_features_at(
                primary_candles=primary,
                by_tf_candles=by_tf,
                primary_idx=idx,
                primary_tf=config.primary_tf,
                feature_tfs=config.feature_tfs,
            )
            if features is None:
                continue

            samples.append(LabeledSample(
                symbol=sym,
                timestamp_ms=int(primary[idx]["timestamp"]),
                bar_idx=idx,
                label=label,
                move_pct_following=move_pct,
                move_direction=direction,
                features=features,
            ))
            n_processed += 1
            if label == 1:
                n_positives += 1
            else:
                n_negatives_kept += 1

        if verbose:
            print(
                f"  [{sym}] processed: {n_processed} samples "
                f"({n_positives} positive, {n_negatives_kept} negative kept, "
                f"{n_negatives_skipped} negative skipped via subsampling)"
            )

    return samples


def write_dataset_csv(samples: List[LabeledSample], path: str) -> Tuple[int, int]:
    """Write samples to a CSV. Returns (n_rows, n_positives)."""
    if not samples:
        return (0, 0)

    # Build the column list from the union of features across all samples
    # (some indicator outputs may be conditional, e.g. divergences only appear
    # if there's enough history).
    feature_keys: set = set()
    for s in samples:
        feature_keys.update(s.features.keys())
    feature_keys = sorted(feature_keys)

    header = ["symbol", "timestamp_ms", "bar_idx", "label",
              "move_pct_following", "move_direction"] + list(feature_keys)

    n_pos = 0
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for s in samples:
            row = {
                "symbol": s.symbol,
                "timestamp_ms": s.timestamp_ms,
                "bar_idx": s.bar_idx,
                "label": s.label,
                "move_pct_following": s.move_pct_following,
                "move_direction": s.move_direction,
            }
            row.update(s.features)
            w.writerow(row)
            if s.label == 1:
                n_pos += 1
    return (len(samples), n_pos)


def summarize_dataset(samples: List[LabeledSample]) -> str:
    if not samples:
        return "(empty dataset)"
    total = len(samples)
    pos = sum(1 for s in samples if s.label == 1)
    neg = total - pos
    by_sym: Dict[str, Dict[str, int]] = {}
    for s in samples:
        b = by_sym.setdefault(s.symbol, {"pos": 0, "neg": 0})
        b["pos" if s.label == 1 else "neg"] += 1

    lines = []
    lines.append("=" * 64)
    lines.append("LABELED DATASET SUMMARY")
    lines.append("=" * 64)
    lines.append(f"  Total samples:     {total}")
    lines.append(f"  Positives (1):     {pos}  ({pos/total*100:.1f}%)")
    lines.append(f"  Negatives (0):     {neg}  ({neg/total*100:.1f}%)")
    lines.append(f"  Class ratio:       1 positive per {neg/pos:.1f} negatives" if pos > 0 else "  (no positives)")
    lines.append("-" * 64)
    lines.append(f"  Feature columns:   {len(samples[0].features)}")
    lines.append("-" * 64)
    lines.append("  Per symbol:")
    for sym, counts in sorted(by_sym.items()):
        t = counts["pos"] + counts["neg"]
        lines.append(
            f"    {sym:<12s}  total={t:>6}  pos={counts['pos']:>4}  "
            f"neg={counts['neg']:>5}  pos_rate={counts['pos']/t*100:.2f}%"
        )
    lines.append("=" * 64)
    return "\n".join(lines)


def extract_features_for_inference(
    candles_by_tf: Dict[str, List[Dict[str, Any]]],
    primary_tf: str = PRIMARY_TF,
    feature_tfs: tuple = FEATURE_TFS,
) -> Dict[str, Any]:
    """Inference-time wrapper around _build_features_at.

    Given the most recent candles_by_tf (as passed to score.py at runtime),
    extract the same feature dict the model was trained on. Anchors at
    the LAST primary-TF bar so the features represent "right now."

    Returns {} if no feature TF has enough history.
    """
    primary_candles = candles_by_tf.get(primary_tf, [])
    if not primary_candles:
        return {}
    last_idx = len(primary_candles) - 1
    features = _build_features_at(
        primary_candles=primary_candles,
        by_tf_candles=candles_by_tf,
        primary_idx=last_idx,
        primary_tf=primary_tf,
        feature_tfs=feature_tfs,
    )
    return features or {}


__all__ = [
    "DatasetBuilderConfig",
    "LabeledSample",
    "build_dataset",
    "write_dataset_csv",
    "summarize_dataset",
    "extract_features_for_inference",
    "_label_at",
    "_build_features_at",
]
