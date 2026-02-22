#!/usr/bin/env python3
"""
label_events.py
===============
Scans candle history and labels pump/dump events.

An event is labeled at bar index `i` (the START bar) if within the next
`horizon_bars` bars the price moves by at least the threshold pct.

NO lookahead: the feature window ends at bar i-1 (strictly before).

Usage:
    from label_events import label_events, HORIZONS, THRESHOLDS
"""

from typing import List, Dict, Tuple, Any
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────
# Map TF string -> approximate minutes per bar
TF_MINUTES: Dict[str, int] = {
    "1": 1, "3": 3, "5": 5, "15": 15,
    "30": 30, "60": 60, "120": 120, "240": 240, "D": 1440
}

# Horizons in minutes. We'll convert to bars per TF.
HORIZON_MINUTES: List[int] = [30, 120, 360]   # 30m, 2h, 6h

# Pump/dump magnitude thresholds (%)
THRESHOLDS: List[float] = [5.0, 10.0, 20.0]

# Minimum cooldown between events (bars) to avoid overlapping labels
COOLDOWN_BARS: int = 5
# ──────────────────────────────────────────────────────────────


def _horizon_bars(tf: str, horizon_minutes: int) -> int:
    """Convert a horizon in minutes to number of bars for the given timeframe."""
    mins = TF_MINUTES.get(str(tf), 1)
    return max(1, horizon_minutes // mins)


def label_events(
    candles: List[Dict[str, Any]],
    tf: str = "15",
    horizon_minutes: int = 120,
    thresholds: List[float] = None,
    cooldown_bars: int = COOLDOWN_BARS
) -> List[Dict[str, Any]]:
    """
    Scan candles and return a list of event dicts.

    Each event dict:
    {
        "event_idx":   int,      # bar index where move STARTS (close of this bar is entry)
        "direction":   "pump"|"dump",
        "magnitude":   float,    # actual % move achieved within horizon
        "bucket":      5|10|20,  # threshold bucket met
        "horizon_bars":int,
        "tf":          str,
        "open_price":  float,    # close of bar at event_idx (used as reference)
        "peak_price":  float,    # max/min within horizon
        "ts_start":    int|None  # timestamp of event start bar
    }

    Bars are indexed 0..N-1. The feature window is candles[0..event_idx-1].
    Minimum candles required: horizon_bars + lookback_window
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    closes = np.array([float(c["close"]) for c in candles])
    highs  = np.array([float(c["high"])  for c in candles])
    lows   = np.array([float(c["low"])   for c in candles])
    n = len(closes)

    horizon = _horizon_bars(tf, horizon_minutes)
    events: List[Dict[str, Any]] = []
    last_event_idx = -cooldown_bars  # allow first bar

    for i in range(n - horizon):
        if i - last_event_idx < cooldown_bars:
            continue

        ref_price = closes[i]
        if ref_price <= 0:
            continue

        future_highs = highs[i + 1 : i + 1 + horizon]
        future_lows  = lows[i + 1 : i + 1 + horizon]

        max_up   = (np.max(future_highs) - ref_price) / ref_price * 100
        max_down = (ref_price - np.min(future_lows))  / ref_price * 100

        # Find highest threshold met
        pump_bucket = None
        for thr in sorted(thresholds, reverse=True):
            if max_up >= thr:
                pump_bucket = thr
                break

        dump_bucket = None
        for thr in sorted(thresholds, reverse=True):
            if max_down >= thr:
                dump_bucket = thr
                break

        # Prefer whichever move is larger
        if pump_bucket is not None and dump_bucket is not None:
            if max_up >= max_down:
                dump_bucket = None
            else:
                pump_bucket = None

        if pump_bucket is not None:
            events.append({
                "event_idx":    i,
                "direction":    "pump",
                "magnitude":    round(max_up, 4),
                "bucket":       pump_bucket,
                "horizon_bars": horizon,
                "tf":           tf,
                "open_price":   ref_price,
                "peak_price":   float(np.max(future_highs)),
                "ts_start":     candles[i].get("ts") or candles[i].get("timestamp")
            })
            last_event_idx = i

        elif dump_bucket is not None:
            events.append({
                "event_idx":    i,
                "direction":    "dump",
                "magnitude":    round(max_down, 4),
                "bucket":       dump_bucket,
                "horizon_bars": horizon,
                "tf":           tf,
                "open_price":   ref_price,
                "peak_price":   float(np.min(future_lows)),
                "ts_start":     candles[i].get("ts") or candles[i].get("timestamp")
            })
            last_event_idx = i

    return events


def label_events_multi_horizon(
    candles: List[Dict[str, Any]],
    tf: str = "15",
    horizons: List[int] = None,
    thresholds: List[float] = None,
    cooldown_bars: int = COOLDOWN_BARS
) -> List[Dict[str, Any]]:
    """
    Run label_events for each horizon and merge results.
    Deduplicates by event_idx (keeps highest bucket).
    """
    if horizons is None:
        horizons = HORIZON_MINUTES

    all_events: Dict[int, Dict] = {}
    for h in horizons:
        evts = label_events(candles, tf=tf, horizon_minutes=h,
                            thresholds=thresholds, cooldown_bars=cooldown_bars)
        for ev in evts:
            idx = ev["event_idx"]
            existing = all_events.get(idx)
            if existing is None or ev["bucket"] > existing["bucket"]:
                all_events[idx] = ev

    return sorted(all_events.values(), key=lambda x: x["event_idx"])


def split_train_test(events: List[Dict], test_ratio: float = 0.2) -> Tuple[List, List]:
    """
    Time-based split. NO random shuffle.
    Returns (train_events, test_events).
    """
    n = len(events)
    split = int(n * (1 - test_ratio))
    return events[:split], events[split:]


def walk_forward_splits(events: List[Dict], n_splits: int = 5,
                        min_train: int = 50) -> List[Tuple[List, List]]:
    """
    Walk-forward cross-validation splits (time-based).
    Returns list of (train_events, val_events) tuples.
    """
    n = len(events)
    fold_size = n // (n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_end = (k + 1) * fold_size
        val_end   = (k + 2) * fold_size
        if train_end < min_train:
            continue
        train = events[:train_end]
        val   = events[train_end:val_end]
        if val:
            splits.append((train, val))
    return splits


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    # Quick smoke test with synthetic data
    np.random.seed(42)
    price = 100.0
    candles = []
    for i in range(500):
        o = price
        change = np.random.randn() * 0.5
        c = o + change
        h = max(o, c) + abs(np.random.randn()) * 0.2
        l = min(o, c) - abs(np.random.randn()) * 0.2
        # Inject a pump at bar 100
        if i == 100:
            h += 8.0
            c += 6.0
        candles.append({"open": o, "high": h, "low": l, "close": c, "volume": 1000 + np.random.rand() * 500, "ts": i})
        price = c

    events = label_events_multi_horizon(candles, tf="15", horizons=[30, 120, 360])
    print(f"Found {len(events)} events")
    for ev in events[:5]:
        print(json.dumps(ev, indent=2))
