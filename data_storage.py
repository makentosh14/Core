#!/usr/bin/env python3
"""
data_storage.py - Dataset Storage & Historical Pattern Analysis
================================================================
Stores all scanner snapshots, signals, and indicator data for:
  - CSV export (for external analysis / ML)
  - JSON pattern database (for internal pattern matching)
  - Historical pattern discovery (what conditions precede pumps/dumps)
  - Statistical summary generation
"""

import csv
import json
import os
import math
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import defaultdict
from dataclasses import asdict

from config import (
    DATA_DIR, EVENTS_CSV, SNAPSHOT_CSV, PATTERN_DB,
    SUMMARY_JSON, SIGNAL_LOG,
)
from mtf_analyzer import MTFSnapshot


# ============================================================
# INITIALIZATION
# ============================================================

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(f"{DATA_DIR}/snapshots", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/signals", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/patterns", exist_ok=True)


# ============================================================
# SNAPSHOT STORAGE (CSV)
# ============================================================

def snapshot_to_flat_dict(snap: MTFSnapshot) -> Dict:
    """Flatten a snapshot into a single-row dict for CSV export."""
    row = {
        "timestamp": snap.timestamp,
        "datetime": snap.dt_str,
        "symbol": snap.symbol,
        "price": snap.price,
        "pump_score": snap.pump_score,
        "dump_score": snap.dump_score,
        "net_score": snap.net_score,
        "direction": snap.direction,
        "signal_type": snap.signal_type,
        "tier": snap.tier,
        "confidence": snap.confidence,
    }

    # Flatten all indicator dicts
    for d in [snap.ind_5m, snap.ind_15m, snap.ind_1h, snap.ind_4h]:
        for k, v in d.items():
            if isinstance(v, bool):
                row[k] = int(v)
            elif isinstance(v, (int, float)):
                if math.isnan(v):
                    row[k] = ""
                else:
                    row[k] = v
            else:
                row[k] = v

    # Flatten MTF signals
    for k, v in snap.mtf_signals.items():
        if isinstance(v, bool):
            row[k] = int(v)
        else:
            row[k] = v

    # Flatten v3 signals
    for k, v in snap.v3_signals.items():
        if isinstance(v, bool):
            row[f"v3_{k}"] = int(v)
        elif isinstance(v, (int, float)):
            row[f"v3_{k}"] = v

    return row


def save_snapshots_csv(snapshots: List[MTFSnapshot], filename: str = None):
    """Save all snapshots to a CSV file for dataset analysis."""
    ensure_data_dir()
    if not snapshots:
        return

    filepath = os.path.join(DATA_DIR, filename or SNAPSHOT_CSV)
    rows = [snapshot_to_flat_dict(s) for s in snapshots]

    # Collect all unique keys
    all_keys = []
    seen_keys = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  [STORAGE] Saved {len(rows)} snapshots → {filepath}")


# ============================================================
# SIGNAL LOG (CSV)
# ============================================================

def save_signal_log(snapshots: List[MTFSnapshot], filename: str = None):
    """Save only snapshots that have an active signal (direction != '')."""
    ensure_data_dir()
    signals = [s for s in snapshots if s.direction]
    if not signals:
        return

    filepath = os.path.join(DATA_DIR, "signals", filename or SIGNAL_LOG)
    rows = [snapshot_to_flat_dict(s) for s in signals]

    all_keys = []
    seen_keys = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_keys:
                all_keys.append(k)
                seen_keys.add(k)

    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  [SIGNAL] Logged {len(signals)} signals → {filepath}")


# ============================================================
# PATTERN DATABASE (JSON)
# ============================================================

def load_pattern_db(filepath: str = None) -> List[Dict]:
    """Load existing pattern database."""
    fp = filepath or os.path.join(DATA_DIR, "patterns", PATTERN_DB)
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return json.load(f)
    return []


def save_pattern_db(patterns: List[Dict], filepath: str = None):
    """Save pattern database."""
    ensure_data_dir()
    fp = filepath or os.path.join(DATA_DIR, "patterns", PATTERN_DB)
    with open(fp, "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    print(f"  [PATTERN] Saved {len(patterns)} patterns → {fp}")


def record_pattern(
    snapshot: MTFSnapshot,
    outcome: Optional[Dict] = None,
    pattern_db: Optional[List] = None,
) -> Dict:
    """
    Record a detected signal pattern for learning.
    outcome: {"moved_pct": float, "direction": str, "bars_to_move": int}
    """
    pattern = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "symbol": snapshot.symbol,
        "timestamp": snapshot.timestamp,
        "price": snapshot.price,
        "signal_type": snapshot.signal_type,
        "direction": snapshot.direction,
        "tier": snapshot.tier,
        "confidence": snapshot.confidence,
        "pump_score": snapshot.pump_score,
        "dump_score": snapshot.dump_score,
        "net_score": snapshot.net_score,
        # Key indicator values at signal time
        "key_indicators": {
            "rsi_5m": snapshot.ind_5m.get("5m_rsi", 0),
            "rsi_15m": snapshot.ind_15m.get("15m_rsi", 0),
            "rsi_1h": snapshot.ind_1h.get("1h_rsi", 0),
            "rsi_4h": snapshot.ind_4h.get("4h_rsi", 0),
            "bb_squeeze_4h": snapshot.ind_4h.get("4h_bb_squeeze", False),
            "st_bull_5m": snapshot.ind_5m.get("5m_st_bull", False),
            "st_bull_4h": snapshot.ind_4h.get("4h_st_bull", False),
            "vol_spike_5m": snapshot.ind_5m.get("5m_vol_spike", False),
            "macd_above_4h": snapshot.ind_4h.get("4h_macd_above", False),
            "adx_4h": snapshot.ind_4h.get("4h_adx", 0),
        },
        "mtf_bull_count": snapshot.mtf_signals.get("mtf_bull_count", 0),
        "mtf_bear_count": snapshot.mtf_signals.get("mtf_bear_count", 0),
        "outcome": outcome,
    }

    if pattern_db is not None:
        pattern_db.append(pattern)

    return pattern


# ============================================================
# PATTERN ANALYSIS ENGINE
# ============================================================

def analyze_historical_patterns(patterns: List[Dict]) -> Dict:
    """
    Study historical patterns to find the best indicator combinations.
    Returns statistical analysis of what works best.
    """
    if not patterns:
        return {"total_patterns": 0, "message": "No patterns to analyze"}

    # Separate by signal type and direction
    by_type = defaultdict(list)
    by_direction = defaultdict(list)
    by_tier = defaultdict(list)

    for p in patterns:
        by_type[p.get("signal_type", "unknown")].append(p)
        by_direction[p.get("direction", "unknown")].append(p)
        by_tier[p.get("tier", 0)].append(p)

    # Calculate win rates (if outcomes exist)
    def calc_win_rate(group: List[Dict]) -> Dict:
        total = len(group)
        with_outcome = [p for p in group if p.get("outcome")]
        if not with_outcome:
            return {"total": total, "evaluated": 0, "win_rate": None}

        wins = sum(1 for p in with_outcome
                    if p["outcome"].get("moved_pct", 0) >= 3.0
                    and p["outcome"].get("direction") == p.get("direction"))
        avg_move = sum(abs(p["outcome"].get("moved_pct", 0)) for p in with_outcome) / len(with_outcome)

        return {
            "total": total,
            "evaluated": len(with_outcome),
            "wins": wins,
            "win_rate": round(wins / len(with_outcome) * 100, 1) if with_outcome else 0,
            "avg_move_pct": round(avg_move, 2),
        }

    # Analyze indicator correlations with success
    indicator_win_rates = {}
    successful = [p for p in patterns if p.get("outcome") and
                  p["outcome"].get("moved_pct", 0) >= 3.0]
    failed = [p for p in patterns if p.get("outcome") and
              p["outcome"].get("moved_pct", 0) < 3.0]

    if successful and failed:
        all_indicator_keys = set()
        for p in patterns:
            ki = p.get("key_indicators", {})
            all_indicator_keys.update(ki.keys())

        for key in all_indicator_keys:
            # Count True/high-value occurrences in wins vs losses
            win_true = sum(1 for p in successful
                          if bool(p.get("key_indicators", {}).get(key, False)))
            loss_true = sum(1 for p in failed
                           if bool(p.get("key_indicators", {}).get(key, False)))

            total_true = win_true + loss_true
            if total_true >= 5:  # minimum sample size
                indicator_win_rates[key] = {
                    "win_count": win_true,
                    "loss_count": loss_true,
                    "win_rate": round(win_true / total_true * 100, 1),
                    "total_occurrences": total_true,
                }

    # Sort indicators by win rate
    best_indicators = sorted(
        indicator_win_rates.items(),
        key=lambda x: x[1]["win_rate"],
        reverse=True,
    )

    # Analyze best RSI ranges for successful signals
    rsi_ranges_success = {"low_rsi": 0, "mid_rsi": 0, "high_rsi": 0}
    for p in successful:
        rsi = p.get("key_indicators", {}).get("rsi_5m", 50)
        if rsi < 35:
            rsi_ranges_success["low_rsi"] += 1
        elif rsi > 65:
            rsi_ranges_success["high_rsi"] += 1
        else:
            rsi_ranges_success["mid_rsi"] += 1

    return {
        "total_patterns": len(patterns),
        "by_signal_type": {k: calc_win_rate(v) for k, v in by_type.items()},
        "by_direction": {k: calc_win_rate(v) for k, v in by_direction.items()},
        "by_tier": {k: calc_win_rate(v) for k, v in by_tier.items()},
        "best_indicators": dict(best_indicators[:20]),
        "rsi_ranges_at_success": rsi_ranges_success,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# SUMMARY REPORT
# ============================================================

def generate_scan_summary(
    snapshots: List[MTFSnapshot],
    pattern_analysis: Optional[Dict] = None,
) -> Dict:
    """Generate a comprehensive summary of the latest scan cycle."""
    ensure_data_dir()

    total = len(snapshots)
    signals = [s for s in snapshots if s.direction]
    pumps = [s for s in signals if s.direction == "pump"]
    dumps = [s for s in signals if s.direction == "dump"]

    # Top signals by tier
    top_pumps = sorted(pumps, key=lambda s: (s.tier, s.confidence), reverse=True)[:10]
    top_dumps = sorted(dumps, key=lambda s: (s.tier, s.confidence), reverse=True)[:10]

    summary = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "total_symbols_scanned": total,
        "total_signals": len(signals),
        "pump_signals": len(pumps),
        "dump_signals": len(dumps),
        "signal_breakdown": {
            "snipers": sum(1 for s in signals if s.tier == 3),
            "elite": sum(1 for s in signals if s.tier == 2),
            "standard": sum(1 for s in signals if s.tier == 1),
        },
        "top_pump_setups": [
            {
                "symbol": s.symbol,
                "signal_type": s.signal_type,
                "tier": s.tier,
                "confidence": s.confidence,
                "price": s.price,
                "pump_score": s.pump_score,
                "net_score": s.net_score,
            }
            for s in top_pumps
        ],
        "top_dump_setups": [
            {
                "symbol": s.symbol,
                "signal_type": s.signal_type,
                "tier": s.tier,
                "confidence": s.confidence,
                "price": s.price,
                "dump_score": s.dump_score,
                "net_score": s.net_score,
            }
            for s in top_dumps
        ],
    }

    if pattern_analysis:
        summary["pattern_analysis"] = pattern_analysis

    filepath = os.path.join(DATA_DIR, SUMMARY_JSON)
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  [SUMMARY] Saved scan summary → {filepath}")
    return summary
