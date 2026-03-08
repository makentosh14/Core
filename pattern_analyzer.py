#!/usr/bin/env python3
"""
pattern_analyzer.py - High-Quality Pattern Mining Engine v3
============================================================
Reads the labeled research dataset and produces statistically rigorous
analysis with anti-overfitting safeguards and base-rate comparison.

CHANGES vs v2:
  - Numeric tier ordering everywhere (TIER_ORDER dict, not raw string sort)
  - test_edge_vs_baseline added to all ranking functions
  - edge_vs_baseline now always = TEST version for final output/sort
  - train_edge_vs_baseline kept as separate column
  - confidence_tier() called with TEST edge, not train edge
  - noisy indicator detection uses test_edge_vs_baseline
  - decision_report uses test_edge_vs_baseline
  - bot_feature_candidates includes baseline_wr, train_edge, test_edge
  - save_csv_ranking uses union of all row keys (no silent column drops)

OUTPUT FILES:
  scanner_data/analysis/pattern_report.json
  scanner_data/analysis/indicator_ranking.csv
  scanner_data/analysis/combo2_ranking.csv
  scanner_data/analysis/combo3_ranking.csv
  scanner_data/analysis/regime_ranking.csv
  scanner_data/analysis/setup_type_ranking.csv
  scanner_data/analysis/top_patterns.csv
  scanner_data/analysis/false_positive_traps.csv
  scanner_data/analysis/research_decision_report.json
  scanner_data/exports/bot_integration.json
  scanner_data/exports/bot_feature_candidates.json

USAGE:
    python pattern_analyzer.py
    python pattern_analyzer.py --min-samples 10
    python pattern_analyzer.py --direction pump
    python pattern_analyzer.py --regime trending_bull
    python pattern_analyzer.py --train-ratio 0.6
"""

import argparse
import json
import csv
import os
import math
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime, timezone
from collections import defaultdict, OrderedDict
from itertools import combinations

from config import DATA_DIR, RESEARCH_PARQUET, OUTCOME_UP_PCTS, OUTCOME_DOWN_PCTS
from data_storage import ensure_data_dir, load_parquet


# ============================================================
# TIER ORDER — single source of truth for sorting
# ============================================================

# Lower number = better tier = appears first in sorted output (ascending key).
# Used everywhere a confidence_tier string needs to be sorted.
TIER_ORDER: Dict[str, int] = {
    "A_reliable": 0,
    "B_promising": 1,
    "C_weak":      2,
    "C_small_n":   3,
    "D_overfit":   4,
    "D_discard":   5,
}

def _tier_rank(tier: str) -> int:
    """Return numeric rank for a confidence_tier string. Lower = better."""
    return TIER_ORDER.get(tier, 5)


# ============================================================
# HELPERS
# ============================================================

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if not math.isnan(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return False


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _stats(values: List[float]) -> Dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    n    = len(values)
    mean = sum(values) / n
    std  = (sum((x - mean) ** 2 for x in values) / n) ** 0.5 if n > 1 else 0.0
    return {
        "count":  n,
        "mean":   round(mean,          3),
        "median": round(_median(values), 3),
        "min":    round(min(values),   3),
        "max":    round(max(values),   3),
        "std":    round(std,           3),
    }


def _wr(wins_list: List[int]) -> float:
    """Win rate % from a list of 1s and 0s."""
    if not wins_list:
        return 0.0
    return sum(wins_list) / len(wins_list) * 100


def _rank_score(wr: float, n: int) -> float:
    """
    Penalizes small samples: wr * log(n+1).
    Higher score = more trustworthy win rate.
    """
    return round(wr * math.log(n + 1), 2)


# ============================================================
# LOAD AND SPLIT
# ============================================================

# Columns the analyzer actually needs — drop all others to save RAM.
# Indicator columns are added dynamically from the header (bool + numeric only).
_ANALYSIS_META_COLS = {
    "event_id", "symbol", "timestamp", "direction", "tier",
    "market_regime", "setup_type", "vol_state", "trend_state", "structure_state",
    "signal_type", "confidence", "pump_score", "dump_score", "net_score",
    "outcome_labeled", "outcome_resolved",
    "outcome_max_up_pct", "outcome_max_dn_pct", "outcome_bars_to_sig",
    "outcome_pct_1bar", "outcome_pct_3bar", "outcome_pct_6bar",
    "outcome_pct_8bar", "outcome_pct_12bar",
    "outcome_hit_up2", "outcome_hit_up3", "outcome_hit_up5", "outcome_hit_up10",
    "outcome_hit_dn2", "outcome_hit_dn3", "outcome_hit_dn5", "outcome_hit_dn10",
    "outcome_went_against", "outcome_first_move",
    "outcome_path_type", "outcome_entry_quality", "outcome_early_vs_signal",
    "outcome_early_strength", "outcome_intrabar_adverse", "outcome_speed",
    "outcome_peak_bar", "outcome_symmetry", "outcome_drawdown_bw",
}

def _is_analysis_col(col: str, all_cols: set) -> bool:
    """
    Keep meta/outcome columns + indicator columns (5m_/15m_/1h_/4h_/mtf_/v3_).
    Drop all other wide columns (raw OHLCV, scan internals, etc.).
    """
    if col in _ANALYSIS_META_COLS:
        return True
    prefixes = ("5m_", "15m_", "1h_", "4h_", "mtf_", "v3_")
    return any(col.startswith(p) for p in prefixes)


def _load_labeled_csv_streaming(
    direction:    Optional[str] = None,
    regime:       Optional[str] = None,
    min_tier:     int           = 0,
    signals_only: bool          = False,
) -> List[Dict]:
    """
    Load labeled events keeping ONLY columns needed for analysis.
    Drops ~450 raw indicator/OHLCV columns that analysis never uses.
    This cuts per-row RAM by ~90%, allowing 1.2M rows to fit in memory.

    Source priority:
    1. outcome_labels.csv  — already filtered to labeled rows, much smaller
    2. research_events.csv — full file, filtered on load
    """
    labels_csv   = os.path.join(DATA_DIR, "outcomes", "outcome_labels.csv")
    parquet_path = os.path.join(DATA_DIR, "raw", RESEARCH_PARQUET)
    csv_path     = parquet_path.replace(".parquet", ".csv")

    if os.path.exists(labels_csv):
        source       = labels_csv
        pre_filtered = True
    elif os.path.exists(csv_path):
        source       = csv_path
        pre_filtered = False
    elif os.path.exists(parquet_path):
        source       = parquet_path
        pre_filtered = False
    else:
        print("    No labeled data file found.")
        return []

    size_mb = os.path.getsize(source) / 1024 / 1024
    print(f"    Source: {os.path.basename(source)} ({size_mb:.0f} MB)")

    result = []
    kept_cols: Optional[set] = None   # built from first row

    with open(source, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_cols = set(reader.fieldnames or [])

        # Decide which columns to keep once (from header)
        kept_cols = {c for c in all_cols if _is_analysis_col(c, all_cols)}

        for i, row in enumerate(reader):
            if not pre_filtered:
                if not _safe_bool(row.get("outcome_labeled", False)):
                    continue
            if signals_only and not row.get("direction"):
                continue
            if direction and row.get("direction") != direction:
                continue
            if regime and row.get("market_regime") != regime:
                continue
            if min_tier and _safe_int(row.get("tier", 0)) < min_tier:
                continue
            # Keep only analysis columns — drops ~450 wide columns
            result.append({k: row[k] for k in kept_cols if k in row})
            if i > 0 and i % 200000 == 0:
                print(f"    ... read {i} rows")

    print(f"    Loaded {len(result)} rows ({len(kept_cols)} cols each)")
    return result


def load_all_labeled_events(
    direction: Optional[str] = None,
    regime:    Optional[str] = None,
    min_tier:  int           = 0,
) -> List[Dict]:
    """Load labeled signal events with optional filters."""
    return _load_labeled_csv_streaming(
        direction=direction, regime=regime, min_tier=min_tier, signals_only=False
    )


def load_all_rows_for_baseline() -> List[Dict]:
    """
    Load ALL labeled rows for base-rate computation.
    Reads outcome_labels.csv, not the full 2.6 GB research_events file.
    """
    return _load_labeled_csv_streaming(signals_only=False)


def split_train_test(
    rows:        List[Dict],
    train_ratio: float = 0.70,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split events by timestamp (chronological). Train = oldest fraction.
    Test = most recent (1 - train_ratio) fraction.
    We NEVER expose future data to training.
    """
    if not rows:
        return [], []
    sorted_rows = sorted(rows, key=lambda r: _safe_int(r.get("timestamp", 0)))
    split_idx   = int(len(sorted_rows) * train_ratio)
    return sorted_rows[:split_idx], sorted_rows[split_idx:]


# ============================================================
# WIN / MOVE HELPERS
# ============================================================

def is_win(row: Dict) -> Optional[bool]:
    dire     = row.get("direction", "")
    resolved = row.get("outcome_resolved", "")
    if not dire or not resolved:
        return None
    if dire == "pump":
        return resolved == "pump"
    if dire == "dump":
        return resolved == "dump"
    return None


def favorable_move(row: Dict) -> float:
    dire = row.get("direction", "")
    if dire == "pump":
        return _safe_float(row.get("outcome_max_up_pct", 0))
    if dire == "dump":
        return abs(_safe_float(row.get("outcome_max_dn_pct", 0)))
    return 0.0


def adverse_move(row: Dict) -> float:
    dire = row.get("direction", "")
    if dire == "pump":
        return abs(_safe_float(row.get("outcome_max_dn_pct", 0)))
    if dire == "dump":
        return _safe_float(row.get("outcome_max_up_pct", 0))
    return 0.0


# ============================================================
# BASE RATE COMPUTATION
# ============================================================

def compute_baseline(
    all_rows:  List[Dict],
    direction: str,
    regime:    Optional[str] = None,
) -> Dict:
    """
    Compute baseline win rate for a direction (and optionally regime).
    This is the benchmark every pattern must beat.
    """
    filtered = [
        r for r in all_rows
        if r.get("direction") == direction
        and (regime is None or r.get("market_regime") == regime)
    ]
    wins = [1 if is_win(r) else 0 for r in filtered if is_win(r) is not None]
    favs = [favorable_move(r) for r in filtered]
    return {
        "n":            len(wins),
        "win_rate":     round(_wr(wins), 2),
        "avg_fav_move": round(sum(favs) / len(favs), 3) if favs else 0.0,
        "regime":       regime or "all",
    }


def compute_all_baselines(all_rows: List[Dict]) -> Dict[str, Dict]:
    """
    Compute baseline win rates for pump and dump, overall and per-regime.
    baselines["pump"]["trending_bull"] = {n, win_rate, avg_fav_move, regime}
    baselines["pump"]["_overall"]      = {n, win_rate, avg_fav_move, regime}
    """
    baselines: Dict[str, Dict] = {}
    for dire in ("pump", "dump"):
        baselines[dire] = {}
        baselines[dire]["_overall"] = compute_baseline(all_rows, dire)
        regimes = set(r.get("market_regime", "") for r in all_rows if r.get("market_regime"))
        for regime in regimes:
            baselines[dire][regime] = compute_baseline(all_rows, dire, regime)
    return baselines


# ============================================================
# OVERFIT DETECTION
# ============================================================

OVERFIT_WR_DIFF_THRESHOLD = 15.0  # train WR > test WR by this much = overfit flag
SMALL_N_THRESHOLD         = 8     # below this test-n = small-n warning

def overfit_flag(train_wr: float, test_wr: float, n_test: int) -> str:
    """
    'high'    = train/test gap > OVERFIT_WR_DIFF_THRESHOLD
    'medium'  = moderate gap OR tiny test sample
    'low'     = test performance is consistent with train
    'no_test' = no test data at all
    """
    if n_test == 0:
        return "no_test"
    diff = train_wr - test_wr
    if diff > OVERFIT_WR_DIFF_THRESHOLD:
        return "high"
    if diff > OVERFIT_WR_DIFF_THRESHOLD * 0.5 or n_test < SMALL_N_THRESHOLD:
        return "medium"
    return "low"


def confidence_tier(
    test_wr:  float,
    test_edge: float,   # test_wr - baseline_wr
    n_test:   int,
    of_flag:  str,
) -> str:
    """
    Research confidence tier based on TEST performance only.

    A_reliable: high test WR + meaningful test edge + good sample + low overfit
    B_promising: decent test WR and edge but marginal in some dimension
    C_weak:    low test WR or barely positive edge
    C_small_n: too few test samples to be confident
    D_overfit: train/test gap is too large to trust
    D_discard: fails all quality gates
    """
    if of_flag == "high":
        return "D_overfit"
    if n_test < SMALL_N_THRESHOLD:
        return "C_small_n"
    if test_wr >= 60 and test_edge >= 10 and of_flag == "low":
        return "A_reliable"
    if test_wr >= 52 and test_edge >= 5:
        return "B_promising"
    if test_wr >= 45 and test_edge >= 0:
        return "C_weak"
    return "D_discard"


# ============================================================
# SINGLE INDICATOR RANKING WITH TRAIN/TEST
# ============================================================

def _get_bool_indicator_cols(rows: List[Dict]) -> List[str]:
    """Return all boolean indicator column names, excluding raw numeric fields."""
    if not rows:
        return []
    exclude_suffixes = (
        "_rsi", "_adx", "_atr", "_bb_bw", "_vol_ratio",
        "_macd_line", "_macd_sig", "_macd_hist",
        "_ema_fast", "_ema_mid", "_ema_slow", "_ema200",
        "_st_val", "_vwap", "_obv", "_stoch_k", "_stoch_d",
        "_cci", "_wr", "_roc", "_mfi", "_cmf",
        "_bb_upper", "_bb_mid", "_bb_lower", "_bb_pct_b",
        "_atr_pct", "_vol_ma", "_plus_di", "_minus_di",
    )
    return [
        k for k in rows[0].keys()
        if (k.startswith("5m_") or k.startswith("15m_") or
            k.startswith("1h_") or k.startswith("4h_") or
            k.startswith("mtf_") or k.startswith("v3_"))
        and not k.endswith(exclude_suffixes)
    ]


def rank_single_indicators(
    train_rows:  List[Dict],
    test_rows:   List[Dict],
    baseline_wr: float,
    min_samples: int = 10,
) -> List[Dict]:
    """
    Rank boolean indicators by test_edge_vs_baseline.

    For each indicator:
      train_edge_vs_baseline = train_wr - baseline_wr  (kept for reference)
      test_edge_vs_baseline  = test_wr  - baseline_wr  (used for ranking)
      edge_vs_baseline       = test_edge_vs_baseline   (canonical field)
      confidence_tier        = based on test_wr and test_edge_vs_baseline
    """
    ind_cols = _get_bool_indicator_cols(train_rows or test_rows)
    results  = []

    for col in ind_cols:
        # Train
        t_wins, t_favs, t_advs = [], [], []
        f_wins = []
        for row in train_rows:
            val = _safe_bool(row.get(col, False))
            w   = is_win(row)
            if w is None:
                continue
            fav = favorable_move(row)
            adv = adverse_move(row)
            if val:
                t_wins.append(1 if w else 0)
                t_favs.append(fav)
                t_advs.append(adv)
            else:
                f_wins.append(1 if w else 0)

        n_train = len(t_wins)
        if n_train < min_samples:
            continue

        train_wr       = _wr(t_wins)
        train_wr_false = _wr(f_wins) if f_wins else 50.0
        train_edge_int = train_wr - train_wr_false   # edge vs "indicator=False" rows
        train_edge_bl  = train_wr - baseline_wr       # edge vs overall baseline

        # Test
        te_wins, te_favs = [], []
        for row in test_rows:
            val = _safe_bool(row.get(col, False))
            w   = is_win(row)
            if w is None or not val:
                continue
            te_wins.append(1 if w else 0)
            te_favs.append(favorable_move(row))

        n_test         = len(te_wins)
        test_wr        = _wr(te_wins)
        test_edge_bl   = test_wr - baseline_wr   # this is the honest edge
        of_flag        = overfit_flag(train_wr, test_wr, n_test)
        # confidence_tier uses TEST edge (fixed from v2)
        tier           = confidence_tier(test_wr, test_edge_bl, n_test, of_flag)

        results.append({
            "indicator":              col,
            "n_train":                n_train,
            "n_test":                 n_test,
            "train_wr":               round(train_wr,       1),
            "test_wr":                round(test_wr,        1),
            "wr_diff":                round(train_wr - test_wr, 1),
            "baseline_wr":            round(baseline_wr,    1),
            "train_edge_vs_baseline": round(train_edge_bl,  1),
            "test_edge_vs_baseline":  round(test_edge_bl,   1),
            "edge_vs_baseline":       round(test_edge_bl,   1),  # canonical = test
            "train_edge_internal":    round(train_edge_int, 1),  # vs indicator=False
            "avg_fav_move":           round(sum(t_favs)/n_train, 3) if n_train else 0.0,
            "avg_adv_move":           round(sum(t_advs)/n_train, 3) if n_train else 0.0,
            "median_fav":             round(_median(t_favs), 3),
            "overfit_flag":           of_flag,
            "confidence_tier":        tier,
        })

    # Sort: best tier first (numeric), then highest test_edge within tier
    results.sort(
        key=lambda x: (_tier_rank(x["confidence_tier"]), -x["test_edge_vs_baseline"]),
    )
    return results


# ============================================================
# NUMERIC BINS WITH TRAIN/TEST
# ============================================================

NUMERIC_BIN_DEFS = {
    "4h_rsi": [
        (0,  20,  "extreme_os"),
        (20, 35,  "oversold"),
        (35, 50,  "neutral_bear"),
        (50, 65,  "neutral_bull"),
        (65, 80,  "overbought"),
        (80, 100, "extreme_ob"),
    ],
    "1h_rsi": [
        (0,  20,  "extreme_os"),
        (20, 35,  "oversold"),
        (35, 50,  "neutral_bear"),
        (50, 65,  "neutral_bull"),
        (65, 80,  "overbought"),
        (80, 100, "extreme_ob"),
    ],
    "4h_adx": [
        (0,  15,  "weak"),
        (15, 25,  "building"),
        (25, 40,  "trending"),
        (40, 60,  "strong"),
        (60, 100, "extreme"),
    ],
    "4h_atr_pct": [
        (0,   0.5, "very_low"),
        (0.5, 1.5, "low"),
        (1.5, 3.0, "medium"),
        (3.0, 5.0, "high"),
        (5.0, 100, "extreme"),
    ],
    "4h_bb_bw": [
        (0,  2,   "squeeze"),
        (2,  4,   "tight"),
        (4,  8,   "normal"),
        (8,  15,  "wide"),
        (15, 100, "very_wide"),
    ],
    "5m_vol_ratio": [
        (0,   0.5, "very_low"),
        (0.5, 1.0, "low"),
        (1.0, 1.5, "normal"),
        (1.5, 2.5, "high"),
        (2.5, 100, "spike"),
    ],
    "15m_vol_ratio": [
        (0,   0.5, "very_low"),
        (0.5, 1.0, "low"),
        (1.0, 1.5, "normal"),
        (1.5, 2.5, "high"),
        (2.5, 100, "spike"),
    ],
    "4h_roc": [
        (-100, -3.0, "strong_bear"),
        (-3.0, -1.0, "mild_bear"),
        (-1.0,  1.0, "flat"),
        ( 1.0,  3.0, "mild_bull"),
        ( 3.0, 100,  "strong_bull"),
    ],
}


def rank_numeric_bins(
    train_rows:  List[Dict],
    test_rows:   List[Dict],
    baseline_wr: float,
    min_samples: int = 8,
) -> List[Dict]:
    """
    Rank numeric indicator bins with train/test split.

    edge_vs_baseline = test_edge_vs_baseline (test-based, canonical).
    Sort by (test_wr DESC, test_edge_vs_baseline DESC).
    """
    results = []

    for col, bins in NUMERIC_BIN_DEFS.items():
        for lo, hi, label in bins:
            # Train
            train_wins, train_favs = [], []
            for row in train_rows:
                v = _safe_float(row.get(col, -999))
                if v == -999:
                    continue
                if lo <= v < hi:
                    w = is_win(row)
                    if w is not None:
                        train_wins.append(1 if w else 0)
                        train_favs.append(favorable_move(row))

            n_train = len(train_wins)
            if n_train < min_samples:
                continue
            train_wr = _wr(train_wins)

            # Test
            test_wins, test_favs = [], []
            for row in test_rows:
                v = _safe_float(row.get(col, -999))
                if v == -999:
                    continue
                if lo <= v < hi:
                    w = is_win(row)
                    if w is not None:
                        test_wins.append(1 if w else 0)
                        test_favs.append(favorable_move(row))

            n_test   = len(test_wins)
            test_wr  = _wr(test_wins)
            of_flag  = overfit_flag(train_wr, test_wr, n_test)

            train_edge_bl = train_wr - baseline_wr
            test_edge_bl  = test_wr  - baseline_wr   # canonical

            results.append({
                "indicator":              col,
                "bin":                    label,
                "bin_range":              f"{lo}-{hi}",
                "n_train":                n_train,
                "n_test":                 n_test,
                "train_wr":               round(train_wr,       1),
                "test_wr":                round(test_wr,        1),
                "wr_diff":                round(train_wr - test_wr, 1),
                "baseline_wr":            round(baseline_wr,    1),
                "train_edge_vs_baseline": round(train_edge_bl,  1),
                "test_edge_vs_baseline":  round(test_edge_bl,   1),
                "edge_vs_baseline":       round(test_edge_bl,   1),  # canonical = test
                "avg_fav_move":           round(sum(train_favs)/n_train, 3) if n_train else 0.0,
                "median_fav":             round(_median(train_favs), 3),
                "overfit_flag":           of_flag,
            })

    # Sort by test_wr DESC, then test_edge DESC
    results.sort(key=lambda x: (x["test_wr"], x["test_edge_vs_baseline"]), reverse=True)
    return results


# ============================================================
# 2-INDICATOR COMBOS WITH TRAIN/TEST
# ============================================================

def rank_combos_2(
    train_rows:  List[Dict],
    test_rows:   List[Dict],
    top_inds:    List[str],
    baseline_wr: float,
    min_samples: int = 8,
) -> List[Dict]:
    """
    Rank 2-indicator combos with train/test split.
    edge_vs_baseline = test-based (canonical).
    """
    results   = []
    candidates = top_inds[:25]

    for col_a, col_b in combinations(candidates, 2):
        t_wins, t_favs = [], []
        for row in train_rows:
            a = _safe_bool(row.get(col_a, False))
            b = _safe_bool(row.get(col_b, False))
            if not (a and b):
                continue
            w = is_win(row)
            if w is not None:
                t_wins.append(1 if w else 0)
                t_favs.append(favorable_move(row))

        n_train = len(t_wins)
        if n_train < min_samples:
            continue
        train_wr = _wr(t_wins)

        te_wins, te_favs = [], []
        for row in test_rows:
            a = _safe_bool(row.get(col_a, False))
            b = _safe_bool(row.get(col_b, False))
            if not (a and b):
                continue
            w = is_win(row)
            if w is not None:
                te_wins.append(1 if w else 0)
                te_favs.append(favorable_move(row))

        n_test   = len(te_wins)
        test_wr  = _wr(te_wins)
        of_flag  = overfit_flag(train_wr, test_wr, n_test)

        train_edge_bl = train_wr - baseline_wr
        test_edge_bl  = test_wr  - baseline_wr

        results.append({
            "combo":                  f"{col_a} + {col_b}",
            "ind_a":                  col_a,
            "ind_b":                  col_b,
            "n_train":                n_train,
            "n_test":                 n_test,
            "train_wr":               round(train_wr,       1),
            "test_wr":                round(test_wr,        1),
            "wr_diff":                round(train_wr - test_wr, 1),
            "baseline_wr":            round(baseline_wr,    1),
            "train_edge_vs_baseline": round(train_edge_bl,  1),
            "test_edge_vs_baseline":  round(test_edge_bl,   1),
            "edge_vs_baseline":       round(test_edge_bl,   1),  # canonical = test
            "avg_fav_move":           round(sum(t_favs)/n_train, 3) if n_train else 0.0,
            "median_fav":             round(_median(t_favs), 3),
            "overfit_flag":           of_flag,
        })

    # Sort by test_wr DESC, then n_test DESC (more test data = more reliable)
    results.sort(key=lambda x: (x["test_wr"], x["n_test"]), reverse=True)
    return results[:30]


# ============================================================
# 3-INDICATOR COMBOS WITH TRAIN/TEST
# ============================================================

def rank_combos_3(
    train_rows:  List[Dict],
    test_rows:   List[Dict],
    top_inds:    List[str],
    baseline_wr: float,
    min_samples: int = 5,
) -> List[Dict]:
    """
    3-indicator combos with train/test split.
    edge_vs_baseline = test-based (canonical).
    """
    results    = []
    candidates = top_inds[:15]

    for col_a, col_b, col_c in combinations(candidates, 3):
        t_wins, t_favs = [], []
        for row in train_rows:
            a = _safe_bool(row.get(col_a, False))
            b = _safe_bool(row.get(col_b, False))
            c = _safe_bool(row.get(col_c, False))
            if not (a and b and c):
                continue
            w = is_win(row)
            if w is not None:
                t_wins.append(1 if w else 0)
                t_favs.append(favorable_move(row))

        n_train = len(t_wins)
        if n_train < min_samples:
            continue
        train_wr = _wr(t_wins)

        te_wins = []
        for row in test_rows:
            a = _safe_bool(row.get(col_a, False))
            b = _safe_bool(row.get(col_b, False))
            c = _safe_bool(row.get(col_c, False))
            if not (a and b and c):
                continue
            w = is_win(row)
            if w is not None:
                te_wins.append(1 if w else 0)

        n_test   = len(te_wins)
        test_wr  = _wr(te_wins)
        of_flag  = overfit_flag(train_wr, test_wr, n_test)

        train_edge_bl = train_wr - baseline_wr
        test_edge_bl  = test_wr  - baseline_wr

        results.append({
            "combo":                  f"{col_a} + {col_b} + {col_c}",
            "n_train":                n_train,
            "n_test":                 n_test,
            "train_wr":               round(train_wr,       1),
            "test_wr":                round(test_wr,        1),
            "wr_diff":                round(train_wr - test_wr, 1),
            "baseline_wr":            round(baseline_wr,    1),
            "train_edge_vs_baseline": round(train_edge_bl,  1),
            "test_edge_vs_baseline":  round(test_edge_bl,   1),
            "edge_vs_baseline":       round(test_edge_bl,   1),  # canonical = test
            "avg_fav_move":           round(sum(t_favs)/n_train, 3) if n_train else 0.0,
            "median_fav":             round(_median(t_favs), 3),
            "overfit_flag":           of_flag,
        })

    results.sort(key=lambda x: (x["test_wr"], x["n_test"]), reverse=True)
    return results[:20]


# ============================================================
# CATEGORICAL RANKING WITH BASELINE
# ============================================================

def rank_categorical(
    rows:        List[Dict],
    col:         str,
    baselines:   Dict[str, Dict],
    direction:   str,
    min_samples: int = 5,
) -> List[Dict]:
    """
    Rank win rates by a categorical column with baseline comparison.
    edge_vs_baseline = win_rate - regime_baseline (no train/test split here
    because categoricals are regime/setup labels not forward-looking indicators).
    """
    buckets: Dict[str, Dict] = defaultdict(lambda: {"wins": [], "favs": [], "advs": []})
    for row in rows:
        val = str(row.get(col, "unknown") or "unknown")
        w   = is_win(row)
        fav = favorable_move(row)
        adv = adverse_move(row)
        if w is not None:
            buckets[val]["wins"].append(1 if w else 0)
            buckets[val]["favs"].append(fav)
            buckets[val]["advs"].append(adv)

    overall_baseline = baselines.get(direction, {}).get("_overall", {}).get("win_rate", 50.0)

    results = []
    for val, d in buckets.items():
        n = len(d["wins"])
        if n < min_samples:
            continue
        wr             = _wr(d["wins"])
        regime_baseline = baselines.get(direction, {}).get(val, {}).get("win_rate", overall_baseline)
        edge           = wr - regime_baseline
        lift           = round(wr / regime_baseline, 3) if regime_baseline > 0 else 0.0

        results.append({
            col:               val,
            "n_samples":       n,
            "win_rate":        round(wr,             1),
            "baseline_wr":     round(regime_baseline, 1),
            "edge_vs_baseline":round(edge,            1),
            "lift":            lift,
            "avg_fav_move":    round(sum(d["favs"])/n, 3) if d["favs"] else 0.0,
            "avg_adv_move":    round(sum(d["advs"])/n, 3) if d["advs"] else 0.0,
            "median_fav":      round(_median(d["favs"]), 3),
        })

    results.sort(key=lambda x: x["edge_vs_baseline"], reverse=True)
    return results


# ============================================================
# TOP PATTERNS WITH TRAIN/TEST + STABILITY
# ============================================================

def extract_top_patterns(
    train_rows:  List[Dict],
    test_rows:   List[Dict],
    all_rows:    List[Dict],
    baselines:   Dict[str, Dict],
    direction:   str,
    top_n:       int   = 20,
    min_wr:      float = 55.0,
    min_samples: int   = 5,
) -> List[Dict]:
    """
    Extract top N patterns (signal_type + market_regime + setup_type).

    confidence_tier is assigned from TEST edge and TEST win rate.
    edge_vs_baseline = test_edge_vs_baseline (canonical).
    Sort: numeric tier first, then rank_score (which uses test_wr).
    """
    train_dir = [r for r in train_rows if r.get("direction") == direction]
    test_dir  = [r for r in test_rows  if r.get("direction") == direction]

    overall_base = baselines.get(direction, {}).get("_overall", {}).get("win_rate", 50.0)

    # Build train stats per pattern key
    train_combos: Dict[tuple, Dict] = defaultdict(
        lambda: {"wins": [], "favs": [], "advs": [],
                 "symbols": set(), "regimes": set(), "hit3": []}
    )
    for row in train_dir:
        key = (
            str(row.get("signal_type",   "unknown")),
            str(row.get("market_regime", "unknown")),
            str(row.get("setup_type",    "unknown")),
        )
        w = is_win(row)
        if w is None:
            continue
        fav = favorable_move(row)
        adv = adverse_move(row)
        train_combos[key]["wins"].append(1 if w else 0)
        train_combos[key]["favs"].append(fav)
        train_combos[key]["advs"].append(adv)
        train_combos[key]["symbols"].add(str(row.get("symbol", "")))
        train_combos[key]["regimes"].add(str(row.get("market_regime", "")))
        col_hit = "outcome_hit_up3" if direction == "pump" else "outcome_hit_dn3"
        train_combos[key]["hit3"].append(_safe_int(row.get(col_hit, 0)))

    # Build test stats per pattern key
    test_combos: Dict[tuple, Dict] = defaultdict(lambda: {"wins": [], "favs": []})
    for row in test_dir:
        key = (
            str(row.get("signal_type",   "unknown")),
            str(row.get("market_regime", "unknown")),
            str(row.get("setup_type",    "unknown")),
        )
        w = is_win(row)
        if w is not None:
            test_combos[key]["wins"].append(1 if w else 0)
            test_combos[key]["favs"].append(favorable_move(row))

    results = []
    for (stype, regime, setup), td in train_combos.items():
        n_train = len(td["wins"])
        if n_train < min_samples:
            continue
        train_wr = _wr(td["wins"])
        if train_wr < min_wr:
            continue

        # Regime-specific baseline
        regime_base = baselines.get(direction, {}).get(regime, {}).get("win_rate", overall_base)

        # Test performance
        tst      = test_combos.get((stype, regime, setup), {"wins": [], "favs": []})
        n_test   = len(tst["wins"])
        test_wr  = _wr(tst["wins"])
        of_flag  = overfit_flag(train_wr, test_wr, n_test)

        # Both edges
        train_edge_bl = train_wr - regime_base
        test_edge_bl  = test_wr  - regime_base

        # confidence_tier uses TEST edge (fixed)
        tier = confidence_tier(test_wr, test_edge_bl, n_test, of_flag)

        # Lift uses train_wr (lift is a train-set measure of signal strength)
        lift = round(train_wr / regime_base, 3) if regime_base > 0 else 0.0

        # Stability
        n_symbols = len(td["symbols"])
        n_regimes = len(td["regimes"])
        stable    = n_symbols >= 3 and of_flag in ("low", "medium")

        # Hit rate for 3% threshold
        hit3_rate  = round(sum(td["hit3"]) / n_train * 100, 1) if td["hit3"] else 0.0
        # False positive rate = 1 - train_wr (conservative, uses train)
        fp_rate    = round(100 - train_wr, 1)

        # rank_score: uses test_wr if we have test data, else discounted train_wr
        rank_wr = test_wr if n_test >= SMALL_N_THRESHOLD else train_wr * 0.7
        rs      = _rank_score(rank_wr, n_train)

        results.append({
            "direction":              direction,
            "signal_type":            stype,
            "market_regime":          regime,
            "setup_type":             setup,
            "n_train":                n_train,
            "n_test":                 n_test,
            "train_wr":               round(train_wr,       1),
            "test_wr":                round(test_wr,        1),
            "wr_diff":                round(train_wr - test_wr, 1),
            "baseline_wr":            round(regime_base,    1),
            "train_edge_vs_baseline": round(train_edge_bl,  1),
            "test_edge_vs_baseline":  round(test_edge_bl,   1),
            "edge_vs_baseline":       round(test_edge_bl,   1),  # canonical = test
            "lift":                   lift,
            "hit_3pct_rate":          hit3_rate,
            "avg_fav_move":           round(sum(td["favs"])/n_train, 3),
            "avg_adv_move":           round(sum(td["advs"])/n_train, 3),
            "median_fav":             round(_median(td["favs"]), 3),
            "false_positive_rate":    fp_rate,
            "n_symbols":              n_symbols,
            "n_regimes_seen":         n_regimes,
            "overfit_flag":           of_flag,
            "confidence_tier":        tier,
            "is_stable":              stable,
            "rank_score":             rs,
        })

    # Sort: best tier first (numeric), then rank_score within tier
    results.sort(key=lambda x: (_tier_rank(x["confidence_tier"]), -x["rank_score"]))
    return results[:top_n]


# ============================================================
# FALSE POSITIVE / TRAP ANALYSIS
# ============================================================

def find_traps(
    all_patterns: List[Dict],
    single_rank:  List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Find patterns / indicators that are misleading.

    overfit_traps:    patterns with large train/test WR gap
    regime_traps:     patterns seen in only one regime
    small_n_traps:    high train WR but tiny sample
    noisy_indicators: low TEST edge vs baseline (not train edge)
    """
    overfit_traps = [
        p for p in all_patterns
        if p.get("overfit_flag") == "high"
        and p.get("train_wr", 0) >= 55
    ]
    overfit_traps.sort(key=lambda x: x.get("wr_diff", 0), reverse=True)

    regime_traps = [
        p for p in all_patterns
        if p.get("n_regimes_seen", 2) <= 1
        and p.get("n_train", 0) >= SMALL_N_THRESHOLD
        and p.get("train_wr", 0) >= 55
    ]

    small_n_traps = [
        p for p in all_patterns
        if p.get("n_train", 0) < SMALL_N_THRESHOLD
        and p.get("train_wr", 0) >= 60
    ]

    # Noisy: adequate sample, low TEST edge (not train edge)
    noisy = [
        r for r in single_rank
        if r.get("n_train", 0) >= 20
        and abs(r.get("test_edge_vs_baseline", r.get("edge_vs_baseline", 0))) < 2.0
    ]
    noisy.sort(key=lambda x: abs(x.get("test_edge_vs_baseline", x.get("edge_vs_baseline", 0))))

    return {
        "overfit_traps":    overfit_traps[:10],
        "regime_traps":     regime_traps[:10],
        "small_n_traps":    small_n_traps[:10],
        "noisy_indicators": noisy[:15],
    }


# ============================================================
# PATH-AWARE ANALYSIS
# ============================================================

def rank_path_types(rows: List[Dict], direction: str, min_samples: int = 5) -> List[Dict]:
    """Rank by outcome_path_type."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {"wins": [], "favs": []})
    for row in [r for r in rows if r.get("direction") == direction]:
        pt = str(row.get("outcome_path_type", "unknown") or "unknown")
        w  = is_win(row)
        if w is not None:
            buckets[pt]["wins"].append(1 if w else 0)
            buckets[pt]["favs"].append(favorable_move(row))

    results = []
    for pt, d in buckets.items():
        n = len(d["wins"])
        if n < min_samples:
            continue
        results.append({
            "path_type":    pt,
            "n_samples":    n,
            "win_rate":     round(_wr(d["wins"]), 1),
            "avg_fav_move": round(sum(d["favs"])/n, 3) if d["favs"] else 0.0,
        })
    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return results


def rank_entry_quality(rows: List[Dict], direction: str, min_samples: int = 5) -> List[Dict]:
    """Rank by outcome_entry_quality."""
    buckets: Dict[str, Dict] = defaultdict(lambda: {"wins": [], "favs": []})
    for row in [r for r in rows if r.get("direction") == direction]:
        eq = str(row.get("outcome_entry_quality", "unknown") or "unknown")
        w  = is_win(row)
        if w is not None:
            buckets[eq]["wins"].append(1 if w else 0)
            buckets[eq]["favs"].append(favorable_move(row))

    results = []
    for eq, d in buckets.items():
        n = len(d["wins"])
        if n < min_samples:
            continue
        results.append({
            "entry_quality": eq,
            "n_samples":     n,
            "win_rate":      round(_wr(d["wins"]), 1),
            "avg_fav_move":  round(sum(d["favs"])/n, 3) if d["favs"] else 0.0,
        })
    results.sort(key=lambda x: x["win_rate"], reverse=True)
    return results


# ============================================================
# RESEARCH DECISION REPORT
# ============================================================

def build_decision_report(
    pump_patterns:   List[Dict],
    dump_patterns:   List[Dict],
    single_rank:     List[Dict],
    regime_rank_pump:List[Dict],
    regime_rank_dump:List[Dict],
    traps:           Dict,
    baselines:       Dict,
    all_rows:        List[Dict],
) -> Dict:
    """Build a decision-useful research summary."""

    # A-tier patterns
    reliable_pumps = [p for p in pump_patterns if p.get("confidence_tier", "").startswith("A")]
    reliable_dumps = [p for p in dump_patterns if p.get("confidence_tier", "").startswith("A")]

    # Early warning setups
    early_pumps = [p for p in pump_patterns
                   if p.get("setup_type") in ("early", "squeeze_release")
                   and p.get("confidence_tier", "").startswith(("A", "B"))]
    early_dumps = [p for p in dump_patterns
                   if p.get("setup_type") in ("early", "squeeze_release")
                   and p.get("confidence_tier", "").startswith(("A", "B"))]

    # Confirmation setups
    confirm_pumps = [p for p in pump_patterns if p.get("setup_type") == "confirmation"]
    confirm_dumps = [p for p in dump_patterns if p.get("setup_type") == "confirmation"]

    # Late / dangerous: setup_type "late" or negative TEST edge
    late_pumps = [p for p in pump_patterns
                  if p.get("setup_type") == "late"
                  or p.get("test_edge_vs_baseline", p.get("edge_vs_baseline", 0)) < 0]
    late_dumps = [p for p in dump_patterns
                  if p.get("setup_type") == "late"
                  or p.get("test_edge_vs_baseline", p.get("edge_vs_baseline", 0)) < 0]

    # Features to integrate: A-tier, test_edge >= 10, test_wr >= 55
    to_integrate = [
        r for r in single_rank
        if r.get("confidence_tier", "").startswith("A")
        and r.get("test_edge_vs_baseline", r.get("edge_vs_baseline", 0)) >= 10
        and r.get("test_wr", 0) >= 55
    ]

    # Features to remove: noisy by TEST edge
    to_remove = traps.get("noisy_indicators", [])[:10]

    # Best early warning indicators (A/B tier, test_wr >= 55)
    early_warning_inds = [
        r for r in single_rank
        if r.get("confidence_tier", "").startswith(("A", "B"))
        and r.get("test_wr", 0) >= 55
    ][:10]

    return {
        "what_happens_before_pumps": {
            "reliable_patterns":    reliable_pumps[:5],
            "early_warning_setups": early_pumps[:5],
            "confirmation_setups":  confirm_pumps[:5],
            "late_dangerous":       late_pumps[:3],
            "best_regime":          regime_rank_pump[:3],
        },
        "what_happens_before_dumps": {
            "reliable_patterns":    reliable_dumps[:5],
            "early_warning_setups": early_dumps[:5],
            "confirmation_setups":  confirm_dumps[:5],
            "late_dangerous":       late_dumps[:3],
            "best_regime":          regime_rank_dump[:3],
        },
        "top_early_warning_indicators": early_warning_inds,
        "top_fakeout_warnings": {
            "overfit_traps": traps.get("overfit_traps", [])[:5],
            "regime_traps":  traps.get("regime_traps",  [])[:5],
            "small_n_traps": traps.get("small_n_traps", [])[:5],
        },
        "most_robust_patterns_train_test": [
            p for p in pump_patterns + dump_patterns
            if p.get("overfit_flag") == "low" and p.get("n_test", 0) >= SMALL_N_THRESHOLD
        ][:10],
        "most_robust_across_regimes": [
            p for p in pump_patterns + dump_patterns
            if p.get("n_regimes_seen", 0) >= 2
        ][:10],
        "features_recommended_for_integration": to_integrate[:10],
        "features_recommended_for_removal":     to_remove[:10],
        "baselines": {
            "pump_overall": baselines.get("pump", {}).get("_overall", {}),
            "dump_overall": baselines.get("dump", {}).get("_overall", {}),
        },
    }


# ============================================================
# BOT FEATURE CANDIDATES EXPORT
# ============================================================

def build_bot_feature_candidates(
    reliable_patterns: List[Dict],
    best_indicators:   List[Dict],
    best_numeric_bins: List[Dict],
    best_combos:       List[Dict],
) -> Dict:
    """
    Practical bot feature candidates export.

    Each entry includes both train_edge and test_edge (honest disclosure),
    and baseline_wr so the bot developer understands the context.
    Sorted by numeric TIER_ORDER first, then test_wr descending.
    """
    candidates = []

    # Pattern combos
    for p in reliable_patterns:
        candidates.append({
            "type":                   "pattern_combo",
            "trigger": {
                "signal_type":        p.get("signal_type"),
                "market_regime":      p.get("market_regime"),
                "setup_type":         p.get("setup_type"),
            },
            "direction":              p.get("direction"),
            "n_train":                p.get("n_train"),
            "n_test":                 p.get("n_test"),
            "train_wr":               p.get("train_wr"),
            "test_wr":                p.get("test_wr"),
            "baseline_wr":            p.get("baseline_wr"),
            "train_edge_vs_baseline": p.get("train_edge_vs_baseline"),
            "test_edge_vs_baseline":  p.get("test_edge_vs_baseline"),
            "edge_vs_baseline":       p.get("edge_vs_baseline"),
            "avg_fav_move":           p.get("avg_fav_move"),
            "avg_adv_move":           p.get("avg_adv_move"),
            "false_positive_rate":    p.get("false_positive_rate"),
            "confidence_tier":        p.get("confidence_tier"),
            "overfit_flag":           p.get("overfit_flag"),
            "is_stable":              p.get("is_stable"),
        })

    # Boolean indicators
    for r in best_indicators[:10]:
        candidates.append({
            "type":                   "boolean_indicator",
            "trigger":                {"indicator": r.get("indicator"), "value": True},
            "direction":              "any",
            "n_train":                r.get("n_train"),
            "n_test":                 r.get("n_test"),
            "train_wr":               r.get("train_wr"),
            "test_wr":                r.get("test_wr"),
            "baseline_wr":            r.get("baseline_wr"),
            "train_edge_vs_baseline": r.get("train_edge_vs_baseline"),
            "test_edge_vs_baseline":  r.get("test_edge_vs_baseline"),
            "edge_vs_baseline":       r.get("edge_vs_baseline"),
            "avg_fav_move":           r.get("avg_fav_move"),
            "confidence_tier":        r.get("confidence_tier"),
            "overfit_flag":           r.get("overfit_flag"),
        })

    # Numeric bins
    for r in best_numeric_bins[:8]:
        candidates.append({
            "type":                   "numeric_bin",
            "trigger": {
                "indicator":          r.get("indicator"),
                "range":              r.get("bin_range"),
                "bin_label":          r.get("bin"),
            },
            "direction":              "any",
            "n_train":                r.get("n_train"),
            "n_test":                 r.get("n_test"),
            "train_wr":               r.get("train_wr"),
            "test_wr":                r.get("test_wr"),
            "baseline_wr":            r.get("baseline_wr"),
            "train_edge_vs_baseline": r.get("train_edge_vs_baseline"),
            "test_edge_vs_baseline":  r.get("test_edge_vs_baseline"),
            "edge_vs_baseline":       r.get("edge_vs_baseline"),
            "overfit_flag":           r.get("overfit_flag"),
        })

    # Indicator combos
    for r in best_combos[:8]:
        candidates.append({
            "type":                   "indicator_combo",
            "trigger":                {"combo": r.get("combo"), "all_must_be_true": True},
            "direction":              "any",
            "n_train":                r.get("n_train"),
            "n_test":                 r.get("n_test"),
            "train_wr":               r.get("train_wr"),
            "test_wr":                r.get("test_wr"),
            "baseline_wr":            r.get("baseline_wr"),
            "train_edge_vs_baseline": r.get("train_edge_vs_baseline"),
            "test_edge_vs_baseline":  r.get("test_edge_vs_baseline"),
            "edge_vs_baseline":       r.get("edge_vs_baseline"),
            "avg_fav_move":           r.get("avg_fav_move"),
            "overfit_flag":           r.get("overfit_flag"),
        })

    # Sort numerically: best tier first, then highest test_wr
    candidates.sort(
        key=lambda x: (
            _tier_rank(x.get("confidence_tier", "D_discard")),
            -(x.get("test_wr") or 0),
        )
    )

    return {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "description":    "Bot feature candidates — integrate tier A first, discard D",
        "total_candidates": len(candidates),
        "tier_A_count":   sum(1 for c in candidates
                              if str(c.get("confidence_tier","")).startswith("A")),
        "tier_B_count":   sum(1 for c in candidates
                              if str(c.get("confidence_tier","")).startswith("B")),
        "candidates":     candidates,
    }


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_report(report: Dict):
    ensure_data_dir()
    fp = os.path.join(DATA_DIR, "analysis", "pattern_report.json")
    with open(fp, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  [SAVE] pattern_report.json → {fp}")


def save_csv_ranking(rows: List[Dict], filename: str):
    """
    Save ranking CSV using union of all row keys as fieldnames.
    Overwrites (not appends) so no schema drift across runs.
    """
    ensure_data_dir()
    if not rows:
        return
    # Build union of all keys across every row (preserving first-seen order)
    seen_keys: dict = {}
    for row in rows:
        for k in row.keys():
            seen_keys.setdefault(k, None)
    fieldnames = list(seen_keys.keys())

    fp = os.path.join(DATA_DIR, "analysis", filename)
    with open(fp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore",
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [SAVE] {filename} → {fp}")


def save_json(obj: Dict, filepath: str):
    ensure_data_dir()
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  [SAVE] {os.path.basename(filepath)} → {filepath}")


# ============================================================
# MAIN ANALYSIS PASS
# ============================================================

def _sample_rows_chronological(rows: List[Dict], max_rows: int) -> List[Dict]:
    """
    Sample rows while preserving chronological order.
    Takes every Nth row (stride sampling) so train/test split stays meaningful.
    Never shuffles — time order must be preserved for valid train/test split.
    """
    if len(rows) <= max_rows:
        return rows
    stride = len(rows) // max_rows
    return rows[::stride][:max_rows]


def run_analysis(
    min_samples:  int           = 10,
    direction:    Optional[str] = None,
    regime:       Optional[str] = None,
    train_ratio:  float         = 0.70,
    max_rows:     int           = 100_000,
) -> Dict:
    """Full analysis pass.

    max_rows: cap on signal rows loaded into RAM.
    100k rows gives statistically reliable results and uses ~400 MB RAM.
    1.2M rows gives the same results but uses 5+ GB RAM — not worth it.
    Sampling is chronological (stride-based) to preserve train/test validity.
    """
    print(f"\n{'='*65}")
    print(f"  PATTERN MINING ENGINE v3")
    print(f"{'='*65}")

    signal_rows  = load_all_labeled_events(direction=direction, regime=regime)
    all_labeled  = load_all_rows_for_baseline()

    if not signal_rows:
        print("  No labeled signal events found. Run outcome_labeler.py first.")
        return {}

    # Cap rows to avoid RAM spike — 100k is statistically sufficient
    if len(signal_rows) > max_rows:
        print(f"  Sampling {max_rows:,} from {len(signal_rows):,} signal rows (chronological stride)")
        signal_rows = _sample_rows_chronological(signal_rows, max_rows)
    if len(all_labeled) > max_rows:
        all_labeled = _sample_rows_chronological(all_labeled, max_rows)

    print(f"  Signal events:   {len(signal_rows)}")
    print(f"  All labeled:     {len(all_labeled)}  (used for baseline)")
    if direction:
        print(f"  Direction filter: {direction}")
    if regime:
        print(f"  Regime filter:    {regime}")
    print(f"  Train ratio:     {train_ratio:.0%}")

    train_rows, test_rows = split_train_test(signal_rows, train_ratio)
    print(f"  Train:           {len(train_rows)}  |  Test: {len(test_rows)}")

    if len(test_rows) < 5:
        print("  WARNING: Very few test events. Collect more data before trusting results.")

    # Baselines
    print(f"\n  [0/9] Computing baselines ...")
    baselines   = compute_all_baselines(all_labeled)
    pump_base   = baselines.get("pump", {}).get("_overall", {}).get("win_rate", 50.0)
    dump_base   = baselines.get("dump", {}).get("_overall", {}).get("win_rate", 50.0)
    overall_base = pump_base if direction == "pump" else (
                   dump_base if direction == "dump" else (pump_base + dump_base) / 2)
    print(f"         pump baseline: {pump_base:.1f}%  |  dump baseline: {dump_base:.1f}%")

    # 1. Single indicator ranking
    print(f"  [1/9] Ranking single boolean indicators ...")
    single_rank = rank_single_indicators(
        train_rows, test_rows, overall_base, min_samples=min_samples
    )
    print(f"         {len(single_rank)} indicators ranked")
    tier_a = sum(1 for r in single_rank if r.get("confidence_tier","").startswith("A"))
    print(f"         Tier-A reliable: {tier_a}")

    # 2. Numeric bins
    print(f"  [2/9] Ranking numeric indicator bins ...")
    numeric_rank = rank_numeric_bins(
        train_rows, test_rows, overall_base, min_samples=min_samples
    )
    print(f"         {len(numeric_rank)} bins ranked")

    # 3. 2-indicator combos (use only non-D candidates)
    top_bool_inds = [
        r["indicator"] for r in single_rank[:25]
        if r.get("confidence_tier", "") not in ("D_overfit", "D_discard")
    ]
    print(f"  [3/9] Ranking 2-indicator combos ({len(top_bool_inds)} candidates) ...")
    combos_2 = rank_combos_2(
        train_rows, test_rows, top_bool_inds, overall_base, min_samples
    )
    print(f"         {len(combos_2)} combos ranked")

    # 4. 3-indicator combos
    print(f"  [4/9] Ranking 3-indicator combos ...")
    combos_3 = rank_combos_3(
        train_rows, test_rows, top_bool_inds[:15], overall_base,
        max(5, min_samples // 2)
    )
    print(f"         {len(combos_3)} combos ranked")

    # 5. Categorical rankings
    print(f"  [5/9] Ranking categorical fields with baseline ...")
    dire_for_cat     = direction or "pump"
    regime_rank_pump = rank_categorical(
        [r for r in signal_rows if r.get("direction") == "pump"],
        "market_regime", baselines, "pump", min_samples=5)
    regime_rank_dump = rank_categorical(
        [r for r in signal_rows if r.get("direction") == "dump"],
        "market_regime", baselines, "dump", min_samples=5)
    setup_rank  = rank_categorical(signal_rows, "setup_type",      baselines, dire_for_cat, min_samples=5)
    vol_rank    = rank_categorical(signal_rows, "vol_state",        baselines, dire_for_cat, min_samples=5)
    trend_rank  = rank_categorical(signal_rows, "trend_state",      baselines, dire_for_cat, min_samples=5)
    struct_rank = rank_categorical(signal_rows, "structure_state",  baselines, dire_for_cat, min_samples=5)

    # 6. Top patterns
    print(f"  [6/9] Extracting top patterns ...")
    top_pumps = extract_top_patterns(
        train_rows, test_rows, signal_rows, baselines,
        "pump", top_n=20, min_wr=55.0, min_samples=min_samples)
    top_dumps = extract_top_patterns(
        train_rows, test_rows, signal_rows, baselines,
        "dump", top_n=20, min_wr=55.0, min_samples=min_samples)
    print(f"         Pump: {len(top_pumps)}  |  Dump: {len(top_dumps)}")

    # 7. Path-aware analysis
    print(f"  [7/9] Path-aware analysis ...")
    path_rank_pump = rank_path_types(signal_rows, "pump", min_samples=5)
    path_rank_dump = rank_path_types(signal_rows, "dump", min_samples=5)
    eq_rank_pump   = rank_entry_quality(signal_rows, "pump", min_samples=5)
    eq_rank_dump   = rank_entry_quality(signal_rows, "dump", min_samples=5)

    # 8. Trap analysis
    print(f"  [8/9] False-positive and trap analysis ...")
    traps = find_traps(top_pumps + top_dumps, single_rank)
    print(f"         Overfit traps: {len(traps['overfit_traps'])}")
    print(f"         Regime traps:  {len(traps['regime_traps'])}")
    print(f"         Noisy inds:    {len(traps['noisy_indicators'])}")

    # 9. Summary stats
    print(f"  [9/9] Summary statistics ...")
    all_wins   = [1 if is_win(r) else 0 for r in signal_rows if is_win(r) is not None]
    all_favs   = [favorable_move(r) for r in signal_rows]
    overall_wr = _wr(all_wins)

    # Decision report
    decision_report = build_decision_report(
        top_pumps, top_dumps, single_rank,
        regime_rank_pump, regime_rank_dump,
        traps, baselines, signal_rows,
    )

    # Bot feature candidates (A+B tier only)
    reliable_all = [
        p for p in top_pumps + top_dumps
        if p.get("confidence_tier", "").startswith(("A", "B"))
    ]
    bot_features = build_bot_feature_candidates(
        reliable_all,
        [r for r in single_rank if r.get("confidence_tier", "").startswith(("A", "B"))],
        numeric_rank[:15],
        combos_2[:10],
    )

    # Full report dict
    report = {
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "total_signal_events":     len(signal_rows),
        "total_all_labeled":       len(all_labeled),
        "train_events":            len(train_rows),
        "test_events":             len(test_rows),
        "overall_win_rate":        round(overall_wr, 1),
        "pump_baseline_wr":        round(pump_base,  1),
        "dump_baseline_wr":        round(dump_base,  1),
        "favorable_move_stats":    _stats(all_favs),
        # Rankings
        "single_indicator_top20":  single_rank[:20],
        "numeric_bins_top20":      numeric_rank[:20],
        "combo_2_top20":           combos_2[:20],
        "combo_3_top20":           combos_3[:20],
        "regime_rank_pump":        regime_rank_pump,
        "regime_rank_dump":        regime_rank_dump,
        "setup_rank":              setup_rank,
        "vol_rank":                vol_rank,
        "trend_rank":              trend_rank,
        "struct_rank":             struct_rank,
        # Patterns
        "top_pump_patterns":       top_pumps,
        "top_dump_patterns":       top_dumps,
        # Path analysis
        "path_types_pump":         path_rank_pump,
        "path_types_dump":         path_rank_dump,
        "entry_quality_pump":      eq_rank_pump,
        "entry_quality_dump":      eq_rank_dump,
        # Traps
        "traps":                   traps,
        "noisy_indicators":        traps["noisy_indicators"],
        "recommended_removals": [
            r["indicator"] for r in traps["noisy_indicators"][:10]
        ],
        # Decision report
        "decision_report":         decision_report,
    }

    # Save outputs
    ensure_data_dir()
    save_report(report)
    save_csv_ranking(single_rank[:50],              "indicator_ranking.csv")
    save_csv_ranking(combos_2[:30],                 "combo2_ranking.csv")
    save_csv_ranking(combos_3[:20],                 "combo3_ranking.csv")
    save_csv_ranking(regime_rank_pump + regime_rank_dump, "regime_ranking.csv")
    save_csv_ranking(setup_rank,                    "setup_type_ranking.csv")
    save_csv_ranking(top_pumps + top_dumps,         "top_patterns.csv")
    save_csv_ranking(
        traps["overfit_traps"] + traps["regime_traps"],
        "false_positive_traps.csv",
    )

    bot_dir = os.path.join(DATA_DIR, "exports")
    os.makedirs(bot_dir, exist_ok=True)
    save_json(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "top_pump_patterns": top_pumps,
         "top_dump_patterns": top_dumps},
        os.path.join(bot_dir, "bot_integration.json"),
    )
    save_json(bot_features, os.path.join(bot_dir, "bot_feature_candidates.json"))
    save_json(decision_report, os.path.join(DATA_DIR, "analysis", "research_decision_report.json"))

    # Console summary
    print(f"\n{'='*65}")
    print(f"  ANALYSIS SUMMARY")
    print(f"{'='*65}")
    print(f"  Signal events:  {len(signal_rows)}  (train={len(train_rows)} test={len(test_rows)})")
    print(f"  Overall WR:     {overall_wr:.1f}%")
    print(f"  Pump baseline:  {pump_base:.1f}%  |  Dump baseline: {dump_base:.1f}%")

    print(f"\n  Top 5 indicators (Tier-A, by test edge):")
    for r in [x for x in single_rank if x.get("confidence_tier","").startswith("A")][:5]:
        print(f"    {r['indicator']:<42} train={r['train_wr']:.1f}%"
              f" test={r['test_wr']:.1f}%"
              f" test_edge={r['test_edge_vs_baseline']:+.1f}"
              f"  [{r['confidence_tier']}]")

    print(f"\n  Top 5 pump patterns:")
    for p in top_pumps[:5]:
        print(f"    {p['signal_type']:<22} {p['market_regime']:<18} {p['setup_type']:<18}"
              f"  train={p['train_wr']:.1f}% test={p['test_wr']:.1f}%"
              f"  test_edge={p['test_edge_vs_baseline']:+.1f}"
              f"  [{p['confidence_tier']}]")

    print(f"\n  Top 5 dump patterns:")
    for p in top_dumps[:5]:
        print(f"    {p['signal_type']:<22} {p['market_regime']:<18} {p['setup_type']:<18}"
              f"  train={p['train_wr']:.1f}% test={p['test_wr']:.1f}%"
              f"  test_edge={p['test_edge_vs_baseline']:+.1f}"
              f"  [{p['confidence_tier']}]")

    if traps["overfit_traps"]:
        print(f"\n  ⚠  Overfit traps (high train WR, failing test):")
        for t in traps["overfit_traps"][:3]:
            print(f"    {t.get('signal_type',''):<22}"
                  f"  train={t['train_wr']:.1f}% test={t['test_wr']:.1f}%"
                  f"  diff={t['wr_diff']:.1f}pp")

    print(f"\n  Outputs:  {DATA_DIR}/analysis/")
    print(f"  Bot:      {DATA_DIR}/exports/bot_feature_candidates.json")
    print(f"{'='*65}")

    return report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pattern Mining Engine v3")
    parser.add_argument("--min-samples",  type=int,   default=10)
    parser.add_argument("--direction",    type=str,   default=None,
                        choices=["pump", "dump"])
    parser.add_argument("--regime",       type=str,   default=None)
    parser.add_argument("--train-ratio",  type=float, default=0.70,
                        help="Fraction of data for training (0.5-0.85)")
    parser.add_argument("--max-rows",     type=int,   default=100_000,
                        help="Max signal rows to load (default 100000). "
                             "Reduces RAM. 100k gives reliable results.")
    args = parser.parse_args()
    run_analysis(
        min_samples = args.min_samples,
        direction   = args.direction,
        regime      = args.regime,
        train_ratio = args.train_ratio,
        max_rows    = args.max_rows,
    )
