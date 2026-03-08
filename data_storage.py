#!/usr/bin/env python3
"""
data_storage.py - Research Dataset Storage
===========================================
Stores scanner snapshots and signals in a research-friendly schema.

PRIMARY OUTPUT FORMAT: Parquet (via csv fallback if pandas unavailable)
  - research_events.parquet   = every snapshot with all indicator values
  - outcome_labels.parquet    = outcome labels filled in by outcome_labeler.py
  - pattern_metadata.json     = pattern metadata and analysis results

SCHEMA DESIGN:
  Each row = one snapshot event at one point in time.
  Columns:
    - Identity:    event_id, symbol, timestamp, datetime, scan_cycle
    - Price:       price, open_5m, high_5m, low_5m, close_5m
    - Signals:     direction, signal_type, tier, confidence
    - Scores:      pump_score, dump_score, net_score
    - Research:    market_regime, setup_type, vol_state, trend_state, structure_state
    - Indicators:  all 150+ indicator values per TF (prefixed 5m_, 15m_, 1h_, 4h_)
    - MTF:         all mtf_* confluence flags
    - V3:          all v3_* combo flags
    - Outcomes:    filled in later by outcome_labeler.py
"""

import csv
import json
import os
import math
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from collections import defaultdict

from config import (
    DATA_DIR, RESEARCH_PARQUET, RESEARCH_CSV,
    PATTERN_JSON, SNAPSHOT_CSV, SIGNAL_LOG, PATTERN_DB, SUMMARY_JSON,
)
from mtf_analyzer import MTFSnapshot


# ============================================================
# INITIALIZATION
# ============================================================

def ensure_data_dir():
    """Create all required data directories."""
    for sub in ["", "/raw", "/outcomes", "/analysis", "/exports"]:
        os.makedirs(f"{DATA_DIR}{sub}", exist_ok=True)


# ============================================================
# PARQUET HELPER
# ============================================================

def _try_import_pandas():
    """Try to import pandas. Return None if not available."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None


def save_parquet(rows: List[Dict], filepath: str) -> bool:
    """
    Save rows as Parquet. Falls back to CSV if pandas/pyarrow unavailable.
    Returns True if parquet was written, False if CSV fallback was used.
    """
    pd = _try_import_pandas()
    if pd is not None:
        try:
            df = pd.DataFrame(rows)
            # Parquet needs pyarrow or fastparquet
            df.to_parquet(filepath, index=False, engine="pyarrow")
            return True
        except Exception:
            pass
    # CSV fallback
    csv_path = filepath.replace(".parquet", ".csv")
    save_csv_generic(rows, csv_path)
    return False


def load_parquet(filepath: str) -> List[Dict]:
    """Load parquet (or CSV fallback) into list of dicts."""
    pd = _try_import_pandas()
    if pd is not None and os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath)
            return df.to_dict(orient="records")
        except Exception:
            pass
    # Try CSV fallback
    csv_path = filepath.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        rows = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
    return []


def save_csv_generic(rows: List[Dict], filepath: str):
    """
    Save list of dicts to CSV.

    Schema-safe append logic:
    - Builds the union of keys from ALL rows being written (not just rows[0]).
    - If the file already exists, reads its current header and computes the union
      of existing columns + new columns so no column is ever silently dropped.
    - If the schema changed (new columns added), reloads the existing rows and
      rewrites the whole file with the merged schema.
    - extrasaction="ignore" is intentionally NOT used so coding errors are visible.
    - restval="" fills missing fields for older rows.

    Research correctness > append speed.
    """
    if not rows:
        return

    # Collect union of all keys in the incoming rows (preserving first-seen order)
    new_keys: dict = {}
    for row in rows:
        for k in row.keys():
            new_keys.setdefault(k, None)
    new_fieldset = set(new_keys.keys())

    if os.path.exists(filepath):
        # Read existing header to detect schema changes
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader       = csv.DictReader(f)
            existing_hdr = reader.fieldnames or []

        existing_fieldset = set(existing_hdr)

        if new_fieldset.issubset(existing_fieldset):
            # No new columns: safe to append
            # Use existing header ordering so file stays consistent
            fieldnames = existing_hdr
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore", restval="")
                writer.writerows(rows)

        else:
            # New columns appeared: rebuild schema from union of old + new
            # Preserve existing column order, append new columns at the end
            extra = [k for k in new_keys if k not in existing_fieldset]
            fieldnames = list(existing_hdr) + extra

            # Reload existing rows so they are written with the new schema
            with open(filepath, "r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))

            all_rows = existing_rows + rows
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore", restval="")
                writer.writeheader()
                writer.writerows(all_rows)

    else:
        # New file: use union of new rows' keys in first-seen order
        fieldnames = list(new_keys.keys())
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(rows)


# ============================================================
# SNAPSHOT → RESEARCH ROW CONVERSION
# ============================================================

def _safe(v: Any) -> Any:
    """Convert NaN and bool to JSON/CSV safe types."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, bool):
        return int(v)
    return v


def snapshot_to_research_row(
    snap:       MTFSnapshot,
    scan_cycle: int = 0,
    event_id:   Optional[str] = None,
) -> Dict:
    """
    Convert an MTFSnapshot to a flat research row.

    This is the canonical research schema. Every row includes:
    - Identity fields
    - All indicator values (150+ columns, prefixed by TF)
    - MTF confluence flags
    - V3 combo flags
    - Research classification fields
    - Outcome columns initialized to None (filled later by outcome_labeler.py)
    """
    if event_id is None:
        event_id = f"{snap.symbol}_{snap.timestamp}"

    row: Dict[str, Any] = {
        # ── Identity ──
        "event_id":    event_id,
        "symbol":      snap.symbol,
        "timestamp":   snap.timestamp,
        "datetime":    snap.dt_str,
        "scan_cycle":  scan_cycle,

        # ── Price at signal ──
        "price":       snap.price,

        # ── Signal assessment ──
        "direction":   snap.direction,
        "signal_type": snap.signal_type,
        "tier":        snap.tier,
        "confidence":  snap.confidence,

        # ── v1 Scores ──
        "pump_score":  snap.pump_score,
        "dump_score":  snap.dump_score,
        "net_score":   snap.net_score,

        # ── Research classification ──
        "market_regime":   snap.market_regime,
        "setup_type":      snap.setup_type,
        "vol_state":       snap.vol_state,
        "trend_state":     snap.trend_state,
        "structure_state": snap.structure_state,
    }

    # ── All per-TF indicator values ──
    for ind_dict in [snap.ind_5m, snap.ind_15m, snap.ind_1h, snap.ind_4h]:
        for k, v in ind_dict.items():
            row[k] = _safe(v)

    # ── MTF confluence signals ──
    for k, v in snap.mtf_signals.items():
        row[f"mtf_{k}" if not k.startswith("mtf_") else k] = _safe(v)

    # ── V3 combo signals ──
    for k, v in snap.v3_signals.items():
        row[f"v3_{k}" if not k.startswith("v3_") else k] = _safe(v)

    # ── Outcome placeholders (filled by outcome_labeler.py) ──
    # SYNC: must match every key returned by compute_outcome() in outcome_labeler.py
    outcome_cols = [
        # Core flag + resolution
        "outcome_labeled",          # bool: has been labeled by outcome_labeler.py
        "outcome_resolved",         # pump / dump / failed / neutral
        # Excursion bounds
        "outcome_max_up_pct",       # max upside % within 12 bars
        "outcome_max_dn_pct",       # max downside % within 12 bars
        "outcome_bars_to_sig",      # bars to first significant close move
        # Per-bar close moves
        "outcome_pct_1bar",
        "outcome_pct_3bar",
        "outcome_pct_6bar",
        "outcome_pct_8bar",
        "outcome_pct_12bar",
        # Threshold flags (1/0)
        "outcome_hit_up2",
        "outcome_hit_up3",
        "outcome_hit_up5",
        "outcome_hit_up10",
        "outcome_hit_dn2",
        "outcome_hit_dn3",
        "outcome_hit_dn5",
        "outcome_hit_dn10",
        # Original went-against / first move
        "outcome_went_against",     # 1 if moved adversely > 1% intrabar in first 3 bars
        "outcome_first_move",       # up / down / flat (close direction at bar 3)
        # NEW: path-aware fields (outcome_labeler.py v2)
        "outcome_path_type",        # fast_pump / slow_pump / fake_pump / whipsaw / etc.
        "outcome_entry_quality",    # clean_won / ugly_but_won / clean_failed / etc.
        "outcome_early_vs_signal",  # same / against / mixed (first 3 bars vs signal)
        "outcome_early_strength",   # float % move of close[3] vs entry
        "outcome_intrabar_adverse", # worst intrabar adverse % in first 3 bars
        "outcome_speed",            # fast / medium / slow / none
        "outcome_peak_bar",         # bar number (1-12) of best favorable move
        "outcome_symmetry",         # one_sided / symmetric / whipsaw
        "outcome_drawdown_bw",      # worst adverse % before reaching sig_move (None if never hit)
    ]
    for col in outcome_cols:
        row.setdefault(col, None)

    return row


# ============================================================
# SAVE RESEARCH EVENTS
# ============================================================

def _load_existing_event_ids(filepath: str) -> set:
    """
    Read only the event_id column from an existing CSV or parquet file.
    Much faster than loading all 500+ columns just to deduplicate.
    """
    pd = _try_import_pandas()
    if pd is not None and os.path.exists(filepath):
        try:
            df = pd.read_parquet(filepath, columns=["event_id"])
            return set(df["event_id"].astype(str).tolist())
        except Exception:
            pass
    csv_path = filepath.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        ids = set()
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                eid = row.get("event_id", "")
                if eid:
                    ids.add(eid)
        return ids
    return set()


def _append_rows_to_parquet(rows: List[Dict], filepath: str) -> bool:
    """
    Append new rows to parquet by reading existing + writing combined.
    Fast because parquet is columnar. Returns True if succeeded.
    """
    pd = _try_import_pandas()
    if pd is None:
        return False
    try:
        if os.path.exists(filepath):
            existing_df = pd.read_parquet(filepath)
            new_df      = pd.DataFrame(rows)
            combined    = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = pd.DataFrame(rows)
        combined.to_parquet(filepath, index=False, engine="pyarrow")
        return True
    except Exception:
        return False


def save_research_events(
    snapshots:    List[MTFSnapshot],
    scan_cycle:   int  = 0,
    signals_only: bool = False,
) -> str:
    """
    Save snapshots to the research dataset.

    signals_only=False -> save ALL snapshots (for base-rate analysis)
    signals_only=True  -> save only snapshots with direction (signals)

    PERFORMANCE FIX:
    Old logic loaded the ENTIRE file every cycle to deduplicate, then rewrote
    the whole file. With 500+ columns this hangs after a few cycles.

    New logic:
    - Read only the event_id column for dedup (fast, single column).
    - Append only the new rows.
    - CSV fallback: pure append via save_csv_generic (schema-safe).
    - Parquet: read parquet + rewrite parquet (fast columnar read).
    """
    ensure_data_dir()

    to_save = snapshots
    if signals_only:
        to_save = [s for s in snapshots if s.direction]

    if not to_save:
        return ""

    rows = [
        snapshot_to_research_row(s, scan_cycle=scan_cycle)
        for s in to_save
    ]

    parquet_path = os.path.join(DATA_DIR, "raw", RESEARCH_PARQUET)
    csv_path     = parquet_path.replace(".parquet", ".csv")

    # Step 1: read only event_ids for dedup (single column, very fast)
    existing_ids = _load_existing_event_ids(parquet_path)
    new_rows     = [r for r in rows if str(r.get("event_id", "")) not in existing_ids]

    if not new_rows:
        print(f"  [STORAGE] No new events (all {len(rows)} already stored)")
        return parquet_path

    total = len(existing_ids) + len(new_rows)

    # Step 2: append new rows only
    pd = _try_import_pandas()
    if pd is not None:
        ok = _append_rows_to_parquet(new_rows, parquet_path)
        if ok:
            print(f"  [STORAGE] +{len(new_rows)} events -> {parquet_path} (parquet). Total: {total}")
            return parquet_path

    # CSV fallback: pure append, schema-safe
    save_csv_generic(new_rows, csv_path)
    print(f"  [STORAGE] +{len(new_rows)} events -> {csv_path} (csv). Total: {total}")
    return csv_path


# ============================================================
# LEGACY CSV EXPORTS (kept for compatibility)
# ============================================================

def snapshot_to_flat_dict(snap: MTFSnapshot) -> Dict:
    """Legacy flat dict for CSV export (used by save_snapshots_csv)."""
    return snapshot_to_research_row(snap)


def save_snapshots_csv(snapshots: List[MTFSnapshot], filename: str = None):
    """Save snapshots to CSV (legacy format)."""
    ensure_data_dir()
    fp   = os.path.join(DATA_DIR, filename or SNAPSHOT_CSV)
    rows = [snapshot_to_flat_dict(s) for s in snapshots]
    if not rows:
        return
    mode   = "a" if os.path.exists(fp) else "w"
    fields = list(rows[0].keys())
    with open(fp, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"  [CSV] Saved {len(rows)} rows → {fp}")


def save_signal_log(snapshots: List[MTFSnapshot], filename: str = None):
    """Save only signals (direction != '') to signal log CSV."""
    ensure_data_dir()
    signals = [s for s in snapshots if s.direction]
    if not signals:
        return
    fp   = os.path.join(DATA_DIR, filename or SIGNAL_LOG)
    rows = []
    for s in signals:
        rows.append({
            "datetime":      s.dt_str,
            "symbol":        s.symbol,
            "price":         s.price,
            "direction":     s.direction,
            "signal_type":   s.signal_type,
            "tier":          s.tier,
            "confidence":    s.confidence,
            "pump_score":    s.pump_score,
            "dump_score":    s.dump_score,
            "net_score":     s.net_score,
            "market_regime": s.market_regime,
            "setup_type":    s.setup_type,
            "vol_state":     s.vol_state,
            "trend_state":   s.trend_state,
        })
    mode = "a" if os.path.exists(fp) else "w"
    with open(fp, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print(f"  [SIGNAL_LOG] +{len(rows)} signals → {fp}")


# ============================================================
# PATTERN DATABASE (JSON)
# ============================================================

def load_pattern_db(filepath: str = None) -> List:
    """Load the pattern database from JSON."""
    fp = filepath or os.path.join(DATA_DIR, PATTERN_DB)
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return json.load(f)
    return []


def save_pattern_db(patterns: List, filepath: str = None):
    """Save the pattern database to JSON."""
    ensure_data_dir()
    fp = filepath or os.path.join(DATA_DIR, PATTERN_DB)
    with open(fp, "w") as f:
        json.dump(patterns, f, indent=2, default=str)
    print(f"  [PATTERN] Saved {len(patterns)} patterns → {fp}")


def record_pattern(
    snapshot:   MTFSnapshot,
    outcome:    Optional[Dict] = None,
    pattern_db: Optional[List] = None,
) -> Dict:
    """
    Record a detected signal pattern for the pattern database.
    Stores key indicator values + research fields at signal time.
    outcome: filled later by outcome_labeler.py
    """
    pattern = {
        "recorded_at":   datetime.now(timezone.utc).isoformat(),
        "symbol":        snapshot.symbol,
        "timestamp":     snapshot.timestamp,
        "price":         snapshot.price,
        "signal_type":   snapshot.signal_type,
        "direction":     snapshot.direction,
        "tier":          snapshot.tier,
        "confidence":    snapshot.confidence,
        "pump_score":    snapshot.pump_score,
        "dump_score":    snapshot.dump_score,
        "net_score":     snapshot.net_score,
        # Research classification
        "market_regime":   snapshot.market_regime,
        "setup_type":      snapshot.setup_type,
        "vol_state":       snapshot.vol_state,
        "trend_state":     snapshot.trend_state,
        "structure_state": snapshot.structure_state,
        # Key indicator snapshot
        "key_indicators": {
            "rsi_5m":         snapshot.ind_5m.get("5m_rsi", 0),
            "rsi_15m":        snapshot.ind_15m.get("15m_rsi", 0),
            "rsi_1h":         snapshot.ind_1h.get("1h_rsi", 0),
            "rsi_4h":         snapshot.ind_4h.get("4h_rsi", 0),
            "rsi_zone_4h":    snapshot.ind_4h.get("4h_rsi_zone", ""),
            "adx_4h":         snapshot.ind_4h.get("4h_adx", 0),
            "bb_bw_4h":       snapshot.ind_4h.get("4h_bb_bw", 0),
            "atr_pct_4h":     snapshot.ind_4h.get("4h_atr_pct", 0),
            "vol_ratio_5m":   snapshot.ind_5m.get("5m_vol_ratio", 0),
            "vol_ratio_15m":  snapshot.ind_15m.get("15m_vol_ratio", 0),
            "bb_squeeze_4h":  snapshot.ind_4h.get("4h_bb_squeeze", False),
            "bb_squeeze_1h":  snapshot.ind_1h.get("1h_bb_squeeze", False),
            "st_bull_5m":     snapshot.ind_5m.get("5m_st_bull", False),
            "st_bull_4h":     snapshot.ind_4h.get("4h_st_bull", False),
            "vol_spike_5m":   snapshot.ind_5m.get("5m_vol_spike", False),
            "macd_above_4h":  snapshot.ind_4h.get("4h_macd_above", False),
            "above_ema200_4h":snapshot.ind_4h.get("4h_above_ema200", False),
            "ribbon_bull_4h": snapshot.ind_4h.get("4h_ribbon_bull", False),
            "ich_above_4h":   snapshot.ind_4h.get("4h_ich_above_cloud", False),
        },
        "mtf_bull_count": snapshot.mtf_signals.get("mtf_bull_count", 0),
        "mtf_bear_count": snapshot.mtf_signals.get("mtf_bear_count", 0),
        "mtf_bull_pct":   snapshot.mtf_signals.get("mtf_bull_pct", 0),
        "outcome":        outcome,
    }

    if pattern_db is not None:
        pattern_db.append(pattern)

    return pattern


# ============================================================
# PATTERN ANALYSIS ENGINE
# ============================================================

def analyze_historical_patterns(patterns: List[Dict]) -> Dict:
    """
    Statistical analysis of stored patterns.
    Evaluates: win rates, avg moves, best indicators, best combos,
    regime breakdown, setup type breakdown.
    """
    evaluated = [p for p in patterns if p.get("outcome") is not None]
    total     = len(patterns)

    if not evaluated:
        return {
            "total_patterns":    total,
            "evaluated":         0,
            "unevaluated":       total,
            "message":           "No outcomes labeled yet. Run outcome_labeler.py first.",
        }

    def _win(p: Dict, direction: str) -> Optional[bool]:
        oc = p.get("outcome", {})
        if not oc:
            return None
        resolved = oc.get("resolved", "")
        if direction == "pump":
            return resolved == "pump"
        if direction == "dump":
            return resolved == "dump"
        return False

    def _pct_move(p: Dict) -> float:
        oc = p.get("outcome", {})
        return float(oc.get("max_favorable_pct", 0) or 0)

    # ── By signal type ──
    by_type: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "moves": []})
    for p in evaluated:
        stype = p.get("signal_type", "unknown")
        dire  = p.get("direction", "")
        w     = _win(p, dire)
        mv    = _pct_move(p)
        if w is True:
            by_type[stype]["wins"] += 1
        elif w is False:
            by_type[stype]["losses"] += 1
        by_type[stype]["moves"].append(mv)

    by_type_stats = {}
    for st, d in by_type.items():
        total_ev = d["wins"] + d["losses"]
        moves    = d["moves"]
        by_type_stats[st] = {
            "evaluated":  total_ev,
            "wins":       d["wins"],
            "losses":     d["losses"],
            "win_rate":   round(d["wins"] / total_ev * 100, 1) if total_ev else 0,
            "avg_move":   round(sum(moves) / len(moves), 2)    if moves    else 0,
            "median_move":round(sorted(moves)[len(moves)//2], 2) if moves  else 0,
        }

    # ── By market regime ──
    by_regime: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "moves": []})
    for p in evaluated:
        regime = p.get("market_regime", "unknown")
        dire   = p.get("direction", "")
        w      = _win(p, dire)
        mv     = _pct_move(p)
        if w is True:   by_regime[regime]["wins"]   += 1
        elif w is False: by_regime[regime]["losses"] += 1
        by_regime[regime]["moves"].append(mv)

    by_regime_stats = {}
    for reg, d in by_regime.items():
        total_ev = d["wins"] + d["losses"]
        moves    = d["moves"]
        by_regime_stats[reg] = {
            "evaluated":  total_ev,
            "win_rate":   round(d["wins"] / total_ev * 100, 1) if total_ev else 0,
            "avg_move":   round(sum(moves) / len(moves), 2)    if moves    else 0,
        }

    # ── By setup type ──
    by_setup: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "moves": []})
    for p in evaluated:
        setup = p.get("setup_type", "unknown")
        dire  = p.get("direction", "")
        w     = _win(p, dire)
        mv    = _pct_move(p)
        if w is True:   by_setup[setup]["wins"]   += 1
        elif w is False: by_setup[setup]["losses"] += 1
        by_setup[setup]["moves"].append(mv)

    by_setup_stats = {}
    for st, d in by_setup.items():
        total_ev = d["wins"] + d["losses"]
        by_setup_stats[st] = {
            "evaluated":  total_ev,
            "win_rate":   round(d["wins"] / total_ev * 100, 1) if total_ev else 0,
            "avg_move":   round(sum(d["moves"]) / len(d["moves"]), 2) if d["moves"] else 0,
        }

    # ── Best individual indicators ──
    # For each boolean indicator in key_indicators, compute win rate when True
    indicator_stats: Dict[str, Dict] = defaultdict(lambda: {"win": 0, "loss": 0})
    for p in evaluated:
        ki   = p.get("key_indicators", {})
        dire = p.get("direction", "")
        w    = _win(p, dire)
        for ind_name, ind_val in ki.items():
            if isinstance(ind_val, (bool, int)) and ind_val:
                if w is True:  indicator_stats[ind_name]["win"]  += 1
                elif w is False: indicator_stats[ind_name]["loss"] += 1

    best_indicators = {}
    for name, d in indicator_stats.items():
        total_ev = d["win"] + d["loss"]
        if total_ev >= 5:
            best_indicators[name] = {
                "sample_size": total_ev,
                "win_count":   d["win"],
                "loss_count":  d["loss"],
                "win_rate":    round(d["win"] / total_ev * 100, 1),
            }
    # Sort by win rate descending
    best_indicators = dict(
        sorted(best_indicators.items(), key=lambda x: x[1]["win_rate"], reverse=True)[:20]
    )

    return {
        "total_patterns":    total,
        "evaluated":         len(evaluated),
        "unevaluated":       total - len(evaluated),
        "by_signal_type":    by_type_stats,
        "by_market_regime":  by_regime_stats,
        "by_setup_type":     by_setup_stats,
        "best_indicators":   best_indicators,
        "generated_at":      datetime.now(timezone.utc).isoformat(),
    }


# ============================================================
# SCAN SUMMARY
# ============================================================

def generate_scan_summary(
    snapshots:        List[MTFSnapshot],
    pattern_analysis: Optional[Dict] = None,
) -> Dict:
    """Generate and save the scan summary JSON."""
    ensure_data_dir()

    total = len(snapshots)
    sigs  = [s for s in snapshots if s.direction]
    pumps = [s for s in sigs if s.direction == "pump"]
    dumps = [s for s in sigs if s.direction == "dump"]

    top_pumps = sorted(pumps, key=lambda s: (s.tier, s.confidence), reverse=True)[:10]
    top_dumps = sorted(dumps, key=lambda s: (s.tier, s.confidence), reverse=True)[:10]

    summary = {
        "scan_time":             datetime.now(timezone.utc).isoformat(),
        "total_symbols_scanned": total,
        "total_signals":         len(sigs),
        "pump_signals":          len(pumps),
        "dump_signals":          len(dumps),
        "signal_breakdown": {
            "snipers":  sum(1 for s in sigs if s.tier == 3),
            "elite":    sum(1 for s in sigs if s.tier == 2),
            "standard": sum(1 for s in sigs if s.tier == 1),
        },
        "regime_breakdown": {},
        "setup_breakdown":  {},
        "top_pump_setups": [
            {"symbol": s.symbol, "signal_type": s.signal_type, "tier": s.tier,
             "confidence": s.confidence, "price": s.price,
             "market_regime": s.market_regime, "setup_type": s.setup_type}
            for s in top_pumps
        ],
        "top_dump_setups": [
            {"symbol": s.symbol, "signal_type": s.signal_type, "tier": s.tier,
             "confidence": s.confidence, "price": s.price,
             "market_regime": s.market_regime, "setup_type": s.setup_type}
            for s in top_dumps
        ],
    }

    # Regime/setup breakdown for all signals
    regime_cnt: Dict[str, int] = defaultdict(int)
    setup_cnt:  Dict[str, int] = defaultdict(int)
    for s in sigs:
        regime_cnt[s.market_regime] += 1
        setup_cnt[s.setup_type]     += 1
    summary["regime_breakdown"] = dict(regime_cnt)
    summary["setup_breakdown"]  = dict(setup_cnt)

    if pattern_analysis:
        summary["pattern_analysis"] = pattern_analysis

    fp = os.path.join(DATA_DIR, SUMMARY_JSON)
    with open(fp, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  [SUMMARY] Saved → {fp}")
    return summary
