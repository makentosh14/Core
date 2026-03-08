#!/usr/bin/env python3
"""
outcome_labeler.py - Path-Aware Future Outcome Labeling Engine v2
==================================================================
For every stored research event, fetches future 5m candles and labels:

BASIC METRICS (backward-compatible):
  - price move at 1, 3, 6, 8, 12 bars
  - max upside / max downside excursion
  - threshold flags: +/-2/3/5/10% reached
  - final resolution: pump / dump / failed / neutral

NEW: PATH-AWARE METRICS:
  outcome_path_type:
    fast_pump / slow_pump / fast_dump / slow_dump /
    fake_pump / fake_dump / pump_reversal / dump_reversal /
    whipsaw / neutral_drift

  outcome_entry_quality:
    clean_won / clean_failed / moderate_won / moderate_failed /
    ugly_but_won / ugly_and_failed

  outcome_early_vs_signal: same / against / mixed
  outcome_early_strength:  float %, close[2] vs entry
  outcome_intrabar_adverse: worst intrabar adverse % in first 3 bars
  outcome_speed:           fast / medium / slow / none
  outcome_peak_bar:        which bar hit best favorable move
  outcome_symmetry:        one_sided / symmetric / whipsaw
  outcome_drawdown_bw:     worst adverse % before reaching sig_move

WENT-AGAINST FIX:
  Old: checked only first bar close direction.
  New: checks intrabar adverse move across bars 1-3.
       A pump only counts as "went against" if the low in first 3 bars
       dipped more than 1% below entry (not just a tick).

DUPLICATE PREVENTION FIX:
  Events are indexed by event_id before labeling.
  Save is done once at the end from the deduped index dict.

USAGE:
    python outcome_labeler.py              # label unlabeled mature signal events
    python outcome_labeler.py --rerun      # re-label all mature events
    python outcome_labeler.py --all        # also label non-signal baseline events
"""

import asyncio
import aiohttp
import argparse
import time
import os
from typing import List, Dict, Optional

from config import (
    BYBIT_API, RATE_DELAY, API_TIMEOUT, MAX_RETRIES,
    DATA_DIR, RESEARCH_PARQUET,
    OUTCOME_BARS, OUTCOME_UP_PCTS, OUTCOME_DOWN_PCTS, OUTCOME_SIG_MOVE,
)
from data_storage import (
    ensure_data_dir, load_parquet, save_parquet, save_csv_generic,
)
from indicators import Candle


# ============================================================
# CONSTANTS
# ============================================================

FUTURE_BARS_TO_FETCH = 16   # extra buffer beyond the 12 we analyze
MATURITY_BARS        = 12   # bars of future data required
BAR_SECONDS          = 300  # 5-minute bars


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("true", "1", "yes")
    try:
        return bool(int(v))
    except (TypeError, ValueError):
        return False


# ============================================================
# FUTURE CANDLE FETCH
# ============================================================

async def fetch_candles_after(
    session:  aiohttp.ClientSession,
    symbol:   str,
    after_ts: int,
    n_bars:   int = FUTURE_BARS_TO_FETCH,
    interval: str = "5",
) -> List[Candle]:
    """
    Fetch n_bars of 5m candles with timestamp strictly > after_ts.
    Bybit returns newest-first; we sort to chronological order.
    """
    url    = f"{BYBIT_API}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol":   symbol,
        "interval": interval,
        "limit":    str(n_bars + 5),
        "start":    str(after_ts),
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get("retCode") != 0:
                    await asyncio.sleep(RATE_DELAY)
                    continue
                raw = data.get("result", {}).get("list", [])
                if not raw:
                    return []
                candles = []
                for row in raw:
                    ts = int(row[0])
                    if ts > after_ts:
                        candles.append(Candle(
                            timestamp = ts,
                            open      = float(row[1]),
                            high      = float(row[2]),
                            low       = float(row[3]),
                            close     = float(row[4]),
                            volume    = float(row[5]),
                        ))
                candles.sort(key=lambda c: c.timestamp)
                return candles[:n_bars]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RATE_DELAY * (attempt + 1))
    return []


# ============================================================
# EARLY PATH ANALYSIS
# ============================================================

def analyze_early_path(
    entry_price: float,
    candles:     List[Candle],
    direction:   str,
) -> Dict:
    """
    Analyze the first 1-3 bars after signal to characterize entry quality.

    Uses intrabar highs/lows (not just close-to-close) to detect whether
    price moved adversely before eventually recovering.

    Returns:
      early_direction:   "up" / "down" / "flat"
      early_strength:    float % (close[last_of_3] vs entry)
      early_vs_signal:   "same" / "against" / "mixed"
      intrabar_adverse:  worst intrabar adverse % in first 3 bars
                         (negative for pump = how far LOW went below entry;
                          positive for dump = how far HIGH went above entry)
      entry_quality:     "clean" / "moderate" / "ugly"
    """
    if not candles or entry_price <= 0:
        return {
            "early_direction":  "unknown",
            "early_strength":   0.0,
            "early_vs_signal":  "unknown",
            "intrabar_adverse": 0.0,
            "entry_quality":    "unknown",
        }

    def pct(p):
        return (p - entry_price) / entry_price * 100

    first_bars = candles[:min(3, len(candles))]

    # Worst intrabar adverse across the first 3 bars
    intrabar_adverse = 0.0
    for c in first_bars:
        if direction == "pump":
            adv = pct(c.low)          # negative = adverse for pump
            intrabar_adverse = min(intrabar_adverse, adv)
        elif direction == "dump":
            adv = pct(c.high)         # positive = adverse for dump
            intrabar_adverse = max(intrabar_adverse, adv)

    # Overall early direction: use last available close
    final_move = pct(first_bars[-1].close)
    if final_move > 0.3:
        early_direction = "up"
    elif final_move < -0.3:
        early_direction = "down"
    else:
        early_direction = "flat"

    # Compare early direction vs signal direction
    if direction == "pump":
        early_vs_signal = ("same" if early_direction == "up"
                           else "against" if early_direction == "down"
                           else "mixed")
    elif direction == "dump":
        early_vs_signal = ("same" if early_direction == "down"
                           else "against" if early_direction == "up"
                           else "mixed")
    else:
        early_vs_signal = "mixed"

    # Entry quality based on intrabar adverse magnitude
    adv_abs = abs(intrabar_adverse)
    if adv_abs < 0.5:
        entry_quality = "clean"
    elif adv_abs < 1.5:
        entry_quality = "moderate"
    else:
        entry_quality = "ugly"

    return {
        "early_direction":  early_direction,
        "early_strength":   round(final_move, 4),
        "early_vs_signal":  early_vs_signal,
        "intrabar_adverse": round(intrabar_adverse, 4),
        "entry_quality":    entry_quality,
    }


# ============================================================
# PATH TYPE CLASSIFICATION
# ============================================================

def classify_path_type(
    direction:   str,
    max_up:      float,
    max_dn:      float,
    bars_to_sig: Optional[int],
    move_12:     Optional[float],
    sig_move:    float,
) -> str:
    """
    Classify HOW the move happened.

    fast_pump / slow_pump     = pump reached sig_move quickly or slowly
    fast_dump / slow_dump     = dump reached sig_move quickly or slowly
    pump_reversal             = pumped then reversed back below sig_move close
    dump_reversal             = dumped then reversed back above sig_move close
    fake_pump                 = signal was pump but price dumped instead
    fake_dump                 = signal was dump but price pumped instead
    whipsaw                   = both +sig_move AND -sig_move were hit
    neutral_drift             = nothing significant happened
    """
    up_hit = max_up >=  sig_move
    dn_hit = max_dn <= -sig_move

    if up_hit and dn_hit:
        return "whipsaw"

    m12 = move_12 if move_12 is not None else 0.0

    if direction == "pump":
        if up_hit:
            # Reversal: peaked but closed well below peak
            if max_up > sig_move * 1.5 and m12 < sig_move * 0.3:
                return "pump_reversal"
            return "fast_pump" if (bars_to_sig and bars_to_sig <= 3) else "slow_pump"
        elif dn_hit:
            return "fake_pump"
        return "neutral_drift"

    elif direction == "dump":
        if dn_hit:
            if max_dn < -sig_move * 1.5 and m12 > -sig_move * 0.3:
                return "dump_reversal"
            return "fast_dump" if (bars_to_sig and bars_to_sig <= 3) else "slow_dump"
        elif up_hit:
            return "fake_dump"
        return "neutral_drift"

    else:
        # Baseline event (no direction signal)
        if up_hit and not dn_hit:
            return "fast_pump" if (bars_to_sig and bars_to_sig <= 3) else "slow_pump"
        if dn_hit and not up_hit:
            return "fast_dump" if (bars_to_sig and bars_to_sig <= 3) else "slow_dump"
        if up_hit and dn_hit:
            return "whipsaw"
        return "neutral_drift"


# ============================================================
# SPEED / PEAK / SYMMETRY
# ============================================================

def classify_speed(bars_to_sig: Optional[int]) -> str:
    if bars_to_sig is None:   return "none"
    if bars_to_sig <= 3:      return "fast"
    if bars_to_sig <= 6:      return "medium"
    return "slow"


def find_peak_bar(entry_price: float, candles: List[Candle], direction: str) -> int:
    """Bar number (1-indexed) that achieved the best favorable move."""
    if not candles:
        return 0
    best_val, best_bar = 0.0, 1
    for i, c in enumerate(candles):
        if direction == "pump":
            val = (c.high  - entry_price) / entry_price * 100
        elif direction == "dump":
            val = (entry_price - c.low) / entry_price * 100
        else:
            val = abs(c.close - entry_price) / entry_price * 100
        if val > best_val:
            best_val, best_bar = val, i + 1
    return best_bar


def classify_symmetry(max_up: float, max_dn: float, sig_move: float) -> str:
    both_hit     = max_up >= sig_move and max_dn <= -sig_move
    both_notable = max_up > sig_move * 0.5 and abs(max_dn) > sig_move * 0.5
    if both_hit:      return "whipsaw"
    if both_notable:  return "symmetric"
    return "one_sided"


def compute_drawdown_before_win(
    entry_price: float,
    candles:     List[Candle],
    direction:   str,
    sig_move:    float,
) -> Optional[float]:
    """
    Worst adverse % before the first bar where favorable move >= sig_move.
    Returns None if sig_move was never reached.
    Returns 0.0 if first bar already hit sig_move.
    """
    worst = 0.0
    for c in candles:
        fav = ((c.high - entry_price) / entry_price * 100 if direction == "pump"
               else (entry_price - c.low) / entry_price * 100)
        if fav >= sig_move:
            return round(abs(worst), 4)
        if direction == "pump":
            worst = min(worst, (c.low  - entry_price) / entry_price * 100)
        else:
            worst = max(worst, (c.high - entry_price) / entry_price * 100)
    return None


# ============================================================
# MAIN OUTCOME COMPUTATION
# ============================================================

def compute_outcome(
    entry_price:    float,
    direction:      str,
    future_candles: List[Candle],
    bars:           List[int]   = OUTCOME_BARS,
    up_pcts:        List[float] = OUTCOME_UP_PCTS,
    dn_pcts:        List[float] = OUTCOME_DOWN_PCTS,
    sig_move:       float       = OUTCOME_SIG_MOVE,
) -> Dict:
    """
    Compute full outcome dict for one signal event.
    Backward-compatible with v1 outcome columns plus new path-aware fields.
    """
    if not future_candles or entry_price <= 0:
        return {"outcome_labeled": False}

    n      = len(future_candles)
    closes = [c.close for c in future_candles]
    highs  = [c.high  for c in future_candles]
    lows   = [c.low   for c in future_candles]

    def pct(p):
        return (p - entry_price) / entry_price * 100

    # Per-bar close moves
    bar_moves: Dict[int, Optional[float]] = {}
    for b in bars:
        bar_moves[b] = round(pct(closes[b - 1]), 4) if b <= n else None

    # Max excursions over available bars
    window = min(n, max(bars))
    max_up = round(pct(max(highs[:window])), 4) if window >= 1 else 0.0
    max_dn = round(pct(min(lows[:window])),  4) if window >= 1 else 0.0

    # Threshold flags
    hit_up = {p: max_up >=  p for p in up_pcts}
    hit_dn = {p: max_dn <= -p for p in dn_pcts}

    # Bars to first significant close move
    bars_to_sig: Optional[int] = None
    for i, c in enumerate(future_candles):
        if abs(pct(c.close)) >= sig_move:
            bars_to_sig = i + 1
            break

    # Early path (first 3 bars)
    early = analyze_early_path(entry_price, future_candles[:3], direction)

    # Went-against: intrabar adverse > 1% in first 3 bars
    if direction == "pump":
        went_against = early["intrabar_adverse"] < -1.0
    elif direction == "dump":
        went_against = early["intrabar_adverse"] > 1.0
    else:
        went_against = False

    # Final resolution
    move_8 = bar_moves.get(8) or bar_moves.get(6) or bar_moves.get(3) or 0.0
    if move_8 is None:
        move_8 = 0.0

    if direction == "pump":
        if max_up >= sig_move and (max_dn > -sig_move or max_up > abs(max_dn)):
            resolved = "pump"
        elif max_dn <= -sig_move and abs(max_dn) > max_up:
            resolved = "failed"
        elif abs(move_8) < 1.0:
            resolved = "neutral"
        else:
            resolved = "failed"
    elif direction == "dump":
        if max_dn <= -sig_move and (max_up < sig_move or abs(max_dn) > max_up):
            resolved = "dump"
        elif max_up >= sig_move and max_up > abs(max_dn):
            resolved = "failed"
        elif abs(move_8) < 1.0:
            resolved = "neutral"
        else:
            resolved = "failed"
    else:
        if max_up >= sig_move:
            resolved = "pump"
        elif max_dn <= -sig_move:
            resolved = "dump"
        else:
            resolved = "neutral"

    # Entry quality label combining quality + resolution
    eq = early["entry_quality"]
    won = resolved in ("pump", "dump")
    entry_quality_label = f"{eq}_{'won' if won else 'failed'}"

    # Path type
    path_type = classify_path_type(
        direction, max_up, max_dn, bars_to_sig, bar_moves.get(12), sig_move
    )

    return {
        # Core
        "outcome_labeled":          True,
        "outcome_resolved":         resolved,
        "outcome_max_up_pct":       max_up,
        "outcome_max_dn_pct":       max_dn,
        "outcome_bars_to_sig":      bars_to_sig,
        # Per-bar
        "outcome_pct_1bar":         bar_moves.get(1),
        "outcome_pct_3bar":         bar_moves.get(3),
        "outcome_pct_6bar":         bar_moves.get(6),
        "outcome_pct_8bar":         bar_moves.get(8),
        "outcome_pct_12bar":        bar_moves.get(12),
        # Thresholds
        "outcome_hit_up2":          int(hit_up.get(2.0,  False)),
        "outcome_hit_up3":          int(hit_up.get(3.0,  False)),
        "outcome_hit_up5":          int(hit_up.get(5.0,  False)),
        "outcome_hit_up10":         int(hit_up.get(10.0, False)),
        "outcome_hit_dn2":          int(hit_dn.get(2.0,  False)),
        "outcome_hit_dn3":          int(hit_dn.get(3.0,  False)),
        "outcome_hit_dn5":          int(hit_dn.get(5.0,  False)),
        "outcome_hit_dn10":         int(hit_dn.get(10.0, False)),
        # Went-against (improved)
        "outcome_went_against":     int(went_against),
        "outcome_first_move":       early["early_direction"],
        # NEW path-aware fields
        "outcome_path_type":        path_type,
        "outcome_entry_quality":    entry_quality_label,
        "outcome_early_vs_signal":  early["early_vs_signal"],
        "outcome_early_strength":   early["early_strength"],
        "outcome_intrabar_adverse": early["intrabar_adverse"],
        "outcome_speed":            classify_speed(bars_to_sig),
        "outcome_peak_bar":         find_peak_bar(entry_price, future_candles, direction),
        "outcome_symmetry":         classify_symmetry(max_up, max_dn, sig_move),
        "outcome_drawdown_bw":      compute_drawdown_before_win(
                                        entry_price, future_candles, direction, sig_move),
    }


# ============================================================
# MATURITY CHECK
# ============================================================

def event_is_mature(event_ts_ms: int) -> bool:
    """True if (MATURITY_BARS + 1) × 5min has elapsed since the event."""
    now_ms  = int(time.time() * 1000)
    need_ms = (MATURITY_BARS + 1) * BAR_SECONDS * 1000
    return (now_ms - event_ts_ms) >= need_ms


# ============================================================
# MAIN LABELING PASS
# ============================================================

async def label_events(rerun: bool = False, label_all: bool = False):
    """
    Load research events → find unlabeled mature ones →
    fetch future candles → compute outcomes → save back once.
    """
    ensure_data_dir()
    parquet_path = os.path.join(DATA_DIR, "raw", RESEARCH_PARQUET)

    print(f"\n{'='*60}")
    print(f"  OUTCOME LABELER  (path-aware v2)")
    print(f"{'='*60}")

    rows = load_parquet(parquet_path)
    if not rows:
        print("  No research events found. Run scanner first.")
        return

    print(f"  Loaded {len(rows)} events")

    # Build event_id → row index for O(1) mutation and dedup safety
    id_to_row: Dict[str, Dict] = {}
    for row in rows:
        eid = row.get("event_id") or f"{row.get('symbol')}_{row.get('timestamp')}"
        id_to_row[eid] = row

    # Decide which events to label
    to_label = []
    for row in rows:
        has_label = _safe_bool(row.get("outcome_labeled", False))
        if has_label and not rerun:
            continue
        ts = row.get("timestamp")
        if not ts:
            continue
        if not event_is_mature(int(ts)):
            continue
        if not label_all and not row.get("direction"):
            continue
        to_label.append(row)

    mature_ct = sum(1 for r in rows
                    if r.get("timestamp") and event_is_mature(int(r["timestamp"])))
    print(f"  Mature events total:  {mature_ct}")
    print(f"  Need labeling:        {len(to_label)}")

    if not to_label:
        print("  Nothing to label (need 65+ min future data, or use --rerun).")
        return

    # Group by symbol
    by_symbol: Dict[str, List[Dict]] = {}
    for row in to_label:
        sym = str(row.get("symbol", "UNKNOWN"))
        by_symbol.setdefault(sym, []).append(row)
    print(f"  Symbols to process:   {len(by_symbol)}")

    labeled_count = 0
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for sym, sym_rows in by_symbol.items():
            sym_rows.sort(key=lambda r: r.get("timestamp", 0))
            for row in sym_rows:
                ts    = int(row.get("timestamp", 0))
                price = float(row.get("price", 0) or 0)
                dire  = str(row.get("direction", "") or "")

                if price <= 0:
                    continue

                future_candles = await fetch_candles_after(
                    session, sym, ts, n_bars=FUTURE_BARS_TO_FETCH
                )
                await asyncio.sleep(RATE_DELAY)

                if len(future_candles) < 3:
                    continue

                outcome = compute_outcome(price, dire, future_candles)

                # Mutate via the indexed reference
                eid = row.get("event_id") or f"{sym}_{ts}"
                if eid in id_to_row:
                    id_to_row[eid].update(outcome)

                labeled_count += 1
                if labeled_count % 50 == 0:
                    print(f"  Labeled {labeled_count} / {len(to_label)} ...")

    print(f"\n  Labeled {labeled_count} events.")
    if labeled_count == 0:
        return

    # Single dedup-safe save from the indexed dict
    final_rows = list(id_to_row.values())
    ok  = save_parquet(final_rows, parquet_path)
    fmt = "parquet" if ok else "csv"
    print(f"  Saved {len(final_rows)} rows → {parquet_path} ({fmt})")

    labeled_rows = [r for r in final_rows if _safe_bool(r.get("outcome_labeled", False))]
    out_csv = os.path.join(DATA_DIR, "outcomes", "outcome_labels.csv")
    save_csv_generic(labeled_rows, out_csv)
    print(f"  Exported {len(labeled_rows)} labeled rows → {out_csv}")
    print("  Done.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path-aware outcome labeler"
    )
    parser.add_argument("--rerun", action="store_true",
                        help="Re-label already-labeled events")
    parser.add_argument("--all",   action="store_true",
                        help="Also label non-signal baseline events")
    args = parser.parse_args()
    asyncio.run(label_events(rerun=args.rerun, label_all=args.all))
