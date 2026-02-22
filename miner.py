#!/usr/bin/env python3
"""
miner.py
========
Discovers rule combinations that predict pump/dump events.

Approach:
- Each rule is a set of boolean conditions on features.
- We score each candidate rule by: precision, recall, FPR, avg/median move.
- Walk-forward validation (time-based) to avoid overfitting.
- Minimum sample size enforced.

Usage:
    from miner import RuleMiner
    miner = RuleMiner()
    rules = miner.mine(samples, direction="pump", bucket=5)
"""

import json
import itertools
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ── RULE CANDIDATES DEFINITION ───────────────────────────────
# Each candidate is a (feature_name, operator, threshold) triple.
# Operator: "gt", "lt", "eq" (for bool: gt 0.5 = True)

BOOL_FEATURES = [
    "rsi_overbought", "rsi_oversold",
    "macd_cross_up", "macd_cross_down",
    "ema50_gt_ema200",
    "supertrend_bull", "supertrend_bear",
    "supertrend_flipped_bull", "supertrend_flipped_bear",
    "bb_squeeze",
    "bb_width_contracting",
    "close_above_upper_bb", "close_below_lower_bb",
    "atr_contracting",
    "atr_ratio_lt_09",
    "obv_rising", "obv_falling",
    "volume_spike_18", "volume_spike_25",
    "volume_contracting",
    "range_compressed_vs_prior",
    "breakout_up", "breakout_down",
    "liquidity_sweep_high", "liquidity_sweep_low",
    "candle_bull_engulf", "candle_bear_engulf",
    "candle_bull_pinbar", "candle_bear_pinbar",
    "candle_inside_bar",
    "candle_ib_breakout_up", "candle_ib_breakout_down",
    "ema_golden_cross", "ema_death_cross",
    "close_gt_ema50", "close_gt_ema200",
]

NUMERIC_CANDIDATES = [
    ("rsi",               "gt", [40, 50, 55, 60]),
    ("rsi",               "lt", [45, 50, 55, 60]),
    ("rsi_slope",         "gt", [0, 1, 2]),
    ("rsi_slope",         "lt", [0, -1, -2]),
    ("macd_hist",         "gt", [0]),
    ("macd_hist",         "lt", [0]),
    ("macd_hist_slope",   "gt", [0]),
    ("macd_hist_slope",   "lt", [0]),
    ("volume_spike_ratio","gt", [1.5, 1.8, 2.5]),
    ("bb_width_pct",      "lt", [3.0, 4.0, 5.0]),
    ("atr_ratio",         "lt", [0.8, 0.9, 1.0]),
    ("ema_dist_pct",      "gt", [0.5, 1.0]),
    ("ema_dist_pct",      "lt", [-0.5, -1.0]),
    ("range_compression", "lt", [0.03, 0.05, 0.08]),
]


@dataclass
class Condition:
    feature: str
    operator: str       # "gt", "lt", "eq", "is_true", "is_false"
    threshold: float = 0.5

    def evaluate(self, feats: Dict[str, Any]) -> Optional[bool]:
        val = feats.get(self.feature)
        if val is None:
            return None
        if self.operator == "is_true":
            return bool(val)
        if self.operator == "is_false":
            return not bool(val)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        if self.operator == "gt":
            return val > self.threshold
        if self.operator == "lt":
            return val < self.threshold
        if self.operator == "eq":
            return abs(val - self.threshold) < 1e-9
        return None

    def to_dict(self) -> Dict:
        return {"feature": self.feature, "operator": self.operator, "threshold": self.threshold}


@dataclass
class Rule:
    name: str
    conditions: List[Condition]
    weight: float = 1.0

    # Stats (filled after evaluation)
    sample_size:     int   = 0
    precision:       float = 0.0
    recall:          float = 0.0
    fpr:             float = 0.0
    avg_move:        float = 0.0
    median_move:     float = 0.0
    score:           float = 0.0   # composite score

    def matches(self, feats: Dict[str, Any]) -> bool:
        """Returns True only if ALL conditions pass (no None allowed = skip)."""
        for cond in self.conditions:
            result = cond.evaluate(feats)
            if result is None or not result:
                return False
        return True

    def to_dict(self) -> Dict:
        return {
            "name":        self.name,
            "weight":      self.weight,
            "conditions":  [c.to_dict() for c in self.conditions],
            "stats": {
                "sample_size": self.sample_size,
                "precision":   round(self.precision, 4),
                "recall":      round(self.recall, 4),
                "fpr":         round(self.fpr, 4),
                "avg_move":    round(self.avg_move, 4),
                "median_move": round(self.median_move, 4),
                "score":       round(self.score, 4),
            }
        }


# ── SAMPLE SCHEMA ─────────────────────────────────────────────
# Each sample:
# {
#   "features": {feat_name: value, ...},   # from compute_features()
#   "is_event": bool,                       # True if this bar started a pump/dump
#   "magnitude": float,                     # move % (0 if not event)
#   "direction": "pump"|"dump"|"none",
#   "bucket": float|None,
# }


class RuleMiner:
    def __init__(self,
                 min_sample_size: int = 200,
                 min_precision: float = 0.55,
                 max_fpr: float = 0.15,
                 max_conditions: int = 6,
                 top_k: int = 20):
        self.min_sample_size = min_sample_size
        self.min_precision   = min_precision
        self.max_fpr         = max_fpr
        self.max_conditions  = max_conditions
        self.top_k           = top_k

    # ── CANDIDATE GENERATION ──────────────────────────────────

    def _build_candidates(self) -> List[Condition]:
        cands: List[Condition] = []
        # Bool features
        for feat in BOOL_FEATURES:
            cands.append(Condition(feat, "is_true"))
            cands.append(Condition(feat, "is_false"))
        # Numeric features
        for feat, op, thresholds in NUMERIC_CANDIDATES:
            for thr in thresholds:
                cands.append(Condition(feat, op, thr))
        return cands

    # ── EVALUATION ────────────────────────────────────────────

    def evaluate_rule(self,
                      rule: Rule,
                      samples: List[Dict],
                      direction: str,
                      bucket: float) -> Rule:
        """
        Evaluate rule on samples.
        TP = triggered and is_event with correct direction+bucket
        FP = triggered but not event
        FN = not triggered but is_event
        TN = not triggered, not event
        """
        tp_moves: List[float] = []
        fp_count = 0
        fn_count = 0
        tn_count = 0

        for s in samples:
            triggered = rule.matches(s["features"])
            is_event  = (s["direction"] == direction and
                         s["bucket"] is not None and
                         s["bucket"] >= bucket)

            if triggered and is_event:
                tp_moves.append(s["magnitude"])
            elif triggered and not is_event:
                fp_count += 1
            elif not triggered and is_event:
                fn_count += 1
            else:
                tn_count += 1

        tp_count = len(tp_moves)
        total_triggered = tp_count + fp_count
        total_events    = tp_count + fn_count
        total_neg       = fp_count + tn_count

        rule.sample_size = total_triggered
        rule.precision   = tp_count / total_triggered if total_triggered > 0 else 0.0
        rule.recall      = tp_count / total_events    if total_events > 0    else 0.0
        rule.fpr         = fp_count / total_neg       if total_neg > 0       else 0.0
        rule.avg_move    = float(np.mean(tp_moves))   if tp_moves else 0.0
        rule.median_move = float(np.median(tp_moves)) if tp_moves else 0.0

        # Composite score: F-beta (beta=0.5 favours precision) * (1-FPR) * log(sample+1)
        beta = 0.5
        if rule.precision + rule.recall > 0:
            fbeta = (1 + beta**2) * rule.precision * rule.recall / (beta**2 * rule.precision + rule.recall)
        else:
            fbeta = 0.0
        import math
        rule.score = fbeta * (1 - rule.fpr) * math.log1p(rule.sample_size)

        return rule

    # ── SINGLE-CONDITION PASS ─────────────────────────────────

    def _single_condition_pass(self,
                               samples: List[Dict],
                               direction: str,
                               bucket: float) -> List[Rule]:
        """Screen individual conditions. Keep those above min_precision."""
        candidates = self._build_candidates()
        passing: List[Rule] = []
        for i, cond in enumerate(candidates):
            rule = Rule(name=f"single_{i}", conditions=[cond])
            rule = self.evaluate_rule(rule, samples, direction, bucket)
            if (rule.sample_size >= self.min_sample_size and
                    rule.precision >= self.min_precision and
                    rule.fpr <= self.max_fpr):
                passing.append(rule)
        # Sort by score
        passing.sort(key=lambda r: r.score, reverse=True)
        return passing

    # ── GREEDY COMBINATION ────────────────────────────────────

    def _greedy_combine(self,
                        base_conds: List[Condition],
                        all_cands: List[Condition],
                        samples: List[Dict],
                        direction: str,
                        bucket: float,
                        max_conds: int) -> Rule:
        """
        Greedy forward selection: start with best single condition,
        add one condition at a time if it improves score.
        """
        chosen = list(base_conds)
        best_score = -1.0

        # Evaluate base
        r = Rule("combo", chosen)
        r = self.evaluate_rule(r, samples, direction, bucket)
        best_score = r.score

        for _ in range(max_conds - len(chosen)):
            improved = False
            for cand in all_cands:
                # Avoid duplicating existing conditions
                if any(c.feature == cand.feature and c.operator == cand.operator
                       for c in chosen):
                    continue
                trial = Rule("trial", chosen + [cand])
                trial = self.evaluate_rule(trial, samples, direction, bucket)
                if (trial.sample_size >= self.min_sample_size and
                        trial.precision >= self.min_precision and
                        trial.fpr <= self.max_fpr and
                        trial.score > best_score):
                    best_score = trial.score
                    best_cand  = cand
                    improved   = True
            if improved:
                chosen.append(best_cand)
            else:
                break

        final = Rule("final", chosen)
        return self.evaluate_rule(final, samples, direction, bucket)

    # ── WALK-FORWARD VALIDATION ───────────────────────────────

    def _wf_validate(self,
                     rule: Rule,
                     all_samples: List[Dict],
                     direction: str,
                     bucket: float,
                     n_splits: int = 5) -> Dict[str, float]:
        """
        Walk-forward validate a rule. Returns mean/std of precision/recall/fpr.
        """
        n = len(all_samples)
        fold = n // (n_splits + 1)
        precisions, recalls, fprs = [], [], []

        for k in range(n_splits):
            val_start = (k + 1) * fold
            val_end   = (k + 2) * fold
            val = all_samples[val_start:val_end]
            if not val:
                continue
            r = Rule("wf", rule.conditions)
            r = self.evaluate_rule(r, val, direction, bucket)
            if r.sample_size >= 5:  # at least some triggers
                precisions.append(r.precision)
                recalls.append(r.recall)
                fprs.append(r.fpr)

        return {
            "wf_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "wf_precision_std":  float(np.std(precisions))  if precisions else 1.0,
            "wf_recall_mean":    float(np.mean(recalls))    if recalls    else 0.0,
            "wf_fpr_mean":       float(np.mean(fprs))       if fprs       else 1.0,
        }

    # ── MAIN MINE ─────────────────────────────────────────────

    def mine(self,
             samples: List[Dict],
             direction: str = "pump",
             bucket: float = 5.0,
             n_wf_splits: int = 5) -> List[Dict]:
        """
        Full mining run.

        Args:
            samples: list of sample dicts (see schema above)
            direction: "pump" or "dump"
            bucket: minimum magnitude bucket (5, 10, or 20)
            n_wf_splits: walk-forward splits

        Returns:
            List of top rule dicts, sorted by wf-validated score.
        """
        print(f"\n[Miner] Mining {direction.upper()} >= {bucket}% | "
              f"samples={len(samples)}")

        all_cands = self._build_candidates()

        # 1) single condition screen on training portion (first 80%)
        split = int(len(samples) * 0.8)
        train = samples[:split]

        print(f"[Miner] Single condition screen on {len(train)} train samples...")
        single_rules = self._single_condition_pass(train, direction, bucket)
        print(f"[Miner] {len(single_rules)} single-condition rules passed screen.")

        if not single_rules:
            print("[Miner] No rules passed single-condition screen. Lowering thresholds?")
            return []

        # 2) Greedy combination from top-K single conditions
        top_singles = single_rules[:min(20, len(single_rules))]
        combined_rules: List[Rule] = []

        for i, base_rule in enumerate(top_singles):
            print(f"[Miner] Combining from base condition {i+1}/{len(top_singles)}...")
            combo = self._greedy_combine(
                base_conds=list(base_rule.conditions),
                all_cands=all_cands,
                samples=train,
                direction=direction,
                bucket=bucket,
                max_conds=self.max_conditions
            )
            combined_rules.append(combo)

        # Deduplicate by condition set fingerprint
        seen = set()
        unique_rules: List[Rule] = []
        for r in combined_rules:
            fp = frozenset((c.feature, c.operator, c.threshold) for c in r.conditions)
            if fp not in seen:
                seen.add(fp)
                unique_rules.append(r)

        # Sort by training score
        unique_rules.sort(key=lambda r: r.score, reverse=True)
        top_rules = unique_rules[:self.top_k]

        # 3) Walk-forward validate top rules
        print(f"[Miner] Walk-forward validating {len(top_rules)} candidate rules...")
        results = []
        for idx, rule in enumerate(top_rules):
            rule.name = f"{direction.upper()}_{int(bucket)}PCT_RULE_{idx+1:02d}"
            rule.weight = round(rule.precision * 5, 2)
            wf = self._wf_validate(rule, samples, direction, bucket, n_wf_splits)

            rule_dict = rule.to_dict()
            rule_dict["wf_validation"] = wf
            # Filter: only keep rules with consistent wf precision
            if (wf["wf_precision_mean"] >= self.min_precision and
                    wf["wf_precision_std"] < 0.15):
                results.append(rule_dict)

        results.sort(key=lambda r: r["wf_validation"]["wf_precision_mean"], reverse=True)
        print(f"[Miner] {len(results)} rules passed walk-forward validation.")
        return results


def build_samples(
    candles_by_symbol: Dict[str, List[Dict]],
    events_by_symbol:  Dict[str, List[Dict]],
    feature_windows:   List[int] = [5, 10, 20, 30],
    window_to_use:     int = 20
) -> List[Dict]:
    """
    Build a flat list of samples for the miner.

    For each symbol:
      - For each event: create a POSITIVE sample using candles[event_idx-window:event_idx]
      - For every Nth non-event bar: create a NEGATIVE sample

    candles_by_symbol: {symbol: [candle_dict, ...]}
    events_by_symbol:  {symbol: [event_dict, ...]}  (from label_events)
    """
    from features import compute_features

    samples = []
    neg_step = 3  # take every 3rd non-event bar as negative

    for symbol, candles in candles_by_symbol.items():
        events = events_by_symbol.get(symbol, [])
        event_indices = {ev["event_idx"]: ev for ev in events}

        for i in range(window_to_use, len(candles)):
            window = candles[i - window_to_use : i]  # strictly before bar i
            feats = compute_features(window)

            if i in event_indices:
                ev = event_indices[i]
                samples.append({
                    "symbol":    symbol,
                    "bar_idx":   i,
                    "features":  feats,
                    "is_event":  True,
                    "direction": ev["direction"],
                    "bucket":    ev["bucket"],
                    "magnitude": ev["magnitude"],
                })
            elif i % neg_step == 0:
                samples.append({
                    "symbol":    symbol,
                    "bar_idx":   i,
                    "features":  feats,
                    "is_event":  False,
                    "direction": "none",
                    "bucket":    None,
                    "magnitude": 0.0,
                })

    # Sort by (symbol, bar_idx) to preserve time order
    samples.sort(key=lambda s: (s["symbol"], s["bar_idx"]))
    return samples
