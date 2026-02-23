#!/usr/bin/env python3
"""
miner.py
========
Discovers rule combinations that predict pump/dump events.

Approach:
- Each rule is a set of boolean conditions on features.
- We score each candidate rule by: precision, recall, FPR, lift, avg/median move.
- Walk-forward validation (time-based) to avoid overfitting.
- Minimum sample size enforced.

KEY FIX v2: Thresholds are now auto-scaled based on actual base rate.
  With a 9% base rate, a rule at 55% precision is amazing (6x lift).
  The old hardcoded 55% min_precision was unreachable.
  Now min_precision = max(base_rate * min_lift, absolute_floor).
  Default min_lift = 2.5x, absolute_floor = 0.15 (15%).

Usage:
    from miner import RuleMiner
    miner = RuleMiner()
    rules = miner.mine(samples, direction="pump", bucket=5)
"""

import json
import math
import itertools
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


# ── RULE CANDIDATES DEFINITION ───────────────────────────────────────────────
# Each candidate is a (feature_name, operator, threshold) triple.
# Operator: "gt", "lt", "eq", "is_true", "is_false"

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

    def __str__(self) -> str:
        if self.operator in ("is_true", "is_false"):
            return f"{self.feature}={self.operator}"
        return f"{self.feature} {self.operator} {self.threshold}"


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
    lift:            float = 0.0   # precision / base_rate
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
                "lift":        round(self.lift, 4),
                "recall":      round(self.recall, 4),
                "fpr":         round(self.fpr, 4),
                "avg_move":    round(self.avg_move, 4),
                "median_move": round(self.median_move, 4),
                "score":       round(self.score, 4),
            }
        }


# ── SAMPLE SCHEMA ──────────────────────────────────────────────────────────
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
                 min_sample_size: int   = 50,
                 min_lift:        float = 2.0,
                 min_precision:   float = 0.15,   # absolute floor (overridden by lift-based calc)
                 max_fpr:         float = 0.30,
                 wf_min_lift:     float = 1.5,    # walk-forward: rule must have lift >= this
                 wf_max_std:      float = 0.25,   # walk-forward: precision std tolerance
                 max_conditions:  int   = 6,
                 top_k:           int   = 20):
        """
        Args:
            min_sample_size: minimum number of rule triggers (TPs + FPs)
            min_lift:        minimum lift = precision / base_rate (e.g. 2.0 = 2x better than random)
            min_precision:   absolute floor for precision regardless of lift calc
            max_fpr:         maximum false positive rate
            wf_min_lift:     minimum lift during walk-forward validation
            wf_max_std:      maximum std of precision across WF folds
            max_conditions:  maximum conditions per rule
            top_k:           number of top rules to walk-forward validate
        """
        self.min_sample_size = min_sample_size
        self.min_lift        = min_lift
        self.min_precision   = min_precision
        self.max_fpr         = max_fpr
        self.wf_min_lift     = wf_min_lift
        self.wf_max_std      = wf_max_std
        self.max_conditions  = max_conditions
        self.top_k           = top_k
        self._base_rate      = 0.0   # set during mine()

    def _effective_min_precision(self) -> float:
        """
        Compute effective min precision based on base rate.
        Rule must be at least min_lift × base_rate precise.
        Also enforces absolute floor min_precision.
        """
        lift_based = self._base_rate * self.min_lift
        return max(lift_based, self.min_precision)

    def _effective_max_fpr(self) -> float:
        """
        Compute effective max FPR.
        With a low base rate, even a small FPR means many false positives.
        Scale max_fpr proportionally but never let it go below 0.05.
        """
        # Allow FPR up to max_fpr, but cap it at a reasonable absolute level
        return min(self.max_fpr, max(0.05, self._base_rate * 4))

    # ── CANDIDATE GENERATION ─────────────────────────────────────────────────

    def _build_candidates(self) -> List[Condition]:
        cands: List[Condition] = []
        # Bool features: both is_true and is_false variants
        for feat in BOOL_FEATURES:
            cands.append(Condition(feat, "is_true"))
            cands.append(Condition(feat, "is_false"))
        # Numeric features
        for feat, op, thresholds in NUMERIC_CANDIDATES:
            for thr in thresholds:
                cands.append(Condition(feat, op, thr))
        return cands

    # ── EVALUATION ────────────────────────────────────────────────────────────

    def evaluate_rule(self,
                      rule: Rule,
                      samples: List[Dict],
                      direction: str,
                      bucket: float) -> Rule:
        """
        Evaluate rule on samples.
        TP = triggered and is_event with correct direction+bucket
        FP = triggered but not event (or wrong direction/bucket)
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

        tp_count        = len(tp_moves)
        total_triggered = tp_count + fp_count
        total_events    = tp_count + fn_count
        total_neg       = fp_count + tn_count

        rule.sample_size = total_triggered
        rule.precision   = tp_count / total_triggered if total_triggered > 0 else 0.0
        rule.recall      = tp_count / total_events    if total_events > 0    else 0.0
        rule.fpr         = fp_count / total_neg       if total_neg > 0       else 0.0
        rule.avg_move    = float(np.mean(tp_moves))   if tp_moves else 0.0
        rule.median_move = float(np.median(tp_moves)) if tp_moves else 0.0
        rule.lift        = rule.precision / self._base_rate if self._base_rate > 0 else 0.0

        # Composite score: F-beta (beta=0.5 favours precision) * (1 - FPR) * log(sample + 1) * lift
        # Using lift in the score rewards rules that beat random by more
        beta = 0.5
        if rule.precision + rule.recall > 0:
            fbeta = (1 + beta**2) * rule.precision * rule.recall / (beta**2 * rule.precision + rule.recall)
        else:
            fbeta = 0.0
        rule.score = fbeta * (1 - rule.fpr) * math.log1p(rule.sample_size) * max(rule.lift, 1.0)

        return rule

    # ── DIAGNOSTIC: show best single conditions even if nothing passes ────────

    def _diagnose_top_conditions(self,
                                  samples: List[Dict],
                                  direction: str,
                                  bucket: float,
                                  n_show: int = 10) -> None:
        """
        Print top N single conditions by lift regardless of thresholds.
        Helps understand what the data supports.
        """
        candidates = self._build_candidates()
        all_results = []

        for i, cond in enumerate(candidates):
            rule = Rule(name=f"diag_{i}", conditions=[cond])
            rule = self.evaluate_rule(rule, samples, direction, bucket)
            if rule.sample_size >= 10:   # need at least 10 triggers to be informative
                all_results.append(rule)

        all_results.sort(key=lambda r: r.lift, reverse=True)

        eff_min_prec = self._effective_min_precision()
        eff_max_fpr  = self._effective_max_fpr()

        print(f"\n[Miner] DIAGNOSTIC — Top {n_show} single conditions for {direction.upper()} >= {bucket}%")
        print(f"        Base rate={self._base_rate:.3f} | "
              f"Eff min precision={eff_min_prec:.3f} | "
              f"Min lift={self.min_lift}x | "
              f"Eff max FPR={eff_max_fpr:.3f}")
        print(f"        {'Condition':<45} {'N':>6} {'Prec':>7} {'Lift':>6} {'FPR':>7} {'Pass?':>6}")
        print(f"        {'-'*45} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*6}")

        for rule in all_results[:n_show]:
            cond_str = str(rule.conditions[0])
            passes = (rule.sample_size >= self.min_sample_size and
                      rule.precision >= eff_min_prec and
                      rule.fpr <= eff_max_fpr)
            mark = "✅" if passes else "❌"
            print(f"        {cond_str:<45} {rule.sample_size:>6} "
                  f"{rule.precision:>7.3f} {rule.lift:>6.2f}x "
                  f"{rule.fpr:>7.3f} {mark:>6}")
        print()

    # ── SINGLE-CONDITION SCREEN ───────────────────────────────────────────────

    def _single_condition_pass(self,
                               samples: List[Dict],
                               direction: str,
                               bucket: float) -> List[Rule]:
        """Screen individual conditions. Keep those above effective thresholds."""
        candidates = self._build_candidates()
        passing: List[Rule] = []

        eff_min_prec = self._effective_min_precision()
        eff_max_fpr  = self._effective_max_fpr()

        for i, cond in enumerate(candidates):
            rule = Rule(name=f"single_{i}", conditions=[cond])
            rule = self.evaluate_rule(rule, samples, direction, bucket)
            if (rule.sample_size >= self.min_sample_size and
                    rule.precision >= eff_min_prec and
                    rule.fpr <= eff_max_fpr):
                passing.append(rule)

        # Sort by score
        passing.sort(key=lambda r: r.score, reverse=True)
        return passing

    # ── GREEDY COMBINATION ────────────────────────────────────────────────────

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
        chosen     = list(base_conds)
        best_score = -1.0

        eff_min_prec = self._effective_min_precision()
        eff_max_fpr  = self._effective_max_fpr()

        # Evaluate base
        r = Rule("combo", chosen)
        r = self.evaluate_rule(r, samples, direction, bucket)
        best_score = r.score
        best_cand  = None

        for _ in range(max_conds - len(chosen)):
            improved = False
            for cand in all_cands:
                # Avoid duplicating conditions on same feature+operator
                if any(c.feature == cand.feature and c.operator == cand.operator
                       for c in chosen):
                    continue
                trial = Rule("trial", chosen + [cand])
                trial = self.evaluate_rule(trial, samples, direction, bucket)
                if (trial.sample_size >= self.min_sample_size and
                        trial.precision >= eff_min_prec and
                        trial.fpr <= eff_max_fpr and
                        trial.score > best_score):
                    best_score = trial.score
                    best_cand  = cand
                    improved   = True
            if improved and best_cand is not None:
                chosen.append(best_cand)
                best_cand = None
            else:
                break

        final = Rule("final", chosen)
        return self.evaluate_rule(final, samples, direction, bucket)

    # ── WALK-FORWARD VALIDATION ───────────────────────────────────────────────

    def _wf_validate(self,
                     rule: Rule,
                     all_samples: List[Dict],
                     direction: str,
                     bucket: float,
                     n_splits: int = 5) -> Dict[str, float]:
        """
        Walk-forward validate a rule. Returns mean/std of precision/lift/recall/fpr.
        """
        n    = len(all_samples)
        fold = n // (n_splits + 1)
        precisions, lifts, recalls, fprs = [], [], [], []

        for k in range(n_splits):
            val_start = (k + 1) * fold
            val_end   = (k + 2) * fold
            val       = all_samples[val_start:val_end]
            if not val:
                continue
            # Recalculate base rate for this fold
            fold_events = sum(1 for s in val
                              if s["direction"] == direction and
                              s["bucket"] is not None and
                              s["bucket"] >= bucket)
            fold_base_rate = fold_events / len(val) if val else self._base_rate

            r = Rule("wf", rule.conditions)
            # Temporarily set base rate to fold rate for lift calc
            saved_br        = self._base_rate
            self._base_rate = fold_base_rate if fold_base_rate > 0 else saved_br
            r = self.evaluate_rule(r, val, direction, bucket)
            self._base_rate = saved_br

            if r.sample_size >= 3:  # at least some triggers in this fold
                precisions.append(r.precision)
                lifts.append(r.lift)
                recalls.append(r.recall)
                fprs.append(r.fpr)

        return {
            "wf_precision_mean": float(np.mean(precisions)) if precisions else 0.0,
            "wf_precision_std":  float(np.std(precisions))  if precisions else 1.0,
            "wf_lift_mean":      float(np.mean(lifts))      if lifts      else 0.0,
            "wf_recall_mean":    float(np.mean(recalls))    if recalls    else 0.0,
            "wf_fpr_mean":       float(np.mean(fprs))       if fprs       else 1.0,
            "wf_folds_with_data": len(precisions),
        }

    # ── MAIN MINE ─────────────────────────────────────────────────────────────

    def mine(self,
             samples: List[Dict],
             direction: str = "pump",
             bucket: float  = 5.0,
             n_wf_splits: int = 5) -> List[Dict]:
        """
        Full mining run.

        Args:
            samples:     list of sample dicts (see schema above)
            direction:   "pump" or "dump"
            bucket:      minimum magnitude bucket (5, 10, or 20)
            n_wf_splits: walk-forward splits

        Returns:
            List of top rule dicts, sorted by wf-validated lift.
        """
        print(f"\n[Miner] Mining {direction.upper()} >= {bucket}% | "
              f"samples={len(samples)}")

        # ── Calculate base rate on FULL sample set ─────────────────
        n_events = sum(1 for s in samples
                       if s["direction"] == direction and
                       s["bucket"] is not None and
                       s["bucket"] >= bucket)
        self._base_rate = n_events / len(samples) if samples else 0.0

        eff_min_prec = self._effective_min_precision()
        eff_max_fpr  = self._effective_max_fpr()

        print(f"[Miner] Base rate={self._base_rate:.3f} ({n_events}/{len(samples)}) | "
              f"Eff min_precision={eff_min_prec:.3f} (={self.min_lift}x lift) | "
              f"Eff max_fpr={eff_max_fpr:.3f}")

        all_cands = self._build_candidates()

        # ── 1) Single condition screen on training portion (first 80%) ──
        split = int(len(samples) * 0.8)
        train = samples[:split]

        print(f"[Miner] Single condition screen on {len(train)} train samples "
              f"(min_sample={self.min_sample_size}, min_prec={eff_min_prec:.3f}, max_fpr={eff_max_fpr:.3f})...")
        single_rules = self._single_condition_pass(train, direction, bucket)
        print(f"[Miner] {len(single_rules)} single-condition rules passed screen.")

        if not single_rules:
            # Show diagnostic so user can understand the data
            self._diagnose_top_conditions(train, direction, bucket, n_show=15)
            print("[Miner] No single conditions met thresholds. "
                  "Consider lowering --min-precision or --min-sample.")
            return []

        # ── 2) Greedy combination from top-K single conditions ──────
        top_singles    = single_rules[:min(20, len(single_rules))]
        combined_rules: List[Rule] = []

        for i, base_rule in enumerate(top_singles):
            print(f"[Miner] Combining from base {i+1}/{len(top_singles)}: "
                  f"{base_rule.conditions[0]}  (lift={base_rule.lift:.2f}x)...")
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
        seen          = set()
        unique_rules: List[Rule] = []
        for r in combined_rules:
            fp = frozenset((c.feature, c.operator, c.threshold) for c in r.conditions)
            if fp not in seen:
                seen.add(fp)
                unique_rules.append(r)

        # Sort by training score
        unique_rules.sort(key=lambda r: r.score, reverse=True)
        top_rules = unique_rules[:self.top_k]

        # ── 3) Walk-forward validate top rules ─────────────────────
        print(f"[Miner] Walk-forward validating {len(top_rules)} candidate rules...")
        results = []

        for idx, rule in enumerate(top_rules):
            rule.name   = f"{direction.upper()}_{int(bucket)}PCT_RULE_{idx+1:02d}"
            rule.weight = round(rule.lift, 2)   # weight = lift ratio now (more meaningful)
            wf          = self._wf_validate(rule, samples, direction, bucket, n_wf_splits)

            rule_dict              = rule.to_dict()
            rule_dict["wf_validation"] = wf

            # Pass filter: WF lift must be >= wf_min_lift, std must be acceptable
            wf_precision_mean = wf["wf_precision_mean"]
            wf_base_rate      = self._base_rate   # approximation
            wf_lift           = wf_precision_mean / wf_base_rate if wf_base_rate > 0 else 0.0
            wf_passes         = (wf_lift >= self.wf_min_lift and
                                 wf["wf_precision_std"] < self.wf_max_std and
                                 wf["wf_folds_with_data"] >= 2)

            if wf_passes:
                results.append(rule_dict)
            else:
                print(f"   ❌ {rule.name}: wf_lift={wf_lift:.2f}x "
                      f"std={wf['wf_precision_std']:.3f} "
                      f"folds={wf['wf_folds_with_data']}")

        results.sort(
            key=lambda r: r["wf_validation"]["wf_lift_mean"]
                if "wf_lift_mean" in r["wf_validation"]
                else r["wf_validation"]["wf_precision_mean"],
            reverse=True
        )
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

    samples  = []
    neg_step = 3   # take every 3rd non-event bar as negative sample

    for symbol, candles in candles_by_symbol.items():
        events        = events_by_symbol.get(symbol, [])
        event_indices = {ev["event_idx"]: ev for ev in events}

        for i in range(window_to_use, len(candles)):
            window = candles[i - window_to_use : i]   # strictly before bar i — NO lookahead
            feats  = compute_features(window)

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

    # Sort by (symbol, bar_idx) to preserve time order — critical for walk-forward
    samples.sort(key=lambda s: (s["symbol"], s["bar_idx"]))
    return samples
