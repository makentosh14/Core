#!/usr/bin/env python3
"""
train_score_model.py — Phase 7 Turn 2.

Train a classifier on labeled_dataset_*.csv to replace the hand-tuned
score.py:WEIGHTS. Uses temporal train/test split (NO shuffling — preserves
the time order so test-set performance reflects what a deployed model
would actually see going forward, not what an in-sample-shuffled fit
would achieve).

Two model families supported:
  * 'logreg'  — scikit-learn LogisticRegression with class_weight='balanced'
                (interpretable, robust to small samples, no tuning needed)
  * 'gbdt'    — LightGBM if available, else GradientBoostingClassifier
                (higher capacity, can find non-linear feature interactions)

Outputs:
  score_model.pkl       — joblib-pickled model
  feature_columns.json  — list of features used (order matters for inference)
  training_report.txt   — AUC, precision/recall at various thresholds, top features

USAGE:
  python train_score_model.py --dataset labeled_dataset_30d.csv
  python train_score_model.py --dataset labeled_dataset_30d.csv --model gbdt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Lazy imports inside main so we can fail gracefully if sklearn isn't installed.


def main():
    p = argparse.ArgumentParser(description="Train scoring classifier on labeled dataset.")
    p.add_argument("--dataset", required=True,
                   help="Path to labeled CSV from build_labeled_dataset.py")
    p.add_argument("--model", choices=("logreg", "gbdt"), default="logreg")
    p.add_argument("--test-split", type=float, default=0.3,
                   help="Fraction held out for test (temporal, default 0.3)")
    p.add_argument("--output-prefix", default="score_model",
                   help="Output file prefix (default 'score_model')")
    p.add_argument("--target-precision", type=float, default=0.5,
                   help="Target precision when picking the prediction threshold")
    args = p.parse_args()

    try:
        import pandas as pd
        import numpy as np
        from sklearn.metrics import roc_auc_score, precision_recall_curve
        import joblib
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install with: pip install scikit-learn pandas joblib")
        return 1

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"  rows: {len(df)}, columns: {len(df.columns)}")

    if "label" not in df.columns:
        print(f"[ERROR] 'label' column not found")
        return 1

    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    print(f"  positives: {n_pos}  negatives: {n_neg}  ratio: 1:{n_neg/max(n_pos,1):.1f}")
    if n_pos < 30:
        print(f"[ERROR] Only {n_pos} positive samples — too few to train. "
              f"Re-run build_labeled_dataset.py with --days 90 (or more) "
              f"or lower --min-move-pct to enlarge the positive class.")
        return 1

    # Temporal train/test split — preserve time order.
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    split_idx = int(len(df) * (1 - args.test_split))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nTemporal split:")
    print(f"  train: {len(train_df)} rows ({(train_df['label']==1).sum()} positives)")
    print(f"  test:  {len(test_df)} rows ({(test_df['label']==1).sum()} positives)")

    # Feature columns = everything except metadata + label.
    meta_cols = {"symbol", "timestamp_ms", "bar_idx", "label",
                 "move_pct_following", "move_direction"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Drop columns that are entirely NaN or have a single value (no signal).
    drop_cols = []
    for c in feature_cols:
        if train_df[c].isna().all() or train_df[c].nunique(dropna=True) <= 1:
            drop_cols.append(c)
    if drop_cols:
        print(f"  dropping {len(drop_cols)} constant/empty columns: {drop_cols[:5]}{'...' if len(drop_cols)>5 else ''}")
        feature_cols = [c for c in feature_cols if c not in drop_cols]
    print(f"  features after filtering: {len(feature_cols)}")

    # Convert bools to ints, fill NaN with 0.
    def prep(d):
        x = d[feature_cols].copy()
        for c in feature_cols:
            if x[c].dtype == bool:
                x[c] = x[c].astype(int)
        x = x.apply(pd.to_numeric, errors="coerce").fillna(0)
        return x

    X_train = prep(train_df)
    y_train = train_df["label"].values
    X_test = prep(test_df)
    y_test = test_df["label"].values

    print(f"\nTraining {args.model} ...")
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="liblinear",
                random_state=42,
            )),
        ])
    else:
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                num_leaves=31, class_weight="balanced",
                random_state=42, verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            print("  (lightgbm not installed; using sklearn GradientBoostingClassifier)")
            model = GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42,
            )

    model.fit(X_train, y_train)

    # Evaluate on test set.
    test_proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        auc = float("nan")
    print(f"\nTest AUC: {auc:.4f}")

    # Find threshold that achieves target precision (if possible).
    precision, recall, thresholds = precision_recall_curve(y_test, test_proba)
    chosen_threshold = 0.5
    chosen_precision = 0.0
    chosen_recall = 0.0
    for p_val, r_val, t_val in zip(precision, recall, list(thresholds) + [1.0]):
        if p_val >= args.target_precision and r_val > chosen_recall:
            chosen_threshold = t_val
            chosen_precision = p_val
            chosen_recall = r_val

    print(f"  At target precision {args.target_precision*100:.0f}%:")
    print(f"    threshold:  {chosen_threshold:.4f}")
    print(f"    precision:  {chosen_precision:.4f}")
    print(f"    recall:     {chosen_recall:.4f}")

    # Save model + columns + report.
    model_path = f"{args.output_prefix}.pkl"
    columns_path = f"{args.output_prefix}_columns.json"
    report_path = f"{args.output_prefix}_report.txt"

    joblib.dump(model, model_path)
    with open(columns_path, "w") as f:
        json.dump({
            "feature_columns": feature_cols,
            "chosen_threshold": float(chosen_threshold),
            "model_type": args.model,
        }, f, indent=2)
    with open(report_path, "w") as f:
        f.write("TRAINING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"dataset:           {args.dataset}\n")
        f.write(f"rows:              {len(df)}\n")
        f.write(f"positives:         {n_pos}\n")
        f.write(f"negatives:         {n_neg}\n")
        f.write(f"features:          {len(feature_cols)}\n")
        f.write(f"model:             {args.model}\n")
        f.write(f"test AUC:          {auc:.4f}\n")
        f.write(f"chosen threshold:  {chosen_threshold:.4f}\n")
        f.write(f"precision @ thr:   {chosen_precision:.4f}\n")
        f.write(f"recall    @ thr:   {chosen_recall:.4f}\n")

        # For logistic regression, dump top features by coefficient magnitude.
        if args.model == "logreg":
            try:
                clf = model.named_steps["clf"]
                coefs = clf.coef_[0]
                top_pos = sorted(zip(feature_cols, coefs), key=lambda kv: -kv[1])[:15]
                top_neg = sorted(zip(feature_cols, coefs), key=lambda kv: kv[1])[:15]
                f.write("\nTOP POSITIVE COEFFICIENTS (predict pump):\n")
                for name, c in top_pos:
                    f.write(f"  {c:+8.4f}  {name}\n")
                f.write("\nTOP NEGATIVE COEFFICIENTS (predict NO pump):\n")
                for name, c in top_neg:
                    f.write(f"  {c:+8.4f}  {name}\n")
            except Exception as e:
                f.write(f"\n(could not extract coefficients: {e})\n")

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {columns_path}")
    print(f"  {report_path}")

    # Final verdict — same thresholds I called out in the audit:
    print()
    print("=" * 60)
    if auc < 0.55:
        print(f"  VERDICT: AUC {auc:.3f} is at or below noise level.")
        print(f"  This indicator set fundamentally CANNOT predict moves on")
        print(f"  this data. Don't proceed to wire this model into score.py.")
        print(f"  Consider: longer training window, different indicators, or")
        print(f"  abandon the model-based scoring approach.")
    elif auc < 0.6:
        print(f"  VERDICT: AUC {auc:.3f} shows weak predictive signal.")
        print(f"  Might be usable with very tight precision threshold (low")
        print(f"  recall, few trades, but each one credible).")
    elif auc < 0.7:
        print(f"  VERDICT: AUC {auc:.3f} is genuine predictive signal.")
        print(f"  Proceed to wire into score.py and validate via backtest.")
    else:
        print(f"  VERDICT: AUC {auc:.3f} is strong predictive signal.")
        print(f"  Verify it's not data leakage (re-check temporal split,")
        print(f"  feature engineering for lookahead). If clean, this is a")
        print(f"  deployable scoring model.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
