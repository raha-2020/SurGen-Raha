"""
Stage 2: 5-year survival using LightGBM on Stage-1 features (SR386).

Same inputs and CSV layout as survival_stage2_main.py (emb_* + prob_* or ablations).
Uses LightGBMClassifier with optional early stopping on the validation set.

Requires: pip install lightgbm

Run from reproducibility/:
  python survival_stage2_lightgbm.py \\
    --train_probs_csv exports/stage1_probs_train.csv \\
    --val_probs_csv exports/stage1_probs_validate.csv \\
    --test_probs_csv exports/stage1_probs_test.csv \\
    --surv_train_csv dataset_csv/SR386_5y_sur_train.csv \\
    --surv_val_csv dataset_csv/SR386_5y_sur_validate.csv \\
    --surv_test_csv dataset_csv/SR386_5y_sur_test.csv

Optional (Stage 1 test AUROCs + JSON):
  --report_stage1_test_auroc \\
  --out_metrics_json exports/stage2_lgbm_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from survival_stage2_main import (
    _merge_probs_survival,
    _stage1_test_aurocs,
    infer_stage2_feature_columns,
)

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except ImportError as e:
    raise ImportError(
        "Install LightGBM: pip install lightgbm"
    ) from e


def build_lgbm_classifier(
    random_state: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2 survival (5Y_SUR) with LightGBM")
    p.add_argument("--train_probs_csv", type=str, required=True)
    p.add_argument("--val_probs_csv", type=str, required=True)
    p.add_argument("--test_probs_csv", type=str, required=True)
    p.add_argument("--surv_train_csv", type=str, required=True)
    p.add_argument("--surv_val_csv", type=str, required=True)
    p.add_argument("--surv_test_csv", type=str, required=True)
    p.add_argument(
        "--prob_only",
        action="store_true",
        help="Use only prob_* columns (ignore emb_* even if present).",
    )
    p.add_argument(
        "--emb_only",
        action="store_true",
        help="Use only emb_* slide embeddings.",
    )
    p.add_argument(
        "--report_stage1_test_auroc",
        action="store_true",
        help="Print Stage 1 per-task test AUROCs from test_probs_csv (and include them in --out_metrics_json when set).",
    )
    p.add_argument(
        "--out_metrics_json",
        type=str,
        default="",
        help="Optional path to save metrics JSON (Stage 1 AUROCs included only if --report_stage1_test_auroc)",
    )
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=31)
    p.add_argument("--min_child_samples", type=int, default=10)
    p.add_argument("--subsample", type=float, default=0.85)
    p.add_argument("--colsample_bytree", type=float, default=0.85)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=50,
        help="Stop if validation metric does not improve for this many rounds. Set 0 to disable.",
    )
    args = p.parse_args()

    if args.prob_only and args.emb_only:
        raise SystemExit("Cannot use both --prob_only and --emb_only.")

    train_p = pd.read_csv(args.train_probs_csv)
    val_p = pd.read_csv(args.val_probs_csv)
    test_p = pd.read_csv(args.test_probs_csv)

    feat_cols = infer_stage2_feature_columns(
        train_p,
        prob_only=args.prob_only,
        emb_only=args.emb_only,
    )
    preview = feat_cols[:6] if len(feat_cols) > 6 else feat_cols
    suf = " ..." if len(feat_cols) > 6 else ""
    print("Stage 2 (LightGBM) feature columns (%d): %s%s" % (len(feat_cols), preview, suf))
    for c in feat_cols:
        for name, df in [("train", train_p), ("val", val_p), ("test", test_p)]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {name} probs CSV")

    train_m = _merge_probs_survival(train_p, Path(args.surv_train_csv))
    val_m = _merge_probs_survival(val_p, Path(args.surv_val_csv))
    test_m = _merge_probs_survival(test_p, Path(args.surv_test_csv))

    print("Merged rows (probs ∩ survival): train=%d val=%d test=%d" % (len(train_m), len(val_m), len(test_m)))

    X_train = train_m[feat_cols].values.astype(np.float32)
    y_train = train_m["y_surv"].values.astype(int)
    X_val = val_m[feat_cols].values.astype(np.float32)
    y_val = val_m["y_surv"].values.astype(int)
    X_test = test_m[feat_cols].values.astype(np.float32)
    y_test = test_m["y_surv"].values.astype(int)

    clf = build_lgbm_classifier(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
    )

    fit_kwargs: Dict[str, Any] = {}
    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            early_stopping(stopping_rounds=args.early_stopping_rounds),
            log_evaluation(period=0),
        ]

    clf.fit(X_train, y_train, **fit_kwargs)

    def _report(split: str, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
            "accuracy": float(accuracy_score(y, pred)),
            "precision_death": float(precision_score(y, pred, pos_label=1, zero_division=0)),
            "recall_death": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        }
        print(f"\n--- Stage 2 [{split}] (LightGBM) ---")
        print(f"  AUROC (5y death vs surv): {metrics['auroc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (death): {metrics['precision_death']:.4f}")
        print(f"  Recall (death): {metrics['recall_death']:.4f}")
        return metrics["auroc"], metrics

    val_auroc, _ = _report("validation", X_val, y_val)
    test_auroc, test_metrics = _report("test (final)", X_test, y_test)

    best_iter = getattr(clf, "best_iteration_", None)
    if best_iter is not None:
        print(f"\nLightGBM best_iteration (early stopping): {best_iter}")

    stage1_test: Dict[str, float] = {}
    if args.report_stage1_test_auroc:
        stage1_test = _stage1_test_aurocs(Path(args.test_probs_csv))
        print("\n--- Stage 1 [test] per-task AUROC (pred vs label in export CSV) ---")
        for k, v in stage1_test.items():
            print(f"  {k}: {v:.4f}")

    def _json_float(x: float) -> Optional[float]:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return float(x)

    if args.out_metrics_json:
        stage1_json = stage1_test if args.report_stage1_test_auroc else {}
        out = {
            "stage2_model": "lightgbm",
            "stage2_feature_columns": feat_cols,
            "stage2_prob_only": bool(args.prob_only),
            "stage2_emb_only": bool(args.emb_only),
            "lgbm_best_iteration": int(best_iter) if best_iter is not None else None,
            "stage2_val_auroc": _json_float(val_auroc),
            "stage2_test_auroc": _json_float(test_auroc),
            "stage2_test_metrics": {k: _json_float(v) for k, v in test_metrics.items()},
            "stage1_test_auroc_per_task": {k: _json_float(v) for k, v in stage1_json.items()},
        }
        Path(args.out_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_metrics_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote metrics to {args.out_metrics_json}")


if __name__ == "__main__":
    main()
