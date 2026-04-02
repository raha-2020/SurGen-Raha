"""
Stage 2: 5-year survival from Stage-1 features (SR386).

Default: **pooled slide embedding (emb_*) + predicted biomarker probabilities (prob_*)** when
`biomarker_multitask_eval.py` was run with `--export_embeddings`. If only prob_* columns exist,
Stage 2 uses probabilities alone (morphologic signal is dropped unless embeddings are exported).

Joins stage1 CSVs with SR386_5y_sur_*.csv on case_id; trains sklearn logistic regression on train.

Usage (after Stage 1 eval CSVs exist):
  python survival_stage2_main.py \\
    --train_probs_csv exports/stage1_probs_train.csv \\
    --val_probs_csv exports/stage1_probs_validate.csv \\
    --test_probs_csv exports/stage1_probs_test.csv \\
    --surv_train_csv dataset_csv/SR386_5y_sur_train.csv \\
    --surv_val_csv dataset_csv/SR386_5y_sur_validate.csv \\
    --surv_test_csv dataset_csv/SR386_5y_sur_test.csv \\
    --report_stage1_test_auroc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datasets.biomarker_multitask_loader import STAGE1_TASK_ORDER


def _infer_prob_columns(train_df: pd.DataFrame) -> List[str]:
    """Use prob_* columns in canonical order (only tasks present in the CSV)."""
    preferred = [f"prob_{t}" for t in STAGE1_TASK_ORDER]
    return [c for c in preferred if c in train_df.columns]


def _infer_emb_columns(train_df: pd.DataFrame) -> List[str]:
    """emb_0, emb_1, ... in numeric order."""
    cols = [c for c in train_df.columns if c.startswith("emb_")]
    return sorted(cols, key=lambda c: int(c.split("_", 1)[1]))


def infer_stage2_feature_columns(
    train_df: pd.DataFrame,
    prob_only: bool,
    emb_only: bool,
) -> List[str]:
    """
    Build feature column list: [emb_*] + [prob_*] by default when embeddings exist.
    Otherwise prob_* only (requires at least one prob column).
    """
    emb_cols = _infer_emb_columns(train_df)
    prob_cols = _infer_prob_columns(train_df)

    if emb_only:
        if not emb_cols:
            raise ValueError("No emb_* columns in train CSV; export Stage 1 with --export_embeddings.")
        return emb_cols
    if prob_only:
        if not prob_cols:
            raise ValueError("No prob_* columns in train CSV.")
        return prob_cols
    if emb_cols and prob_cols:
        return emb_cols + prob_cols
    if emb_cols:
        return emb_cols
    if prob_cols:
        return prob_cols
    raise ValueError("Train CSV must contain prob_* and/or emb_* columns.")


def _merge_probs_survival(probs_df: pd.DataFrame, surv_path: Path) -> pd.DataFrame:
    surv = pd.read_csv(surv_path)
    if "case_id" not in surv.columns or "label" not in surv.columns:
        raise ValueError(f"{surv_path} must have case_id and label")
    surv = surv[["case_id", "label"]].copy()
    surv = surv.rename(columns={"label": "y_surv"})
    surv["y_surv"] = pd.to_numeric(surv["y_surv"], errors="coerce")
    merged = probs_df.merge(surv, on="case_id", how="inner")
    merged = merged.dropna(subset=["y_surv"])
    return merged


def _stage1_test_aurocs(test_probs_csv: Path) -> Dict[str, float]:
    """Per-task AUROC on test using prob_* vs label_* from eval export."""
    df = pd.read_csv(test_probs_csv)
    out: Dict[str, float] = {}
    for col in df.columns:
        if not col.startswith("prob_"):
            continue
        task = col[len("prob_") :]
        lc = f"label_{task}"
        if lc not in df.columns:
            continue
        y = df[lc].values
        s = df[col].values
        if len(np.unique(y)) < 2:
            out[task] = float("nan")
        else:
            out[task] = float(roc_auc_score(y, s))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2 survival (5Y_SUR) from Stage-1 probs")
    p.add_argument("--train_probs_csv", type=str, required=True)
    p.add_argument("--val_probs_csv", type=str, required=True)
    p.add_argument("--test_probs_csv", type=str, required=True)
    p.add_argument("--surv_train_csv", type=str, required=True)
    p.add_argument("--surv_val_csv", type=str, required=True)
    p.add_argument("--surv_test_csv", type=str, required=True)
    p.add_argument(
        "--prob_only",
        action="store_true",
        help="Use only prob_* columns (ignore emb_* even if present). Ablation for prob-only Stage 2.",
    )
    p.add_argument(
        "--emb_only",
        action="store_true",
        help="Use only emb_* slide embeddings (requires Stage 1 export with --export_embeddings).",
    )
    p.add_argument(
        "--report_stage1_test_auroc",
        action="store_true",
        help="Print Stage 1 per-task test AUROCs from test_probs_csv",
    )
    p.add_argument(
        "--out_metrics_json",
        type=str,
        default="",
        help="Optional path to save metrics JSON (val/test AUROC, Stage 1 test AUROCs)",
    )
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()

    if args.prob_only and args.emb_only:
        raise SystemExit("Cannot use both --prob_only and --emb_only.")

    train_p = pd.read_csv(args.train_probs_csv)
    val_p = pd.read_csv(args.val_probs_csv)
    test_p = pd.read_csv(args.test_probs_csv)

    feat_cols = infer_stage2_feature_columns(
        train_p, prob_only=args.prob_only, emb_only=args.emb_only
    )
    preview = feat_cols[:6] if len(feat_cols) > 6 else feat_cols
    suf = " ..." if len(feat_cols) > 6 else ""
    print("Stage 2 feature columns (%d): %s%s" % (len(feat_cols), preview, suf))
    for c in feat_cols:
        for name, df in [("train", train_p), ("val", val_p), ("test", test_p)]:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {name} probs CSV")

    train_m = _merge_probs_survival(train_p, Path(args.surv_train_csv))
    val_m = _merge_probs_survival(val_p, Path(args.surv_val_csv))
    test_m = _merge_probs_survival(test_p, Path(args.surv_test_csv))

    print("Merged rows (probs ∩ survival): train=%d val=%d test=%d" % (len(train_m), len(val_m), len(test_m)))

    X_train = train_m[feat_cols].values
    y_train = train_m["y_surv"].values.astype(int)
    X_val = val_m[feat_cols].values
    y_val = val_m["y_surv"].values.astype(int)
    X_test = test_m[feat_cols].values
    y_test = test_m["y_surv"].values.astype(int)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=args.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    def _report(split: str, X: np.ndarray, y: np.ndarray) -> Tuple[float, Dict[str, float]]:
        proba = clf.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "auroc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
            "accuracy": float(accuracy_score(y, pred)),
            "precision_death": float(precision_score(y, pred, pos_label=1, zero_division=0)),
            "recall_death": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        }
        print(f"\n--- Stage 2 [{split}] ---")
        print(f"  AUROC (5y death vs surv): {metrics['auroc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (death): {metrics['precision_death']:.4f}")
        print(f"  Recall (death): {metrics['recall_death']:.4f}")
        return metrics["auroc"], metrics

    val_auroc, _ = _report("validation", X_val, y_val)
    test_auroc, test_metrics = _report("test (final)", X_test, y_test)

    stage1_test: Dict[str, float] = {}
    if args.report_stage1_test_auroc:
        stage1_test = _stage1_test_aurocs(Path(args.test_probs_csv))
        print("\n--- Stage 1 [test] per-task AUROC ---")
        for k, v in stage1_test.items():
            print(f"  {k}: {v:.4f}")

    def _json_float(x: float) -> Optional[float]:
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return float(x)

    if args.out_metrics_json:
        out = {
            "stage2_feature_columns": feat_cols,
            "stage2_prob_only": bool(args.prob_only),
            "stage2_emb_only": bool(args.emb_only),
            "stage2_val_auroc": _json_float(val_auroc),
            "stage2_test_auroc": _json_float(test_auroc),
            "stage2_test_metrics": {k: _json_float(v) for k, v in test_metrics.items()},
            "stage1_test_auroc_per_task": {k: _json_float(v) for k, v in stage1_test.items()},
        }
        Path(args.out_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_metrics_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote metrics to {args.out_metrics_json}")


if __name__ == "__main__":
    main()
