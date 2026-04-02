"""SR386 Stage-1 multi-task loader: MMR_LOSS + KRAS + NRAS + BRAF from merged CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.dataset_constants import LABEL_ENCODINGS

# Default: all four biomarkers (classifier output order)
STAGE1_TASK_ORDER: List[str] = ["MMR_LOSS", "KRAS_M", "NRAS_M", "BRAF_M"]

ALLOWED_STAGE1_TASKS: frozenset = frozenset(STAGE1_TASK_ORDER)

# CSV stem per task (under dataset_csv/), e.g. SR386_msi_train.csv for MMR on train split
_TASK_TO_CSV_STEM = {
    "MMR_LOSS": "SR386_msi",
    "KRAS_M": "SR386_kras",
    "NRAS_M": "SR386_nras",
    "BRAF_M": "SR386_braf",
}


def normalize_stage1_tasks(tasks: List[str]) -> List[str]:
    """Validate and return a non-empty ordered subset of biomarker tasks."""
    if not tasks:
        raise ValueError("At least one task is required.")
    out: List[str] = []
    for t in tasks:
        t = t.strip()
        if t not in ALLOWED_STAGE1_TASKS:
            raise ValueError(f"Unknown task {t!r}. Allowed: {sorted(ALLOWED_STAGE1_TASKS)}")
        if t not in out:
            out.append(t)
    return out


def parse_stage1_tasks_arg(s: str) -> List[str]:
    """Comma-separated list, e.g. 'MMR_LOSS,KRAS_M'."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return normalize_stage1_tasks(parts)


def _encode_column(series: pd.Series, task: str) -> pd.Series:
    enc = LABEL_ENCODINGS["SR386"][task]
    s = series.copy()
    for key, val in enc.items():
        s = s.replace(key, val)
    return pd.to_numeric(s, errors="coerce")


def load_sr386_multitask_dataframe(csv_dir: Path, split: str, tasks: List[str]) -> pd.DataFrame:
    """
    Merge per-task CSVs on case_id. split is 'train', 'validate', or 'test'.
    Returns columns: case_id, slide_id, and one float column per task.
    """
    tasks = normalize_stage1_tasks(tasks)
    csv_dir = Path(csv_dir)
    frames = []
    for task in tasks:
        stem = _TASK_TO_CSV_STEM[task]
        path = csv_dir / f"{stem}_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing multi-task CSV: {path}")
        df = pd.read_csv(path)
        if "case_id" not in df.columns or "label" not in df.columns:
            raise ValueError(f"{path} must have case_id and label columns")
        sub = df[["case_id", "label"]].copy()
        sub = sub.rename(columns={"label": f"label_{task}"})
        sub[f"label_{task}"] = _encode_column(sub[f"label_{task}"], task)
        frames.append(sub[["case_id", f"label_{task}"]])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="case_id", how="inner")

    first = tasks[0]
    slide_ref = pd.read_csv(csv_dir / f"{_TASK_TO_CSV_STEM[first]}_{split}.csv")
    slide_map = slide_ref.set_index("case_id")["slide_id"]
    merged["slide_id"] = merged["case_id"].map(slide_map)

    before = len(merged)
    merged = merged.dropna(subset=[f"label_{t}" for t in tasks])
    if len(merged) < before:
        print(f"Warning: dropped {before - len(merged)} rows with NaN labels after merge.")

    return merged.reset_index(drop=True)


class Dataset_MultiTask_Biomarkers_SR386(Dataset):
    """One row per slide; __getitem__ returns (case_id, label_vector of length len(tasks))."""

    def __init__(self, csv_dir: Path, split: str, tasks: Optional[List[str]] = None, shuffle: bool = False):
        self.task_order = normalize_stage1_tasks(tasks if tasks is not None else list(STAGE1_TASK_ORDER))
        self.df = load_sr386_multitask_dataframe(Path(csv_dir), split, self.task_order)
        if shuffle:
            self.df = self.df.sample(frac=1.0, random_state=None).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        row = self.df.iloc[idx]
        case_id = str(row["case_id"])
        labels = torch.tensor(
            [float(row[f"label_{t}"]) for t in self.task_order],
            dtype=torch.float32,
        )
        return case_id, labels
