#!/usr/bin/env python3
"""
Copy all CSVs from dataset_csv -> dataset_csv1. For SR386_40X_HE_T{digits} in case_id/slide_id:
  - Pad T-number to 3 digits
  - Append _01 if missing (matches slide filenames e.g. ..._T001_01.czi)
  Examples: T1 -> T001_01, T10 -> T010_01, T103 -> T103_01
  Already ..._T001_01.czi unchanged.

Usage (from reproducibility/):
  python pad_sr386_case_slide_ids.py
  python pad_sr386_case_slide_ids.py --src dataset_csv_old --dst dataset_csv
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

# SR386_40X_HE_T1, SR386_40X_HE_T10_01.czi -> pad the first run of digits after SR386_40X_HE_T
_SR386_T = re.compile(r"^(SR386_40X_HE_T)(\d+)(.*)$", re.IGNORECASE)


def pad_sr386_40x_he_t(value: str) -> str:
    if not value or not isinstance(value, str):
        return value
    s = value.strip()
    m = _SR386_T.match(s)
    if not m:
        return value
    prefix = m.group(1)
    num = int(m.group(2))
    rest = m.group(3) or ""
    padded = prefix + str(num).zfill(3)
    if rest.startswith("_01"):
        return padded + rest
    if rest == "":
        return padded + "_01"
    # Rare suffix (e.g. _02); keep numeric pad only
    return padded + rest


def process_cell(col: str, value: str, cols_to_fix: set[str]) -> str:
    if col not in cols_to_fix or value is None:
        return value
    return pad_sr386_40x_he_t(value)


def convert_file(src: Path, dst: Path, cols_to_fix: set[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, newline="", encoding="utf-8", errors="replace") as f_in:
        reader = csv.reader(f_in)
        rows = list(reader)
    if not rows:
        with open(dst, "w", newline="", encoding="utf-8") as f_out:
            pass
        return
    header = rows[0]
    fix_cols = {c for c in cols_to_fix if c in header}
    if not fix_cols:
        shutil.copy2(src, dst)
        return
    col_idx = {name: i for i, name in enumerate(header)}
    out_rows = [header]
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[: len(header)]
        for c in fix_cols:
            i = col_idx[c]
            if i < len(row):
                row[i] = pad_sr386_40x_he_t(row[i])
        out_rows.append(row)
    with open(dst, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out, lineterminator="\n")
        w.writerows(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_csv",
    )
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_csv1",
    )
    args = ap.parse_args()
    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        sys.exit(f"Source not found: {src}")

    cols_to_fix = {"case_id", "slide_id"}
    csv_files = sorted(src.glob("*.csv"))
    if not csv_files:
        sys.exit(f"No CSV files in {src}")

    for path in csv_files:
        convert_file(path, dst / path.name, cols_to_fix)
        print(path.name)

    print(f"\nWrote {len(csv_files)} file(s) to {dst.resolve()}")


if __name__ == "__main__":
    main()
