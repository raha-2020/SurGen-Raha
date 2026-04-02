# Full pipeline: Stage 1 (multi-task biomarkers) → Stage 2 (5y survival)

This document walks through training, evaluation exports, survival modelling, and final metrics for **SR386** using the scripts in this folder.

**Prerequisites**

- Python env with `requirements.txt` installed (PyTorch, pandas, scikit-learn, zarr, tqdm, etc.).
- UNI (or other) **Zarr** features: one folder per split **or** one tree with `train` / `val` / `test` subfolders — see below.
- Working directory for commands: `SurGen-Dataset-main/reproducibility/`.

---

## Paths you must set

| Variable | Meaning |
|----------|--------|
| `TRAIN_ZARR` | Directory containing `SR386_*.zarr` for **training** slides (must match `dataset_csv` train `case_id`s). |
| `VAL_ZARR` | Same for **validation** slides. |
| `TEST_ZARR` | Same for **test** slides (only for **final** eval — not used during Stage 1 training). |
| `CSV_DIR` | Usually `./dataset_csv` (contains `SR386_*` CSVs). |

If all Zarr files live in **one** directory (flat: `case_id.zarr`), use the **same** path for train/val/test exports and `--flat_fv_path`. If you mirror the paper’s split folders, point each eval at the right subfolder.

---

### Subset of biomarkers (optional)

Train only some tasks, e.g. **MMR + KRAS** (exclude NRAS and BRAF):

```bash
python biomarker_multitask_main.py ... \
  --tasks "MMR_LOSS,KRAS_M"
```

Eval exports read `stage1_tasks` from the checkpoint automatically. Stage 2 uses whatever `prob_*` columns appear in the train probs CSV.

---

## Step 1 — Train Stage 1 and keep `best_multitask_*.pth`

From `reproducibility/`:

```bash
python biomarker_multitask_main.py \
  --train_fv_path "${TRAIN_ZARR}" \
  --val_fv_path "${VAL_ZARR}" \
  --train_dataset_csv_path ./dataset_csv \
  --val_dataset_csv_path ./dataset_csv \
  --feature_extractor uni \
  --flat_fv_path \
  --epochs 50 \
  --balance_classes \
  --disable_wandb \
  --results_dir ./results_multitask_biomarkers
```

- Checkpoints are saved under:  
  `results_multitask_biomarkers/SR386/MULTITASK_BIOMARKERS/uni/<run_id>/best_multitask_epoch_*_auroc_*.pth`
- Note the **full path** to the best checkpoint for the next steps.

**Optional:** `bash run_biomarker_multitask_example.sh` after editing paths inside the script.

---

## Step 2 — Run `biomarker_multitask_eval.py` for train / validate / test

For **each** split, point `--fv_path` at the directory that actually contains the Zarrs for slides in that split.

### Option A — helper script (three runs)

```bash
export CHECKPOINT="/path/to/best_multitask_epoch_X_auroc_Y.pth"
export FV_PATH="/path/to/zarr_root"   # same root if flat layout for all splits
export CSV_DIR="./dataset_csv"
export OUT_DIR="./exports/stage1_probs"
chmod +x export_stage1_all_splits.sh
bash export_stage1_all_splits.sh
```

If **train/val/test Zarrs live in different folders**, run `biomarker_multitask_eval.py` three times manually with different `--fv_path` per split (see Option B).

### Option B — manual commands

```bash
CKPT="/path/to/best_multitask_....pth"
Z_TRAIN="/path/to/zarr_train"
Z_VAL="/path/to/zarr_val"
Z_TEST="/path/to/zarr_test"

python biomarker_multitask_eval.py --checkpoint "$CKPT" --fv_path "$Z_TRAIN" \
  --dataset_csv_path ./dataset_csv --split train --feature_extractor uni --flat_fv_path \
  --out_csv ./exports/stage1_probs/stage1_probs_train.csv

python biomarker_multitask_eval.py --checkpoint "$CKPT" --fv_path "$Z_VAL" \
  --dataset_csv_path ./dataset_csv --split validate --feature_extractor uni --flat_fv_path \
  --out_csv ./exports/stage1_probs/stage1_probs_validate.csv

python biomarker_multitask_eval.py --checkpoint "$CKPT" --fv_path "$Z_TEST" \
  --dataset_csv_path ./dataset_csv --split test --feature_extractor uni --flat_fv_path \
  --out_csv ./exports/stage1_probs/stage1_probs_test.csv
```

Outputs are CSVs with `case_id`, `prob_MMR_LOSS`, `prob_KRAS_M`, `prob_NRAS_M`, `prob_BRAF_M`, and `label_*` ground truth for Stage 1.

### Slide embeddings for Stage 2 (recommended)

Using **probabilities alone** in Stage 2 drops most **morphologic** signal (only four numbers per slide). To match the **embedding + probabilities** design, export the **pooled Transformer slide vector** (`d_model` = 512) as `emb_0` … `emb_511`:

```bash
python biomarker_multitask_eval.py ... --export_embeddings --out_csv ./exports/stage1_probs_train.csv
```

Or set `export EXPORT_EMBEDDINGS=1` before `bash export_stage1_all_splits.sh`.

**Stage 2** (`survival_stage2_main.py`) automatically uses **`emb_*` + `prob_*`** when both are present. Use **`--prob_only`** to force probabilities alone (ablation). Use **`--emb_only`** for embedding-only.

---

## Step 3 — Join with survival (automatic in Step 4)

Merging on `case_id` is done inside `survival_stage2_main.py` (inner join between Stage 1 prob CSVs and `SR386_5y_sur_*.csv`). No separate join script is required.

---

## Step 4 — Train + validate Stage 2; one final test evaluation

```bash
python survival_stage2_main.py \
  --train_probs_csv ./exports/stage1_probs/stage1_probs_train.csv \
  --val_probs_csv ./exports/stage1_probs/stage1_probs_validate.csv \
  --test_probs_csv ./exports/stage1_probs/stage1_probs_test.csv \
  --surv_train_csv ./dataset_csv/SR386_5y_sur_train.csv \
  --surv_val_csv ./dataset_csv/SR386_5y_sur_validate.csv \
  --surv_test_csv ./dataset_csv/SR386_5y_sur_test.csv \
  --report_stage1_test_auroc \
  --out_metrics_json ./exports/final_metrics.json
```


