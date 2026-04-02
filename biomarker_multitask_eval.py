"""Export Stage-1 multi-task biomarker probabilities for a split (e.g. for Stage-2 survival)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.cuda.amp import autocast

from biomarker_multitask_main import TransformerMultiTaskBiomarkers
from datasets.biomarker_multitask_loader import STAGE1_TASK_ORDER, Dataset_MultiTask_Biomarkers_SR386, parse_stage1_tasks_arg
from datasets.dataset_loader import SlideDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="best_multitask_*.pth from biomarker_multitask_main.py")
    parser.add_argument("--fv_path", type=str, required=True, help="Folder containing case_id.zarr")
    parser.add_argument("--dataset_csv_path", type=str, required=True, help="dataset_csv/ folder")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validate", "test"])
    parser.add_argument("--feature_extractor", type=str, default="uni", choices=["ctranspath", "owkin", "resnet50", "resnet50-b", "uni"])
    parser.add_argument("--flat_fv_path", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated tasks; default is read from checkpoint 'stage1_tasks'.",
    )
    parser.add_argument(
        "--export_embeddings",
        action="store_true",
        help="Add emb_0..emb_{D-1} pooled Transformer slide features (d_model=512) for Stage 2 embedding+prob models.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ft = ckpt["ft_extractor_size"]
    fe = ckpt.get("feature_extractor", args.feature_extractor)
    if args.tasks.strip():
        stage1_tasks = parse_stage1_tasks_arg(args.tasks)
    else:
        stage1_tasks = list(ckpt.get("stage1_tasks", STAGE1_TASK_ORDER))

    fv_path = Path(args.fv_path)
    if not args.flat_fv_path:
        fv_path = fv_path / fe

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMultiTaskBiomarkers(fv_extractor_size=ft, num_tasks=len(stage1_tasks))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    ds = Dataset_MultiTask_Biomarkers_SR386(Path(args.dataset_csv_path), args.split, tasks=stage1_tasks, shuffle=False)

    rows = []
    with torch.no_grad():
        for slide_id, label_vec in ds:
            z = fv_path / f"{slide_id}.zarr"
            if not z.exists():
                continue
            slide_dataset = SlideDataset(slide=z, cohort="SR386", task="MMR_LOSS", batch_size=args.batch_size)
            features, _ = next(iter(slide_dataset))
            features = features.to(device)
            row = {"case_id": slide_id}
            with autocast(enabled=args.use_amp):
                if args.export_embeddings:
                    logits, emb = model.forward_logits_and_embedding(features)
                    emb_np = emb.detach().float().cpu().numpy()
                    for j, v in enumerate(emb_np):
                        row[f"emb_{j}"] = float(v)
                else:
                    logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i, name in enumerate(stage1_tasks):
                row[f"prob_{name}"] = float(probs[i])
                row[f"label_{name}"] = float(label_vec[i].item())
            rows.append(row)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
