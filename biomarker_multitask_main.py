"""
Stage 1: train one shared Transformer encoder on SR386 with four biomarker heads
(MMR_LOSS, KRAS_M, NRAS_M, BRAF_M). Patch features from UNI Zarr; labels from merged CSVs.

Run from reproducibility/:
  python biomarker_multitask_main.py --train_fv_path /path/to/zarr --val_fv_path /path/to/zarr \\
    --train_dataset_csv_path ./dataset_csv --val_dataset_csv_path ./dataset_csv \\
    --feature_extractor uni --flat_fv_path --epochs 50
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb
import zarr
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets.biomarker_multitask_loader import (
    Dataset_MultiTask_Biomarkers_SR386,
    parse_stage1_tasks_arg,
)
from datasets.dataset_loader import SlideDataset


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_gpu_memory_stats(tag: str = "") -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        logging.debug(f"{tag} - Allocated: {allocated:.4f} GB, Cached: {cached:.4f} GB")


class TransformerMultiTaskBiomarkers(nn.Module):
    """Shared Transformer encoder + 4 binary logits (BCE)."""

    def __init__(
        self,
        fv_extractor_size: int,
        num_tasks: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.fc = nn.Linear(fv_extractor_size, d_model)
        self.relu = nn.ReLU()
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps=layer_norm_eps,
        )
        self.classifier = nn.Linear(d_model, num_tasks)
        self.d_model = d_model

    def forward_logits_and_embedding(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns (logits per task, pooled slide embedding of shape (d_model,))."""
        src = self.relu(self.fc(src))
        memory = self.transformer.encoder(src)
        pooled = memory.mean(dim=0)  # (1, d_model)
        logits = self.classifier(pooled).view(-1)
        emb = pooled.view(-1)
        return logits, emb

    def forward(self, src: Tensor) -> Tensor:
        logits, _ = self.forward_logits_and_embedding(src)
        return logits


def delete_previous_best_models(directory: Path, metric_name: str) -> None:
    pattern = str(directory / f"best_{metric_name}_epoch_*")
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
            print(f"Deleted previous best model: {filepath}")
        except OSError as e:
            print(f"Error deleting file {filepath}: {e}")


def train_step_multitask(
    model: nn.Module,
    criterion: nn.Module,
) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
    def step(features: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = model(features)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        return loss, labels, probs, preds

    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="SR386 multi-task biomarker training (Stage 1).")
    parser.add_argument("--train_fv_path", type=str, required=True, help="Directory of training .zarr folders")
    parser.add_argument("--val_fv_path", type=str, required=True, help="Directory of val .zarr folders")
    parser.add_argument(
        "--train_dataset_csv_path",
        type=str,
        required=True,
        help="Folder containing SR386_*_{train,validate,test}.csv (e.g. dataset_csv/)",
    )
    parser.add_argument("--val_dataset_csv_path", type=str, required=True, help="Same structure for val split CSVs")
    parser.add_argument("--results_dir", type=str, default="./results_multitask_biomarkers")
    parser.add_argument("--log_dir", type=str, default="./log_dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu"])
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Use per-task pos_weight in BCE (neg/pos per task on train)",
    )
    parser.add_argument("--feature_extractor", type=str, default="uni", choices=["ctranspath", "owkin", "resnet50", "resnet50-b", "uni"])
    parser.add_argument("--flat_fv_path", action="store_true", help="Zarr files live directly under train/val paths")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--tasks",
        type=str,
        default="MMR_LOSS,KRAS_M,NRAS_M,BRAF_M",
        help="Comma-separated biomarkers to train (subset of MMR_LOSS,KRAS_M,NRAS_M,BRAF_M). Example: MMR_LOSS,KRAS_M",
    )

    args = parser.parse_args()
    stage1_tasks = parse_stage1_tasks_arg(args.tasks)
    setup_logging(args.verbose)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_fv_path = Path(args.train_fv_path)
    val_fv_path = Path(args.val_fv_path)
    train_csv_dir = Path(args.train_dataset_csv_path)
    val_csv_dir = Path(args.val_dataset_csv_path)
    if not getattr(args, "flat_fv_path", False):
        train_fv_path = train_fv_path / args.feature_extractor
        val_fv_path = val_fv_path / args.feature_extractor

    results_dir = Path(args.results_dir).expanduser()
    log_dir = Path(args.log_dir)
    for p in (results_dir, log_dir):
        p.mkdir(parents=True, exist_ok=True)

    if args.feature_extractor in ("ctranspath", "owkin"):
        ft_extractor_size = 768
    elif args.feature_extractor in ("resnet50-b", "uni"):
        ft_extractor_size = 1024
    elif args.feature_extractor == "resnet50":
        ft_extractor_size = 2048
    else:
        raise ValueError(args.feature_extractor)

    num_tasks = len(stage1_tasks)
    model = TransformerMultiTaskBiomarkers(
        fv_extractor_size=ft_extractor_size,
        num_tasks=num_tasks,
        d_model=512,
        nhead=args.heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=args.dropout,
        activation=args.activation,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = Dataset_MultiTask_Biomarkers_SR386(train_csv_dir, "train", tasks=stage1_tasks, shuffle=False)
    val_ds = Dataset_MultiTask_Biomarkers_SR386(val_csv_dir, "validate", tasks=stage1_tasks, shuffle=False)

    print("Stage-1 tasks:", stage1_tasks)
    print(f"Train slides: {len(train_ds)}, Val slides: {len(val_ds)}")

    # pos_weight per task: n_neg / n_pos for each label column
    pos_weights: List[float] = []
    for i, task in enumerate(stage1_tasks):
        col = f"label_{task}"
        positives = train_ds.df[col].sum()
        negatives = len(train_ds.df) - positives
        if positives > 0:
            pos_weights.append(float(negatives / positives))
        else:
            pos_weights.append(1.0)
        print(f"  {task}: pos={int(positives)}, neg={int(negatives)}, pos_weight={pos_weights[-1]:.4f}")

    pos_weight_tensor = torch.tensor(pos_weights, device=device) if args.balance_classes else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    process_batch = train_step_multitask(model, criterion)
    use_amp = args.use_amp
    scaler = GradScaler() if use_amp else None

    if not args.disable_wandb:
        wandb.init(
            project="vit-path",
            tags=["multitask_biomarkers", "SR386", args.feature_extractor],
            config=vars(args),
        )
        run_id = wandb.run.id
    else:
        run_id = "local"

    run_dir = results_dir / "SR386" / "MULTITASK_BIOMARKERS" / args.feature_extractor / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    def _max_patches(root: Path, ds: Dataset_MultiTask_Biomarkers_SR386) -> int:
        sizes = []
        for cid, _ in ds:
            p = root / f"{cid}.zarr"
            if p.exists():
                sizes.append(zarr.open(str(p), mode="r")["features"].shape[0])
        if not sizes:
            raise FileNotFoundError(f"No .zarr slides found under {root} for this CSV.")
        return max(sizes)

    # Dummy init
    largest_num_patches = max(_max_patches(train_fv_path, train_ds), _max_patches(val_fv_path, val_ds))
    print(f"Max patches per slide: {largest_num_patches}")

    dummy_features = torch.randn(largest_num_patches, ft_extractor_size, dtype=torch.float32, device=device)
    dummy_labels = torch.zeros(num_tasks, dtype=torch.float32, device=device)
    model.train()
    with autocast(enabled=use_amp):
        loss, _, _, _ = process_batch(dummy_features, dummy_labels)
    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
    else:
        loss.backward()
        optimiser.step()
    optimiser.zero_grad()
    del dummy_features, dummy_labels, loss
    torch.cuda.empty_cache()

    num_epochs = args.epochs
    best_val_mean_auroc = 0.0

    def _collect_aurocs(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """y_true, y_score: (n, num_tasks). Returns mean AUROC and per-task dict."""
        per: Dict[str, float] = {}
        scores: List[float] = []
        for t, name in enumerate(stage1_tasks):
            yt = y_true[:, t]
            ys = y_score[:, t]
            if len(np.unique(yt)) < 2:
                per[name] = float("nan")
            else:
                a = float(roc_auc_score(yt, ys))
                per[name] = a
                scores.append(a)
        mean_auroc = float(np.nanmean(scores)) if scores else float("nan")
        return mean_auroc, per

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_probs_list: List[np.ndarray] = []
        train_labels_list: List[np.ndarray] = []

        with tqdm(total=len(train_ds), desc=f"Epoch {epoch+1}/{num_epochs} train", unit="WSI") as pbar:
            for slide_id, label_vec in train_ds:
                slide_path = train_fv_path / f"{slide_id}.zarr"
                if not slide_path.exists():
                    print(f"Missing zarr: {slide_path}")
                    pbar.update(1)
                    continue
                slide_dataset = SlideDataset(slide=slide_path, cohort="SR386", task="MMR_LOSS", batch_size=args.batch_size)
                features, _ = next(iter(slide_dataset))
                features = features.to(device)
                labels = label_vec.to(device)

                optimiser.zero_grad()
                with autocast(enabled=use_amp):
                    loss, _, probs, preds = process_batch(features, labels)

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    optimiser.step()

                total_loss += loss.item()
                train_probs_list.append(probs.detach().float().cpu().numpy())
                train_labels_list.append(labels.detach().float().cpu().numpy())
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        if train_labels_list:
            train_y = np.stack(train_labels_list, axis=0)
            train_p = np.stack(train_probs_list, axis=0)
            train_mean_auroc, _ = _collect_aurocs(train_y, train_p)
        else:
            train_mean_auroc = float("nan")

        model.eval()
        total_val_loss = 0.0
        val_probs_list: List[np.ndarray] = []
        val_labels_list: List[np.ndarray] = []

        with torch.no_grad(), tqdm(total=len(val_ds), desc=f"Epoch {epoch+1} val", unit="WSI") as pbar:
            for slide_id, label_vec in val_ds:
                slide_path = val_fv_path / f"{slide_id}.zarr"
                if not slide_path.exists():
                    pbar.update(1)
                    continue
                slide_dataset = SlideDataset(slide=slide_path, cohort="SR386", task="MMR_LOSS", batch_size=args.batch_size)
                features, _ = next(iter(slide_dataset))
                features = features.to(device)
                labels = label_vec.to(device)
                with autocast(enabled=use_amp):
                    loss, _, probs, preds = process_batch(features, labels)
                total_val_loss += loss.item()
                val_probs_list.append(probs.detach().float().cpu().numpy())
                val_labels_list.append(labels.detach().float().cpu().numpy())
                pbar.update(1)

        if val_labels_list:
            val_y = np.stack(val_labels_list, axis=0)
            val_p = np.stack(val_probs_list, axis=0)
            val_mean_auroc, per_val = _collect_aurocs(val_y, val_p)
        else:
            val_mean_auroc, per_val = float("nan"), {t: float("nan") for t in stage1_tasks}

        avg_train_loss = total_loss / max(len(train_ds), 1)
        avg_val_loss = total_val_loss / max(len(val_ds), 1)

        print(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"val_mean_auroc={val_mean_auroc:.4f} | per-task: {per_val}"
        )

        if not args.disable_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mean_auroc": train_mean_auroc,
                "val_mean_auroc": val_mean_auroc,
            }
            for name, v in per_val.items():
                log_dict[f"val_auroc_{name}"] = v
            wandb.log(log_dict)

        if not np.isnan(val_mean_auroc) and val_mean_auroc >= best_val_mean_auroc:
            best_val_mean_auroc = val_mean_auroc
            path = run_dir / f"best_multitask_epoch_{epoch+1}_auroc_{val_mean_auroc:.4f}.pth"
            delete_previous_best_models(run_dir, "multitask")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "stage1_tasks": stage1_tasks,
                    "ft_extractor_size": ft_extractor_size,
                    "feature_extractor": args.feature_extractor,
                },
                path,
            )
            print(f"Saved {path}")

        model.train()


if __name__ == "__main__":
    main()
