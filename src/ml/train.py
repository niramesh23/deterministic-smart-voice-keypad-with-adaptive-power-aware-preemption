"""
src/ml/train.py
---------------
Comprehensive training pipeline for DS-CNN and Residual DS-CNN keyword spotters.

Features:
- Deterministic seeding across Python, NumPy, and PyTorch.
- Class-balanced inverse frequency loss weighting with label smoothing (0.05).
- Macro F1-score tracking per epoch to prevent class collapse.
- Dynamic learning rate scheduling via ReduceLROnPlateau or CosineAnnealing.
- SpecAugment data augmentation during training.
- Checkpoint management with separate improved filenames (never overwriting baseline).
- Export of comprehensive metrics to CSV, JSON, and YAML.
- Precomputed tensor caching for ultra-fast CPU training.
"""

import argparse
import csv
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.classes import CLASS_NAMES, NUM_CLASSES
from src.ml.dataset import SpeechCommandsDataset, precompute_dataset_features
from src.ml.manifests import build_manifests
from src.ml.model import DSCNNKeywordSpotter, ResidualDSCNNKeywordSpotter, get_model


def set_seed(seed: int = 42):
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_epoch_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int = NUM_CLASSES) -> Dict[str, float]:
    """Computes Top-1 Accuracy and Macro F1 score."""
    total = len(targets)
    correct = int(np.sum(preds == targets))
    acc = (correct / max(1, total)) * 100.0

    f1_list = []
    for c in range(num_classes):
        tp = int(np.sum((preds == c) & (targets == c)))
        fp = int(np.sum((preds == c) & (targets != c)))
        fn = int(np.sum((preds != c) & (targets == c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) * 100.0
    return {"accuracy": acc, "macro_f1": macro_f1}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Executes a single training epoch and returns (loss, accuracy, macro_f1)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to ensure numerical stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_preds.append(preds)
        all_targets.append(targets)

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)

    epoch_loss = running_loss / max(1, len(all_targets_arr))
    metrics = compute_epoch_metrics(all_preds_arr, all_targets_arr)
    return epoch_loss, metrics["accuracy"], metrics["macro_f1"]


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluates validation loss, accuracy, and macro F1."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            running_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            targets = labels.detach().cpu().numpy()

            all_preds.append(preds)
            all_targets.append(targets)

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)

    val_loss = running_loss / max(1, len(all_targets_arr))
    metrics = compute_epoch_metrics(all_preds_arr, all_targets_arr)
    return val_loss, metrics["accuracy"], metrics["macro_f1"]


def train(
    architecture: str = "ResidualDS-CNN",
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.003,
    weight_decay: float = 1e-3,
    label_smoothing: float = 0.05,
    patience: int = 5,
    manifest_dir: str = "data/manifests",
    save_dir: str = "models/checkpoints",
    model_save_filename: str = "dscnn_improved.pt",
    pretrained_checkpoint: Optional[str] = None,
    cache_dir: str = "data/processed",
    use_precomputed_cache: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main training function with complete logging and artifact generation.
    """
    set_seed(seed)
    device = torch.device("cpu")

    # Determine configuration / history naming tag
    stem = pathlib.Path(model_save_filename).stem
    if stem.startswith("dscnn_"):
        tag = stem[len("dscnn_"):]
    elif stem.startswith("resdscnn_"):
        tag = stem[len("resdscnn_"):]
    else:
        tag = stem

    print("=" * 80)
    print(f"  TRAINING PIPELINE: {architecture.upper()} ({NUM_CLASSES} CLASSES, TAG: {tag})")
    print("=" * 80)
    print(f"Architecture:      {architecture}")
    print(f"Num Classes:       {NUM_CLASSES}")
    print(f"Device:            {device} (Threads: {torch.get_num_threads()})")
    print(f"Max Epochs:        {epochs}")
    print(f"Batch Size:        {batch_size}")
    print(f"Learning Rate:     {learning_rate}")
    print(f"Weight Decay:      {weight_decay}")
    print(f"Label Smoothing:   {label_smoothing}")
    print(f"Patience:          {patience} epochs")
    print(f"Pretrained Ckpt:   {pretrained_checkpoint}")
    print(f"Seed:              {seed}")
    print("-" * 80)

    manifest_path = pathlib.Path(manifest_dir)
    train_manifest = manifest_path / "train_manifest.json"
    val_manifest = manifest_path / "val_manifest.json"
    test_manifest = manifest_path / "test_manifest.json"

    if not train_manifest.is_file() or not val_manifest.is_file():
        print("Manifests missing. Generating balanced manifests...")
        build_manifests(output_dir=manifest_dir)

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    # 1. Feature Pre-Extraction & Caching for CPU Acceleration
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_suffix = f"_{NUM_CLASSES}class" if NUM_CLASSES != 12 else ""
    train_cache_file = cache_path / f"train_features{cache_suffix}.pt"
    val_cache_file = cache_path / f"val_features{cache_suffix}.pt"
    test_cache_file = cache_path / f"test_features{cache_suffix}.pt"

    if use_precomputed_cache:
        if not train_cache_file.is_file():
            print(f"[INFO] Precomputing training features cache ({train_cache_file.name})...")
            precompute_dataset_features(str(train_manifest), str(train_cache_file))
        if not val_cache_file.is_file():
            print(f"[INFO] Precomputing validation features cache ({val_cache_file.name})...")
            precompute_dataset_features(str(val_manifest), str(val_cache_file))
        if not test_cache_file.is_file() and test_manifest.is_file():
            print(f"[INFO] Precomputing test features cache ({test_cache_file.name})...")
            precompute_dataset_features(str(test_manifest), str(test_cache_file))

    train_cached = str(train_cache_file) if train_cache_file.is_file() else None
    val_cached = str(val_cache_file) if val_cache_file.is_file() else None

    # 2. Datasets & DataLoaders
    print("Loading training and validation datasets...")
    train_dataset = SpeechCommandsDataset(
        manifest_path=str(train_manifest),
        is_training=True,
        augment=True,
        cached_tensor_file=train_cached,
    )
    val_dataset = SpeechCommandsDataset(
        manifest_path=str(val_manifest),
        is_training=False,
        augment=False,
        cached_tensor_file=val_cached,
    )

    print(f"Dataset Loaded: Train = {len(train_dataset):,} samples | Val = {len(val_dataset):,} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 3. Model, Loss, Optimizer, Scheduler
    model = get_model(architecture=architecture, num_classes=NUM_CLASSES)
    model.to(device)

    # Load backbone weights if pretrained checkpoint is provided
    if pretrained_checkpoint and pathlib.Path(pretrained_checkpoint).is_file():
        print(f"[INFO] Loading backbone weights from pretrained checkpoint: {pretrained_checkpoint}")
        ckpt_data = torch.load(pretrained_checkpoint, map_location=device)
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        if hasattr(model, "load_backbone_weights"):
            n_loaded = model.load_backbone_weights(state_dict)
            print(f"[INFO] Warm-started {n_loaded} backbone parameter tensors from {pretrained_checkpoint}")
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded pretrained state dict (missing: {len(missing)}, unexpected: {len(unexpected)})")

    total_params = model.count_parameters()
    print(f"Model:             {architecture} ({total_params:,} parameters)")

    # Compute inverse frequency class weights to combat class imbalance
    class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for r in train_records:
        class_counts[r["label_idx"]] += 1.0

    total_samples = len(train_records)
    class_weights = total_samples / (NUM_CLASSES * class_counts.clamp(min=1.0))
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print(f"Class Weights:     Min={class_weights.min():.3f}, Max={class_weights.max():.3f}, Mean={class_weights.mean():.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    )

    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_file = save_path / model_save_filename
    results_dir = pathlib.Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save training configuration
    train_config = {
        "architecture": architecture,
        "parameter_count": total_params,
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "patience": patience,
        "random_seed": seed,
        "class_counts_train": {CLASS_NAMES[i]: int(class_counts[i].item()) for i in range(NUM_CLASSES)},
        "class_weights": {CLASS_NAMES[i]: round(float(class_weights[i].item()), 4) for i in range(NUM_CLASSES)},
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=2)",
    }

    with open(save_path / f"train_config_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)

    with open(results_dir / f"training_config_{tag}.yaml", "w", encoding="utf-8") as f:
        yaml.dump(train_config, f, sort_keys=False)

    with open(save_path / f"class_mapping_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "core_commands": [c for c in CLASS_NAMES if not c.startswith("_")],
        }, f, indent=2)

    with open(save_path / f"feature_config_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_type": "log_mel",
            "sample_rate": 16000,
            "frame_size": 640,
            "hop_size": 320,
            "num_mel_bins": 40,
            "target_length": 16000,
            "shape": [1, 49, 40],
        }, f, indent=2)

    # 4. Training Loop
    print("\nStarting Training Loop...")
    t_train_start = time.perf_counter()

    best_val_macro_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        # Step scheduler based on Validation Macro F1
        scheduler.step(val_f1)

        epoch_time = time.perf_counter() - t_epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train F1: {train_f1:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | "
            f"LR: {lr_current:.6f} | Time: {epoch_time:.2f}s"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "train_macro_f1": round(train_f1, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_macro_f1": round(val_f1, 2),
            "learning_rate": lr_current,
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(record)

        # Check for best validation Macro F1 improvement
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "model_architecture": architecture,
                "parameter_count": total_params,
                "training_seed": seed,
            }, best_model_file)
            print(f"  --> Saved new best checkpoint: {best_model_file.name} (Val Macro F1: {val_f1:.2f}%, Val Acc: {val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING] Validation Macro F1 ceased improving for {patience} epochs.")
                break

    total_training_time = time.perf_counter() - t_train_start
    print("-" * 80)
    print(f"Training completed in {total_training_time:.2f}s ({total_training_time / 60:.2f} min).")
    print(f"Best Epoch:               {best_epoch}")
    print(f"Best Val Macro F1:        {best_val_macro_f1:.2f}%")
    print(f"Best Val Accuracy:        {best_val_acc:.2f}%")
    print(f"Saved Checkpoint:         {best_model_file}")

    # 5. Export History Files (JSON & CSV)
    history_json_file = save_path / f"training_history_{tag}.json"
    with open(history_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "best_val_acc": best_val_acc,
            "total_training_time_s": total_training_time,
            "parameter_count": total_params,
            "model_architecture": architecture,
            "random_seed": seed,
            "saved_checkpoint": str(best_model_file),
        }, f, indent=2)

    history_csv_file = results_dir / f"training_history_{tag}.csv"
    with open(history_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc", "train_macro_f1",
            "val_loss", "val_acc", "val_macro_f1", "learning_rate", "epoch_time_s"
        ])
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved history CSV:        {history_csv_file}")
    print(f"Saved history JSON:       {history_json_file}")

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_acc": best_val_acc,
        "best_model_path": str(best_model_file),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Keyword Spotting Model.")
    parser.add_argument("--arch", type=str, default="ResidualDS-CNN", help="Architecture: ResidualDS-CNN or DS-CNN")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch limit")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--save-name", type=str, default="dscnn_improved.pt", help="Checkpoint filename")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="Pretrained checkpoint for backbone warm-start")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train(
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        model_save_filename=args.save_name,
        pretrained_checkpoint=args.pretrained_checkpoint,
        seed=args.seed,
    )
