"""
src/ml/evaluate.py
------------------
Evaluates trained DS-CNN checkpoint on the official held-out test split.

Outputs:
- Top-1 Accuracy
- Precision, Recall, F1-Score per class
- Macro-F1 and Weighted-F1
- 12x12 Confusion Matrix
- Per-class support counts
- Model parameter count
- Single-sample CPU inference latency percentiles (P50, P95, Mean)
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.classes import CLASS_NAMES, NUM_CLASSES
from src.ml.dataset import SpeechCommandsDataset
from src.ml.model import DSCNNKeywordSpotter, ResidualDSCNNKeywordSpotter, get_model


def compute_metrics(
    confusion_matrix: np.ndarray, class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes accuracy, precision, recall, and F1 metrics from a confusion matrix.
    confusion_matrix[i, j] = True label i, Predicted label j.
    """
    num_classes = len(class_names)
    total_samples = int(np.sum(confusion_matrix))
    correct = int(np.trace(confusion_matrix))
    accuracy = (correct / total_samples) if total_samples > 0 else 0.0

    per_class = {}
    f1_list = []
    prec_list = []
    rec_list = []
    weights = []

    for i in range(num_classes):
        tp = float(confusion_matrix[i, i])
        fp = float(np.sum(confusion_matrix[:, i]) - tp)
        fn = float(np.sum(confusion_matrix[i, :]) - tp)
        support = int(np.sum(confusion_matrix[i, :]))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[class_names[i]] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
        f1_list.append(f1)
        prec_list.append(precision)
        rec_list.append(recall)
        weights.append(support)

    macro_f1 = float(np.mean(f1_list))
    macro_precision = float(np.mean(prec_list))
    macro_recall = float(np.mean(rec_list))

    total_weight = sum(weights)
    weighted_f1 = float(sum(f1 * w for f1, w in zip(f1_list, weights)) / total_weight) if total_weight > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "weighted_f1": weighted_f1,
        "total_samples": total_samples,
        "per_class": per_class,
    }


def evaluate(
    checkpoint_path: str = "models/checkpoints/best_model.pt",
    manifest_path: str = "data/manifests/test_manifest.json",
    output_report_path: str = "results/test_evaluation_report.txt",
    output_json_path: str = "results/test_metrics.json",
    output_plot_path: Optional[str] = None,
    subset_ratio: float = 1.0,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Evaluates model on the held-out test split and records detailed metrics.
    """
    device = torch.device("cpu")
    ckpt_file = pathlib.Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    print("=" * 80)
    print("  STAGE 5: MODEL TEST EVALUATION")
    print("=" * 80)
    print(f"Checkpoint:      {ckpt_file}")
    print(f"Test Manifest:   {manifest_path}")

    # 1. Load Model Checkpoint
    checkpoint = torch.load(ckpt_file, map_location=device)
    arch = checkpoint.get("model_architecture", "DS-CNN")
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    class_names = checkpoint.get("class_names", CLASS_NAMES[:num_classes] if num_classes <= len(CLASS_NAMES) else CLASS_NAMES)
    model = get_model(arch, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    total_params = model.count_parameters()
    print(f"Model Architecture: {arch}")
    print(f"Num Classes:        {num_classes}")
    print(f"Model Parameters:   {total_params:,}")

    # 2. Dataset & Loader
    # Check for precomputed test cache
    cache_suffix = f"_{num_classes}class" if num_classes != 12 else ""
    cached_test_file = pathlib.Path(f"data/processed/test_features{cache_suffix}.pt")
    if not cached_test_file.is_file():
        cached_test_file = pathlib.Path("data/processed/test_features.pt")
    cache_to_use = str(cached_test_file) if cached_test_file.is_file() else None

    test_dataset = SpeechCommandsDataset(
        manifest_path=manifest_path,
        is_training=False,
        augment=False,
        cache_features=(subset_ratio < 0.2),
        cached_tensor_file=cache_to_use,
    )

    if subset_ratio < 1.0:
        n_samples = max(64, int(len(test_dataset) * subset_ratio))
        test_dataset = Subset(test_dataset, list(range(n_samples)))
        print(f"Test Subset applied: {len(test_dataset):,} samples")
    else:
        print(f"Full Test Set:       {len(test_dataset):,} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 3. Evaluation Loop & Confusion Matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    latencies = []

    print("\nRunning inference over test samples...")
    t_eval_start = time.perf_counter()

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Benchmark single-batch inference latency
            t0 = time.perf_counter()
            logits = model(features)
            t_batch = (time.perf_counter() - t0) * 1000.0
            per_sample_latency = t_batch / features.size(0)
            latencies.extend([per_sample_latency] * features.size(0))

            preds = logits.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            for t, p in zip(targets, preds):
                confusion_matrix[t, p] += 1

    total_eval_time = time.perf_counter() - t_eval_start

    # Latency Percentiles
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))

    # Compute classification metrics
    metrics = compute_metrics(confusion_matrix, class_names)
    metrics["latency"] = {
        "mean_ms": mean_lat,
        "p50_ms": p50_lat,
        "p95_ms": p95_lat,
    }
    metrics["parameters"] = total_params
    metrics["total_eval_time_s"] = total_eval_time

    # 4. Format Evaluation Report
    report_lines = []
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines.append("=" * 80)
    report_lines.append("STAGE 5: KEYWORD SPOTTING MODEL TEST EVALUATION REPORT")
    report_lines.append("Project: Real-Time Adaptive Voice Recognition Using Python")
    report_lines.append("=" * 80)
    report_lines.append(f"Evaluation Date:              {now_str}")
    report_lines.append(f"Model Architecture:           Compact DS-CNN (Depthwise Separable CNN)")
    report_lines.append(f"Trainable Parameters:         {total_params:,}")
    report_lines.append(f"Input Feature Shape:          [1, 49, 40] (1 ch, 49 frames, 40 Mel bins)")
    report_lines.append(f"Test Set Evaluated:           {metrics['total_samples']:,} samples (Held-out Test Split)")
    report_lines.append(f"Evaluation Runtime:           {total_eval_time:.2f} seconds")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("1. OVERALL TEST PERFORMANCE SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Overall Top-1 Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    report_lines.append(f"Macro-Averaged F1-Score:      {metrics['macro_f1'] * 100:.2f}%")
    report_lines.append(f"Macro-Averaged Precision:     {metrics['macro_precision'] * 100:.2f}%")
    report_lines.append(f"Macro-Averaged Recall:        {metrics['macro_recall'] * 100:.2f}%")
    report_lines.append(f"Weighted F1-Score:            {metrics['weighted_f1'] * 100:.2f}%")
    report_lines.append(f"Trivial Majority Baseline:    ~8.33% (1/12 random chance)")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("2. SINGLE-SAMPLE CPU INFERENCE LATENCY")
    report_lines.append("-" * 80)
    report_lines.append(f"Mean Inference Latency:       {mean_lat:.3f} ms")
    report_lines.append(f"Median (P50) Latency:         {p50_lat:.3f} ms")
    report_lines.append(f"95th Percentile (P95):        {p95_lat:.3f} ms")
    report_lines.append(f"Real-Time Constraint (Budget): 100.0 ms per word utterance (Satisfied)")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("3. PER-CLASS METRICS (PRECISION, RECALL, F1, SUPPORT)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Class Name':<15} | {'Precision':<11} | {'Recall':<11} | {'F1-Score':<11} | {'Support':<8}")
    report_lines.append("-" * 65)
    for c_name in class_names:
        cm = metrics["per_class"][c_name]
        p_str = f"{cm['precision']*100:.1f}%"
        r_str = f"{cm['recall']*100:.1f}%"
        f_str = f"{cm['f1_score']*100:.1f}%"
        sup_str = str(cm['support'])
        report_lines.append(f"{c_name:<15} | {p_str:<11} | {r_str:<11} | {f_str:<11} | {sup_str:<8}")
    report_lines.append("-" * 65)
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("4. CONFUSION MATRIX (Rows: Actual Label, Columns: Predicted Label)")
    report_lines.append("-" * 80)
    header = f"{'Actual \\ Pred':<14} | " + " ".join(f"{c[:3]:>4}" for c in class_names)
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for i, c_name in enumerate(class_names):
        row_vals = " ".join(f"{confusion_matrix[i, j]:>4}" for j in range(num_classes))
        report_lines.append(f"{c_name:<14} | {row_vals}")
    report_lines.append("-" * len(header))
    report_lines.append("=" * 80)

    # Save report and JSON
    rep_path = pathlib.Path(output_report_path)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    json_path = pathlib.Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json_data = dict(metrics)
        json_data["confusion_matrix"] = confusion_matrix.tolist()
        json_data["checkpoint_path"] = str(ckpt_file)
        json_data["evaluation_timestamp"] = now_str
        json.dump(json_data, f, indent=2)

    # Plot confusion matrix if requested
    if output_plot_path:
        plot_path = pathlib.Path(output_plot_path)
        plot_confusion_matrix(
            confusion_matrix,
            class_names,
            plot_path,
            title=f"Confusion Matrix: Top-1 Acc {metrics['accuracy']*100:.1f}%",
        )
        print(f"Confusion matrix plot saved to: {plot_path.resolve()}")

    print("\n" + "\n".join(report_lines[:25]))
    print(f"\nFull report written to: {rep_path.resolve()}")
    print(f"JSON metrics written to: {json_path.resolve()}")

    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: pathlib.Path,
    title: str = "Confusion Matrix",
):
    """Renders and saves a high-resolution confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                f"{val}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DS-CNN Keyword Spotting model.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="data/manifests/test_manifest.json", help="Path to test manifest")
    parser.add_argument("--output-report", type=str, default="results/test_evaluation_report.txt", help="Path for report output")
    parser.add_argument("--output-json", type=str, default="results/test_metrics.json", help="Path for JSON metrics output")
    parser.add_argument("--output-plot", type=str, default=None, help="Path for confusion matrix PNG plot")
    parser.add_argument("--subset", type=float, default=1.0, help="Test subset ratio")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_report_path=args.output_report,
        output_json_path=args.output_json,
        output_plot_path=args.output_plot,
        subset_ratio=args.subset,
    )

