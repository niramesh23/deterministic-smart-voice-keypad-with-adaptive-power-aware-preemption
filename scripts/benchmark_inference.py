"""
scripts/benchmark_inference.py
------------------------------
Rigorous, empirical CPU inference and end-to-end pipeline latency benchmarking
for real-time voice recognition on the Windows host machine.

Benchmarks:
1. Model-only unbatched inference (Batch Size = 1) across 1,000 iterations.
   - Statistics: Mean, Median (P50), P95, P99, Min, Max, Standard Deviation.
   - Evaluation against RTOS 20 ms frame hop and 100 ms deadline.
2. End-to-end pipeline latency (1.000s audio -> Log-Mel extraction -> NN forward -> Softmax -> Argmax).
   - Breakdown: DSP Feature Extraction vs. Model Inference vs. Postprocessing.
3. Batched inference throughput (Batch Size = 32, 64).
4. Memory and parameter footprint.
5. Saves results to `results/inference_benchmark_improved.json`.
"""

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List
import numpy as np
import torch

# Ensure project root in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dsp.features import LogMelSpectrogramExtractor
from src.ml.classes import CLASS_NAMES, NUM_CLASSES
from src.ml.model import get_model


def benchmark_cpu_latency(
    checkpoint_path: str = "models/checkpoints/dscnn_improved.pt",
    output_file: str = "results/inference_benchmark_improved.json",
    num_iterations: int = 1000,
    warmup_runs: int = 50,
) -> Dict[str, Any]:
    ckpt_file = pathlib.Path(checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file}")

    device = torch.device("cpu")
    print("=" * 80)
    print("  CPU INFERENCE LATENCY & EFFICIENCY BENCHMARK")
    print("=" * 80)
    print(f"Checkpoint:       {ckpt_file}")
    print(f"Host Device:      CPU (Threads: {torch.get_num_threads()})")
    print(f"Benchmark Runs:   {num_iterations} iterations ({warmup_runs} warmup)")
    print("-" * 80)

    # 1. Load Model
    checkpoint = torch.load(ckpt_file, map_location=device)
    arch = checkpoint.get("model_architecture", "ResidualDS-CNN")
    model = get_model(arch, num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    total_params = model.count_parameters()
    model_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_file_size_kb = ckpt_file.stat().st_size / 1024.0

    print(f"Architecture:     {arch}")
    print(f"Trainable Params: {total_params:,}")
    print(f"Model File Size:  {model_file_size_kb:.2f} KB")
    print(f"Parameter Memory: {model_param_bytes / 1024.0:.2f} KB")

    # 2. Warmup
    dummy_single = torch.randn(1, 1, 49, 40, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_single)

    # 3. Model-Only Single Sample Latency (B=1)
    single_latencies_ns = []
    with torch.no_grad():
        for _ in range(num_iterations):
            t0 = time.perf_counter_ns()
            _ = model(dummy_single)
            t1 = time.perf_counter_ns()
            single_latencies_ns.append(t1 - t0)

    single_latencies_ms = np.array(single_latencies_ns, dtype=np.float64) / 1e6
    mean_ms = float(np.mean(single_latencies_ms))
    p50_ms = float(np.percentile(single_latencies_ms, 50))
    p95_ms = float(np.percentile(single_latencies_ms, 95))
    p99_ms = float(np.percentile(single_latencies_ms, 99))
    min_ms = float(np.min(single_latencies_ms))
    max_ms = float(np.max(single_latencies_ms))
    std_ms = float(np.std(single_latencies_ms))

    print("\n--- Single-Sample (B=1) Neural Network Latency ---")
    print(f"Mean:             {mean_ms:.3f} ms")
    print(f"Median (P50):     {p50_ms:.3f} ms")
    print(f"P95:              {p95_ms:.3f} ms")
    print(f"P99:              {p99_ms:.3f} ms")
    print(f"Min:              {min_ms:.3f} ms")
    print(f"Max:              {max_ms:.3f} ms")
    print(f"Std Dev:          {std_ms:.3f} ms")

    # Real-time checks
    frame_hop_ms = 20.0
    deadline_ms = 100.0
    hop_utilization_pct = (mean_ms / frame_hop_ms) * 100.0
    deadline_utilization_pct = (mean_ms / deadline_ms) * 100.0
    meets_hop_deadline = bool(p99_ms < frame_hop_ms)
    meets_task_deadline = bool(p99_ms < deadline_ms)

    print(f"Frame Hop (20 ms) Utilization:     {hop_utilization_pct:.2f}% (Meets Hop: {meets_hop_deadline})")
    print(f"Task Deadline (100 ms) Utilization:{deadline_utilization_pct:.2f}% (Meets Deadline: {meets_task_deadline})")

    # 4. End-to-End Pipeline Latency (DSP + Model + Decision)
    extractor = LogMelSpectrogramExtractor(sample_rate=16000, frame_size=640, hop_size=320, num_mel_bins=40)
    dummy_audio = np.random.randn(16000).astype(np.float32)

    dsp_latencies_ms = []
    nn_latencies_ms = []
    post_latencies_ms = []
    total_pipeline_latencies_ms = []

    with torch.no_grad():
        for _ in range(num_iterations // 2):
            t_start = time.perf_counter_ns()

            # Step 1: DSP
            t_dsp_0 = time.perf_counter_ns()
            spec = extractor.extract(dummy_audio)
            t_dsp_1 = time.perf_counter_ns()

            # Step 2: NN Forward
            inp_tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()
            t_nn_0 = time.perf_counter_ns()
            logits = model(inp_tensor)
            t_nn_1 = time.perf_counter_ns()

            # Step 3: Softmax + Decision
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())
            is_command = (confidence >= 0.60 and pred_idx >= 2)
            t_end = time.perf_counter_ns()

            dsp_latencies_ms.append((t_dsp_1 - t_dsp_0) / 1e6)
            nn_latencies_ms.append((t_nn_1 - t_nn_0) / 1e6)
            post_latencies_ms.append((t_end - t_nn_1) / 1e6)
            total_pipeline_latencies_ms.append((t_end - t_start) / 1e6)

    dsp_mean_ms = float(np.mean(dsp_latencies_ms))
    nn_mean_ms = float(np.mean(nn_latencies_ms))
    post_mean_ms = float(np.mean(post_latencies_ms))
    pipeline_mean_ms = float(np.mean(total_pipeline_latencies_ms))
    pipeline_p99_ms = float(np.percentile(total_pipeline_latencies_ms, 99))

    print("\n--- End-to-End Recognition Pipeline Latency ---")
    print(f"DSP Feature Extraction: {dsp_mean_ms:.3f} ms ({dsp_mean_ms / pipeline_mean_ms * 100:.1f}%)")
    print(f"Neural Network Forward: {nn_mean_ms:.3f} ms ({nn_mean_ms / pipeline_mean_ms * 100:.1f}%)")
    print(f"Postprocessing/Softmax: {post_mean_ms:.3f} ms ({post_mean_ms / pipeline_mean_ms * 100:.1f}%)")
    print(f"Total Pipeline Mean:    {pipeline_mean_ms:.3f} ms")
    print(f"Total Pipeline P99:     {pipeline_p99_ms:.3f} ms")
    print(f"Total Deadline Budget:  {pipeline_mean_ms / deadline_ms * 100:.2f}% of 100 ms")

    # 5. Batched Inference Throughput
    batch_benchmarks = {}
    for bs in [32, 64]:
        dummy_batch = torch.randn(bs, 1, 49, 40, dtype=torch.float32)
        batch_times_ms = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.perf_counter_ns()
                _ = model(dummy_batch)
                t1 = time.perf_counter_ns()
                batch_times_ms.append((t1 - t0) / 1e6)
        mean_batch_time = float(np.mean(batch_times_ms))
        latency_per_sample = mean_batch_time / bs
        throughput_samples_per_sec = (bs * 1000.0) / mean_batch_time
        batch_benchmarks[f"batch_{bs}"] = {
            "batch_size": bs,
            "mean_batch_time_ms": round(mean_batch_time, 3),
            "latency_per_sample_ms": round(latency_per_sample, 3),
            "throughput_samples_per_sec": round(throughput_samples_per_sec, 1),
        }
        print(f"\nBatch Size {bs}: {mean_batch_time:.2f} ms/batch | {latency_per_sample:.3f} ms/sample | {throughput_samples_per_sec:.1f} samples/sec")

    # 6. Assemble Benchmark Artifact
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": "CPU",
            "cpu_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "model_architecture": arch,
        "parameter_count": total_params,
        "model_file_size_kb": round(model_file_size_kb, 2),
        "parameter_memory_kb": round(model_param_bytes / 1024.0, 2),
        "single_sample_latency_ms": {
            "mean": round(mean_ms, 3),
            "median_p50": round(p50_ms, 3),
            "p95": round(p95_ms, 3),
            "p99": round(p99_ms, 3),
            "min": round(min_ms, 3),
            "max": round(max_ms, 3),
            "std": round(std_ms, 3),
        },
        "end_to_end_pipeline_ms": {
            "dsp_feature_extraction_mean": round(dsp_mean_ms, 3),
            "neural_network_mean": round(nn_mean_ms, 3),
            "postprocessing_mean": round(post_mean_ms, 3),
            "total_mean": round(pipeline_mean_ms, 3),
            "total_p99": round(pipeline_p99_ms, 3),
        },
        "real_time_deadlines": {
            "frame_hop_ms": frame_hop_ms,
            "meets_frame_hop": meets_hop_deadline,
            "frame_hop_utilization_pct": round(hop_utilization_pct, 2),
            "deadline_ms": deadline_ms,
            "meets_deadline": meets_task_deadline,
            "deadline_utilization_pct": round(deadline_utilization_pct, 2),
        },
        "batched_throughput": batch_benchmarks,
    }

    out_p = pathlib.Path(output_file).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"\nSaved benchmark results to: {out_p}")

    return benchmark_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark CPU inference latency.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/dscnn_improved.pt", help="Checkpoint to benchmark")
    parser.add_argument("--output", type=str, default="results/inference_benchmark_improved.json", help="Output JSON path")
    parser.add_argument("--runs", type=int, default=1000, help="Number of benchmark iterations")
    args = parser.parse_args()

    benchmark_cpu_latency(checkpoint_path=args.checkpoint, output_file=args.output, num_iterations=args.runs)

