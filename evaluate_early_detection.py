#!/usr/bin/env python3
"""
Comprehensive Evaluation Script with Early Detection Metrics.

This script evaluates trained DySTGAT models with a focus on early/incipient
fault detection capabilities. It computes:

1. Standard Metrics: AUC-ROC, Best F1, F1* (at 95th percentile threshold)
2. Early Detection Metrics:
   - Detection delay (samples to first detection)
   - Persistent detection delay (N consecutive detections required)
   - Severity at detection (for progressive faults)
   - Normalized detection time (% of fault duration)
3. TEA-Enhanced Metrics: Temporal Evidence Accumulation for gradual faults

Usage:
    python evaluate_early_detection.py \
        --checkpoint checkpoints/pronto/best.pt \
        --dataset-key pronto \
        --output-dir results/early_detection
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast

# Add project paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dystgat"))

from datasets import get_adapter
from dystgat.src.config import cfg
from dystgat.src.utils.early_detection import (
    compute_early_detection_metrics,
    batch_evaluate_early_detection,
    format_early_detection_results,
)
from dystgat.src.utils.tea import (
    TemporalEvidenceAccumulator,
    compute_tea_metrics,
)


# Dataset-specific configurations
DATASET_CONFIG = {
    "pronto": {"n_nodes": 11, "n_control": 6},
    "tep": {"n_nodes": 52, "n_control": 12},
    "ims": {"n_nodes": 4, "n_control": 0},
    "ashrae": {"n_nodes": 28, "n_control": 5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Early Detection Evaluation")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--dataset-key", required=True,
                        choices=["pronto", "tep", "ims", "ashrae"],
                        help="Dataset to evaluate")

    # Output
    parser.add_argument("--output-dir", default="results/early_detection",
                        help="Output directory for results")

    # Data parameters
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--data-dir", default=None, help="Override data directory")

    # Model parameters
    parser.add_argument("--use-spectral-view", action="store_true")
    parser.add_argument("--freq-embed-dim", type=int, default=16)
    parser.add_argument("--freq-band-mix", default="mlp")
    parser.add_argument("--freq-use-log", action="store_true", default=True)
    parser.add_argument("--freq-use-spectral-features", action="store_true")
    parser.add_argument("--fuse-mode", default="concat")
    parser.add_argument("--divergence-type", default="js")

    # Early detection parameters
    parser.add_argument("--threshold-percentile", type=float, default=95.0,
                        help="Percentile of baseline for threshold")
    parser.add_argument("--persistence", type=int, default=3,
                        help="Consecutive detections required for persistent delay")

    # TEA parameters
    parser.add_argument("--use-tea", action="store_true",
                        help="Also compute TEA-enhanced metrics")
    parser.add_argument("--tea-window-sizes", type=int, nargs="+",
                        default=[30, 60, 180],
                        help="Window sizes for TEA")

    # Execution
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def load_model(args, n_nodes: int, n_control: int, device: torch.device):
    """Load trained model from checkpoint."""
    from train_dystgat import init_model

    model = init_model(
        device=device,
        window_size=args.window_size,
        n_control_vars=n_control,
        n_measurement_vars=n_nodes,
        model_args=args,
        task="reconstruction",
        pred_horizon=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_scores(
    model,
    loader,
    device: torch.device,
    amp_enabled: bool,
    n_nodes: int,
) -> Dict[str, np.ndarray]:
    """
    Collect anomaly scores, reconstruction errors, labels, and divergences.

    Returns a dictionary with:
    - 'scores': Anomaly scores (topology-aware if available)
    - 'recon_errors': Raw reconstruction MSE
    - 'labels': Ground truth labels
    - 'divergences': Spectral-temporal divergence scores
    """
    scores = []
    recon_errors = []
    labels = []
    divergences = []

    for batch in loader:
        batch = batch.to(device)
        with autocast("cuda", enabled=amp_enabled):
            outputs = model(batch, return_graph=True)

        if isinstance(outputs, tuple) and len(outputs) >= 4:
            recon, edge_index, edge_attr, aux = outputs[:4]
        else:
            recon = outputs
            aux = {}

        # Reconstruction error
        target = batch.x
        b = target.shape[0] // n_nodes

        target_reshaped = target.view(b, n_nodes, -1)
        recon_reshaped = recon.view(b, n_nodes, -1)

        # Per-sample MSE
        per_sample_mse = ((target_reshaped - recon_reshaped) ** 2).mean(dim=(1, 2))
        recon_errors.extend(per_sample_mse.cpu().numpy().tolist())

        # Anomaly score (use topology-aware if available)
        if isinstance(aux, dict) and "anomaly_score" in aux and aux["anomaly_score"] is not None:
            score = aux["anomaly_score"].cpu().numpy().flatten()
        else:
            score = per_sample_mse.cpu().numpy()
        scores.extend(score.tolist())

        # Divergence score
        if isinstance(aux, dict) and "divergence_score" in aux and aux["divergence_score"] is not None:
            div = aux["divergence_score"].cpu().numpy().flatten()
            divergences.extend(div.tolist())
        else:
            divergences.extend([0.0] * b)

        # Labels
        if hasattr(batch, "y") and batch.y is not None:
            y = batch.y.cpu().numpy()
            if len(y) == b:
                labels.extend(y.tolist())
            else:
                labels.extend(y[::n_nodes][:b].tolist())
        else:
            labels.extend([0] * b)

    return {
        'scores': np.array(scores),
        'recon_errors': np.array(recon_errors),
        'labels': np.array(labels),
        'divergences': np.array(divergences),
    }


def compute_standard_metrics(
    normal_scores: np.ndarray,
    fault_scores: np.ndarray,
    quantile_threshold: float = 0.95,
) -> Dict[str, float]:
    """Compute AUC, F1, and F1* metrics."""
    all_scores = np.concatenate([normal_scores, fault_scores])
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(fault_scores))
    ])

    # AUC
    try:
        auc_score = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc_score = 0.5

    # Best F1 (sweep thresholds)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores[np.isfinite(f1_scores)])
    best_threshold = thresholds[np.argmax(f1_scores[:-1])] if len(thresholds) > 0 else 0

    # F1* at percentile threshold of normal
    threshold_star = np.percentile(normal_scores, quantile_threshold * 100)
    pred_star = (all_scores >= threshold_star).astype(int)
    f1_star = f1_score(all_labels, pred_star, zero_division=0)

    return {
        "auc": float(auc_score),
        "f1": float(best_f1),
        "f1_star": float(f1_star),
        "threshold_best": float(best_threshold),
        "threshold_star": float(threshold_star),
        "n_normal": len(normal_scores),
        "n_fault": len(fault_scores),
    }


def evaluate_fault(
    baseline_data: Dict[str, np.ndarray],
    fault_data: Dict[str, np.ndarray],
    threshold_percentile: float = 95.0,
    persistence: int = 3,
    use_tea: bool = False,
    tea_window_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a single fault type.

    Returns metrics including:
    - Standard: AUC, F1, F1*
    - Early detection: delay, persistent delay, severity, normalized time
    - TEA-enhanced (if requested)
    """
    baseline_scores = baseline_data['scores']
    fault_scores = fault_data['scores']
    fault_labels = fault_data['labels']

    # Standard metrics
    standard = compute_standard_metrics(
        baseline_scores, fault_scores, threshold_percentile / 100.0
    )

    # Early detection metrics
    # Create binary labels (any non-zero label is a fault)
    binary_labels = (fault_labels > 0).astype(int)

    early = compute_early_detection_metrics(
        baseline_scores=baseline_scores,
        fault_scores=fault_scores,
        fault_labels=binary_labels,
        threshold_percentile=threshold_percentile,
        persistence=persistence,
    )

    result = {
        **standard,
        'delay_samples': early['delay_samples'],
        'delay_persistent': early['delay_persistent'],
        'normalized_delay_pct': early['normalized_delay_pct'],
        'detection_rate': early['detection_rate'],
    }

    # Divergence statistics
    if 'divergences' in baseline_data and 'divergences' in fault_data:
        baseline_div = baseline_data['divergences']
        fault_div = fault_data['divergences']
        result['baseline_div_mean'] = float(np.mean(baseline_div))
        result['baseline_div_std'] = float(np.std(baseline_div))
        result['fault_div_mean'] = float(np.mean(fault_div))
        result['fault_div_std'] = float(np.std(fault_div))
        result['div_ratio'] = float(np.mean(fault_div) / (np.mean(baseline_div) + 1e-8))

    # TEA-enhanced metrics
    if use_tea and tea_window_sizes:
        tea_metrics = compute_tea_metrics(
            baseline_scores=baseline_scores,
            fault_scores=fault_scores,
            window_sizes=tea_window_sizes,
        )
        result['tea_auc'] = tea_metrics['auc']
        result['tea_f1'] = tea_metrics['best_f1']
        result['tea_best_window'] = tea_metrics.get('best_window', tea_window_sizes[-1])

    return result


def print_results_table(results: Dict[str, Dict[str, Any]], dataset_key: str):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print(f"EVALUATION RESULTS - {dataset_key.upper()}")
    print("=" * 100)

    # Header
    print(f"{'Fault Type':<30} {'AUC':>8} {'F1':>8} {'F1*':>8} "
          f"{'Delay':>8} {'Persist':>8} {'Norm%':>8} {'DetRate':>8}")
    print("-" * 100)

    for fault_name, metrics in results.items():
        delay = metrics.get('delay_samples', -1)
        delay_str = str(delay) if delay >= 0 else "N/A"

        persist = metrics.get('delay_persistent', -1)
        persist_str = str(persist) if persist >= 0 else "N/A"

        print(f"{fault_name:<30} "
              f"{metrics.get('auc', 0):>8.4f} "
              f"{metrics.get('f1', 0):>8.4f} "
              f"{metrics.get('f1_star', 0):>8.4f} "
              f"{delay_str:>8} "
              f"{persist_str:>8} "
              f"{metrics.get('normalized_delay_pct', 100):>7.1f}% "
              f"{metrics.get('detection_rate', 0):>7.1%}")

    print("=" * 100)

    # TEA results if available
    tea_available = any('tea_auc' in m for m in results.values())
    if tea_available:
        print("\nTEA-ENHANCED METRICS")
        print("-" * 60)
        print(f"{'Fault Type':<30} {'TEA AUC':>10} {'TEA F1':>10} {'Window':>10}")
        print("-" * 60)
        for fault_name, metrics in results.items():
            if 'tea_auc' in metrics:
                print(f"{fault_name:<30} "
                      f"{metrics['tea_auc']:>10.4f} "
                      f"{metrics['tea_f1']:>10.4f} "
                      f"{metrics.get('tea_best_window', 'N/A'):>10}")
        print("-" * 60)


def main():
    args = parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    amp_enabled = args.use_amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EARLY DETECTION EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_key}")
    print(f"Device: {device}")
    print(f"Threshold percentile: {args.threshold_percentile}%")
    print(f"Persistence requirement: {args.persistence} consecutive detections")
    print(f"TEA enabled: {args.use_tea}")
    if args.use_tea:
        print(f"TEA window sizes: {args.tea_window_sizes}")
    print(f"Output: {output_dir}")

    # Get dataset configuration
    dataset_cfg = DATASET_CONFIG.get(args.dataset_key)
    if dataset_cfg is None:
        raise ValueError(f"Unknown dataset: {args.dataset_key}")

    n_nodes = dataset_cfg["n_nodes"]
    n_control = dataset_cfg["n_control"]

    # Setup config
    cfg.set_dataset_params(
        n_nodes=n_nodes,
        window_size=args.window_size,
        ocvar_dim=n_control,
    )

    # Load adapter and data
    adapter = get_adapter(args.dataset_key)
    data_dir = args.data_dir or adapter.default_data_dir

    print("\nLoading data...")
    train_loader, val_loader, test_loaders = adapter.create_dataloaders(
        data_dir=data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_stride=args.stride,
        val_stride=args.stride,
        test_stride=args.stride,
        num_workers=args.num_workers,
    )

    # Load model
    print("Loading model...")
    model = load_model(args, n_nodes, n_control, device)

    # Collect baseline scores
    print("\nCollecting baseline scores...")
    baseline_data = collect_scores(
        model, test_loaders["baseline"], device, amp_enabled, n_nodes
    )
    print(f"  Baseline: {len(baseline_data['scores'])} samples, "
          f"mean_score={baseline_data['scores'].mean():.4f}")

    # Evaluate each fault type
    results = {}
    print("\nEvaluating fault types...")

    for fault_name, loader in test_loaders.items():
        if fault_name == "baseline":
            continue

        print(f"  Processing {fault_name}...")
        fault_data = collect_scores(model, loader, device, amp_enabled, n_nodes)

        metrics = evaluate_fault(
            baseline_data=baseline_data,
            fault_data=fault_data,
            threshold_percentile=args.threshold_percentile,
            persistence=args.persistence,
            use_tea=args.use_tea,
            tea_window_sizes=args.tea_window_sizes,
        )

        results[fault_name] = metrics

    # Print results
    print_results_table(results, args.dataset_key)

    # Save results
    results_path = output_dir / f"{args.dataset_key}_early_detection_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Create summary CSV
    summary_data = []
    for fault_name, metrics in results.items():
        row = {"fault": fault_name}
        row.update(metrics)
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"{args.dataset_key}_early_detection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary CSV saved to: {summary_path}")

    # Compute and save aggregate statistics
    aggregate = {
        'dataset': args.dataset_key,
        'checkpoint': args.checkpoint,
        'threshold_percentile': args.threshold_percentile,
        'n_fault_types': len(results),
        'mean_auc': float(np.mean([m['auc'] for m in results.values()])),
        'mean_f1': float(np.mean([m['f1'] for m in results.values()])),
        'mean_f1_star': float(np.mean([m['f1_star'] for m in results.values()])),
        'mean_detection_rate': float(np.mean([m['detection_rate'] for m in results.values()])),
    }

    # Count successful early detections
    early_detections = [m['delay_samples'] for m in results.values() if m['delay_samples'] >= 0]
    if early_detections:
        aggregate['mean_delay_samples'] = float(np.mean(early_detections))
        aggregate['median_delay_samples'] = float(np.median(early_detections))

    if args.use_tea:
        tea_aucs = [m.get('tea_auc', 0) for m in results.values()]
        aggregate['mean_tea_auc'] = float(np.mean(tea_aucs))

    aggregate_path = output_dir / f"{args.dataset_key}_aggregate_stats.json"
    with open(aggregate_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Aggregate statistics saved to: {aggregate_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
