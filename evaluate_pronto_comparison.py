#!/usr/bin/env python3
"""
Comprehensive evaluation script for comparing DySTGAT vs DyEdgeGAT on PRONTO.

Computes:
- AUC, F1, F1* (matching DyEdgeGAT paper metrics)
- Per-fault breakdown
- Per-severity analysis
- Statistical significance tests

Usage:
    python evaluate_pronto_comparison.py \
        --checkpoint checkpoints/pronto/dystgat_best.pt \
        --output-dir results/pronto_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRONTO comparison evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--output-dir", default="results/pronto_comparison", help="Output directory")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use-spectral-view", action="store_true")
    parser.add_argument("--freq-embed-dim", type=int, default=16)
    parser.add_argument("--freq-band-mix", default="mlp")
    parser.add_argument("--freq-use-log", action="store_true", default=True)
    parser.add_argument("--freq-use-spectral-features", action="store_true")
    parser.add_argument("--fuse-mode", default="concat")
    parser.add_argument("--divergence-type", default="js")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
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

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


@torch.no_grad()
def collect_scores(model, loader, device, amp_enabled: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect anomaly scores, reconstruction errors, and labels."""
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
        n_nodes = cfg.dataset.n_nodes
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
                # Take one label per graph
                labels.extend(y[::n_nodes][:b].tolist())
        else:
            labels.extend([0] * b)

    return np.array(scores), np.array(recon_errors), np.array(labels), np.array(divergences)


def compute_metrics(
    normal_scores: np.ndarray,
    fault_scores: np.ndarray,
    quantile_threshold: float = 0.95,
) -> Dict[str, float]:
    """
    Compute AUC, F1, and F1* metrics.

    F1: Best F1 across all thresholds
    F1*: F1 at 95th percentile of normal scores (paper definition)
    """
    # Combine for ROC/AUC
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

    # F1* at 95th percentile of normal
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


def main():
    args = parse_args()

    # Setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    amp_enabled = args.use_amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRONTO EVALUATION - DySTGAT vs DyEdgeGAT Comparison")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load adapter and data
    adapter = get_adapter("pronto")
    data_dir = adapter.default_data_dir
    n_nodes = 11
    n_control = 6

    cfg.set_dataset_params(
        n_nodes=n_nodes,
        window_size=args.window_size,
        ocvar_dim=n_control,
    )

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loaders = adapter.create_dataloaders(
        data_dir=data_dir,
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_stride=args.stride,
        val_stride=args.stride,
        test_stride=args.stride,
        num_workers=0,
    )

    # Load model
    print("Loading model...")
    model = load_model(args, n_nodes, n_control, device)

    # Collect baseline (normal) scores
    print("\nCollecting baseline scores...")
    baseline_scores, baseline_recon, baseline_labels, baseline_div = collect_scores(
        model, test_loaders["baseline"], device, amp_enabled
    )
    print(f"  Baseline: {len(baseline_scores)} samples, mean_score={baseline_scores.mean():.4f}")

    # Results storage
    results = {
        "model": args.checkpoint,
        "window_size": args.window_size,
        "use_spectral_view": args.use_spectral_view,
        "baseline_mean": float(baseline_scores.mean()),
        "baseline_std": float(baseline_scores.std()),
        "per_fault": {},
    }

    # Evaluate each fault type
    print("\n" + "=" * 70)
    print("PER-FAULT METRICS")
    print("=" * 70)
    print(f"{'Fault Type':<30} {'AUC':>8} {'F1':>8} {'F1*':>8} {'N':>8}")
    print("-" * 70)

    # Use the new fault key structure from pronto.py
    # Available keys: baseline, slugging, blockage, leakage, diverted, faults_all
    fault_loaders = {
        "slugging": test_loaders["slugging"],
        "blockage": test_loaders["blockage"],
        "leakage": test_loaders["leakage"],
        "diverted": test_loaders["diverted"],
        "faults_all": test_loaders["faults_all"],
    }

    for fault_name, loader in fault_loaders.items():
        fault_scores, fault_recon, fault_labels, fault_div = collect_scores(
            model, loader, device, amp_enabled
        )

        metrics = compute_metrics(baseline_scores, fault_scores)
        results["per_fault"][fault_name] = metrics

        print(f"{fault_name:<30} {metrics['auc']:>8.4f} {metrics['f1']:>8.4f} {metrics['f1_star']:>8.4f} {metrics['n_fault']:>8}")

    # Summary comparison with DyEdgeGAT
    print("\n" + "=" * 70)
    print("COMPARISON WITH DyEdgeGAT (Table VII)")
    print("=" * 70)

    # DyEdgeGAT reported metrics (from paper)
    dyedgegat_metrics = {
        "auc": 0.83,
        "f1": 0.64,
        "f1_star": 0.75,
    }

    # Your metrics on faults_all
    your_metrics = results["per_fault"]["faults_all"]

    print(f"{'Metric':<15} {'DyEdgeGAT':>12} {'DySTGAT':>12} {'Delta':>12} {'Better?':>10}")
    print("-" * 70)

    for metric in ["auc", "f1", "f1_star"]:
        dyedge_val = dyedgegat_metrics[metric]
        your_val = your_metrics[metric]
        delta = your_val - dyedge_val
        better = "YES" if delta > 0 else "NO"
        print(f"{metric:<15} {dyedge_val:>12.4f} {your_val:>12.4f} {delta:>+12.4f} {better:>10}")

    # Save results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create summary CSV
    summary_data = []
    for fault_name, metrics in results["per_fault"].items():
        summary_data.append({
            "fault": fault_name,
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "f1_star": metrics["f1_star"],
            "n_samples": metrics["n_fault"],
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "per_fault_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Per-fault metrics saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
