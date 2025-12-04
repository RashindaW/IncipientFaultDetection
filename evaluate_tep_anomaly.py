#!/usr/bin/env python3
"""
Evaluate DyEdgeGAT on the TEP dataset with labeled faults.

Workflow:
- Train/val stats come from FaultFree training.
- Validation: FaultFree testing (label 0).
- Test: Faulty testing (faultNumber 1..20).
- Sweep anomaly-score thresholds to maximise F1 (binary: normal vs any fault).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch.amp import autocast
from torch_geometric.loader import DataLoader

from datasets import get_adapter
from train_dyedgegat import (
    forward_model,
    init_model,
    resolve_devices,
    unwrap_model,
    unpack_model_outputs,
)
from dyedgegat.src.data.tep_column_config import (
    FAULT_FREE_TEST_FILE,
    FAULT_FREE_TRAIN_FILE,
    FAULTY_TEST_FILE,
    MEASUREMENT_VARS,
)
from dyedgegat.src.data.tep_dataset import TEPDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DyEdgeGAT on TEP with best-F1 threshold search.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained DyEdgeGAT checkpoint.")
    parser.add_argument("--data-dir", default=None, help="Path to TEP RData files (defaults to adapter).")
    parser.add_argument("--window-size", type=int, default=60, help="Sliding window size.")
    parser.add_argument("--train-stride", type=int, default=1, help="Stride for fault-free training windows.")
    parser.add_argument("--val-stride", type=int, default=1, help="Stride for fault-free validation windows.")
    parser.add_argument("--test-stride", type=int, default=1, help="Stride for faulty testing windows.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Computation device.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index when using GPU.")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument(
        "--quantiles",
        type=int,
        default=201,
        help="Number of quantiles to sample for threshold search (higher = finer).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for saving metrics; defaults to checkpoint stem + timestamp.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tep_eval",
        help="Directory to store metrics JSON/text (will be created).",
    )
    parser.add_argument("--use-spectral-view", action="store_true", help="Enable spectral branch for evaluation.")
    parser.add_argument("--freq-embed-dim", type=int, default=16, help="Spectral embedding dimension.")
    parser.add_argument("--freq-bins", type=int, default=0, help="Number of rFFT bins to keep (0 = all).")
    parser.add_argument(
        "--freq-band-mix",
        type=str,
        default="none",
        choices=["none", "conv", "mlp"],
        help="Optional band mixer for spectral encoder.",
    )
    parser.add_argument("--freq-topk", type=int, default=None, help="Top-k neighbors for spectral graph (default=temporal topk).")
    parser.add_argument(
        "--share-gnn-weights",
        action="store_true",
        help="Share temporal GNN weights with spectral branch.",
    )
    parser.add_argument(
        "--fuse-mode",
        type=str,
        default="concat",
        choices=["concat", "sum", "gated"],
        help="Fusion strategy for temporal/spectral embeddings.",
    )
    parser.add_argument(
        "--divergence-type",
        type=str,
        default="js",
        choices=["js", "kl"],
        help="Divergence metric used during training; needed to build the model.",
    )
    parser.add_argument(
        "--faults",
        type=int,
        nargs="+",
        default=None,
        help="List of specific fault IDs to evaluate (e.g., --faults 3 9 15). If omitted, all faults are evaluated.",
    )
    return parser.parse_args()


def build_datasets(args: argparse.Namespace, data_dir: str):
    print("\nLoading datasets...")
    train_dataset = TEPDataset(
        data_files=[FAULT_FREE_TRAIN_FILE],
        window_size=args.window_size,
        stride=args.train_stride,
        data_dir=data_dir,
        normalize=True,
        fault_filter=[0],
    )
    norm_stats = train_dataset.get_normalization_stats()

    val_dataset = TEPDataset(
        data_files=[FAULT_FREE_TEST_FILE],
        window_size=args.window_size,
        stride=args.val_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        fault_filter=[0],
    )
    
    fault_filter = args.faults if args.faults is not None else None
    faulty_dataset = TEPDataset(
        data_files=[FAULTY_TEST_FILE],
        window_size=args.window_size,
        stride=args.test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        fault_filter=fault_filter,
    )
    return train_dataset, val_dataset, faulty_dataset


def collect_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    base_model = unwrap_model(model)
    scores = []
    labels = []
    model.eval()

    with torch.no_grad():
        for raw_batch in loader:
            with autocast("cuda", enabled=amp_enabled):
                outputs, batch_obj = forward_model(
                    model,
                    raw_batch,
                    device,
                    return_graph=True,
                )
                recon, edge_index, edge_attr, aux = unpack_model_outputs(outputs)
                target = batch_obj.x.unsqueeze(-1)
                batch_scores = base_model.compute_anomaly_scores_per_sample(
                    target, recon, edge_index, edge_attr
                )
            scores.append(batch_scores.cpu().numpy())
            labels.append(batch_obj.y.view(-1).cpu().numpy())

    all_scores = np.concatenate(scores) if scores else np.array([])
    all_labels = np.concatenate(labels) if labels else np.array([])
    return all_scores, all_labels


def best_f1(scores: np.ndarray, labels: np.ndarray, quantiles: int = 201) -> Dict[str, float]:
    if scores.size == 0:
        raise ValueError("No scores collected for threshold search.")
    binary_labels = (labels > 0).astype(np.int64)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, quantiles)))

    best = {"threshold": thresholds[0], "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for thr in thresholds:
        preds = scores >= thr
        tp = np.sum(preds & (binary_labels == 1))
        fp = np.sum(preds & (binary_labels == 0))
        fn = np.sum((~preds) & (binary_labels == 1))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        if f1 > best["f1"]:
            best.update({"threshold": float(thr), "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    return best


def per_fault_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    preds = scores >= threshold
    binary_labels = labels > 0

    # Overall
    tp = np.sum(preds & binary_labels)
    fp = np.sum(preds & ~binary_labels)
    fn = np.sum(~preds & binary_labels)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    metrics["overall"] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    # Incipient Faults subset (Faults 3, 9, 15)
    incipient_faults = {3, 9, 15}
    incipient_mask = np.isin(labels, list(incipient_faults))
    if incipient_mask.any():
        # Predictions on incipient faults
        inc_preds = preds[incipient_mask]
        # True labels on incipient faults (all are anomalies)
        inc_labels = binary_labels[incipient_mask]
        
        # False positives are counted from the global normal pool (labels == 0)
        # We use the same False Positives as "overall" because in an anomaly detection setting,
        # the "Normal" class is shared.
        # However, Precision = TP / (TP + FP). TP comes from Incipient, FP comes from Normal.
        
        inc_tp = np.sum(inc_preds)
        # FP is calculated against normal data only (labels == 0)
        normal_mask = labels == 0
        inc_fp = np.sum(preds[normal_mask])
        
        inc_fn = np.sum(~inc_preds)
        
        inc_prec = inc_tp / (inc_tp + inc_fp + 1e-9)
        inc_rec = inc_tp / (inc_tp + inc_fn + 1e-9)
        inc_f1 = 2 * inc_prec * inc_rec / (inc_prec + inc_rec + 1e-9)
        
        metrics["incipient_subset"] = {
            "precision": float(inc_prec),
            "recall": float(inc_rec),
            "f1": float(inc_f1),
            "support": int(incipient_mask.sum())
        }

    # Per fault class (>0)
    for fault_id in sorted(set(int(x) for x in labels.tolist() if x > 0)):
        mask = labels == fault_id
        if not mask.any():
            continue
        fault_preds = preds[mask]
        tp = np.sum(fault_preds)
        # FP is always relative to the normal dataset
        normal_mask = labels == 0
        fp = np.sum(preds[normal_mask])
        fn = np.sum(~fault_preds)
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        metrics[f"fault_{fault_id:02d}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(mask.sum()),
        }
    return metrics


def main() -> None:
    args = parse_args()

    adapter = get_adapter("tep")
    adapter.ensure("testing")
    data_dir = args.data_dir or adapter.get_default_data_dir()

    device, _ = resolve_devices(args.device, args.cuda_device, None)
    amp_enabled = args.use_amp and device.type == "cuda"

    control_vars = adapter.get_control_variables(data_dir)
    model = init_model(
        device,
        args.window_size,
        len(control_vars),
        len(MEASUREMENT_VARS),
        model_args=args,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    load_result = model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")
    if load_result.missing_keys:
        print(f"  Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"  Unexpected keys: {load_result.unexpected_keys}")

    if device.type == "cuda" and args.cuda_device is not None:
        torch.cuda.set_device(args.cuda_device)

    train_ds, val_ds, faulty_ds = build_datasets(args, data_dir)

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    faulty_loader = DataLoader(
        faulty_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print("\nCollecting anomaly scores on fault-free validation...")
    val_scores, val_labels = collect_scores(val_loader, model, device, amp_enabled)
    print(f"  Collected {len(val_scores)} samples (labels sum={val_labels.sum()})")

    print("Collecting anomaly scores on faulty testing...")
    test_scores, test_labels = collect_scores(faulty_loader, model, device, amp_enabled)
    print(f"  Collected {len(test_scores)} samples with faults present.")

    all_scores = np.concatenate([val_scores, test_scores])
    all_labels = np.concatenate([val_labels, test_labels])

    print("\nSearching for best F1 threshold...")
    best = best_f1(all_scores, all_labels, quantiles=args.quantiles)
    metrics = per_fault_metrics(all_scores, all_labels, best["threshold"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"tep_eval_{Path(args.checkpoint).stem}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "window_size": args.window_size,
        "train_stride": args.train_stride,
        "val_stride": args.val_stride,
        "test_stride": args.test_stride,
        "device": str(device),
        "faults_evaluated": args.faults if args.faults else "all",
        "best_threshold": best,
        "metrics": metrics,
    }

    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))

    txt_path = output_dir / "metrics.txt"
    lines = [
        f"Best threshold: {best['threshold']:.6f}",
        f"Precision: {best['precision']:.4f}",
        f"Recall   : {best['recall']:.4f}",
        f"F1       : {best['f1']:.4f}",
        "",
    ]
    lines.append("Per-fault metrics:")
    for fault_key, stats in metrics.items():
        if fault_key == "overall":
            continue
        lines.append(
            f"{fault_key}: prec={stats['precision']:.4f} "
            f"rec={stats['recall']:.4f} f1={stats['f1']:.4f} support={stats.get('support', 0)}"
        )
    txt_path.write_text("\n".join(lines))

    print(f"\nBest F1 threshold: {best['threshold']:.6f}")
    print(f"Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, F1={best['f1']:.4f}")
    print(f"Saved metrics to {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
