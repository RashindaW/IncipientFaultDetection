#!/usr/bin/env python3
"""Unified training script for baseline anomaly detection methods.

This script provides a consistent interface for training and evaluating
baseline methods to compare against DySTGAT on the IMS-raw bearing dataset.

Usage:
    # Train LSTM-VAE
    python train_baselines.py --method lstm_vae --dataset-key ims-raw --epochs 100

    # Train USAD
    python train_baselines.py --method usad --dataset-key ims-raw --epochs 100

    # Train GDN with custom hyperparameters
    python train_baselines.py --method gdn --dataset-key ims-raw --epochs 100 --topk 5

    # Evaluate only (load checkpoint)
    python train_baselines.py --method lstm_vae --dataset-key ims-raw --eval-only \
        --checkpoint checkpoints/lstm_vae_best.pt
"""

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from baselines import (
    create_baseline,
    list_baselines,
    get_baseline_description,
    get_default_hyperparams,
    BaselineModel,
)
from baselines.utils.metrics import (
    compute_all_metrics,
    format_metrics_table,
)
from datasets import get_adapter, list_adapter_keys

# Optional: import TEA if available
try:
    from dystgat.src.utils.tea import compute_tea_metrics as compute_tea_metrics_dystgat
    HAS_TEA = True
except ImportError:
    HAS_TEA = False


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    available_datasets = list_adapter_keys()
    available_methods = list_baselines()

    parser = argparse.ArgumentParser(
        description="Train baseline anomaly detection methods for DySTGAT comparison"
    )

    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=available_methods,
        help=f"Baseline method to train. Available: {', '.join(available_methods)}",
    )

    # Dataset selection
    parser.add_argument(
        "--dataset-key",
        type=str,
        required=True,
        choices=available_datasets,
        help=f"Dataset to use. Available: {', '.join(available_datasets)}",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay (L2 regularization)"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Early stopping patience (epochs)",
    )

    # Data parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Temporal window size (default: 1024 for IMS-raw)",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=1,
        help="Sliding window stride for training",
    )
    parser.add_argument(
        "--val-stride",
        type=int,
        default=5,
        help="Sliding window stride for validation/test",
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=None,
        help="Sliding window stride for test (defaults to val-stride)",
    )

    # Data splitting
    parser.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=["temporal", "window_shuffle", "segment_shuffle"],
        help="Data splitting strategy",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=10,
        help="Number of segments for segment_shuffle mode",
    )
    parser.add_argument(
        "--train-segments",
        type=str,
        default=None,
        help="Comma-separated segment indices for training (e.g. '2,3,4,6,7,8,9')",
    )
    parser.add_argument(
        "--val-segments",
        type=str,
        default=None,
        help="Comma-separated segment indices for validation (e.g. '0,1')",
    )
    parser.add_argument(
        "--test-segments",
        type=str,
        default=None,
        help="Comma-separated segment indices for testing (e.g. '5')",
    )

    # Device
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes",
    )

    # Checkpointing
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for per-epoch checkpoints and metrics",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to load for evaluation or warm-start",
    )

    # Evaluation
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate checkpoint",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test set evaluation",
    )
    parser.add_argument(
        "--baseline-from",
        type=str,
        choices=["val", "train", "test"],
        default="test",
        help="Source for baseline (healthy) scores",
    )

    # Method-specific hyperparameters (parsed as key=value pairs)
    parser.add_argument(
        "--hyperparam",
        "-H",
        action="append",
        default=[],
        help="Method-specific hyperparameters as key=value (can be repeated)",
    )

    # Verbose
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print training progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress most output",
    )

    args = parser.parse_args()

    # Parse hyperparameters
    args.hyperparams = {}
    for hp in args.hyperparam:
        if "=" not in hp:
            parser.error(f"Invalid hyperparam format: {hp}. Use key=value.")
        key, value = hp.split("=", 1)
        # Try to parse as number
        try:
            if "." in value:
                args.hyperparams[key] = float(value)
            else:
                args.hyperparams[key] = int(value)
        except ValueError:
            args.hyperparams[key] = value

    if args.quiet:
        args.verbose = False

    return args


def resolve_device(
    device_flag: str, cuda_index: Optional[int]
) -> torch.device:
    """Resolve computation device."""
    if device_flag == "cpu":
        return torch.device("cpu")

    if device_flag == "cuda" or (device_flag == "auto" and torch.cuda.is_available()):
        if cuda_index is not None:
            if cuda_index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {cuda_index} not available. "
                    f"Only {torch.cuda.device_count()} devices visible."
                )
            return torch.device(f"cuda:{cuda_index}")
        return torch.device("cuda")

    return torch.device("cpu")


def evaluate_model(
    model: BaselineModel,
    test_loaders: Dict,
    baseline_loader,
    val_loader,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on test datasets using unified IQR-normalized scoring.

    Per the DyEdgeGAT paper (Section V.B): all baselines use GDN's scoring
    function — per-feature residuals normalized by validation median/IQR,
    then mean-aggregated across features.

    Args:
        model: Trained baseline model
        test_loaders: Dict of test data loaders
        baseline_loader: DataLoader for healthy/baseline samples
        val_loader: DataLoader for validation set (IQR normalization params)
        device: Computation device
        verbose: Print results

    Returns:
        Dict mapping test set names to metrics dicts
    """
    # Step 1: Compute IQR normalization parameters from validation set
    val_residuals = model.compute_per_feature_residuals(val_loader, device)
    val_median = np.median(val_residuals, axis=0)
    val_iqr = (
        np.percentile(val_residuals, 75, axis=0)
        - np.percentile(val_residuals, 25, axis=0)
    )
    val_iqr = np.clip(val_iqr, 1e-8, None)  # avoid division by zero

    if verbose:
        print(f"  IQR normalization: median range [{val_median.min():.4f}, "
              f"{val_median.max():.4f}], IQR range [{val_iqr.min():.4f}, "
              f"{val_iqr.max():.4f}]")

    # Step 2: Compute normalized baseline scores
    base_residuals = model.compute_per_feature_residuals(
        baseline_loader, device
    )
    baseline_scores = BaselineModel.iqr_normalize_scores(
        base_residuals, val_median, val_iqr
    )

    if verbose:
        print(f"  Baseline scores (IQR-norm): mean={baseline_scores.mean():.4f}, "
              f"std={baseline_scores.std():.4f}")

    # Step 3: Evaluate each fault loader
    all_metrics = {}

    for name, loader in test_loaders.items():
        if name == "baseline":
            continue

        # Compute normalized fault scores
        fault_residuals = model.compute_per_feature_residuals(loader, device)
        fault_scores = BaselineModel.iqr_normalize_scores(
            fault_residuals, val_median, val_iqr
        )

        # Compute metrics
        metrics = compute_all_metrics(
            baseline_scores, fault_scores, include_tea=True
        )

        all_metrics[name] = metrics

        if verbose:
            print(f"\n[{name}] Metrics:")
            print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Best F1:   {metrics['best_f1']:.4f}")
            if 'tea_auc' in metrics:
                print(f"  TEA AUC:   {metrics['tea_auc']:.4f}")
                print(f"  TEA F1:    {metrics['tea_best_f1']:.4f}")

    return all_metrics


def save_metrics_csv(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    method_name: str,
) -> None:
    """Save metrics to CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for test_name, test_metrics in metrics.items():
        row = {"method": method_name, "test_set": test_name}
        row.update(test_metrics)
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Main training and evaluation loop."""
    args = parse_args()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        if args.verbose:
            print(f"Random seed: {args.seed}")

    # Resolve device
    device = resolve_device(args.device, args.cuda_device)
    if args.verbose:
        print(f"Using device: {device}")

    # Load dataset adapter
    adapter = get_adapter(args.dataset_key)
    if args.verbose:
        print(f"Dataset: {adapter.description}")

    # Parse segment lists
    train_segments = (
        [int(s) for s in args.train_segments.split(",")]
        if args.train_segments else None
    )
    val_segments = (
        [int(s) for s in args.val_segments.split(",")]
        if args.val_segments else None
    )
    test_segments = (
        [int(s) for s in args.test_segments.split(",")]
        if args.test_segments else None
    )

    # Build extra kwargs for segment-based splitting
    split_kwargs = {}
    if args.split_mode:
        split_kwargs["split_mode"] = args.split_mode
    if train_segments is not None:
        split_kwargs["train_segments"] = train_segments
    if val_segments is not None:
        split_kwargs["val_segments"] = val_segments
    if test_segments is not None:
        split_kwargs["test_segments"] = test_segments
    if args.n_segments != 10:
        split_kwargs["n_segments"] = args.n_segments
    if args.seed is not None:
        split_kwargs["random_seed"] = args.seed

    # Create data loaders
    train_loader, val_loader, test_loaders = adapter.create_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride or args.val_stride,
        data_dir=adapter.default_data_dir,
        num_workers=args.num_workers,
        baseline_from=args.baseline_from,
        **split_kwargs,
    )

    if args.verbose:
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test loaders: {list(test_loaders.keys())}")

    # Get number of features from data (measurement + control)
    sample_batch = next(iter(train_loader))
    if hasattr(sample_batch, 'x'):
        # PyG format
        batch_size = sample_batch.num_graphs if hasattr(sample_batch, 'num_graphs') else 1
        n_measurement = sample_batch.x.shape[0] // batch_size
        window_size = sample_batch.x.shape[1]
        # Check for control variables
        n_control = 0
        if hasattr(sample_batch, 'c') and sample_batch.c is not None:
            c = sample_batch.c
            if c.dim() == 2:
                n_control = c.shape[0] // batch_size
            elif c.dim() == 3:
                n_control = c.shape[1]
        n_features = n_measurement + n_control
    else:
        n_features = sample_batch[0].shape[1]
        n_measurement = n_features
        n_control = 0
        window_size = sample_batch[0].shape[2]

    if args.verbose:
        print(f"Measurement vars: {n_measurement}, Control vars: {n_control}, "
              f"Total features: {n_features}, Window size: {window_size}")

    # Create model
    method_hyperparams = get_default_hyperparams(args.method)
    method_hyperparams.update(args.hyperparams)

    model = create_baseline(
        args.method,
        n_features=n_features,
        window_size=window_size,
        n_measurement_vars=n_measurement,
        **method_hyperparams,
    )

    if args.verbose:
        print(f"\nModel: {model.name}")
        print(f"Description: {get_baseline_description(args.method)}")
        info = model.get_model_info()
        print(f"Parameters: {info['n_parameters']:,}")

    # Load checkpoint if provided
    if args.checkpoint:
        if args.verbose:
            print(f"Loading checkpoint: {args.checkpoint}")
        model.load(args.checkpoint, device)

    # Training
    if not args.eval_only:
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Training {model.name}")
            print(f"{'='*50}")

        start_time = time.time()

        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            verbose=args.verbose,
        )

        train_time = time.time() - start_time
        if args.verbose:
            print(f"\nTraining completed in {train_time:.1f}s")
            print(f"Best validation loss: {model._train_stats.get('best_val_loss', 'N/A'):.6f}")

        # Save model
        if args.save_model:
            os.makedirs(os.path.dirname(args.save_model) or ".", exist_ok=True)
            model.save(args.save_model)
            if args.verbose:
                print(f"Model saved to: {args.save_model}")

    # Evaluation
    if not args.skip_test and test_loaders:
        if args.verbose:
            print(f"\n{'='*50}")
            print(f"Evaluation")
            print(f"{'='*50}")

        # Get baseline loader
        baseline_loader = test_loaders.get("baseline", val_loader)

        # Evaluate on fault datasets with unified IQR-normalized scoring
        all_metrics = evaluate_model(
            model, test_loaders, baseline_loader, val_loader,
            device, verbose=args.verbose
        )

        # Save metrics
        if args.checkpoint_dir:
            metrics_path = os.path.join(
                args.checkpoint_dir, f"{args.method}_metrics.csv"
            )
            save_metrics_csv(all_metrics, metrics_path, model.name)
            if args.verbose:
                print(f"\nMetrics saved to: {metrics_path}")

        # Print summary table
        if args.verbose and all_metrics:
            print(f"\n{'='*50}")
            print("Summary")
            print(f"{'='*50}")
            print(format_metrics_table({model.name: all_metrics.get("faults_all", {})}, "markdown"))

    if args.verbose:
        print("\nDone.")


if __name__ == "__main__":
    main()
