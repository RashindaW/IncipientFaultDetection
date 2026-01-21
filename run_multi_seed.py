#!/usr/bin/env python3
"""
Multi-Seed Training Runner for DySTGAT

Runs training for multiple seeds and aggregates results with mean ± std
for reproducible research paper results.

Usage:
    python run_multi_seed.py --dataset-key pronto --seeds 42,123,456,789,1024 --epochs 20
    python run_multi_seed.py --dataset-key tep --seeds 42,123,456 --epochs 50 --use-spectral-view
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import aggregation utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dystgat"))
from src.utils.aggregate_results import (
    load_run_results,
    compute_statistics,
    format_markdown_table,
    format_latex_table,
    save_aggregate_results,
)


DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DySTGAT training across multiple seeds with result aggregation"
    )

    # Multi-seed specific arguments
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(map(str, DEFAULT_SEEDS)),
        help=f"Comma-separated list of random seeds (default: {','.join(map(str, DEFAULT_SEEDS))})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for multi-seed run (default: results/{dataset}/multi_seed_{timestamp})",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only aggregate existing results",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step (only train)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    # Pass-through arguments for train_dystgat.py
    parser.add_argument("--dataset-key", type=str, required=True, help="Dataset adapter to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--window-size", type=int, default=60, help="Temporal window size")
    parser.add_argument("--train-stride", type=int, default=1, help="Training stride")
    parser.add_argument("--val-stride", type=int, default=5, help="Validation stride")
    parser.add_argument("--test-stride", type=int, default=None, help="Test stride")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--anomaly-weight", type=float, default=0.0, help="Anomaly score weight")
    parser.add_argument("--use-spectral-view", action="store_true", help="Enable spectral view")
    parser.add_argument("--freq-embed-dim", type=int, default=16, help="Spectral embedding dim")
    parser.add_argument("--freq-band-mix", type=str, default="none", help="Band mixing mode")
    parser.add_argument("--freq-use-log", action="store_true", help="Use log-magnitude")
    parser.add_argument("--freq-use-spectral-features", action="store_true", help="Use spectral features")
    parser.add_argument("--lambda-div", type=float, default=0.0, help="Divergence loss weight")
    parser.add_argument("--fuse-mode", type=str, default="concat", help="Fusion mode")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--task", type=str, default="reconstruction", help="Task type")
    parser.add_argument("--pred-horizon", type=int, default=0, help="Prediction horizon")
    parser.add_argument("--severity-range", type=str, default=None, help="Severity range for TEP")
    parser.add_argument("--baseline-from", type=str, default="test", help="Baseline source")

    return parser.parse_args()


def build_train_command(
    seed: int,
    checkpoint_path: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    """Build the training command for a single seed."""
    cmd = [
        sys.executable,
        "train_dystgat.py",
        "--dataset-key", args.dataset_key,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--window-size", str(args.window_size),
        "--train-stride", str(args.train_stride),
        "--val-stride", str(args.val_stride),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--anomaly-weight", str(args.anomaly_weight),
        "--lambda-div", str(args.lambda_div),
        "--fuse-mode", args.fuse_mode,
        "--device", args.device,
        "--num-workers", str(args.num_workers),
        "--task", args.task,
        "--pred-horizon", str(args.pred_horizon),
        "--baseline-from", args.baseline_from,
        "--save-model", str(checkpoint_path),
        "--checkpoint-dir", str(checkpoint_dir),
    ]

    if args.test_stride is not None:
        cmd.extend(["--test-stride", str(args.test_stride)])

    if args.use_spectral_view:
        cmd.append("--use-spectral-view")
        cmd.extend(["--freq-embed-dim", str(args.freq_embed_dim)])
        cmd.extend(["--freq-band-mix", args.freq_band_mix])
        if args.freq_use_log:
            cmd.append("--freq-use-log")
        if args.freq_use_spectral_features:
            cmd.append("--freq-use-spectral-features")

    if args.use_amp:
        cmd.append("--use-amp")

    if args.cuda_device is not None:
        cmd.extend(["--cuda-device", str(args.cuda_device)])

    if args.severity_range:
        cmd.extend(["--severity-range", args.severity_range])

    return cmd


def run_training(
    seed: int,
    seed_dir: Path,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> bool:
    """Run training for a single seed."""
    checkpoint_path = seed_dir / f"dystgat_seed{seed}.pt"
    checkpoint_dir = seed_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_train_command(seed, checkpoint_path, checkpoint_dir, args)

    print(f"\n{'='*60}")
    print(f"Training seed {seed}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd[:10])}...")

    if dry_run:
        print("[DRY RUN] Would execute command")
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed for seed {seed}: {e}")
        return False


def collect_results_from_seed_dir(seed_dir: Path, seed: int) -> Optional[Dict[str, Any]]:
    """Collect results from a seed directory's checkpoint run."""
    # Find the most recent run directory in checkpoints
    checkpoint_dir = seed_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    # Look for run directories (format: YYYYMMDD_HHMMSS)
    run_dirs = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
        reverse=True,
    )

    if not run_dirs:
        return None

    latest_run = run_dirs[0]

    # Look for detailed_test_metrics.csv in plots/
    plots_dir = latest_run / "plots"
    metrics_path = plots_dir / "detailed_test_metrics.csv"

    if not metrics_path.exists():
        # Try looking for metrics.csv in the run directory directly
        metrics_path = latest_run / "metrics.csv"
        if not metrics_path.exists():
            print(f"Warning: No metrics found for seed {seed} in {latest_run}")
            return None

    # Parse the CSV file
    results = {"seed": seed, "run_dir": str(latest_run)}

    try:
        import csv
        with open(metrics_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                return None

            # Store all test set results
            results["test_sets"] = {}
            for row in rows:
                test_name = row.get("test_set", "unknown")
                results["test_sets"][test_name] = {
                    k: float(v) if v and k != "test_set" else v
                    for k, v in row.items()
                    if v and k != "test_set"
                }

            # Also look for epoch metrics
            epoch_metrics_path = latest_run / "metrics.csv"
            if epoch_metrics_path.exists():
                with open(epoch_metrics_path, "r") as ef:
                    epoch_reader = csv.DictReader(ef)
                    epoch_rows = list(epoch_reader)
                    if epoch_rows:
                        # Get final epoch metrics
                        final_epoch = epoch_rows[-1]
                        results["final_epoch"] = {
                            k: float(v) if v else v
                            for k, v in final_epoch.items()
                            if v
                        }
                        # Get best validation metrics
                        best_val_loss = min(
                            float(r.get("val_loss", float("inf")))
                            for r in epoch_rows
                            if r.get("val_loss")
                        )
                        results["best_val_loss"] = best_val_loss

    except Exception as e:
        print(f"Error parsing results for seed {seed}: {e}")
        return None

    return results


def aggregate_multi_seed_results(
    output_dir: Path,
    seeds: List[int],
) -> Dict[str, Any]:
    """Aggregate results from all seed runs."""
    all_results = []

    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        result = collect_results_from_seed_dir(seed_dir, seed)
        if result:
            all_results.append(result)

    if not all_results:
        return {"error": "No results found", "n_seeds": 0}

    # Compute aggregate statistics
    aggregate = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "timestamp": datetime.now().isoformat(),
    }

    # Aggregate test set metrics
    test_set_names = set()
    for r in all_results:
        if "test_sets" in r:
            test_set_names.update(r["test_sets"].keys())

    aggregate["metrics"] = {}
    for test_name in test_set_names:
        aggregate["metrics"][test_name] = {}

        # Collect all metric names for this test set
        metric_names = set()
        for r in all_results:
            if "test_sets" in r and test_name in r["test_sets"]:
                metric_names.update(r["test_sets"][test_name].keys())

        for metric_name in metric_names:
            values = []
            for r in all_results:
                if "test_sets" in r and test_name in r["test_sets"]:
                    val = r["test_sets"][test_name].get(metric_name)
                    if val is not None and isinstance(val, (int, float)):
                        values.append(val)

            if values:
                aggregate["metrics"][test_name][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values,
                }

    # Aggregate validation metrics
    val_losses = [r.get("best_val_loss") for r in all_results if r.get("best_val_loss") is not None]
    if val_losses:
        aggregate["validation"] = {
            "best_loss": {
                "mean": float(np.mean(val_losses)),
                "std": float(np.std(val_losses)),
                "values": val_losses,
            }
        }

    return aggregate


def main() -> None:
    args = parse_args()

    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        print("Error: No valid seeds provided")
        sys.exit(1)

    print(f"Multi-Seed Training Runner")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_key}")
    print(f"Seeds: {seeds}")
    print(f"Epochs: {args.epochs}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / args.dataset_key / f"multi_seed_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save run configuration
    config = {
        "seeds": seeds,
        "dataset_key": args.dataset_key,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "use_spectral_view": args.use_spectral_view,
        "anomaly_weight": args.anomaly_weight,
        "lambda_div": args.lambda_div,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run training for each seed
    if not args.skip_training:
        successful_seeds = []
        failed_seeds = []

        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            success = run_training(seed, seed_dir, args, dry_run=args.dry_run)
            if success:
                successful_seeds.append(seed)
            else:
                failed_seeds.append(seed)

        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Successful: {len(successful_seeds)}/{len(seeds)} - {successful_seeds}")
        if failed_seeds:
            print(f"Failed: {len(failed_seeds)}/{len(seeds)} - {failed_seeds}")

    # Aggregate results
    if not args.dry_run:
        print(f"\n{'='*60}")
        print("Aggregating Results")
        print(f"{'='*60}")

        aggregate = aggregate_multi_seed_results(output_dir, seeds)

        # Save aggregate results
        aggregate_path = output_dir / "aggregate_results.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        print(f"Saved: {aggregate_path}")

        # Generate summary tables
        if aggregate.get("metrics"):
            # Save markdown summary
            md_table = format_markdown_table(aggregate)
            md_path = output_dir / "aggregate_summary.md"
            with open(md_path, "w") as f:
                f.write(f"# Multi-Seed Results: {args.dataset_key}\n\n")
                f.write(f"Seeds: {aggregate['seeds']}\n\n")
                f.write(md_table)
            print(f"Saved: {md_path}")

            # Save LaTeX table
            latex_table = format_latex_table(aggregate)
            latex_path = output_dir / "aggregate_summary.tex"
            with open(latex_path, "w") as f:
                f.write(latex_table)
            print(f"Saved: {latex_path}")

            # Print summary to console
            print(f"\n{'='*60}")
            print("Results Summary (mean ± std)")
            print(f"{'='*60}")
            print(md_table)
        else:
            print("Warning: No metrics found to aggregate")

    print(f"\nMulti-seed run complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
