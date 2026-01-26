#!/usr/bin/env python3
"""Run all baseline methods sequentially for comparison.

This script trains all baseline methods on the specified dataset and
generates a comparison table at the end.

Usage:
    # Run all baselines on IMS-raw with default settings
    python run_all_baselines.py --dataset-key ims-raw --cuda-device 0

    # Run specific baselines
    python run_all_baselines.py --dataset-key ims-raw --methods lstm_vae,usad,gdn

    # Run with custom epochs
    python run_all_baselines.py --dataset-key ims-raw --epochs 200 --cuda-device 0
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from baselines import list_baselines, get_baseline_description


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all baseline methods for DySTGAT comparison"
    )

    parser.add_argument(
        "--dataset-key",
        type=str,
        required=True,
        help="Dataset to use (e.g., ims-raw)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to run (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per method",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1024,
        help="Temporal window size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="CUDA device index (None for CPU)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/<dataset-key>/<timestamp>)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip methods that already have saved checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--include-dystgat",
        action="store_true",
        help="Also train DySTGAT (temporal-only and full spectral)",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="rashindaNew-torch-env",
        help="Conda environment to use",
    )

    return parser.parse_args()


def run_command(cmd: list, dry_run: bool = False, conda_env: str = None) -> int:
    """Run a command and return exit code."""
    if conda_env:
        # Wrap command with conda activation
        cmd_str = " ".join(cmd)
        full_cmd = f"source /opt/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {cmd_str}"
        cmd = ["bash", "-c", full_cmd]

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0

    print(f"  Running: {' '.join(cmd[:10])}...")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    """Main function to run all baselines."""
    args = parse_args()

    # Determine which methods to run
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = list_baselines()
        # Remove dyedgegat from default list since it requires DySTGAT
        if "dyedgegat" in methods:
            methods.remove("dyedgegat")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/{args.dataset_key}/{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Running All Baselines")
    print("=" * 60)
    print(f"Dataset: {args.dataset_key}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Epochs: {args.epochs}")
    print(f"Output: {output_dir}")
    print(f"Device: {'cuda:' + str(args.cuda_device) if args.cuda_device is not None else 'cpu'}")
    print("=" * 60)

    # Track results
    results = {}
    failed = []

    # Run each baseline
    for i, method in enumerate(methods, 1):
        print(f"\n[{i}/{len(methods)}] Training {method.upper()}")
        print("-" * 40)
        print(f"Description: {get_baseline_description(method)}")

        checkpoint_path = checkpoints_dir / f"{method}_best.pt"
        metrics_path = output_dir / f"{method}_metrics.csv"

        # Skip if already exists
        if args.skip_existing and checkpoint_path.exists():
            print(f"  Skipping (checkpoint exists): {checkpoint_path}")
            continue

        # Build command
        cmd = [
            "python", "train_baselines.py",
            "--method", method,
            "--dataset-key", args.dataset_key,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--window-size", str(args.window_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--save-model", str(checkpoint_path),
            "--checkpoint-dir", str(output_dir),
        ]

        if args.cuda_device is not None:
            cmd.extend(["--cuda-device", str(args.cuda_device)])
        else:
            cmd.extend(["--device", "cpu"])

        # Run training
        start_time = time.time()
        exit_code = run_command(cmd, args.dry_run, args.conda_env)
        elapsed = time.time() - start_time

        if exit_code == 0:
            results[method] = {"status": "success", "time": elapsed}
            print(f"  Completed in {elapsed:.1f}s")
        else:
            results[method] = {"status": "failed", "exit_code": exit_code}
            failed.append(method)
            print(f"  FAILED with exit code {exit_code}")

    # Optionally run DySTGAT variants
    if args.include_dystgat:
        print(f"\n[Extra] Training DySTGAT variants")
        print("-" * 40)

        # DyEdgeGAT (temporal-only)
        print("\nTraining DyEdgeGAT (temporal-only DySTGAT)...")
        dyedgegat_checkpoint = checkpoints_dir / "dyedgegat_best.pt"
        cmd = [
            "python", "train_dystgat.py",
            "--dataset-key", args.dataset_key,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--window-size", str(args.window_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--save-model", str(dyedgegat_checkpoint),
            "--checkpoint-dir", str(output_dir / "dyedgegat"),
        ]
        if args.cuda_device is not None:
            cmd.extend(["--cuda-device", str(args.cuda_device)])
        run_command(cmd, args.dry_run, args.conda_env)

        # DySTGAT (full with spectral)
        print("\nTraining DySTGAT (full with spectral view)...")
        dystgat_checkpoint = checkpoints_dir / "dystgat_best.pt"
        cmd = [
            "python", "train_dystgat.py",
            "--dataset-key", args.dataset_key,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--window-size", str(args.window_size),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(args.seed),
            "--use-spectral-view",
            "--freq-embed-dim", "32",
            "--lambda-div", "0.5",
            "--save-model", str(dystgat_checkpoint),
            "--checkpoint-dir", str(output_dir / "dystgat"),
        ]
        if args.cuda_device is not None:
            cmd.extend(["--cuda-device", str(args.cuda_device)])
        run_command(cmd, args.dry_run, args.conda_env)

    # Generate comparison table
    if not args.dry_run and not failed:
        print(f"\n{'=' * 60}")
        print("Generating Comparison Table")
        print("=" * 60)

        cmd = [
            "python", "compare_baselines.py",
            "--results-dir", str(output_dir),
            "--format", "all",
        ]
        run_command(cmd, args.dry_run, args.conda_env)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Successful: {len(results) - len(failed)}/{len(methods)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    for method, info in results.items():
        status = "✓" if info["status"] == "success" else "✗"
        time_str = f"{info.get('time', 0):.1f}s" if "time" in info else "N/A"
        print(f"  {status} {method}: {time_str}")

    print("\nDone.")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
