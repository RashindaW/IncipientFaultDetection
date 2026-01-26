#!/usr/bin/env python3
"""Run all baseline experiments across all datasets.

This script trains all baseline methods on all available datasets and
saves results in a structured format for later analysis.

Usage:
    # Run all experiments
    python run_all_experiments.py --cuda-device 3

    # Run specific datasets
    python run_all_experiments.py --cuda-device 3 --datasets ims-raw,tep

    # Run specific methods
    python run_all_experiments.py --cuda-device 3 --methods lstm_vae,gdn
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from baselines import list_baselines, get_baseline_description

# Dataset configurations with reasonable defaults
DATASET_CONFIGS = {
    "ims-raw": {
        "window_size": 1024,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "IMS Bearing raw accelerometer (20kHz)",
    },
    "ims": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "IMS Bearing feature-engineered",
    },
    "tep": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "description": "Tennessee Eastman Process",
    },
    "ashrae": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "ASHRAE RP-1043 Chiller",
    },
    "co2": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "CO2 Refrigeration",
    },
    "pronto": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "PRONTO dataset",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all baseline experiments across datasets"
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=3,
        help="CUDA device index (default: 3)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated datasets (default: all configured)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods (default: all baselines)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baselines",
        help="Base output directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments with existing checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--include-dystgat",
        action="store_true",
        help="Also train DySTGAT variants",
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="rashindaNew-torch-env",
        help="Conda environment",
    )

    return parser.parse_args()


def run_command(cmd: list, dry_run: bool = False, conda_env: str = None) -> tuple:
    """Run a command and return (exit_code, output)."""
    if conda_env:
        cmd_str = " ".join(cmd)
        full_cmd = f"source /opt/anaconda3/etc/profile.d/conda.sh && conda activate {conda_env} && {cmd_str}"
        cmd = ["bash", "-c", full_cmd]

    if dry_run:
        print(f"    [DRY RUN] Would run: {cmd_str if conda_env else ' '.join(cmd)}")
        return 0, ""

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def save_experiment_config(output_dir: Path, args, datasets: list, methods: list):
    """Save experiment configuration for reproducibility."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "cuda_device": args.cuda_device,
        "seed": args.seed,
        "datasets": datasets,
        "methods": methods,
        "dataset_configs": {d: DATASET_CONFIGS[d] for d in datasets if d in DATASET_CONFIGS},
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config_path


def save_results_summary(output_dir: Path, results: dict):
    """Save results summary as JSON."""
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    return summary_path


def main():
    args = parse_args()

    # Determine datasets and methods
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = list(DATASET_CONFIGS.keys())

    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = list_baselines()
        # Remove dyedgegat as it requires DySTGAT setup
        if "dyedgegat" in methods:
            methods.remove("dyedgegat")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BASELINE EXPERIMENTS")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Methods ({len(methods)}): {', '.join(methods)}")
    print(f"Total experiments: {len(datasets) * len(methods)}")
    print("=" * 70)

    # Save experiment config
    if not args.dry_run:
        config_path = save_experiment_config(output_dir, args, datasets, methods)
        print(f"Config saved: {config_path}")

    # Track all results
    all_results = {}
    failed_experiments = []
    total_start = time.time()

    experiment_num = 0
    total_experiments = len(datasets) * len(methods)

    for dataset in datasets:
        if dataset not in DATASET_CONFIGS:
            print(f"\nWarning: No config for {dataset}, using defaults")
            config = {"window_size": 60, "epochs": 100, "batch_size": 32, "learning_rate": 1e-3}
        else:
            config = DATASET_CONFIGS[dataset]

        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        checkpoints_dir = dataset_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset}")
        print(f"  {config.get('description', 'No description')}")
        print(f"  Window: {config['window_size']}, Epochs: {config['epochs']}, Batch: {config['batch_size']}")
        print("=" * 70)

        dataset_results = {}

        for method in methods:
            experiment_num += 1
            print(f"\n[{experiment_num}/{total_experiments}] {dataset} / {method}")
            print("-" * 50)

            checkpoint_path = checkpoints_dir / f"{method}_best.pt"
            metrics_path = dataset_dir / f"{method}_metrics.csv"

            # Skip if exists
            if args.skip_existing and checkpoint_path.exists():
                print(f"  Skipping (exists): {checkpoint_path}")
                dataset_results[method] = {"status": "skipped", "checkpoint": str(checkpoint_path)}
                continue

            # Build command
            cmd = [
                "python", "train_baselines.py",
                "--method", method,
                "--dataset-key", dataset,
                "--epochs", str(config["epochs"]),
                "--batch-size", str(config["batch_size"]),
                "--window-size", str(config["window_size"]),
                "--learning-rate", str(config["learning_rate"]),
                "--seed", str(args.seed),
                "--cuda-device", str(args.cuda_device),
                "--save-model", str(checkpoint_path),
                "--checkpoint-dir", str(dataset_dir),
            ]

            start_time = time.time()
            exit_code, output = run_command(cmd, args.dry_run, args.conda_env)
            elapsed = time.time() - start_time

            if exit_code == 0:
                print(f"  ✓ Completed in {elapsed:.1f}s")
                dataset_results[method] = {
                    "status": "success",
                    "time": elapsed,
                    "checkpoint": str(checkpoint_path),
                    "metrics": str(metrics_path),
                }
            else:
                print(f"  ✗ FAILED (exit code {exit_code})")
                if output:
                    # Print last few lines of error
                    error_lines = output.strip().split("\n")[-5:]
                    for line in error_lines:
                        print(f"    {line}")
                dataset_results[method] = {
                    "status": "failed",
                    "exit_code": exit_code,
                    "error": output[-500:] if output else "",
                }
                failed_experiments.append(f"{dataset}/{method}")

        all_results[dataset] = dataset_results

        # Generate comparison table for this dataset
        if not args.dry_run:
            print(f"\n  Generating comparison table for {dataset}...")
            cmd = [
                "python", "compare_baselines.py",
                "--results-dir", str(dataset_dir),
                "--format", "all",
            ]
            run_command(cmd, args.dry_run, args.conda_env)

    # Optionally run DySTGAT
    if args.include_dystgat:
        print(f"\n{'='*70}")
        print("RUNNING DYSTGAT VARIANTS")
        print("=" * 70)

        for dataset in datasets:
            config = DATASET_CONFIGS.get(dataset, {"window_size": 60, "epochs": 100, "batch_size": 32, "learning_rate": 1e-3})
            dataset_dir = output_dir / dataset
            checkpoints_dir = dataset_dir / "checkpoints"

            # DyEdgeGAT (temporal-only)
            print(f"\n  {dataset} / DyEdgeGAT (temporal-only)...")
            cmd = [
                "python", "train_dystgat.py",
                "--dataset-key", dataset,
                "--epochs", str(config["epochs"]),
                "--batch-size", str(config["batch_size"]),
                "--window-size", str(config["window_size"]),
                "--learning-rate", str(config["learning_rate"]),
                "--seed", str(args.seed),
                "--cuda-device", str(args.cuda_device),
                "--save-model", str(checkpoints_dir / "dyedgegat_best.pt"),
                "--checkpoint-dir", str(dataset_dir / "dyedgegat"),
            ]
            run_command(cmd, args.dry_run, args.conda_env)

            # DySTGAT (full spectral)
            print(f"  {dataset} / DySTGAT (spectral)...")
            cmd = [
                "python", "train_dystgat.py",
                "--dataset-key", dataset,
                "--epochs", str(config["epochs"]),
                "--batch-size", str(config["batch_size"]),
                "--window-size", str(config["window_size"]),
                "--learning-rate", str(config["learning_rate"]),
                "--seed", str(args.seed),
                "--cuda-device", str(args.cuda_device),
                "--use-spectral-view",
                "--freq-embed-dim", "32",
                "--lambda-div", "0.5",
                "--save-model", str(checkpoints_dir / "dystgat_best.pt"),
                "--checkpoint-dir", str(dataset_dir / "dystgat"),
            ]
            run_command(cmd, args.dry_run, args.conda_env)

    # Save results summary
    if not args.dry_run:
        summary_path = save_results_summary(output_dir, all_results)
        print(f"\nResults summary saved: {summary_path}")

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    print(f"Successful: {total_experiments - len(failed_experiments)}/{total_experiments}")

    if failed_experiments:
        print(f"\nFailed experiments ({len(failed_experiments)}):")
        for exp in failed_experiments:
            print(f"  ✗ {exp}")

    print(f"\nResults structure:")
    print(f"  {output_dir}/")
    print(f"  ├── experiment_config.json")
    print(f"  ├── results_summary.json")
    for dataset in datasets:
        print(f"  ├── {dataset}/")
        print(f"  │   ├── checkpoints/")
        print(f"  │   │   ├── lstm_vae_best.pt")
        print(f"  │   │   └── ...")
        print(f"  │   ├── lstm_vae_metrics.csv")
        print(f"  │   ├── comparison.md")
        print(f"  │   └── comparison.tex")

    print("\nDone!")
    return 0 if not failed_experiments else 1


if __name__ == "__main__":
    sys.exit(main())
