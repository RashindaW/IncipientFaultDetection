#!/usr/bin/env python3
"""Run baseline experiments in parallel on a specific GPU.

This script is designed to be run in parallel across multiple GPUs,
with each instance handling a specific dataset.

Usage:
    # Run ims-raw on GPU 0
    python run_baselines_parallel.py --cuda-device 0 --datasets ims-raw --output-dir results/baselines/20260126_071141

    # Run tep on GPU 1 with skip-existing
    python run_baselines_parallel.py --cuda-device 1 --datasets tep --skip-existing --output-dir results/baselines/20260126_071141
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

from baselines import list_baselines

# Dataset configurations - only the 4 target datasets (excluding ims and co2)
DATASET_CONFIGS = {
    "ims-raw": {
        "window_size": 1024,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "IMS Bearing raw accelerometer (20kHz)",
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
    "pronto": {
        "window_size": 60,
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "description": "PRONTO dataset",
    },
}

# Baseline methods to run (excludes dyedgegat)
BASELINE_METHODS = ["lstm_vae", "usad", "omnianomaly", "gdn", "mtad_gat"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline experiments on a specific GPU"
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        required=True,
        help="CUDA device index",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        help="Comma-separated datasets to run (from: ims-raw, tep, ashrae, pronto)",
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
        required=True,
        help="Output directory (use existing results dir for unified comparison)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments with existing checkpoints",
    )

    return parser.parse_args()


def run_experiment(
    method: str,
    dataset: str,
    config: dict,
    checkpoint_path: Path,
    dataset_dir: Path,
    cuda_device: int,
    seed: int,
) -> tuple:
    """Run a single experiment and return (success, elapsed_time, output)."""
    cmd = [
        sys.executable, "train_baselines.py",
        "--method", method,
        "--dataset-key", dataset,
        "--epochs", str(config["epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--window-size", str(config["window_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--seed", str(seed),
        "--cuda-device", str(cuda_device),
        "--save-model", str(checkpoint_path),
        "--checkpoint-dir", str(dataset_dir),
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    return result.returncode == 0, elapsed, result.stdout + result.stderr


def main():
    args = parse_args()

    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split(",")]
    for d in datasets:
        if d not in DATASET_CONFIGS:
            print(f"Error: Unknown dataset '{d}'. Valid: {list(DATASET_CONFIGS.keys())}")
            return 1

    # Parse methods
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = BASELINE_METHODS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PARALLEL BASELINE TRAINING - GPU {args.cuda_device}")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"CUDA Device: {args.cuda_device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 70)

    # Track results
    all_results = {}
    failed_experiments = []
    total_start = time.time()

    experiment_num = 0
    total_experiments = len(datasets) * len(methods)

    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]

        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        checkpoints_dir = dataset_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        print(f"\n{'='*70}")
        print(f"DATASET: {dataset}")
        print(f"  {config['description']}")
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

            # Run experiment
            success, elapsed, output = run_experiment(
                method=method,
                dataset=dataset,
                config=config,
                checkpoint_path=checkpoint_path,
                dataset_dir=dataset_dir,
                cuda_device=args.cuda_device,
                seed=args.seed,
            )

            if success:
                print(f"  Completed in {elapsed:.1f}s")
                dataset_results[method] = {
                    "status": "success",
                    "time": elapsed,
                    "checkpoint": str(checkpoint_path),
                    "metrics": str(metrics_path),
                }
            else:
                print(f"  FAILED")
                # Print last few lines of error
                if output:
                    error_lines = output.strip().split("\n")[-10:]
                    for line in error_lines:
                        print(f"    {line}")
                dataset_results[method] = {
                    "status": "failed",
                    "error": output[-1000:] if output else "",
                }
                failed_experiments.append(f"{dataset}/{method}")

        all_results[dataset] = dataset_results

    # Save results for this GPU
    results_path = output_dir / f"results_gpu{args.cuda_device}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"GPU {args.cuda_device} SUMMARY")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {total_experiments - len(failed_experiments)}/{total_experiments}")

    if failed_experiments:
        print(f"\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")

    print(f"\nResults saved: {results_path}")
    print("\nDone!")

    return 0 if not failed_experiments else 1


if __name__ == "__main__":
    sys.exit(main())
