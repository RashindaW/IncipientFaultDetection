#!/usr/bin/env python3
"""
Batch script to test all 6 fault datasets and generate anomaly score plots.
This helps visualize incipient faults across different failure modes.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
CHECKPOINT = "checkpoints/dyedgegat_stride10.pt"
OUTPUT_DIR = "outputs/anomaly_scores"
STRIDE = 1
BATCH_SIZE = 64
NUM_WORKERS = 4

from dyedgegat.src.data.column_config import FAULT_FILES

# All fault datasets (auto-discovered from column config)
FAULTS = sorted(FAULT_FILES.keys())


def main():
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT):
        print(f"Error: Checkpoint not found at {CHECKPOINT}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 50)
    print("Testing Model on All Fault Datasets")
    print("=" * 50)
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Number of Faults: {len(FAULTS)}")
    print("=" * 50)
    print()
    
    base_env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" not in base_env or not base_env["CUDA_VISIBLE_DEVICES"]:
        base_env["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    
    # Process each fault dataset
    for i, fault in enumerate(FAULTS, 1):
        print(f"[{i}/{len(FAULTS)}] Processing: {fault}")
        print("-" * 50)
        
        cmd = [
            "python", "plot_anomaly_scores.py",
            "--checkpoint", CHECKPOINT,
            "--dataset", fault,
            "--stride", str(STRIDE),
            "--batch-size", str(BATCH_SIZE),
            "--num-workers", str(NUM_WORKERS),
            "--output-dir", OUTPUT_DIR,
            "--line-plot",
            "--use-amp",
        ]
        
        try:
            subprocess.run(cmd, check=True, env=base_env)
            print(f"\n✓ Completed: {fault}\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed: {fault}")
            print(f"Error: {e}")
            print("Continuing with remaining faults...\n")
    
    print("=" * 50)
    print("All fault tests completed!")
    print("=" * 50)
    print()
    print(f"Generated files in {OUTPUT_DIR}:")
    
    # List generated files
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        files = sorted(output_path.glob("*"))
        for f in files:
            size = f.stat().st_size / 1024  # Size in KB
            print(f"  {f.name} ({size:.1f} KB)")
    
    print()
    print("Files generated per fault:")
    print("  - <fault>_anomaly_scores.csv      : Per-timestep scores")
    print("  - <fault>_timestamp_mean.csv      : Timestamp-averaged scores")
    print("  - <fault>_heatmap.png             : Heatmap visualization")
    print("  - <fault>_line.png                : Line plot (temporal progression)")


if __name__ == "__main__":
    main()
