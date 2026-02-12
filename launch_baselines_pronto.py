#!/usr/bin/env python3
"""GPU-aware parallel launcher for all baselines on PRONTO.

Distributes baseline experiments across specified GPUs, checking memory
before launching each one. Designed to run with nohup.

Usage:
    nohup python launch_baselines_pronto.py > baselines_pronto.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ──────────── Configuration ────────────

CUDA_DEVICES = [1, 2, 3]
RESERVE_MB = 3072  # Leave 3 GB free per GPU
ESTIMATED_MB = 4000  # Conservative estimate per baseline
POLL_INTERVAL = 30  # Seconds between GPU memory checks
SETTLE_TIME = 60  # Seconds to wait after launch for memory to settle

PYTHON = "/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"

METHODS = [
    "ae", "fnn", "lstm", "lstm_ae",
    "usad", "mtad_gat", "gdn", "grelen", "dyedgegat",
]

# PRONTO config matching DySTGAT baseline
PRONTO_CONFIG = {
    "dataset-key": "pronto",
    "window-size": 30,
    "epochs": 200,
    "batch-size": 64,
    "learning-rate": 1e-3,
    "weight-decay": 1e-5,
    "early-stopping": 20,
    "train-stride": 1,
    "val-stride": 1,
    "test-stride": 1,
    "split-mode": "segment_shuffle",
    "n-segments": 10,
    "train-segments": "2,3,4,6,7,8,9",
    "val-segments": "0,1",
    "test-segments": "5",
    "seed": 42,
}


def get_gpu_free_mb(device_id: int) -> int:
    """Query free GPU memory in MB via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", f"--id={device_id}"],
            capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def find_available_gpu(required_mb: int) -> int | None:
    """Find a GPU with enough free memory."""
    for dev in CUDA_DEVICES:
        free = get_gpu_free_mb(dev)
        if free >= required_mb + RESERVE_MB:
            return dev
    return None


def build_command(method: str, cuda_device: int, output_dir: Path) -> list[str]:
    """Build the train_baselines.py command for one method."""
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = method_dir / f"{method}_best.pt"

    cmd = [
        PYTHON, "train_baselines.py",
        "--method", method,
        "--cuda-device", str(cuda_device),
        "--save-model", str(checkpoint_path),
        "--checkpoint-dir", str(method_dir),
    ]
    for key, val in PRONTO_CONFIG.items():
        cmd.extend([f"--{key}", str(val)])
    return cmd


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/pronto/baselines_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "launch_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "methods": METHODS,
            "pronto_config": PRONTO_CONFIG,
            "cuda_devices": CUDA_DEVICES,
            "reserve_mb": RESERVE_MB,
            "timestamp": timestamp,
        }, f, indent=2)

    print("=" * 70)
    print("PRONTO BASELINE LAUNCHER")
    print("=" * 70)
    print(f"Timestamp   : {timestamp}")
    print(f"Methods     : {', '.join(METHODS)} ({len(METHODS)} total)")
    print(f"GPUs        : {CUDA_DEVICES}")
    print(f"Reserve     : {RESERVE_MB} MB per GPU")
    print(f"Output      : {output_dir}")
    print("=" * 70)
    sys.stdout.flush()

    # Track state
    pending = list(METHODS)
    running = {}   # method -> {proc, gpu, start}
    completed = {}  # method -> {gpu, elapsed, returncode}

    while pending or running:
        # ── Check for finished processes ──
        done = []
        for method, info in running.items():
            rc = info["proc"].poll()
            if rc is not None:
                elapsed = time.time() - info["start"]
                completed[method] = {
                    "gpu": info["gpu"],
                    "elapsed": elapsed,
                    "returncode": rc,
                }
                status = "OK" if rc == 0 else f"FAIL (rc={rc})"
                print(f"[{datetime.now():%H:%M:%S}] DONE  {method:12s} GPU {info['gpu']}  "
                      f"{elapsed/60:.1f}min  {status}")
                sys.stdout.flush()
                done.append(method)
        for m in done:
            del running[m]

        # ── Try to launch pending experiments ──
        launched = True
        while pending and launched:
            launched = False
            gpu = find_available_gpu(ESTIMATED_MB)
            if gpu is not None:
                method = pending.pop(0)
                cmd = build_command(method, gpu, output_dir)
                method_dir = output_dir / method

                stdout_f = open(method_dir / "stdout.log", "w")
                stderr_f = open(method_dir / "stderr.log", "w")
                proc = subprocess.Popen(
                    cmd, stdout=stdout_f, stderr=stderr_f,
                    cwd=str(Path(__file__).parent),
                )
                running[method] = {
                    "proc": proc,
                    "gpu": gpu,
                    "start": time.time(),
                    "stdout_f": stdout_f,
                    "stderr_f": stderr_f,
                }
                print(f"[{datetime.now():%H:%M:%S}] START {method:12s} GPU {gpu}  "
                      f"PID {proc.pid}  "
                      f"(pending={len(pending)} running={len(running)} done={len(completed)})")
                sys.stdout.flush()
                launched = True
                time.sleep(SETTLE_TIME)  # Let memory settle

        # ── Wait before next check ──
        if pending or running:
            time.sleep(POLL_INTERVAL)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_time = sum(c["elapsed"] for c in completed.values())
    n_ok = sum(1 for c in completed.values() if c["returncode"] == 0)
    n_fail = len(completed) - n_ok

    for method in METHODS:
        c = completed.get(method, {})
        rc = c.get("returncode", -1)
        elapsed = c.get("elapsed", 0)
        gpu = c.get("gpu", "?")
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  {method:12s}  GPU {gpu}  {elapsed/60:6.1f}min  {status}")

    print(f"\nTotal: {n_ok} succeeded, {n_fail} failed")
    print(f"Wall time captured: {total_time/60:.1f} min (sum of all)")
    print(f"Results in: {output_dir}")

    # Save summary JSON
    summary = {method: completed.get(method, {}) for method in METHODS}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Close file handles
    for info in running.values():
        info["stdout_f"].close()
        info["stderr_f"].close()

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
