#!/usr/bin/env python3
"""
Hyperparameter Search & Ablation Study Script for DySTGAT

Runs comprehensive experiments covering:
- Reference configurations (original and balanced)
- Sensitivity analysis (window size, anomaly weight, lambda, LR, embed dim, batch, decay, epochs)
- Ablation studies (spectral components, band mixing, fusion modes)
- Alternative configurations (divergence type, prediction task)
- Optimized combinations

Usage:
    # Run all 40 experiments for ASHRAE
    python run_hpsearch.py --dataset-key ashrae --cuda-device 0

    # Run specific experiments by number
    python run_hpsearch.py --dataset-key ashrae --experiments 1,2,34,40

    # Run experiments by name
    python run_hpsearch.py --dataset-key ashrae --experiments original,balanced,combo_best_guess

    # Run only sensitivity analysis experiments
    python run_hpsearch.py --dataset-key ashrae --category sensitivity

    # Dry run (print commands without executing)
    python run_hpsearch.py --dataset-key ashrae --dry-run

    # Only aggregate existing results
    python run_hpsearch.py --dataset-key ashrae --aggregate-only --results-dir results/ashrae/hpsearch_20260120_123456

    # Resume from specific experiment
    python run_hpsearch.py --dataset-key ashrae --resume --start-from 15
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Conda environment to use for running experiments
CONDA_ENV = "rashindaNew-torch-env"
CONDA_PYTHON = "/home/rashinda/.conda/envs/rashindaNew-torch-env/bin/python"


# =============================================================================
# Dataset-Specific Base Configurations
# =============================================================================

# ASHRAE Base configuration (balanced baseline)
ASHRAE_BASE_CONFIG = {
    # Dataset
    "dataset-key": "ashrae",
    "ashrae-feature-option": "a",
    # Training
    "epochs": 300,
    "batch-size": 128,
    "learning-rate": 3e-4,
    "weight-decay": 1e-5,
    # Window
    "window-size": 180,
    "train-stride": 4,
    "val-stride": 8,
    # Spectral
    "use-spectral-view": True,
    "freq-embed-dim": 24,
    "freq-band-mix": "mlp",
    "freq-use-log": True,
    "freq-use-spectral-features": True,
    "fuse-mode": "concat",
    "divergence-type": "js",
    # Loss
    "anomaly-weight": 1.0,
    "lambda-div": 0.2,
    # Task
    "task": "reconstruction",
    # AMP
    "use-amp": False,
}

# ASHRAE User's original configuration for reference
ASHRAE_ORIGINAL_CONFIG = {
    "window-size": 300,
    "train-stride": 8,
    "val-stride": 16,
    "epochs": 500,
    "batch-size": 256,
    "learning-rate": 1e-4,
    "anomaly-weight": 0.5,
    "lambda-div": 0.1,
    "freq-embed-dim": 16,
}

# TEP Base configuration (balanced baseline)
TEP_BASE_CONFIG = {
    # Dataset
    "dataset-key": "tep",
    "data-dir": "data/tep/raw",
    # Training
    "epochs": 150,
    "batch-size": 128,
    "learning-rate": 3e-4,
    "weight-decay": 1e-5,
    # Window (smaller for TEP's faster dynamics - 3s sampling)
    "window-size": 48,
    "train-stride": 4,
    "val-stride": 8,
    # Spectral
    "use-spectral-view": True,
    "freq-embed-dim": 24,
    "freq-band-mix": "mlp",
    "freq-use-log": True,
    "freq-use-spectral-features": True,
    "fuse-mode": "concat",
    "divergence-type": "js",
    # Loss
    "anomaly-weight": 1.0,
    "lambda-div": 0.2,
    # Task
    "task": "reconstruction",
    # AMP (TEP benefits from AMP)
    "use-amp": True,
}

# TEP User's original configuration for reference
TEP_ORIGINAL_CONFIG = {
    "window-size": 64,
    "train-stride": 8,
    "val-stride": 16,
    "epochs": 100,
    "batch-size": 256,
    "learning-rate": 1e-3,
    "anomaly-weight": 0.5,
    "lambda-div": 0.1,
    "freq-embed-dim": 16,
}

# IMS Bearing Dataset Base configuration (balanced baseline)
IMS_BASE_CONFIG = {
    # Dataset
    "dataset-key": "ims",
    # Training
    "epochs": 300,
    "batch-size": 32,
    "learning-rate": 3e-4,
    "weight-decay": 1e-5,
    # Window (small for high-frequency vibration data)
    "window-size": 15,
    "train-stride": 1,
    "val-stride": 1,
    "test-stride": 1,
    # Spectral
    "use-spectral-view": True,
    "freq-embed-dim": 24,
    "freq-band-mix": "mlp",
    "freq-use-log": True,
    "freq-use-spectral-features": True,
    "fuse-mode": "concat",
    "divergence-type": "js",
    # Loss
    "anomaly-weight": 1.0,
    "lambda-div": 0.2,
    # Task
    "task": "reconstruction",
    # AMP
    "use-amp": True,
}

# IMS User's original configuration for reference
IMS_ORIGINAL_CONFIG = {
    "window-size": 15,
    "train-stride": 1,
    "val-stride": 1,
    "test-stride": 1,
    "epochs": 500,
    "batch-size": 32,
    "learning-rate": 1e-4,
    "anomaly-weight": 0.5,
    "lambda-div": 0.1,
    "freq-embed-dim": 16,
}

# PRONTO Dataset Base configuration (balanced baseline)
PRONTO_BASE_CONFIG = {
    # Dataset
    "dataset-key": "pronto",
    # Training
    "epochs": 100,
    "batch-size": 64,
    "learning-rate": 3e-4,
    "weight-decay": 1e-5,
    # Window (small for process data)
    "window-size": 15,
    "train-stride": 1,
    "val-stride": 5,
    "test-stride": 1,
    # Spectral
    "use-spectral-view": True,
    "freq-embed-dim": 24,
    "freq-band-mix": "mlp",
    "freq-use-log": True,
    "freq-use-spectral-features": True,
    "fuse-mode": "concat",
    "divergence-type": "js",
    # Loss
    "anomaly-weight": 1.0,
    "lambda-div": 0.2,
    # Task
    "task": "reconstruction",
    # AMP
    "use-amp": True,
}

# PRONTO User's original configuration for reference
PRONTO_ORIGINAL_CONFIG = {
    "window-size": 15,
    "train-stride": 1,
    "val-stride": 5,
    "test-stride": 1,
    "epochs": 100,
    "batch-size": 64,
    "learning-rate": 1e-3,
    "anomaly-weight": 0.5,
    "lambda-div": 0.1,
    "freq-embed-dim": 16,
}


# Dataset parameter presets for experiment generation
DATASET_PARAMS = {
    "ashrae": {
        "window_small": 60,
        "window_base": 180,
        "window_large": 300,
        "window_xlarge": 450,
        "epochs_short": 150,
        "epochs_base": 300,
        "epochs_long": 500,
        "stride_dense": (2, 4),
        "stride_normal": (4, 8),
        "stride_sparse": (8, 16),
        "stride_very_sparse": (12, 24),
        "combo_fast_window": 100,
        "combo_patient_window": 350,
        "combo_regularized_window": 150,
        "combo_aggressive_window": 120,
        "combo_dense_window": 180,
        "combo_minimal_window": 200,
        "combo_best_window": 150,
    },
    "tep": {
        "window_small": 32,
        "window_base": 48,
        "window_large": 64,
        "window_xlarge": 96,
        "epochs_short": 75,
        "epochs_base": 150,
        "epochs_long": 200,
        "stride_dense": (2, 4),
        "stride_normal": (4, 8),
        "stride_sparse": (8, 16),
        "stride_very_sparse": (12, 24),
        "combo_fast_window": 32,
        "combo_patient_window": 80,
        "combo_regularized_window": 40,
        "combo_aggressive_window": 36,
        "combo_dense_window": 48,
        "combo_minimal_window": 56,
        "combo_best_window": 40,
    },
    "ims": {
        "window_small": 10,
        "window_base": 15,
        "window_large": 20,
        "window_xlarge": 30,
        "epochs_short": 150,
        "epochs_base": 300,
        "epochs_long": 500,
        "stride_dense": (1, 1),
        "stride_normal": (1, 1),
        "stride_sparse": (1, 2),
        "stride_very_sparse": (2, 3),
        "combo_fast_window": 10,
        "combo_patient_window": 25,
        "combo_regularized_window": 12,
        "combo_aggressive_window": 10,
        "combo_dense_window": 15,
        "combo_minimal_window": 18,
        "combo_best_window": 12,
    },
    "pronto": {
        "window_small": 10,
        "window_base": 15,
        "window_large": 20,
        "window_xlarge": 30,
        "epochs_short": 50,
        "epochs_base": 100,
        "epochs_long": 150,
        "stride_dense": (1, 2),
        "stride_normal": (1, 5),
        "stride_sparse": (2, 5),
        "stride_very_sparse": (3, 8),
        "combo_fast_window": 10,
        "combo_patient_window": 25,
        "combo_regularized_window": 12,
        "combo_aggressive_window": 10,
        "combo_dense_window": 15,
        "combo_minimal_window": 18,
        "combo_best_window": 12,
    },
}


def get_base_config(dataset_key: str) -> Dict[str, Any]:
    """Get the base configuration for a specific dataset."""
    configs = {
        "ashrae": ASHRAE_BASE_CONFIG,
        "tep": TEP_BASE_CONFIG,
        "ims": IMS_BASE_CONFIG,
        "pronto": PRONTO_BASE_CONFIG,
    }
    return configs.get(dataset_key, ASHRAE_BASE_CONFIG).copy()


def get_original_config(dataset_key: str) -> Dict[str, Any]:
    """Get the original (user's) configuration for a specific dataset."""
    configs = {
        "ashrae": ASHRAE_ORIGINAL_CONFIG,
        "tep": TEP_ORIGINAL_CONFIG,
        "ims": IMS_ORIGINAL_CONFIG,
        "pronto": PRONTO_ORIGINAL_CONFIG,
    }
    return configs.get(dataset_key, ASHRAE_ORIGINAL_CONFIG).copy()


def get_dataset_params(dataset_key: str) -> Dict[str, Any]:
    """Get the dataset-specific parameter ranges for experiment generation."""
    return DATASET_PARAMS.get(dataset_key, DATASET_PARAMS["ashrae"]).copy()


# Legacy alias for backward compatibility
BASE_CONFIG = ASHRAE_BASE_CONFIG.copy()
ORIGINAL_CONFIG = ASHRAE_ORIGINAL_CONFIG.copy()


def define_experiments(dataset_key: str = "ashrae") -> List[Dict[str, Any]]:
    """
    Define all 40 experiments with dataset-specific configurations.

    Args:
        dataset_key: One of "ashrae", "tep", "ims", or "pronto"

    Returns:
        List of experiment dictionaries with appropriate parameter ranges for the dataset.
    """
    experiments = []

    # Get dataset-specific parameters
    params = get_dataset_params(dataset_key)

    # Extract parameters
    window_small = params["window_small"]
    window_base = params["window_base"]
    window_large = params["window_large"]
    window_xlarge = params["window_xlarge"]
    epochs_short = params["epochs_short"]
    epochs_base = params["epochs_base"]
    epochs_long = params["epochs_long"]

    # Strides
    stride_dense_train, stride_dense_val = params["stride_dense"]
    stride_normal_train, stride_normal_val = params["stride_normal"]
    stride_sparse_train, stride_sparse_val = params["stride_sparse"]
    stride_very_sparse_train, stride_very_sparse_val = params["stride_very_sparse"]

    # Combo windows
    combo_fast_window = params["combo_fast_window"]
    combo_patient_window = params["combo_patient_window"]
    combo_regularized_window = params["combo_regularized_window"]
    combo_aggressive_window = params["combo_aggressive_window"]
    combo_dense_window = params["combo_dense_window"]
    combo_minimal_window = params["combo_minimal_window"]
    combo_best_window = params["combo_best_window"]

    # =========================================================================
    # Reference Experiments (2)
    # =========================================================================
    experiments.append({
        "id": 1,
        "name": "original",
        "category": "reference",
        "description": "User's original configuration",
        "overrides": get_original_config(dataset_key),
    })

    experiments.append({
        "id": 2,
        "name": "balanced",
        "category": "reference",
        "description": "Proposed balanced baseline",
        "overrides": {},  # Uses base config as-is
    })

    # =========================================================================
    # Sensitivity Analysis - Window Size (3)
    # =========================================================================
    experiments.append({
        "id": 3,
        "name": "window_small",
        "category": "sensitivity",
        "description": f"Small window ({window_small}) for fast transients",
        "overrides": {
            "window-size": window_small,
            "train-stride": stride_dense_train,
            "val-stride": stride_dense_val,
        },
    })

    experiments.append({
        "id": 4,
        "name": "window_large",
        "category": "sensitivity",
        "description": f"Large window ({window_large}) for slow dynamics",
        "overrides": {
            "window-size": window_large,
            "train-stride": stride_sparse_train,
            "val-stride": stride_sparse_val,
        },
    })

    experiments.append({
        "id": 5,
        "name": "window_xlarge",
        "category": "sensitivity",
        "description": f"Very large window ({window_xlarge}) for long-term patterns",
        "overrides": {
            "window-size": window_xlarge,
            "train-stride": stride_very_sparse_train,
            "val-stride": stride_very_sparse_val,
        },
    })

    # =========================================================================
    # Sensitivity Analysis - Anomaly Weight (3)
    # =========================================================================
    experiments.append({
        "id": 6,
        "name": "anomaly_01",
        "category": "sensitivity",
        "description": "Low anomaly weight (reconstruction focus)",
        "overrides": {"anomaly-weight": 0.1},
    })

    experiments.append({
        "id": 7,
        "name": "anomaly_20",
        "category": "sensitivity",
        "description": "High anomaly weight (detection focus)",
        "overrides": {"anomaly-weight": 2.0},
    })

    experiments.append({
        "id": 8,
        "name": "anomaly_30",
        "category": "sensitivity",
        "description": "Very high anomaly weight",
        "overrides": {"anomaly-weight": 3.0},
    })

    # =========================================================================
    # Sensitivity Analysis - Lambda Divergence (3)
    # =========================================================================
    experiments.append({
        "id": 9,
        "name": "lambda_00",
        "category": "sensitivity",
        "description": "No divergence loss (disabled)",
        "overrides": {"lambda-div": 0.0},
    })

    experiments.append({
        "id": 10,
        "name": "lambda_04",
        "category": "sensitivity",
        "description": "Moderate divergence weight",
        "overrides": {"lambda-div": 0.4},
    })

    experiments.append({
        "id": 11,
        "name": "lambda_06",
        "category": "sensitivity",
        "description": "High divergence weight",
        "overrides": {"lambda-div": 0.6},
    })

    # =========================================================================
    # Sensitivity Analysis - Learning Rate (3)
    # =========================================================================
    experiments.append({
        "id": 12,
        "name": "lr_5e5",
        "category": "sensitivity",
        "description": "Conservative learning rate",
        "overrides": {"learning-rate": 5e-5},
    })

    experiments.append({
        "id": 13,
        "name": "lr_8e4",
        "category": "sensitivity",
        "description": "Aggressive learning rate",
        "overrides": {"learning-rate": 8e-4},
    })

    experiments.append({
        "id": 14,
        "name": "lr_1e3",
        "category": "sensitivity",
        "description": "Very aggressive learning rate",
        "overrides": {"learning-rate": 1e-3},
    })

    # =========================================================================
    # Sensitivity Analysis - Embedding Dimension (3)
    # =========================================================================
    experiments.append({
        "id": 15,
        "name": "embed_8",
        "category": "sensitivity",
        "description": "Minimal embedding dimension",
        "overrides": {"freq-embed-dim": 8},
    })

    experiments.append({
        "id": 16,
        "name": "embed_32",
        "category": "sensitivity",
        "description": "Moderate embedding dimension",
        "overrides": {"freq-embed-dim": 32},
    })

    experiments.append({
        "id": 17,
        "name": "embed_48",
        "category": "sensitivity",
        "description": "Large embedding dimension",
        "overrides": {"freq-embed-dim": 48},
    })

    # =========================================================================
    # Sensitivity Analysis - Batch Size (2)
    # =========================================================================
    experiments.append({
        "id": 18,
        "name": "batch_64",
        "category": "sensitivity",
        "description": "Small batch size (more noise, more updates)",
        "overrides": {"batch-size": 64},
    })

    experiments.append({
        "id": 19,
        "name": "batch_256",
        "category": "sensitivity",
        "description": "Large batch size (smoother gradients)",
        "overrides": {"batch-size": 256},
    })

    # =========================================================================
    # Sensitivity Analysis - Weight Decay (2)
    # =========================================================================
    experiments.append({
        "id": 20,
        "name": "decay_1e4",
        "category": "sensitivity",
        "description": "Higher regularization",
        "overrides": {"weight-decay": 1e-4},
    })

    experiments.append({
        "id": 21,
        "name": "decay_1e6",
        "category": "sensitivity",
        "description": "Lower regularization",
        "overrides": {"weight-decay": 1e-6},
    })

    # =========================================================================
    # Sensitivity Analysis - Epochs (2)
    # =========================================================================
    experiments.append({
        "id": 22,
        "name": "epochs_short",
        "category": "sensitivity",
        "description": f"Shorter training ({epochs_short} epochs)",
        "overrides": {"epochs": epochs_short},
    })

    experiments.append({
        "id": 23,
        "name": "epochs_long",
        "category": "sensitivity",
        "description": f"Longer training ({epochs_long} epochs, risk of overfitting)",
        "overrides": {"epochs": epochs_long},
    })

    # =========================================================================
    # Ablation Studies (6)
    # =========================================================================
    experiments.append({
        "id": 24,
        "name": "no_spectral",
        "category": "ablation",
        "description": "Temporal only (no spectral view)",
        "overrides": {"use-spectral-view": False, "lambda-div": 0.0},
    })

    experiments.append({
        "id": 25,
        "name": "no_spec_features",
        "category": "ablation",
        "description": "Spectral without shape features",
        "overrides": {"freq-use-spectral-features": False},
    })

    experiments.append({
        "id": 26,
        "name": "band_mix_none",
        "category": "ablation",
        "description": "No band mixing layer",
        "overrides": {"freq-band-mix": "none"},
    })

    experiments.append({
        "id": 27,
        "name": "band_mix_conv",
        "category": "ablation",
        "description": "Convolutional band mixing",
        "overrides": {"freq-band-mix": "conv"},
    })

    experiments.append({
        "id": 28,
        "name": "fuse_sum",
        "category": "ablation",
        "description": "Sum fusion mode",
        "overrides": {"fuse-mode": "sum"},
    })

    experiments.append({
        "id": 29,
        "name": "fuse_gated",
        "category": "ablation",
        "description": "Gated fusion mode",
        "overrides": {"fuse-mode": "gated"},
    })

    # =========================================================================
    # Divergence Type (2)
    # =========================================================================
    experiments.append({
        "id": 30,
        "name": "div_kl",
        "category": "sensitivity",
        "description": "KL divergence instead of JS",
        "overrides": {"divergence-type": "kl"},
    })

    experiments.append({
        "id": 31,
        "name": "div_js_high",
        "category": "sensitivity",
        "description": "JS divergence with higher weight",
        "overrides": {"divergence-type": "js", "lambda-div": 0.4},
    })

    # =========================================================================
    # Task Alternative (2)
    # =========================================================================
    experiments.append({
        "id": 32,
        "name": "prediction_h10",
        "category": "alternative",
        "description": "Prediction task (horizon=10)",
        "overrides": {"task": "prediction", "pred-horizon": 10},
    })

    experiments.append({
        "id": 33,
        "name": "prediction_h20",
        "category": "alternative",
        "description": "Prediction task (horizon=20)",
        "overrides": {"task": "prediction", "pred-horizon": 20},
    })

    # =========================================================================
    # Combination Experiments (7)
    # =========================================================================
    experiments.append({
        "id": 34,
        "name": "combo_fast",
        "category": "combination",
        "description": "Fast transient detection config",
        "overrides": {
            "window-size": combo_fast_window,
            "train-stride": stride_dense_train,
            "val-stride": stride_dense_val,
            "anomaly-weight": 2.0,
            "lambda-div": 0.3,
            "learning-rate": 5e-4,
        },
    })

    # For combo_patient, use dataset-specific epochs (133% of base)
    combo_patient_epochs = int(epochs_base * 1.33)
    experiments.append({
        "id": 35,
        "name": "combo_patient",
        "category": "combination",
        "description": "Slow drift detection config",
        "overrides": {
            "window-size": combo_patient_window,
            "train-stride": stride_very_sparse_train,
            "val-stride": stride_very_sparse_val,
            "anomaly-weight": 0.8,
            "lambda-div": 0.15,
            "learning-rate": 1e-4,
            "epochs": combo_patient_epochs,
        },
    })

    experiments.append({
        "id": 36,
        "name": "combo_regularized",
        "category": "combination",
        "description": "Strong regularization config",
        "overrides": {
            "window-size": combo_regularized_window,
            "anomaly-weight": 1.5,
            "lambda-div": 0.4,
            "freq-embed-dim": 32,
            "weight-decay": 1e-4,
        },
    })

    # For combo_aggressive, use dataset-specific epochs (67% of base)
    combo_aggressive_epochs = int(epochs_base * 0.67)
    experiments.append({
        "id": 37,
        "name": "combo_aggressive",
        "category": "combination",
        "description": "Fast convergence config",
        "overrides": {
            "window-size": combo_aggressive_window,
            "learning-rate": 8e-4,
            "anomaly-weight": 2.5,
            "lambda-div": 0.35,
            "epochs": combo_aggressive_epochs,
        },
    })

    experiments.append({
        "id": 38,
        "name": "combo_dense",
        "category": "combination",
        "description": "Maximum data utilization config",
        "overrides": {
            "window-size": combo_dense_window,
            "train-stride": stride_dense_train,
            "val-stride": stride_dense_val,
            "batch-size": 64,
            "anomaly-weight": 1.5,
            "freq-embed-dim": 32,
        },
    })

    experiments.append({
        "id": 39,
        "name": "combo_minimal",
        "category": "combination",
        "description": "Lean model config",
        "overrides": {
            "window-size": combo_minimal_window,
            "freq-use-spectral-features": False,
            "lambda-div": 0.1,
            "anomaly-weight": 0.8,
        },
    })

    # For combo_best_guess, use dataset-specific epochs (83% of base)
    combo_best_epochs = int(epochs_base * 0.83)
    experiments.append({
        "id": 40,
        "name": "combo_best_guess",
        "category": "combination",
        "description": "Optimized best-guess config",
        "overrides": {
            "window-size": combo_best_window,
            "train-stride": stride_dense_train,
            "val-stride": stride_dense_val,
            "anomaly-weight": 1.5,
            "lambda-div": 0.25,
            "learning-rate": 4e-4,
            "freq-embed-dim": 32,
            "epochs": combo_best_epochs,
        },
    })

    return experiments


# =============================================================================
# Command Building
# =============================================================================

def merge_config(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with experiment overrides."""
    merged = base.copy()
    merged.update(overrides)
    return merged


def build_command(
    config: Dict[str, Any],
    checkpoint_dir: str,
    cuda_device: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Build the training command from config dict."""
    cmd = [CONDA_PYTHON, "train_dystgat.py"]

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                # Boolean flags that are True
                cmd.append(f"--{key}")
            else:
                # For False boolean flags, check if there's a no- variant
                if key == "use-spectral-view":
                    pass  # Simply omit the flag
                elif key == "freq-use-spectral-features":
                    pass  # Simply omit the flag
                elif key == "freq-use-log":
                    cmd.append("--no-freq-use-log")
        else:
            cmd.extend([f"--{key}", str(value)])

    cmd.extend(["--checkpoint-dir", checkpoint_dir])

    if cuda_device is not None:
        cmd.extend(["--cuda-device", str(cuda_device)])

    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    return cmd


def format_command_string(cmd: List[str]) -> str:
    """Format command list as a shell-executable string."""
    return shlex.join(cmd)


# =============================================================================
# Result Aggregation
# =============================================================================

def find_latest_run(checkpoint_base: str) -> Optional[Path]:
    """Find the latest run directory in checkpoint base."""
    checkpoint_path = Path(checkpoint_base)
    if not checkpoint_path.exists():
        return None

    # Find all run directories (format: dystgat_{dataset}_YYYYMMDD_HHMMSS)
    run_dirs = [d for d in checkpoint_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by modification time, get most recent
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def load_detailed_metrics(metrics_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load detailed test metrics from CSV."""
    metrics = {}
    if not metrics_csv.exists():
        return metrics

    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_name = row.get("test_set", "unknown")
            metrics[test_name] = {}
            for k, v in row.items():
                if k != "test_set" and v:
                    try:
                        metrics[test_name][k] = float(v)
                    except ValueError:
                        metrics[test_name][k] = v
    return metrics


def load_training_metrics(metrics_csv: Path) -> Dict[str, Any]:
    """Load training/validation metrics from metrics.csv."""
    if not metrics_csv.exists():
        return {}

    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    final = rows[-1]
    best_val_loss = float("inf")
    best_epoch = 0

    for row in rows:
        val_loss = row.get("val_loss")
        if val_loss:
            try:
                val_loss_f = float(val_loss)
                if val_loss_f < best_val_loss:
                    best_val_loss = val_loss_f
                    best_epoch = int(row.get("epoch", 0))
            except (ValueError, TypeError):
                pass

    return {
        "final_epoch": int(final.get("epoch", len(rows))),
        "final_train_loss": float(final.get("train_loss", 0)),
        "final_val_loss": float(final.get("val_loss", 0)),
        "best_val_loss": best_val_loss if best_val_loss < float("inf") else 0,
        "best_epoch": best_epoch,
    }


def aggregate_experiment_results(
    experiments: List[Dict[str, Any]],
    results_base: Path,
    base_config: Optional[Dict[str, Any]] = None,
    dataset_key: str = "ashrae",
) -> Dict[str, Any]:
    """Aggregate results from all experiments."""
    if base_config is None:
        base_config = get_base_config(dataset_key)

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_key": dataset_key,
        "n_experiments": len(experiments),
        "experiments": [],
    }

    for exp in experiments:
        exp_name = exp["name"]
        exp_id = exp["id"]
        exp_dir = results_base / f"exp{exp_id:02d}_{exp_name}"

        exp_result = {
            "id": exp_id,
            "name": exp_name,
            "category": exp["category"],
            "description": exp["description"],
            "config": merge_config(base_config, exp.get("overrides", {})),
            "status": "not_found",
            "test_metrics": {},
            "training_metrics": {},
        }

        # Find the run directory
        run_dir = find_latest_run(str(exp_dir))
        if run_dir:
            exp_result["run_dir"] = str(run_dir)

            # Load detailed test metrics
            plots_dir = run_dir / "plots"
            detailed_csv = plots_dir / "detailed_test_metrics.csv"
            if detailed_csv.exists():
                exp_result["test_metrics"] = load_detailed_metrics(detailed_csv)
                exp_result["status"] = "completed"
            else:
                exp_result["status"] = "incomplete"

            # Load training metrics
            training_csv = run_dir / "metrics.csv"
            if training_csv.exists():
                exp_result["training_metrics"] = load_training_metrics(training_csv)

            # Load command
            cmd_file = run_dir / "train_command.txt"
            if cmd_file.exists():
                exp_result["command"] = cmd_file.read_text().strip()

        results["experiments"].append(exp_result)

    return results


def compute_summary_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create a summary table with key metrics for each experiment."""
    summary = []

    for exp in results["experiments"]:
        row = {
            "id": exp["id"],
            "name": exp["name"],
            "category": exp["category"],
            "status": exp["status"],
            # Key config params
            "window": exp["config"].get("window-size", "-"),
            "lambda_div": exp["config"].get("lambda-div", "-"),
            "anomaly_wt": exp["config"].get("anomaly-weight", "-"),
            "lr": exp["config"].get("learning-rate", "-"),
            "embed_dim": exp["config"].get("freq-embed-dim", "-"),
            "epochs": exp["config"].get("epochs", "-"),
        }

        # Training metrics
        train_metrics = exp.get("training_metrics", {})
        row["best_val_loss"] = train_metrics.get("best_val_loss", "-")
        row["best_epoch"] = train_metrics.get("best_epoch", "-")

        # Average test metrics across all fault types
        test_metrics = exp.get("test_metrics", {})
        if test_metrics:
            aucs = []
            f1s = []
            best_f1s = []
            tea_aucs = []
            delays = []

            for test_name, metrics in test_metrics.items():
                if "baseline" in test_name.lower():
                    continue

                if "auc_roc" in metrics:
                    try:
                        aucs.append(float(metrics["auc_roc"]))
                    except (ValueError, TypeError):
                        pass
                if "f1_score" in metrics:
                    try:
                        f1s.append(float(metrics["f1_score"]))
                    except (ValueError, TypeError):
                        pass
                if "best_f1" in metrics:
                    try:
                        best_f1s.append(float(metrics["best_f1"]))
                    except (ValueError, TypeError):
                        pass
                if "tea_auc" in metrics:
                    try:
                        tea_aucs.append(float(metrics["tea_auc"]))
                    except (ValueError, TypeError):
                        pass
                if "delay_best_thr" in metrics:
                    try:
                        delays.append(float(metrics["delay_best_thr"]))
                    except (ValueError, TypeError):
                        pass

            row["avg_auc"] = f"{np.mean(aucs):.4f}" if aucs else "-"
            row["avg_f1"] = f"{np.mean(f1s):.4f}" if f1s else "-"
            row["avg_best_f1"] = f"{np.mean(best_f1s):.4f}" if best_f1s else "-"
            row["avg_tea_auc"] = f"{np.mean(tea_aucs):.4f}" if tea_aucs else "-"
            row["avg_delay"] = f"{np.mean(delays):.1f}" if delays else "-"
        else:
            row["avg_auc"] = "-"
            row["avg_f1"] = "-"
            row["avg_best_f1"] = "-"
            row["avg_tea_auc"] = "-"
            row["avg_delay"] = "-"

        summary.append(row)

    return summary


# =============================================================================
# Output Formatting
# =============================================================================

def save_csv(summary: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary table as CSV."""
    if not summary:
        return

    fieldnames = list(summary[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def save_markdown(summary: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary table as Markdown."""
    if not summary:
        output_path.write_text("No results available.\n")
        return

    lines = ["# Hyperparameter Search Results\n"]
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Create table header
    cols = ["ID", "Name", "Category", "Window", "λ-div", "Anom Wt", "AUC", "F1", "F1*", "TEA AUC", "Delay"]
    header = "| " + " | ".join(cols) + " |"
    separator = "|" + "|".join([":---:" if i > 0 else ":---" for i in range(len(cols))]) + "|"
    lines.append(header)
    lines.append(separator)

    # Data rows
    for row in summary:
        values = [
            str(row["id"]),
            row["name"],
            row["category"],
            str(row["window"]),
            str(row["lambda_div"]),
            str(row["anomaly_wt"]),
            str(row["avg_auc"]),
            str(row["avg_f1"]),
            str(row["avg_best_f1"]),
            str(row["avg_tea_auc"]),
            str(row["avg_delay"]),
        ]
        lines.append("| " + " | ".join(values) + " |")

    output_path.write_text("\n".join(lines) + "\n")


def save_latex(summary: List[Dict[str, Any]], output_path: Path, caption: str = "Hyperparameter Search Results") -> None:
    """Save summary table as LaTeX."""
    if not summary:
        output_path.write_text("% No results available\n")
        return

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        r"\label{tab:hpsearch_results}",
        r"\begin{tabular}{clcccccccc}",
        r"\toprule",
        r"ID & Name & Cat. & Win & $\lambda$ & $\alpha$ & AUC & F1 & F1* & TEA \\",
        r"\midrule",
    ]

    for row in summary:
        name_escaped = row["name"].replace("_", r"\_")
        cat_short = row["category"][:3].upper()
        line = (
            f"{row['id']} & {name_escaped} & {cat_short} & "
            f"{row['window']} & {row['lambda_div']} & {row['anomaly_wt']} & "
            f"{row['avg_auc']} & {row['avg_f1']} & {row['avg_best_f1']} & {row['avg_tea_auc']} \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.write_text("\n".join(lines) + "\n")


def save_commands(
    experiments: List[Dict[str, Any]],
    output_path: Path,
    cuda_device: int = 0,
    base_config: Optional[Dict[str, Any]] = None,
    dataset_key: str = "ashrae",
) -> None:
    """Save all experiment commands to a text file."""
    if base_config is None:
        base_config = get_base_config(dataset_key)

    lines = [
        "# Hyperparameter Search Commands",
        f"# Dataset: {dataset_key}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for exp in experiments:
        exp_name = exp["name"]
        exp_id = exp["id"]
        config = merge_config(base_config, exp.get("overrides", {}))
        checkpoint_dir = f"results/{dataset_key}/hpsearch/exp{exp_id:02d}_{exp_name}"

        cmd = build_command(config, checkpoint_dir, cuda_device)
        cmd_str = format_command_string(cmd)

        lines.append(f"# Experiment {exp_id}: {exp_name} ({exp['category']})")
        lines.append(f"# {exp['description']}")
        lines.append(cmd_str)
        lines.append("")

    output_path.write_text("\n".join(lines))


# =============================================================================
# Experiment Execution
# =============================================================================

def run_experiment(
    exp: Dict[str, Any],
    results_base: Path,
    cuda_device: Optional[int],
    seed: Optional[int],
    dry_run: bool = False,
    base_config: Optional[Dict[str, Any]] = None,
    dataset_key: str = "ashrae",
) -> Tuple[bool, str]:
    """Run a single experiment and return success status and message."""
    if base_config is None:
        base_config = get_base_config(dataset_key)

    exp_name = exp["name"]
    exp_id = exp["id"]
    config = merge_config(base_config, exp.get("overrides", {}))

    checkpoint_dir = results_base / f"exp{exp_id:02d}_{exp_name}"
    cmd = build_command(config, str(checkpoint_dir), cuda_device, seed)
    cmd_str = format_command_string(cmd)

    if dry_run:
        print(f"  [DRY RUN] Would execute:")
        print(f"  {cmd_str}")
        return True, "dry_run"

    print(f"  Command: {cmd_str[:100]}...")

    # Save command to file
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "planned_command.txt").write_text(cmd_str + "\n")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            cwd=Path(__file__).parent,
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            return True, f"completed in {elapsed/60:.1f} min"
        else:
            return False, f"failed with code {result.returncode}"

    except Exception as e:
        return False, f"exception: {str(e)}"


def filter_experiments(
    experiments: List[Dict[str, Any]],
    experiment_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    start_from: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter experiments by various criteria."""
    filtered = experiments

    # Filter by category
    if category_filter:
        filtered = [e for e in filtered if e["category"].lower() == category_filter.lower()]

    # Filter by specific experiments (comma-separated IDs or names)
    if experiment_filter:
        tokens = [t.strip() for t in experiment_filter.split(",")]
        ids = set()
        names = set()

        for token in tokens:
            if token.isdigit():
                ids.add(int(token))
            else:
                names.add(token.lower())

        filtered = [
            e for e in filtered
            if e["id"] in ids or e["name"].lower() in names
        ]

    # Filter by start-from ID
    if start_from is not None:
        filtered = [e for e in filtered if e["id"] >= start_from]

    return filtered


def check_completed(exp: Dict[str, Any], results_base: Path) -> bool:
    """Check if an experiment has already been completed."""
    exp_name = exp["name"]
    exp_id = exp["id"]
    exp_dir = results_base / f"exp{exp_id:02d}_{exp_name}"

    run_dir = find_latest_run(str(exp_dir))
    if run_dir:
        detailed_csv = run_dir / "plots" / "detailed_test_metrics.csv"
        return detailed_csv.exists()

    return False


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search & Ablation Study Script for DySTGAT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 40 experiments for ASHRAE
  python run_hpsearch.py --dataset-key ashrae --cuda-device 0

  # Run all 40 experiments for TEP on cuda:1
  python run_hpsearch.py --dataset-key tep --cuda-device 1

  # Run for IMS bearing dataset
  python run_hpsearch.py --dataset-key ims --cuda-device 0

  # Run for PRONTO dataset
  python run_hpsearch.py --dataset-key pronto --cuda-device 0

  # Run specific experiments by number
  python run_hpsearch.py --dataset-key ashrae --experiments 1,2,34,40

  # Run experiments by name
  python run_hpsearch.py --dataset-key ashrae --experiments original,balanced,combo_best_guess

  # Run only sensitivity analysis experiments
  python run_hpsearch.py --dataset-key ashrae --category sensitivity

  # Run only ablation studies
  python run_hpsearch.py --dataset-key tep --category ablation

  # Dry run (print commands without executing)
  python run_hpsearch.py --dataset-key tep --cuda-device 1 --dry-run

  # Only aggregate existing results (skip training)
  python run_hpsearch.py --dataset-key ashrae --aggregate-only \\
      --results-dir results/ashrae/hpsearch_20260120_123456

  # Resume from specific experiment (skip completed ones)
  python run_hpsearch.py --dataset-key ashrae --resume --start-from 15

  # List all experiments for a dataset
  python run_hpsearch.py --dataset-key tep --list
        """,
    )

    parser.add_argument(
        "--dataset-key",
        type=str,
        choices=["ashrae", "tep", "ims", "pronto"],
        default="ashrae",
        help="Dataset to use: ashrae, tep, ims, or pronto (default: ashrae)",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment IDs or names to run (default: all)",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["reference", "sensitivity", "ablation", "alternative", "combination"],
        default=None,
        help="Run only experiments in this category",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=None,
        help="Start from experiment ID (skip earlier ones)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already completed experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing results (skip training)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results directory (default: results/{dataset}/hpsearch_{timestamp})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments and exit",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    dataset_key = args.dataset_key

    # Get dataset-specific base configuration
    base_config = get_base_config(dataset_key)

    # Define all experiments with dataset-specific parameters
    all_experiments = define_experiments(dataset_key)

    # List mode
    if args.list:
        print(f"\nAvailable Experiments for {dataset_key.upper()} (40 total):")
        print("=" * 80)
        for exp in all_experiments:
            print(f"  [{exp['id']:2d}] {exp['name']:20s} ({exp['category']:12s}) - {exp['description']}")
        print("\nCategories:")
        categories = {}
        for exp in all_experiments:
            cat = exp["category"]
            categories[cat] = categories.get(cat, 0) + 1
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} experiments")
        print(f"\nBase configuration for {dataset_key}:")
        for key, value in base_config.items():
            print(f"  {key}: {value}")
        return

    # Setup results directory
    if args.results_dir:
        results_base = Path(args.results_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base = Path(f"results/{dataset_key}/hpsearch_{timestamp}")

    results_base.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset: {dataset_key.upper()}")
    print(f"Results directory: {results_base}")

    # Filter experiments
    experiments = filter_experiments(
        all_experiments,
        experiment_filter=args.experiments,
        category_filter=args.category,
        start_from=args.start_from,
    )

    if not experiments:
        print("No experiments match the filter criteria.")
        return

    print(f"Selected {len(experiments)} experiments to run")

    # Aggregate-only mode
    if args.aggregate_only:
        print("\nAggregating existing results...")
        results = aggregate_experiment_results(
            all_experiments, results_base, base_config=base_config, dataset_key=dataset_key
        )
        summary = compute_summary_table(results)

        # Save outputs
        save_csv(summary, results_base / "summary_results.csv")
        save_markdown(summary, results_base / "summary_table.md")
        save_latex(summary, results_base / "summary_table.tex")
        (results_base / "summary_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        print(f"Results saved to {results_base}")
        return

    # Save experiment configs and commands
    (results_base / "experiment_configs.json").write_text(
        json.dumps({
            "dataset_key": dataset_key,
            "experiments": experiments,
            "base_config": base_config,
        }, indent=2)
    )
    save_commands(
        all_experiments,
        results_base / "experiment_commands.txt",
        args.cuda_device,
        base_config=base_config,
        dataset_key=dataset_key,
    )

    # Run experiments
    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER SEARCH - {dataset_key.upper()}")
    print("=" * 80)

    completed = 0
    failed = 0
    skipped = 0

    for i, exp in enumerate(experiments, 1):
        exp_id = exp["id"]
        exp_name = exp["name"]

        print(f"\n[{i}/{len(experiments)}] Experiment {exp_id}: {exp_name}")
        print(f"  Category: {exp['category']}")
        print(f"  Description: {exp['description']}")

        # Check if already completed
        if args.resume and check_completed(exp, results_base):
            print("  Skipping (already completed)")
            skipped += 1
            continue

        # Run experiment
        success, message = run_experiment(
            exp,
            results_base,
            cuda_device=args.cuda_device,
            seed=args.seed,
            dry_run=args.dry_run,
            base_config=base_config,
            dataset_key=dataset_key,
        )

        if success:
            print(f"  OK: {message}")
            completed += 1
        else:
            print(f"  FAIL: {message}")
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Dataset:   {dataset_key}")
    print(f"  Completed: {completed}")
    print(f"  Failed:    {failed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total:     {len(experiments)}")

    # Aggregate results
    if not args.dry_run:
        print("\nAggregating results...")
        results = aggregate_experiment_results(
            all_experiments, results_base, base_config=base_config, dataset_key=dataset_key
        )
        summary = compute_summary_table(results)

        # Save outputs
        save_csv(summary, results_base / "summary_results.csv")
        save_markdown(summary, results_base / "summary_table.md")
        save_latex(summary, results_base / "summary_table.tex")
        (results_base / "summary_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )

        print(f"\nResults saved to {results_base}")
        print("  - summary_results.csv")
        print("  - summary_results.json")
        print("  - summary_table.md")
        print("  - summary_table.tex")
        print("  - experiment_commands.txt")


if __name__ == "__main__":
    main()
