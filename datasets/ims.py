"""
IMS Bearing Dataset Adapter for DySTGAT.

Provides data loading interface for the NASA IMS Bearing run-to-failure dataset.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dystgat.src.data.ims_column_config import MEASUREMENT_VARS, CONTROL_VARS
from dystgat.src.data.ims_dataset import (
    IMSBearingDataset,
    get_ims_control_variables,
    get_ims_measurement_variables,
)
from .registry import DatasetAdapter, register_adapter


# Default dataset set to use
DEFAULT_DATASET_SET = "2nd_test"

# Healthy ratio for train/test split
# First 70% is "healthy", used for training
HEALTHY_RATIO = 0.7


def _resolve_split_files(split_key: str) -> List[str]:
    """Resolve split key to file identifiers."""
    if split_key == "train":
        return ["healthy"]
    elif split_key == "val":
        return ["healthy"]
    elif split_key == "test":
        return ["degraded", "faulty"]
    elif split_key == "all":
        return ["healthy", "degraded", "faulty"]
    else:
        return []


def _create_dataloaders(
    window_size: int,
    batch_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: Optional[int],
    data_dir: str,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    baseline_from: str = "val",
    severity_range: Optional[Tuple[int, int]] = None,
    feature_option: Optional[str] = None,
    fault_keys: Optional[List[str]] = None,
    pred_horizon: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train/val/test dataloaders for IMS Bearing dataset.

    The dataset is split temporally:
    - Train: First 60% of healthy period
    - Val: Next 10% of healthy period
    - Test: Remaining files (degraded + faulty)
    """
    if test_stride is None:
        test_stride = val_stride

    # Determine dataset set from feature_option or default
    dataset_set = feature_option if feature_option in ["1st_test", "2nd_test", "3rd_test"] else DEFAULT_DATASET_SET

    print("=" * 70)
    print("CREATING IMS BEARING DATALOADERS")
    print("=" * 70)
    print(f"Dataset set: {dataset_set}")
    print(f"Data dir: {data_dir}")

    # Calculate file indices for splits
    # Train: files 0 to 60% of healthy (which is 70% of total)
    # Val: files 60% to 70% of healthy
    # Test: files from 70% onwards (degraded + faulty)

    # First, count total files
    set_dir = os.path.join(data_dir, dataset_set)
    if os.path.exists(set_dir):
        n_total = len([f for f in os.listdir(set_dir) if os.path.isfile(os.path.join(set_dir, f))])
    else:
        raise ValueError(f"Dataset directory not found: {set_dir}")

    healthy_end = int(n_total * HEALTHY_RATIO)
    train_end = int(healthy_end * 0.75)  # 75% of healthy for training
    val_end = int(healthy_end * 0.875)   # Next 12.5% of healthy for validation
    baseline_end = healthy_end           # Final 12.5% of healthy for test baseline

    print(f"Total files: {n_total}")
    print(f"Healthy period ends at: {healthy_end} files")
    print(f"Train: 0 - {train_end} ({train_end} files)")
    print(f"Val: {train_end} - {val_end} ({val_end - train_end} files)")
    print(f"Test Baseline: {val_end} - {baseline_end} ({baseline_end - val_end} files)")
    print(f"Test (degraded+faulty): {baseline_end} - {n_total} ({n_total - baseline_end} files)")

    # Training dataset (healthy only, first portion)
    print("\n[1/3] Creating TRAINING dataset (healthy period)...")
    train_dataset = IMSBearingDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=train_stride,
        normalize=True,
        healthy_ratio=1.0,  # All files in train are "healthy" by construction
        fault_filter={0},  # Only healthy
        max_files=train_end,
    )
    norm_stats = train_dataset.get_normalization_stats()

    # Validation dataset (healthy, middle portion)
    print("[2/3] Creating VALIDATION dataset (late healthy period)...")

    # Create a dataset for validation portion
    # We load all files but filter to only the validation window
    class ValidationIMSDataset(IMSBearingDataset):
        """Wrapper to select validation portion of healthy data."""

        def _load_data(self):
            super()._load_data()
            # Keep only validation portion (train_end to val_end)
            self.features = self.features[train_end:val_end]
            self.labels = self.labels[train_end:val_end]
            self.time_indices = self.time_indices[train_end:val_end]
            self.normalized_time = self.normalized_time[train_end:val_end]

    val_dataset = ValidationIMSDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=val_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=1.0,
        fault_filter={0},
        require_stats=True,
    )

    # Test datasets
    print("[3/3] Creating TEST datasets...")
    test_datasets = {}

    # Baseline (separate from validation - late healthy period)
    # IMPORTANT: This must be different from validation to avoid data leakage
    class BaselineIMSDataset(IMSBearingDataset):
        """Wrapper to select baseline (late healthy) portion for test."""

        def _load_data(self):
            super()._load_data()
            # Keep only baseline portion (val_end to baseline_end)
            self.features = self.features[val_end:baseline_end]
            self.labels = self.labels[val_end:baseline_end]
            self.time_indices = self.time_indices[val_end:baseline_end]
            self.normalized_time = self.normalized_time[val_end:baseline_end]

    print("  - Baseline (late healthy, separate from validation)")
    test_datasets["baseline"] = BaselineIMSDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=1.0,
        fault_filter={0},
        require_stats=True,
    )

    # Full test (degraded + faulty) - starts after baseline_end
    class TestIMSDataset(IMSBearingDataset):
        """Wrapper to select test portion of data (degradation period)."""

        def _load_data(self):
            super()._load_data()
            # Keep only test portion (baseline_end onwards = degraded + faulty)
            self.features = self.features[baseline_end:]
            self.labels = self.labels[baseline_end:]
            # Labels are already assigned based on position (1=degraded, 2=faulty)
            self.time_indices = self.time_indices[baseline_end:]
            self.normalized_time = self.normalized_time[baseline_end:]

    test_all = TestIMSDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=HEALTHY_RATIO,  # Use original ratio for label assignment
        require_stats=True,
    )
    test_datasets["faults_all"] = test_all

    # Degraded only
    test_datasets["degraded"] = TestIMSDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=HEALTHY_RATIO,
        fault_filter={1},
        require_stats=True,
    )

    # Faulty only
    test_datasets["faulty"] = TestIMSDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=HEALTHY_RATIO,
        fault_filter={2},
        require_stats=True,
    )

    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_samplers = {
            name: DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
            for name, ds in test_datasets.items()
        }
    else:
        train_sampler = None
        val_sampler = None
        test_samplers = {name: None for name in test_datasets}

    # Create dataloaders
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loaders = {}
    for name, dataset in test_datasets.items():
        if len(dataset) > 0:
            test_loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=test_samplers[name],
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loaders: {list(test_loaders.keys())}")

    return train_loader, val_loader, test_loaders


# Register the adapter
register_adapter(
    DatasetAdapter(
        key="ims",
        description="NASA IMS Bearing run-to-failure dataset (vibration data).",
        default_data_dir=os.path.join("data", "IMS_Bearing"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=IMSBearingDataset,
        control_names_fn=lambda _, __=None: get_ims_control_variables(use_time=False),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ["faults_all", "degraded", "faulty"],
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
