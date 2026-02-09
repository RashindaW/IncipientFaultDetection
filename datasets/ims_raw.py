"""
IMS Bearing Raw Accelerometer Dataset Adapter for DySTGAT.

Provides data loading interface for raw 20 kHz vibration signals from the
NASA IMS Bearing run-to-failure dataset. Unlike the feature-engineered
'ims' adapter, this preserves raw accelerometer values for spectral analysis.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dystgat.src.data.ims_raw_column_config import MEASUREMENT_VARS, CONTROL_VARS
from dystgat.src.data.ims_raw_dataset import (
    IMSRawDataset,
    get_ims_raw_control_variables,
    get_ims_raw_measurement_variables,
)
from .registry import DatasetAdapter, register_adapter


# Default dataset set to use (most commonly used in literature)
DEFAULT_DATASET_SET = "2nd_test"

# Healthy ratio for train/test split (first 70% is healthy)
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
    **kwargs,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train/val/test dataloaders for IMS raw accelerometer data.

    The dataset is split temporally:
    - Train: First 60% of healthy period
    - Val: Next 10% of healthy period
    - Test Baseline: Next 10% of healthy period
    - Test Faults: Remaining files (degraded + faulty)

    For raw data, window_size is interpreted as the sample window size
    (number of accelerometer samples per window, e.g., 1024 for ~50ms).
    """
    if test_stride is None:
        test_stride = val_stride

    # Determine dataset set from feature_option or default
    dataset_set = feature_option if feature_option in ["1st_test", "2nd_test", "3rd_test"] else DEFAULT_DATASET_SET

    # For raw data, we use smaller strides to maintain temporal resolution
    # Sample stride within files (50% overlap recommended for spectral analysis)
    sample_stride = window_size // 2

    print("=" * 70)
    print("CREATING IMS RAW ACCELEROMETER DATALOADERS")
    print("=" * 70)
    print(f"Dataset set: {dataset_set}")
    print(f"Data dir: {data_dir}")
    print(f"Sample window: {window_size} samples")
    print(f"Sample stride: {sample_stride} samples")

    # Count total files
    set_dir = os.path.join(data_dir, dataset_set)
    if os.path.exists(set_dir):
        n_total = len([f for f in os.listdir(set_dir)
                       if os.path.isfile(os.path.join(set_dir, f)) and not f.startswith('.')])
    else:
        raise ValueError(f"Dataset directory not found: {set_dir}")

    # Calculate file index splits (within healthy period)
    healthy_end = int(n_total * HEALTHY_RATIO)
    train_end = int(healthy_end * 0.75)      # 75% of healthy for training
    val_end = int(healthy_end * 0.875)       # Next 12.5% for validation
    baseline_end = healthy_end               # Final 12.5% for test baseline

    print(f"Total files: {n_total}")
    print(f"Healthy period ends at file: {healthy_end}")
    print(f"Train: files 0-{train_end} ({train_end} files)")
    print(f"Val: files {train_end}-{val_end} ({val_end - train_end} files)")
    print(f"Test Baseline: files {val_end}-{baseline_end} ({baseline_end - val_end} files)")
    print(f"Test Faults: files {baseline_end}-{n_total} ({n_total - baseline_end} files)")

    # Training dataset (healthy only, first portion)
    print("\n[1/3] Creating TRAINING dataset (healthy period)...")
    train_dataset = IMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,  # Single file per sample
        file_stride=train_stride,
        normalize=True,
        healthy_ratio=1.0,  # All files considered healthy for label assignment
        fault_filter={0},   # Only healthy labels
        max_files=train_end,
    )
    norm_stats = train_dataset.get_normalization_stats()

    # Validation dataset (healthy, middle portion)
    print("[2/3] Creating VALIDATION dataset (late healthy period)...")

    class ValidationIMSRawDataset(IMSRawDataset):
        """Wrapper to select validation portion of healthy data."""

        def _load_file_list(self):
            super()._load_file_list()
            # Keep only validation portion
            self.file_list = self.file_list[train_end:val_end]
            self.file_labels = self.file_labels[train_end:val_end]
            self.time_indices = self.time_indices[train_end:val_end]
            self.normalized_time = self.normalized_time[train_end:val_end]

    val_dataset = ValidationIMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,
        file_stride=val_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=1.0,
        fault_filter={0},
        require_stats=True,
    )

    # Test datasets
    print("[3/3] Creating TEST datasets...")
    test_datasets = {}

    # Baseline (late healthy, separate from validation)
    class BaselineIMSRawDataset(IMSRawDataset):
        """Wrapper to select baseline (late healthy) portion for test."""

        def _load_file_list(self):
            super()._load_file_list()
            self.file_list = self.file_list[val_end:baseline_end]
            self.file_labels = self.file_labels[val_end:baseline_end]
            self.time_indices = self.time_indices[val_end:baseline_end]
            self.normalized_time = self.normalized_time[val_end:baseline_end]

    print("  - Baseline (late healthy, separate from validation)")
    test_datasets["baseline"] = BaselineIMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,
        file_stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=1.0,
        fault_filter={0},
        require_stats=True,
    )

    # Test (degraded + faulty) - starts after baseline_end
    class TestIMSRawDataset(IMSRawDataset):
        """Wrapper to select test portion of data (degradation period)."""

        def _load_file_list(self):
            super()._load_file_list()
            self.file_list = self.file_list[baseline_end:]
            self.file_labels = self.file_labels[baseline_end:]
            self.time_indices = self.time_indices[baseline_end:]
            self.normalized_time = self.normalized_time[baseline_end:]

    print("  - Faults (degraded + faulty)")
    test_all = TestIMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,
        file_stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=HEALTHY_RATIO,
        require_stats=True,
    )
    test_datasets["faults_all"] = test_all

    # Degraded only
    print("  - Degraded only")
    test_datasets["degraded"] = TestIMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,
        file_stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        healthy_ratio=HEALTHY_RATIO,
        fault_filter={1},
        require_stats=True,
    )

    # Faulty only
    print("  - Faulty only")
    test_datasets["faulty"] = TestIMSRawDataset(
        data_dir=data_dir,
        dataset_set=dataset_set,
        sample_window=window_size,
        sample_stride=sample_stride,
        file_window=1,
        file_stride=test_stride,
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

    print(f"\nTrain loader: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"Val loader: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"Test loaders: {list(test_loaders.keys())}")
    for name, loader in test_loaders.items():
        print(f"  - {name}: {len(loader)} batches ({len(loader.dataset)} samples)")

    return train_loader, val_loader, test_loaders


# Register the adapter
register_adapter(
    DatasetAdapter(
        key="ims-raw",
        description="NASA IMS Bearing raw accelerometer data (20kHz vibration signals for spectral analysis).",
        default_data_dir=os.path.join("data", "IMS_Bearing"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=IMSRawDataset,
        control_names_fn=lambda _, __=None: get_ims_raw_control_variables(use_time=False),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ["faults_all", "degraded", "faulty"],
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
