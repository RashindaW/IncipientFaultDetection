"""
PRONTO Dataset Adapter

Provides DataLoader factory for the PRONTO benchmark dataset,
loading data from raw CSV files to ensure consistent column order.
"""
from __future__ import annotations

import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional

from dystgat.src.data.pronto_column_config import MEASUREMENT_VARS, CONTROL_VARS
from dystgat.src.data.pronto_dataset import PRONTODataset, PRONTODatasetLegacy
from dystgat.src.data.pronto_raw_loader import DATA_SPLITS
from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    """
    Resolve split key to list of source identifiers.

    This is used by reconstruction/plotting scripts for backward compatibility.
    Returns split names that can be used with the new CSV-based loader.
    """
    if split_key == "train":
        return ["train"]
    elif split_key == "val":
        return ["val"]
    elif split_key == "test":
        return ["test_baseline", "test_slugging", "test_blockage", "test_leakage", "test_diverted"]
    elif split_key == "slug" or split_key == "slugging":
        return ["test_slugging"]
    elif split_key == "faults":
        return ["test_blockage", "test_leakage", "test_diverted"]
    elif split_key == "baseline":
        return ["test_baseline"]
    elif split_key in DATA_SPLITS:
        return [split_key]
    else:
        return []


def _create_dataloaders(
    window_size: int,
    batch_size: int,
    train_stride: int,
    val_stride: int,
    test_stride: int | None,
    data_dir: str,
    num_workers: int,
    distributed: bool,
    rank: int,
    world_size: int,
    baseline_from: str = "val",
    severity_range: Tuple[int, int] | None = None,
    feature_option: str | None = None,
    fault_keys: List[str] | None = None,
    pred_horizon: int | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train, validation, and test DataLoaders for PRONTO dataset.

    Uses raw CSV loading to ensure consistent column order matching DyEdgeGAT.

    Args:
        window_size: Number of timesteps per window
        batch_size: Batch size for DataLoader
        train_stride: Stride between windows for training
        val_stride: Stride between windows for validation
        test_stride: Stride between windows for testing (defaults to val_stride)
        data_dir: Path to PRONTO benchmark data directory
        num_workers: Number of workers for DataLoader
        distributed: Whether to use distributed training
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training
        baseline_from: Unused (kept for backward compatibility)
        severity_range: Unused with raw CSV loading
        feature_option: Unused (kept for backward compatibility)
        fault_keys: Unused (kept for backward compatibility)
        pred_horizon: Number of future timesteps for prediction

    Returns:
        Tuple of (train_loader, val_loader, test_loaders_dict)
    """
    if test_stride is None:
        test_stride = val_stride

    # Data directory should point to pronto_benchmark folder
    # which contains the scenario folders (C0, C1, C2, C3)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"PRONTO data directory not found: {data_dir}")

    print("=" * 70)
    print("CREATING PRONTO DATALOADERS (Raw CSV Loading)")
    print("=" * 70)

    # 1. Training Dataset
    print("[1/3] Loading TRAINING dataset (Test11, 70% of Normal data)...")
    train_dataset = PRONTODataset(
        data_dir=data_dir,
        split='train',
        window_size=window_size,
        stride=train_stride,
        normalize=True,
        pred_horizon=pred_horizon or 0,
    )
    norm_stats = train_dataset.get_normalization_stats()

    # 2. Validation Dataset
    print("[2/3] Loading VALIDATION dataset (Test11, 30% of Normal data)...")
    val_dataset = PRONTODataset(
        data_dir=data_dir,
        split='val',
        window_size=window_size,
        stride=val_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # 3. Test Datasets
    print("[3/3] Loading TEST datasets...")
    test_datasets = {}

    # Baseline (Normal from Test9)
    test_datasets["baseline"] = PRONTODataset(
        data_dir=data_dir,
        split='test_baseline',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # Slugging (Novel OC from Test9)
    test_datasets["slugging"] = PRONTODataset(
        data_dir=data_dir,
        split='test_slugging',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # Blockage Faults (Test2 + Test3)
    test_datasets["blockage"] = PRONTODataset(
        data_dir=data_dir,
        split='test_blockage',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # Leakage Faults (Test4 + Test5 + Test6)
    test_datasets["leakage"] = PRONTODataset(
        data_dir=data_dir,
        split='test_leakage',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # Diverted Flow Faults (Test7 + Test8)
    test_datasets["diverted"] = PRONTODataset(
        data_dir=data_dir,
        split='test_diverted',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    # Combined faults for convenience (blockage + leakage + diverted)
    # Uses ConcatDataset to create a single dataset from individual fault datasets
    test_datasets["faults_all"] = ConcatDataset([
        test_datasets["blockage"],
        test_datasets["leakage"],
        test_datasets["diverted"],
    ])
    print(f"    faults_all samples: {len(test_datasets['faults_all'])}")

    # Samplers for Distributed Training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_samplers = {
            name: DistributedSampler(
                ds, num_replicas=world_size, rank=rank, shuffle=False
            )
            for name, ds in test_datasets.items()
        }
    else:
        train_sampler = None
        val_sampler = None
        test_samplers = {name: None for name in test_datasets}

    # Create Loaders
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loaders = {}
    for name, dataset in test_datasets.items():
        test_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_samplers[name],
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    print("=" * 70)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    for name, loader in test_loaders.items():
        print(f"Test [{name}] samples: {len(loader.dataset)}")
    print("=" * 70)

    return train_loader, val_loader, test_loaders


register_adapter(
    DatasetAdapter(
        key="pronto",
        description="PRONTO Benchmark Dataset (Raw CSV loading).",
        default_data_dir=os.path.join("data", "pronto", "pronto_benchmark"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=PRONTODataset,
        control_names_fn=lambda _, __=None: CONTROL_VARS.copy(),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ["blockage", "leakage", "diverted", "slugging"],
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
