"""
PRONTO Merged Dataset Adapter (15 Variables)

Provides DataLoader factory for the PRONTO benchmark dataset using
preprocessed merged CSV files with 15 variables (4 conditioning + 11 measurement).
"""
from __future__ import annotations

import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional

from dystgat.src.data.pronto_merged_config import (
    MEASUREMENT_VARS,
    CONDITIONING_VARS,
)
from dystgat.src.data.pronto_merged_dataset import PRONTOMergedDataset
from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    """
    Resolve split key to list of source identifiers for merged dataset.

    Returns:
        List of fault type names
    """
    if split_key == "train":
        return ["train"]
    elif split_key == "val":
        return ["val"]
    elif split_key == "test":
        return ["normal", "slugging", "air_blockage", "air_leakage", "diverted_flow"]
    elif split_key == "test_normal":
        return ["test_normal"]
    elif split_key == "normal":
        return ["normal"]
    elif split_key == "slugging" or split_key == "slug":
        return ["slugging"]
    elif split_key == "air_blockage" or split_key == "blockage":
        return ["air_blockage"]
    elif split_key == "air_leakage" or split_key == "leakage":
        return ["air_leakage"]
    elif split_key == "diverted_flow" or split_key == "diverted":
        return ["diverted_flow"]
    elif split_key == "faults":
        return ["air_blockage", "air_leakage", "diverted_flow"]
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
    split_mode: str = 'segment_shuffle',
    n_segments: int = 10,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    random_seed: int = 42,
    train_segments: List[int] | None = None,
    val_segments: List[int] | None = None,
    test_segments: List[int] | None = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train, validation, and test DataLoaders for PRONTO merged dataset.

    Uses preprocessed merged CSV files with 15 variables.

    Args:
        window_size: Number of timesteps per window
        batch_size: Batch size for DataLoader
        train_stride: Stride between windows for training
        val_stride: Stride between windows for validation
        test_stride: Stride between windows for testing (defaults to val_stride)
        data_dir: Path to 15var_merged directory
        num_workers: Number of workers for DataLoader
        distributed: Whether to use distributed training
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training
        baseline_from: Unused (kept for backward compatibility)
        severity_range: Unused with merged dataset
        feature_option: Unused (kept for backward compatibility)
        fault_keys: Unused (kept for backward compatibility)
        pred_horizon: Number of future timesteps for prediction
        split_mode: How to split data:
            - 'temporal': Traditional 70/30 temporal split
            - 'segment_shuffle': Shuffle segments to balance operating conditions (default)
        n_segments: Number of segments for segment_shuffle mode (default: 10)
        split_ratios: Ratios for train/val/test when using segment_shuffle (default: 0.7/0.2/0.1)
        random_seed: Random seed for reproducibility
        train_segments: Explicit list of segment indices for training (overrides split_ratios)
        val_segments: Explicit list of segment indices for validation (overrides split_ratios)
        test_segments: Explicit list of segment indices for testing (overrides split_ratios)

    Returns:
        Tuple of (train_loader, val_loader, test_loaders_dict)
    """
    if test_stride is None:
        test_stride = val_stride

    # Data directory should point to 15var_merged folder
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"PRONTO merged data directory not found: {data_dir}")

    # Common kwargs for split mode
    shuffle_kwargs = {
        'split_mode': split_mode,
        'n_segments': n_segments,
        'split_ratios': split_ratios,
        'random_seed': random_seed,
        'train_segments': train_segments,
        'val_segments': val_segments,
        'test_segments': test_segments,
    }

    print("=" * 70)
    print(f"CREATING PRONTO MERGED DATALOADERS (15 Variables, mode={split_mode})")
    print("=" * 70)

    # 1. Training Dataset (Normal only - no fault data in training)
    if split_mode == 'segment_shuffle':
        print(f"[1/3] Loading TRAINING dataset (segment_shuffle: {split_ratios[0]*100:.0f}% of normal)...")
    else:
        print("[1/3] Loading TRAINING dataset (temporal: 70% of normal)...")
    train_dataset = PRONTOMergedDataset(
        data_dir=data_dir,
        split='train',
        window_size=window_size,
        stride=train_stride,
        normalize=True,
        pred_horizon=pred_horizon or 0,
        **shuffle_kwargs,
    )
    norm_stats = train_dataset.compute_normalization_stats()

    # 2. Validation Dataset (Normal only)
    if split_mode == 'segment_shuffle':
        print(f"[2/3] Loading VALIDATION dataset (segment_shuffle: {split_ratios[1]*100:.0f}% of normal)...")
    else:
        print("[2/3] Loading VALIDATION dataset (temporal: 30% of normal)...")
    val_dataset = PRONTOMergedDataset(
        data_dir=data_dir,
        split='val',
        window_size=window_size,
        stride=val_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
        **shuffle_kwargs,
    )

    # 3. Test Datasets
    print("[3/3] Loading TEST datasets...")
    test_datasets = {}

    # Test Normal - uses segment shuffle mode if enabled (for baseline comparison)
    if split_mode == 'segment_shuffle':
        print(f"    test_normal: segment_shuffle (remaining {split_ratios[2]*100:.0f}% of normal)")
        test_datasets["normal"] = PRONTOMergedDataset(
            data_dir=data_dir,
            split='test_normal',
            window_size=window_size,
            stride=test_stride,
            normalize=True,
            normalization_stats=norm_stats,
            require_stats=True,
            pred_horizon=pred_horizon or 0,
            **shuffle_kwargs,
        )
    else:
        test_datasets["normal"] = PRONTOMergedDataset(
            data_dir=data_dir,
            split='normal',
            window_size=window_size,
            stride=test_stride,
            normalize=True,
            normalization_stats=norm_stats,
            require_stats=True,
            pred_horizon=pred_horizon or 0,
            split_mode='temporal',  # Fault sets always use temporal
        )

    # Slugging - full file, temporal mode
    test_datasets["slugging"] = PRONTOMergedDataset(
        data_dir=data_dir,
        split='slugging',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
        split_mode='temporal',  # Fault sets always use temporal
    )

    # Air Blockage - full file, temporal mode
    test_datasets["air_blockage"] = PRONTOMergedDataset(
        data_dir=data_dir,
        split='air_blockage',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
        split_mode='temporal',  # Fault sets always use temporal
    )

    # Air Leakage - full file, temporal mode
    test_datasets["air_leakage"] = PRONTOMergedDataset(
        data_dir=data_dir,
        split='air_leakage',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
        split_mode='temporal',  # Fault sets always use temporal
    )

    # Diverted Flow - full file, temporal mode
    test_datasets["diverted_flow"] = PRONTOMergedDataset(
        data_dir=data_dir,
        split='diverted_flow',
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
        split_mode='temporal',  # Fault sets always use temporal
    )

    # Combined faults for convenience (air_blockage + air_leakage + diverted_flow)
    test_datasets["faults_all"] = ConcatDataset([
        test_datasets["air_blockage"],
        test_datasets["air_leakage"],
        test_datasets["diverted_flow"],
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
        key="pronto_merged",
        description="PRONTO Benchmark Dataset (15 Variables - Merged/Preprocessed CSV).",
        default_data_dir=os.path.join("data", "pronto", "pronto_benchmark", "Pre-processed data", "Process data", "15var_merged"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=PRONTOMergedDataset,
        control_names_fn=lambda _, __=None: CONDITIONING_VARS.copy(),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ["air_blockage", "air_leakage", "diverted_flow", "slugging", "normal"],
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
