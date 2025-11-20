from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dyedgegat.src.data.ashrae_column_config import (
    BASELINE_FILES, 
    FAULT_FILES, 
    MEASUREMENT_VARS,
    BENCHMARK_DIR,
    REFRIGERANT_LEAK_DIR,
    BASELINE_FAULT_CODE_WHITELIST,
    BASELINE_UNIT_STATUS_WHITELIST,
)
from dyedgegat.src.data.ashrae_dataset import (
    ASHRAEDataset, 
    ASHRAEFaultDataset,
    get_ashrae_control_variable_names,
)

from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    """Resolve dataset split key to list of file paths."""
    key = split_key.lower()
    
    # Handle baseline/validation splits
    if key in ("baseline", "val"):
        # Return validation files with full path from benchmark directory
        return [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES["val"]]
    
    if key == "train":
        # Return training files with full path from benchmark directory
        return [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES["train"]]
    
    # Handle fault files
    if split_key in FAULT_FILES:
        # Return fault file with full path from refrigerant leak directory
        return [os.path.join(REFRIGERANT_LEAK_DIR, FAULT_FILES[split_key])]
    
    # Try case-insensitive match for fault names
    for fault_name, fault_file in FAULT_FILES.items():
        if fault_name.lower() == key:
            return [os.path.join(REFRIGERANT_LEAK_DIR, fault_file)]
    
    raise ValueError(
        f"Unknown dataset split '{split_key}'. Valid options: "
        f"'train', 'baseline', 'val', or one of {list(FAULT_FILES.keys())}"
    )


def _list_faults() -> List[str]:
    """List all available fault keys."""
    return sorted(FAULT_FILES.keys())


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
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train, validation, and test dataloaders for ASHRAE dataset.
    
    Args:
        window_size: Length of sliding window
        batch_size: Number of samples per batch
        train_stride: Stride for training data sliding window
        val_stride: Stride for validation data (larger=faster, fewer samples)
        test_stride: Stride for test data (defaults to val_stride when None)
        data_dir: Directory containing ASHRAE XLS files
        num_workers: Number of worker processes for data loading
        distributed: Enable DistributedSampler for multi-process training
        rank: Rank of the current process (used when distributed=True)
        world_size: Total number of processes participating (used when distributed=True)
    
    Returns:
        train_loader: DataLoader for training (benchmark tests)
        val_loader: DataLoader for validation (near normal tests)
        test_loaders: Dict of DataLoaders for testing (baseline + refrigerant leak faults)
    """

    if test_stride is None:
        test_stride = val_stride
    
    print("=" * 70)
    print("CREATING ASHRAE 1043-RP DATALOADERS")
    print("=" * 70)
    
    # ========== Create Training Dataset ==========
    print("\n[1/3] Creating TRAINING dataset (Benchmark Tests)...")
    train_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES['train']]
    filter_kwargs = dict(
        fault_code_whitelist=BASELINE_FAULT_CODE_WHITELIST,
        unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
    )

    train_dataset = ASHRAEDataset(
        data_files=train_files,
        window_size=window_size,
        stride=train_stride,
        data_dir=data_dir,
        normalize=True,
        **filter_kwargs,
    )
    
    # Get normalization statistics from training data
    norm_stats = train_dataset.get_normalization_stats()
    
    # ========== Create Validation Dataset ==========
    print("\n[2/3] Creating VALIDATION dataset (Near Normal Tests)...")
    val_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES['val']]
    val_dataset = ASHRAEDataset(
        data_files=val_files,
        window_size=window_size,
        stride=val_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,  # Use training stats
        **filter_kwargs,
    )
    
    # ========== Create Test Datasets ==========
    print("\n[3/3] Creating TEST datasets...")
    test_datasets = {}
    
    # Baseline test (normal operation)
    print("  - Baseline (near normal operation)")
    baseline_test_dataset = ASHRAEDataset(
        data_files=val_files,
        window_size=window_size,
        stride=test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        **filter_kwargs,
    )
    test_datasets['baseline'] = baseline_test_dataset
    
    # Fault datasets - Refrigerant leak
    for fault_idx, (fault_name, fault_file) in enumerate(FAULT_FILES.items(), start=1):
        print(f"  - {fault_name}")
        fault_file_path = os.path.join(REFRIGERANT_LEAK_DIR, fault_file)
        fault_dataset = ASHRAEFaultDataset(
            data_files=[fault_file_path],
            fault_label=fault_idx,  # Assign sequential fault labels
            window_size=window_size,
            stride=test_stride,
            data_dir=data_dir,
            normalize=True,
            normalization_stats=norm_stats,  # Use training stats
        )
        test_datasets[fault_name] = fault_dataset
    
    # ========== Create DataLoaders ==========
    print(f"\nCreating DataLoaders (batch_size={batch_size})...")

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        test_samplers = {
            name: DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            for name, dataset in test_datasets.items()
        }
    else:
        train_sampler = val_sampler = None
        test_samplers = {name: None for name in test_datasets}

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
        test_loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=test_samplers[name],
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("ASHRAE DATALOADER SUMMARY")
    print("=" * 70)
    print(f"Training (Benchmark Tests):")
    print(f"  Files: {len(train_files)}")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)}")
    print()
    print(f"Validation (Near Normal Tests):")
    print(f"  Files: {len(val_files)}")
    print(f"  Samples: {len(val_dataset)}")
    print(f"  Batches: {len(val_loader)}")
    print()
    print(f"Testing:")
    for name, loader in test_loaders.items():
        print(f"  {name:30s}: {len(loader.dataset):6d} samples, {len(loader):4d} batches")
    print()
    print(f"Data dimensions:")
    print(f"  Measurement variables: {train_dataset.n_measurement_vars}")
    print(f"  Control variables: {train_dataset.n_control_vars}")
    print(f"  Window size: {window_size}")
    if distributed:
        print(f"Distributed samplers enabled (rank {rank}/{world_size}).")
    print("=" * 70)

    return train_loader, val_loader, test_loaders


# Default data directory for ASHRAE dataset
ASHRAE_DEFAULT_DIR = os.path.join("data", "ASHRAE_1043_RP")

# Register the ASHRAE adapter
register_adapter(
    DatasetAdapter(
        key="ashrae",
        description="ASHRAE 1043-RP water-cooled chiller dataset. Training on benchmark tests, testing on refrigerant leak.",
        default_data_dir=ASHRAE_DEFAULT_DIR,
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=ASHRAEDataset,
        control_names_fn=get_ashrae_control_variable_names,
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=_list_faults,
    )
)
