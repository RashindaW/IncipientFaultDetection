"""
DataLoader creation utilities for DualSTAGE.

Creates PyTorch Geometric DataLoaders with proper batching for training and evaluation.
"""

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Optional, Dict
from .dataset import RefrigerationDataset, FaultDataset
from .column_config import BASELINE_FILES, FAULT_FILES


def create_dataloaders(
    window_size: int = 15,
    batch_size: int = 64,
    train_stride: int = 1,
    val_stride: int = 5,
    test_stride: Optional[int] = None,
    data_dir: str = 'Dataset',
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    baseline_from: str = "val",
    pred_horizon: int = 0,
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        window_size: Length of sliding window
        batch_size: Number of samples per batch
        train_stride: Stride for training data sliding window
        val_stride: Stride for validation data (larger=faster, fewer samples)
        test_stride: Stride for test data (defaults to val_stride when None)
        data_dir: Directory containing CSV files
        num_workers: Number of worker processes for data loading
        distributed: Enable DistributedSampler for multi-process training
        rank: Rank of the current process (used when distributed=True)
        world_size: Total number of processes participating (used when distributed=True)
    Returns:
        train_loader: DataLoader for training (baseline data only)
        val_loader: DataLoader for validation (baseline data)
        test_loaders: Dict of DataLoaders for testing (baseline + all fault types)
    """

    if test_stride is None:
        test_stride = val_stride
    
    print("=" * 70)
    print("CREATING DATALOADERS")
    print("=" * 70)
    
    # ========== Create Training Dataset ==========
    print("\n[1/3] Creating TRAINING dataset...")
    train_dataset = RefrigerationDataset(
        data_files=BASELINE_FILES['train'],
        window_size=window_size,
        stride=train_stride,
        data_dir=data_dir,
        normalize=True,
        pred_horizon=pred_horizon,
    )
    
    # Get normalization statistics from training data
    norm_stats = train_dataset.get_normalization_stats()
    
    # ========== Create Validation Dataset ==========
    print("\n[2/3] Creating VALIDATION dataset...")
    val_dataset = RefrigerationDataset(
        data_files=BASELINE_FILES['val'],
        window_size=window_size,
        stride=val_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,  # Use training stats
        pred_horizon=pred_horizon,
    )
    
    # ========== Create Test Datasets ==========
    print("\n[3/3] Creating TEST datasets...")
    test_datasets = {}
    
    # Baseline test (normal operation)
    print("  - Baseline (normal operation)")
    baseline_test_dataset = RefrigerationDataset(
        data_files=BASELINE_FILES['val'],
        window_size=window_size,
        stride=test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        pred_horizon=pred_horizon,
    )
    test_datasets['baseline'] = baseline_test_dataset
    
    # Fault datasets
    for fault_name, fault_file in FAULT_FILES.items():
        print(f"  - {fault_name}")
        fault_dataset = FaultDataset(
            data_files=[fault_file],
            fault_label=int(fault_name[5]),  # Extract fault number from name
            window_size=window_size,
            stride=test_stride,
            data_dir=data_dir,
            normalize=True,
            normalization_stats=norm_stats,  # Use training stats
            pred_horizon=pred_horizon,
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
    print("DATALOADER SUMMARY")
    print("=" * 70)
    print(f"Training:")
    print(f"  Files: {len(BASELINE_FILES['train'])}")
    print(f"  Samples: {len(train_dataset)}")
    print(f"  Batches: {len(train_loader)}")
    print()
    print(f"Validation:")
    print(f"  Files: {len(BASELINE_FILES['val'])}")
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


def create_inference_loader(
    data_file: str,
    window_size: int = 15,
    batch_size: int = 64,
    normalization_stats: Optional[Tuple] = None,
    data_dir: str = 'Dataset',
) -> DataLoader:
    """
    Create a DataLoader for inference on a single file.
    
    Args:
        data_file: Single CSV filename to load
        window_size: Length of sliding window
        batch_size: Batch size for inference
        normalization_stats: Pre-computed normalization statistics
        data_dir: Directory containing CSV files
    
    Returns:
        DataLoader for the specified file
    """
    dataset = RefrigerationDataset(
        data_files=[data_file],
        window_size=window_size,
        stride=1,  # No stride for inference (use all windows)
        data_dir=data_dir,
        normalize=(normalization_stats is not None),
        normalization_stats=normalization_stats,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    return loader


if __name__ == "__main__":
    # Test dataloader creation
    print("Testing dataloader creation...\n")
    
    train_loader, val_loader, test_loaders = create_dataloaders(
        window_size=15,
        batch_size=32,
        train_stride=5,  # Faster for testing
        val_stride=10,   # Faster for testing
        data_dir='../../Dataset',
    )
    
    print("\nTesting batch retrieval...")
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  x shape: {batch.x.shape}")
    print(f"  c shape: {batch.c.shape}")
    print(f"  edge_index shape: {batch.edge_index.shape}")
    print(f"  batch assignment: {batch.batch.shape}")
    print(f"  Unique batch IDs: {batch.batch.unique()}")
