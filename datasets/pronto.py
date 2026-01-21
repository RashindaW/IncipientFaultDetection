from __future__ import annotations

import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Optional

from dystgat.src.data.pronto_column_config import MEASUREMENT_VARS, CONTROL_VARS
from dystgat.src.data.pronto_dataset import PRONTODataset
from .registry import DatasetAdapter, register_adapter

# Define file paths
HEALTHY_FILE = "HealthySet.mat"
SLUG_FILE = "SlugSet.mat"
FAULT_FILES = [
    "Blockage_120air_01water.mat",
    "Blockage_150air_05water.mat",
    "Leakage_120air_01water.mat",
    "Leakage_150air_05water.mat",
    "Diverted_120air_01water.mat",
    "Diverted_150air_05water.mat"
]

def _resolve_split_files(split_key: str) -> List[str]:
    # Used by reconstruction/plotting scripts
    if split_key == "train":
        return [HEALTHY_FILE]
    elif split_key == "val":
        return [HEALTHY_FILE]
    elif split_key == "test":
        return [HEALTHY_FILE, SLUG_FILE] + FAULT_FILES
    elif split_key == "slug":
        return [SLUG_FILE]
    elif split_key == "faults":
        return FAULT_FILES
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
    
    if test_stride is None:
        test_stride = val_stride
        
    full_data_dir = os.path.join(data_dir, "Pre-processed data", "Process data")
    if not os.path.exists(full_data_dir):
         # Fallback if data_dir is already the full path
         full_data_dir = data_dir

    print("=" * 70)
    print("CREATING PRONTO DATALOADERS")
    print("=" * 70)

    # 1. Train Dataset (Healthy: Segment 0 only)
    # NOTE: Using only segment 0 for training to prevent overlap with validation (segment 1)
    print("[1/3] Loading FAULT-FREE TRAINING dataset (Segment 0)...")
    train_dataset = PRONTODataset(
        data_files=[HEALTHY_FILE],
        window_size=window_size,
        stride=train_stride,
        data_dir=full_data_dir,
        normalize=True,
        segments_to_load=[0],  # Train on segment 0 only (val uses segment 1)
        pred_horizon=pred_horizon or 0,
    )
    norm_stats = train_dataset.get_normalization_stats()
    
    # 2. Validation (Healthy: Segment 1 - late training period)
    # NOTE: Segment 1 is used for validation to avoid data leakage with test baseline
    print("[2/3] Loading FAULT-FREE VALIDATION dataset (Segment 1)...")
    val_dataset = PRONTODataset(
        data_files=[HEALTHY_FILE],
        window_size=window_size,
        stride=val_stride,
        data_dir=full_data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        segments_to_load=[1],  # Validate on segment 1 (not same as test baseline)
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )

    print("[3/3] Loading TEST datasets...")
    test_datasets = {}

    # Baseline (Normal) - Use Segment 2 (completely held out from val)
    # IMPORTANT: This must be different from validation segment to avoid data leakage
    test_datasets["baseline"] = PRONTODataset(
        data_files=[HEALTHY_FILE],
        window_size=window_size,
        stride=test_stride,
        data_dir=full_data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        segments_to_load=[2],  # Segment 2 only for test baseline
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )
    
    # Slugging (Novel OC)
    test_datasets["slugging"] = PRONTODataset(
        data_files=[SLUG_FILE],
        window_size=window_size,
        stride=test_stride,
        data_dir=full_data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        pred_horizon=pred_horizon or 0,
    )
    
    # Faults (All Combined)
    test_datasets["faults_all"] = PRONTODataset(
        data_files=FAULT_FILES,
        window_size=window_size,
        stride=test_stride,
        data_dir=full_data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        require_stats=True,
        severity_range=severity_range,
        pred_horizon=pred_horizon or 0,
    )
    
    # Individual Faults
    for f_file in FAULT_FILES:
        name = os.path.splitext(f_file)[0]
        test_datasets[name] = PRONTODataset(
            data_files=[f_file],
            window_size=window_size,
            stride=test_stride,
            data_dir=full_data_dir,
            normalize=True,
            normalization_stats=norm_stats,
            require_stats=True,
            severity_range=severity_range,
            pred_horizon=pred_horizon or 0,
        )

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
        
    return train_loader, val_loader, test_loaders

register_adapter(
    DatasetAdapter(
        key="pronto",
        description="PRONTO Benchmark Dataset (Matlab format).",
        default_data_dir=os.path.join("data", "pronto", "pronto_benchmark"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=PRONTODataset,
        control_names_fn=lambda _, __=None: CONTROL_VARS.copy(),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=lambda: ["faults_all"],
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
