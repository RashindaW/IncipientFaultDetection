from __future__ import annotations

import os
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple

from dyedgegat.src.data.tep_column_config import (
    CONTROL_VARS,
    FAULT_FREE_TEST_FILE,
    FAULT_FREE_TRAIN_FILE,
    FAULTY_TEST_FILE,
    MEASUREMENT_VARS,
)
from dyedgegat.src.data.tep_dataset import TEPDataset

from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    key = split_key.lower()
    if key in ("train", "fault_free_train", "baseline_train"):
        return [FAULT_FREE_TRAIN_FILE]
    if key in ("val", "fault_free_test", "fault_free_testing", "baseline"):
        return [FAULT_FREE_TEST_FILE]
    if key in ("faulty_test", "faulty_testing", "test", "faults"):
        return [FAULTY_TEST_FILE]
    raise ValueError(
        f"Unknown TEP split '{split_key}'. Valid: train, val, faulty_test/faults."
    )


def _list_fault_keys() -> List[str]:
    return ["test_all_faults"]


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
) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    if test_stride is None:
        test_stride = val_stride

    print("=" * 70)
    print("CREATING TEP DATALOADERS")
    print("=" * 70)

    # Training on fault-free data
    print("\n[1/3] Loading FAULT-FREE TRAINING dataset...")
    train_dataset = TEPDataset(
        data_files=[FAULT_FREE_TRAIN_FILE],
        window_size=window_size,
        stride=train_stride,
        data_dir=data_dir,
        normalize=True,
        fault_filter=[0],
    )
    norm_stats = train_dataset.get_normalization_stats()

    # Validation on fault-free testing
    print("\n[2/3] Loading FAULT-FREE VALIDATION dataset...")
    val_dataset = TEPDataset(
        data_files=[FAULT_FREE_TEST_FILE],
        window_size=window_size,
        stride=val_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        fault_filter=[0],
    )

    # Testing on fault-free + all faults
    print("\n[3/3] Loading TEST datasets...")
    test_datasets = {}

    # 1. Baseline (Normal)
    baseline_file = FAULT_FREE_TEST_FILE if baseline_from == "val" else FAULT_FREE_TRAIN_FILE
    print(f"  - Baseline (from {baseline_from})")
    baseline_test = TEPDataset(
        data_files=[baseline_file],
        window_size=window_size,
        stride=test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        fault_filter=[0],
    )
    test_datasets["baseline"] = baseline_test

    # 2. Combined Faulty Test Set
    # This loads EVERYTHING in TEP_Faulty_Testing.RData (Faults 0, 1-20)
    # without filtering, preserving the natural sequence.
    print("  - Combined Faulty Test Set (All Faults + Interspersed Normal)")
    combined_test = TEPDataset(
        data_files=[FAULTY_TEST_FILE],
        window_size=window_size,
        stride=test_stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        fault_filter=None, # No filter = load everything
    )
    test_datasets["test_all_faults"] = combined_test

    pin_memory = torch.cuda.is_available()

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
        train_sampler = val_sampler = None
        test_samplers = {name: None for name in test_datasets}

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

    print("\n" + "=" * 70)
    print("TEP DATALOADER SUMMARY")
    print("=" * 70)
    print(f"Train (fault-free) samples: {len(train_dataset):6d} | batches: {len(train_loader):4d}")
    print(f"Val   (fault-free) samples: {len(val_dataset):6d} | batches: {len(val_loader):4d}")
    for name, loader in test_loaders.items():
        print(f"Test  {name:18s}: {len(loader.dataset):6d} samples | batches: {len(loader):4d}")
    print("=" * 70)

    return train_loader, val_loader, test_loaders


register_adapter(
    DatasetAdapter(
        key="tep",
        description="Tennessee Eastman Process multivariate dataset (RData format).",
        default_data_dir=os.path.join("data", "tep", "raw"),
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=TEPDataset,
        control_names_fn=lambda _: CONTROL_VARS.copy(),
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=_list_fault_keys,
        supports_training=True,
        supports_testing=True,
        supports_plotting=True,
    )
)
