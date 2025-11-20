from __future__ import annotations

import os
from typing import List

from dyedgegat.src.data.column_config import BASELINE_FILES, FAULT_FILES, MEASUREMENT_VARS
from dyedgegat.src.data.dataloader import create_dataloaders
from dyedgegat.src.data.dataset import RefrigerationDataset, get_control_variable_names

from .registry import DatasetAdapter, register_adapter


def _resolve_split_files(split_key: str) -> List[str]:
    key = split_key.lower()
    if key in ("baseline", "val"):
        return list(BASELINE_FILES["val"])
    if key == "train":
        return list(BASELINE_FILES["train"])
    if split_key in FAULT_FILES:
        return [FAULT_FILES[split_key]]
    for fault_name, fault_file in FAULT_FILES.items():
        if fault_name.lower() == key:
            return [fault_file]
    raise ValueError(
        f"Unknown dataset split '{split_key}'. Valid options: "
        f"'train', 'baseline', 'val', or one of {list(FAULT_FILES.keys())}"
    )


def _list_faults() -> List[str]:
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
    baseline_from: str = "val",
):
    return create_dataloaders(
        window_size=window_size,
        batch_size=batch_size,
        train_stride=train_stride,
        val_stride=val_stride,
        test_stride=test_stride,
        data_dir=data_dir,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )


CO2_DEFAULT_DIR = os.path.join("data", "co2", "raw")
CO2_1MIN_DEFAULT_DIR = os.path.join("data", "co2", "1min")

register_adapter(
    DatasetAdapter(
        key="co2",
        description="CO₂ supermarket refrigeration benchmark (original cadence).",
        default_data_dir=CO2_DEFAULT_DIR,
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=RefrigerationDataset,
        control_names_fn=get_control_variable_names,
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=_list_faults,
    )
)

register_adapter(
    DatasetAdapter(
        key="co2_1min",
        description="CO₂ supermarket refrigeration benchmark (1-minute aggregation).",
        default_data_dir=CO2_1MIN_DEFAULT_DIR,
        measurement_vars=MEASUREMENT_VARS,
        dataset_cls=RefrigerationDataset,
        control_names_fn=get_control_variable_names,
        dataloader_factory=_create_dataloaders,
        resolve_split_files_fn=_resolve_split_files,
        list_fault_keys_fn=_list_faults,
    )
)
