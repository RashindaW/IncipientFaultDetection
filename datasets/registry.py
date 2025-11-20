from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

DatasetType = TypeVar("DatasetType")


@dataclass
class DatasetAdapter:
    """
    Adapter describing how to use a particular dataset with the DyEdgeGAT pipeline.

    Attributes:
        key: Short identifier (e.g., "co2", "co2_1min").
        description: Human-readable summary of the dataset.
        default_data_dir: Suggested on-disk location for the dataset files.
        measurement_vars: Ordered list of measurement channel names.
        dataset_cls: Dataset implementation (e.g., RefrigerationDataset).
        control_names_fn: Callable returning control variable names for a given data dir.
        dataloader_factory: Callable creating (train, val, test) dataloaders.
        resolve_split_files_fn: Maps an evaluation split key to source filenames.
        list_fault_keys_fn: Returns the known fault identifiers for convenience.
        supports_training/testing/plotting: Feature flags for readiness.
    """

    key: str
    description: str
    default_data_dir: Optional[str]
    measurement_vars: Sequence[str]
    dataset_cls: Optional[Type[DatasetType]]
    control_names_fn: Optional[Callable[[str], List[str]]]
    dataloader_factory: Optional[
        Callable[
            [int, int, int, int, Optional[int], str, int, bool, int, int, str],
            Tuple[DatasetType, DatasetType, Dict[str, DatasetType]],
        ]
    ]
    resolve_split_files_fn: Optional[Callable[[str], List[str]]]
    list_fault_keys_fn: Optional[Callable[[], List[str]]]
    supports_training: bool = True
    supports_testing: bool = True
    supports_plotting: bool = True

    def measurement_count(self) -> int:
        return len(self.measurement_vars)

    def ensure(self, capability: str) -> None:
        flag = {
            "training": self.supports_training,
            "testing": self.supports_testing,
            "plotting": self.supports_plotting,
        }.get(capability, False)
        if not flag:
            raise NotImplementedError(
                f"Dataset '{self.key}' does not currently support {capability}. {self.description}"
            )

    def get_default_data_dir(self) -> Optional[str]:
        return self.default_data_dir

    def get_control_variables(self, data_dir: str) -> List[str]:
        if self.control_names_fn is None:
            raise NotImplementedError(f"Dataset '{self.key}' has no control-variable resolver yet.")
        return list(self.control_names_fn(data_dir))

    def create_dataloaders(
        self,
        *,
        window_size: int,
        batch_size: int,
        train_stride: int,
        val_stride: int,
        test_stride: Optional[int],
        data_dir: str,
        num_workers: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        baseline_from: str = "val",
    ):
        self.ensure("training")
        if self.dataloader_factory is None:
            raise NotImplementedError(f"Dataset '{self.key}' does not provide dataloader construction yet.")
        return self.dataloader_factory(
            window_size,
            batch_size,
            train_stride,
            val_stride,
            test_stride,
            data_dir,
            num_workers,
            distributed,
            rank,
            world_size,
            baseline_from,
        )

    def resolve_split_files(self, split_key: str) -> List[str]:
        if self.resolve_split_files_fn is None:
            raise NotImplementedError(f"Dataset '{self.key}' does not expose evaluation split resolution.")
        return self.resolve_split_files_fn(split_key)

    def list_fault_keys(self) -> List[str]:
        if self.list_fault_keys_fn is None:
            return []
        return list(self.list_fault_keys_fn())


DATASET_REGISTRY: Dict[str, DatasetAdapter] = {}


def register_adapter(adapter: DatasetAdapter) -> None:
    if adapter.key in DATASET_REGISTRY:
        raise KeyError(f"Dataset adapter '{adapter.key}' already registered.")
    DATASET_REGISTRY[adapter.key] = adapter


def get_adapter(key: str) -> DatasetAdapter:
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset adapter '{key}'. Available adapters: {known}")
    return DATASET_REGISTRY[key]


def list_adapters() -> Dict[str, DatasetAdapter]:
    return dict(DATASET_REGISTRY)


def list_adapter_keys() -> List[str]:
    return sorted(DATASET_REGISTRY)
