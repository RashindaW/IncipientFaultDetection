from .registry import (
    DatasetAdapter,
    DATASET_REGISTRY,
    get_adapter,
    list_adapter_keys,
    list_adapters,
    register_adapter,
)

# Register built-in adapters
from . import co2  # noqa: F401
from . import tep  # noqa: F401

__all__ = [
    "DatasetAdapter",
    "DATASET_REGISTRY",
    "get_adapter",
    "list_adapter_keys",
    "list_adapters",
    "register_adapter",
]
