from __future__ import annotations

import os
from typing import List

from .registry import DatasetAdapter, register_adapter


def _not_implemented(*args, **kwargs):
    raise NotImplementedError(
        "The Tennessee Eastman Process (TEP) dataset pipeline is not yet implemented. "
        "Use this adapter as a scaffold when adding preprocessing, measurement definitions, "
        "and dataloader construction."
    )


register_adapter(
    DatasetAdapter(
        key="tep",
        description=(
            "Tennessee Eastman Process multivariate dataset (RData format). "
            "Adapters pending implementation."
        ),
        default_data_dir=os.path.join("data", "tep", "raw"),
        measurement_vars=[],
        dataset_cls=None,
        control_names_fn=lambda _: _not_implemented(),
        dataloader_factory=lambda *args, **kwargs: _not_implemented(),
        resolve_split_files_fn=lambda split: _not_implemented(split),
        list_fault_keys_fn=lambda: [],
        supports_training=False,
        supports_testing=False,
        supports_plotting=False,
    )
)
