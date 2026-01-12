"""
Training script for DyEdgeGAT using the 1-minute aggregated dataset.

Usage:
    conda run -n rashindaNew-torch-env python train_dyedgegat_1min.py --epochs 10
"""

from train_dyedgegat import main as _main


def _ensure_dataset_args(
    default_key: str = "co2_1min",
    default_path: str = "data/co2/1min",
) -> None:
    import sys

    argv = sys.argv
    if not any(token == "--dataset-key" or token.startswith("--dataset-key=") for token in argv[1:]):
        argv.extend(["--dataset-key", default_key])
    for token in argv[1:]:
        if token == "--data-dir":
            return
        if token.startswith("--data-dir="):
            return

    argv.extend(["--data-dir", default_path])


def main() -> None:
    # Delegate to the standard training entrypoint while overriding the default dataset path
    # when the caller does not explicitly provide one.
    _ensure_dataset_args()
    _main()


if __name__ == "__main__":
    main()
