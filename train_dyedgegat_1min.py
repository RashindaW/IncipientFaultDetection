"""
Training script for DyEdgeGAT using the 1-minute aggregated dataset.

Usage:
    conda run -n rashindaNew-torch-env python train_dyedgegat_1min.py --epochs 10
"""

from train_dyedgegat import main as _main


def _ensure_data_dir_arg(default_path: str = "Dataset_1min") -> None:
    import sys

    argv = sys.argv
    for token in argv[1:]:
        if token == "--data-dir":
            return
        if token.startswith("--data-dir="):
            return

    argv.extend(["--data-dir", default_path])


def main() -> None:
    # Delegate to the standard training entrypoint while overriding the default dataset path
    # when the caller does not explicitly provide one.
    _ensure_data_dir_arg()
    _main()


if __name__ == "__main__":
    main()
