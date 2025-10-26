#!/usr/bin/env python3
"""
Generate per-timestep anomaly score plots for DyEdgeGAT checkpoints.

Usage example:
    python plot_anomaly_scores.py \
        --checkpoint checkpoints/dyedgegat_stride10.pt \
        --dataset baseline \
        --stride 1 \
        --batch-size 64 \
        --num-workers 4 \
        --output-dir outputs/anomaly_scores
"""

import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from torch.amp import autocast
from torch_geometric.loader import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_dyedgegat import forward_model, init_model, resolve_devices, unwrap_model
from dyedgegat.src.data.dataset import RefrigerationDataset
from dyedgegat.src.data.column_config import BASELINE_FILES, FAULT_FILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-timestep anomaly scores.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="baseline",
        help="Dataset to evaluate (baseline, val, train, or specific fault key).",
    )
    parser.add_argument("--data-dir", type=str, default="Dataset", help="Root directory of the CSV files.")
    parser.add_argument("--window-size", type=int, default=60, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=1, help="Stride used to build evaluation windows.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device (‘auto’ picks CUDA when available).",
    )
    parser.add_argument("--cuda-device", type=int, default=None, help="Specific CUDA device index.")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision for CUDA inference.")
    parser.add_argument("--max-samples", type=int, default=12, help="Maximum sequences to plot.")
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit sample indices to plot (overrides --max-samples).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/anomaly_scores",
        help="Directory to store CSV and plot outputs.",
    )
    parser.add_argument("--csv-name", type=str, default=None, help="Optional custom name for the exported CSV.")
    parser.add_argument(
        "--line-plot",
        action="store_true",
        help="Generate a line plot for the selected sequences (default: heatmap + CSV).",
    )
    return parser.parse_args()


def resolve_dataset_files(
    dataset_key: str,
) -> List[str]:
    key = dataset_key.lower()
    if key in ("baseline", "val"):
        return BASELINE_FILES["val"]
    if key == "train":
        return BASELINE_FILES["train"]
    if dataset_key in FAULT_FILES:
        return [FAULT_FILES[dataset_key]]
    for fault_name, fault_file in FAULT_FILES.items():
        if fault_name.lower() == key:
            return [fault_file]
    raise ValueError(
        f"Unknown dataset '{dataset_key}'. Valid options: 'train', 'baseline', 'val', "
        f"or one of {list(FAULT_FILES.keys())}"
    )


def build_dataset(
    data_files: List[str],
    args: argparse.Namespace,
) -> RefrigerationDataset:
    print(f"\nLoading evaluation dataset ({len(data_files)} file(s)):")
    for f in data_files:
        print(f"  - {f}")

    train_dataset = RefrigerationDataset(
        data_files=BASELINE_FILES["train"],
        window_size=args.window_size,
        stride=max(1, args.stride),
        data_dir=args.data_dir,
        normalize=True,
    )
    norm_stats = train_dataset.get_normalization_stats()

    eval_dataset = RefrigerationDataset(
        data_files=data_files,
        window_size=args.window_size,
        stride=args.stride,
        data_dir=args.data_dir,
        normalize=True,
        normalization_stats=norm_stats,
    )
    return eval_dataset


def select_sample_ids(
    available_ids: List[int],
    explicit_ids: Optional[List[int]],
    max_samples: int,
) -> List[int]:
    if explicit_ids:
        filtered = [sid for sid in explicit_ids if sid in available_ids]
        missing = set(explicit_ids) - set(filtered)
        if missing:
            print(f"Warning: sample indices not present in dataset and will be skipped: {sorted(missing)}")
        return filtered
    return available_ids[:max_samples]


def collect_scores(
    model: torch.nn.Module,
    dataset: RefrigerationDataset,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> pd.DataFrame:
    base_model = unwrap_model(model)
    records = []
    sample_counter = 0

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch_size = raw_batch.num_graphs
            sample_indices = list(range(sample_counter, sample_counter + batch_size))
            sample_counter += batch_size

            with autocast("cuda", enabled=amp_enabled):
                (recon, edge_index, edge_attr), batch_obj = forward_model(
                    model,
                    raw_batch,
                    device,
                    return_graph=True,
                )
                target = batch_obj.x.unsqueeze(-1)

            per_timestep = base_model.compute_anomaly_scores_per_timestep(
                target, recon, edge_index, edge_attr
            ).detach().cpu()

            for local_idx, sample_idx in enumerate(sample_indices):
                window_start, window_end = dataset.windows[sample_idx]
                window_timestamps = dataset.data.iloc[window_start:window_end]["Timestamp"].reset_index(drop=True)
                start_ts = window_timestamps.iloc[0]

                for step_idx, score in enumerate(per_timestep[local_idx]):
                    ts = window_timestamps.iloc[step_idx]
                    records.append(
                        {
                            "sample_id": sample_idx,
                            "timestep_index": step_idx,
                            "timestamp": ts,
                            "relative_seconds": (ts - start_ts).total_seconds(),
                            "anomaly_score": float(score.item()),
                        }
                    )

    df = pd.DataFrame.from_records(records)
    df.sort_values(["sample_id", "timestep_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def plot_heatmap(
    df: pd.DataFrame,
    selected_ids: List[int],
    output_path: str,
) -> None:
    if not selected_ids:
        print("No sequences selected for heatmap; skipping plot generation.")
        return

    subset = df[df["sample_id"].isin(selected_ids)]
    pivot = (
        subset.pivot(index="sample_id", columns="timestep_index", values="anomaly_score")
        .reindex(selected_ids)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(12, max(3, len(selected_ids) * 0.4)))
    im = ax.imshow(pivot, aspect="auto", cmap="inferno")
    ax.set_yticks(range(len(selected_ids)))
    ax.set_yticklabels(selected_ids)
    ax.set_ylabel("Sample ID")
    ax.set_xlabel("Timestep")
    ax.set_title("Per-timestep anomaly scores (heatmap)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Anomaly score", rotation=270, labelpad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def plot_line_curves(
    df: pd.DataFrame,
    selected_ids: List[int],
    output_path: str,
) -> None:
    if not selected_ids:
        print("No sequences selected for line plot; skipping line chart generation.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    for sample_id in selected_ids:
        sample_df = df[df["sample_id"] == sample_id]
        ax.plot(
            sample_df["timestamp"],
            sample_df["anomaly_score"],
            marker="o",
            linewidth=1.4,
            label=f"Sample {sample_id}",
        )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Per-timestep anomaly scores (selected sequences)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved line plot to {output_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data_files = resolve_dataset_files(args.dataset)
    dataset = build_dataset(data_files, args)

    if args.device in ("auto", "cuda") and args.cuda_device is None:
        # Avoid GPU 0 when not explicitly requested by the user.
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    device, _ = resolve_devices(args.device, args.cuda_device, None)
    amp_enabled = args.use_amp and device.type == "cuda"

    model = init_model(device, args.window_size)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    missing = model.load_state_dict(checkpoint, strict=False)
    if missing.missing_keys:
        print(f"Missing keys when loading checkpoint: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {missing.unexpected_keys}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    df = collect_scores(model, dataset, loader, device, amp_enabled)
    if df.empty:
        raise RuntimeError("No anomaly scores were collected; check dataset and stride settings.")

    csv_name = args.csv_name or f"{args.dataset}_anomaly_scores.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Wrote per-timestep anomaly scores to {csv_path}")

    available_ids = sorted(df["sample_id"].unique())
    selected_ids = select_sample_ids(available_ids, args.sample_indices, args.max_samples)

    heatmap_path = os.path.join(args.output_dir, f"{args.dataset}_heatmap.png")
    plot_heatmap(df, selected_ids, heatmap_path)

    if args.line_plot:
        line_path = os.path.join(args.output_dir, f"{args.dataset}_line.png")
        plot_line_curves(df, selected_ids, line_path)

    aggregated = df.groupby("timestamp", as_index=False)["anomaly_score"].mean()
    aggregated_path = os.path.join(args.output_dir, f"{args.dataset}_timestamp_mean.csv")
    aggregated.to_csv(aggregated_path, index=False)
    print(f"Saved timestamp-averaged anomaly scores to {aggregated_path}")


if __name__ == "__main__":
    main()
