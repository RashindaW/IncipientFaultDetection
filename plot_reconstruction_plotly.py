#!/usr/bin/env python3
"""
Load a DyEdgeGAT checkpoint, run inference, and visualize original vs reconstructed
sensor readings alongside anomaly scores using Plotly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Iterable, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.amp import autocast
from torch_geometric.loader import DataLoader

from train_dyedgegat import forward_model, init_model, resolve_devices, unwrap_model
from datasets import get_adapter, list_adapter_keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot original vs reconstructed sensor values and anomaly scores "
            "with Plotly for a DyEdgeGAT checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument(
        "--dataset-key",
        default="co2",
        choices=list_adapter_keys(),
        help="Dataset adapter to use for loading data (e.g., 'co2', 'co2_1min').",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override dataset directory (adapter default used when omitted).",
    )
    parser.add_argument(
        "--dataset",
        default="baseline",
        help="Dataset to evaluate (baseline, val, train, or specific fault key). "
        "Ignored when --datasets or --include-all-faults is supplied.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset keys to process. Overrides --dataset.",
    )
    parser.add_argument(
        "--include-all-faults",
        action="store_true",
        help="Process baseline plus all defined fault datasets.",
    )
    parser.add_argument(
        "--dataset-suffix",
        default="",
        help="Optional suffix appended to dataset filenames before '.csv' (e.g., '_1min').",
    )
    parser.add_argument(
        "--sensor",
        default=None,
        help="Measurement variable name to plot (defaults to the first channel of the dataset).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Sliding window length used for the dataset and model.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride applied when generating evaluation windows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the DataLoader.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device preference.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Explicit CUDA device index when using --device cuda/auto.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Enable mixed precision (CUDA only).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit the number of evaluation windows processed (useful for quick previews).",
    )
    parser.add_argument(
        "--denormalize",
        action="store_true",
        help="Convert normalized values back to original scale using dataset statistics.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/plotly",
        help="Directory where run outputs are organised (default: outputs/plotly).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name used to create a unique subdirectory under --output-root.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional explicit HTML output path or directory (overrides run directory).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional explicit CSV output path or directory (overrides run directory).",
    )
    parser.add_argument(
        "--anomaly-only",
        action="store_true",
        help="Plot only anomaly scores over time (omit actual/reconstructed series).",
    )
    return parser.parse_args()


def apply_suffix(file_list: Iterable[str], suffix: str) -> List[str]:
    if not suffix:
        return list(file_list)
    suffixed: List[str] = []
    for name in file_list:
        root, ext = os.path.splitext(name)
        suffixed.append(f"{root}{suffix}{ext}")
    return suffixed


def build_dataset(
    adapter,
    dataset_key: str,
    args: argparse.Namespace,
    data_dir: str,
) -> any:
    dataset_cls = adapter.dataset_cls
    if dataset_cls is None:
        raise NotImplementedError(
            f"Dataset adapter '{adapter.key}' does not yet provide a dataset implementation."
        )

    eval_sources = apply_suffix(adapter.resolve_split_files(dataset_key), args.dataset_suffix)
    train_sources = apply_suffix(adapter.resolve_split_files("train"), args.dataset_suffix)

    print(f"\nLoading training dataset for normalization ({len(train_sources)} file(s)):")
    for f in train_sources:
        print(f"  - {f}")
    train_dataset = dataset_cls(
        data_files=train_sources,
        window_size=args.window_size,
        stride=max(1, args.stride),
        data_dir=data_dir,
        normalize=True,
    )
    norm_stats = train_dataset.get_normalization_stats()
    del train_dataset

    print(f"\nLoading evaluation dataset '{dataset_key}' ({len(eval_sources)} file(s)):")
    for f in eval_sources:
        print(f"  - {f}")

    eval_dataset = dataset_cls(
        data_files=eval_sources,
        window_size=args.window_size,
        stride=max(1, args.stride),
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
    )
    return eval_dataset


def gather_time_series(
    model: torch.nn.Module,
    dataset,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    sensor_index: Optional[int],
    max_windows: Optional[int],
    include_values: bool,
) -> pd.DataFrame:
    base_model = unwrap_model(model)
    records: List[dict] = []
    sample_counter = 0

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch_size = raw_batch.num_graphs
            sample_indices = range(sample_counter, sample_counter + batch_size)
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

            if include_values and sensor_index is not None:
                target_np = target.detach().cpu().numpy()
                recon_np = recon.detach().cpu().numpy()
                window = dataset.window_size
                n_nodes = dataset.n_measurement_vars
                target_np = target_np.reshape(batch_size, n_nodes, window, -1).squeeze(-1)
                recon_np = recon_np.reshape(batch_size, n_nodes, window, -1).squeeze(-1)

            for local_idx, sample_idx in enumerate(sample_indices):
                window_start, window_end = dataset.windows[sample_idx]
                window_timestamps = (
                    dataset.data.iloc[window_start:window_end]["Timestamp"]
                    .reset_index(drop=True)
                )
                anomaly_series = per_timestep[local_idx].numpy()

                if include_values and sensor_index is not None:
                    actual_series = target_np[local_idx, sensor_index]
                    recon_series = recon_np[local_idx, sensor_index]
                for step_idx, ts in enumerate(window_timestamps):
                    entry = {
                        "timestamp": ts,
                        "sample_id": sample_idx,
                        "timestep_index": step_idx,
                        "anomaly_score": float(anomaly_series[step_idx]),
                    }
                    if include_values and sensor_index is not None:
                        entry["actual"] = float(actual_series[step_idx])
                        entry["reconstructed"] = float(recon_series[step_idx])
                    records.append(entry)

                if max_windows is not None and (sample_idx + 1) >= max_windows:
                    break

            if max_windows is not None and sample_counter >= max_windows:
                break

    if not records:
        raise RuntimeError("No data collected; check dataset, stride, and max_windows settings.")

    df = pd.DataFrame.from_records(records)
    df.sort_values(["timestamp", "sample_id", "timestep_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def aggregate_by_timestamp(
    df: pd.DataFrame,
    denormalize: bool,
    dataset,
    sensor_index: Optional[int],
    include_values: bool,
) -> pd.DataFrame:
    # Preserve all timestamps in order before merging aggregated statistics
    timestamps = pd.DataFrame({"timestamp": df["timestamp"].drop_duplicates().sort_values()})
    grouped = timestamps.copy()

    if include_values and sensor_index is not None:
        median_values = (
            df.groupby("timestamp", as_index=False)[["actual", "reconstructed"]]
            .median()
        )
        grouped = grouped.merge(median_values, on="timestamp", how="left")

    # Ignore extreme anomaly spikes (>100) when aggregating
    anomaly_df = df[df["anomaly_score"] <= 100]
    if anomaly_df.empty:
        anomaly_df = df

    anomaly_median = (
        anomaly_df.groupby("timestamp", as_index=False)["anomaly_score"]
        .median()
    )
    anomaly_mean = (
        anomaly_df.groupby("timestamp", as_index=False)["anomaly_score"]
        .mean()
        .rename(columns={"anomaly_score": "anomaly_score_mean_filtered"})
    )

    grouped = grouped.merge(anomaly_median, on="timestamp", how="left")
    grouped = grouped.merge(anomaly_mean, on="timestamp", how="left")

    if denormalize and include_values and sensor_index is not None:
        mean = dataset.measurement_mean[sensor_index]
        std = dataset.measurement_std[sensor_index]
        grouped["actual"] = grouped["actual"] * std + mean
        grouped["reconstructed"] = grouped["reconstructed"] * std + mean

    grouped.sort_values("timestamp", inplace=True)
    grouped.reset_index(drop=True, inplace=True)
    return grouped


def build_plot(
    timeseries: pd.DataFrame,
    sensor_name: str,
    dataset_label: str,
    include_values: bool,
) -> go.Figure:
    if include_values:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=timeseries["timestamp"],
                y=timeseries["actual"],
                name="Actual",
                mode="lines",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=timeseries["timestamp"],
                y=timeseries["reconstructed"],
                name="Reconstructed",
                mode="lines",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=timeseries["timestamp"],
                y=timeseries["anomaly_score"],
                name="Anomaly Score",
                mode="lines",
                line=dict(color="firebrick", dash="dot"),
            ),
            secondary_y=True,
        )

        title = f"{dataset_label} – {sensor_name}"
        fig.update_layout(
            title=title,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text="Sensor Value", secondary_y=False)
        fig.update_yaxes(title_text="Anomaly Score", secondary_y=True)
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=timeseries["timestamp"],
                y=timeseries["anomaly_score"],
                name="Anomaly Score",
                mode="lines",
                line=dict(color="firebrick"),
            )
        )
        title = f"{dataset_label} – Anomaly Score"
        fig.update_layout(
            title=title,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Timestamp")
        fig.update_yaxes(title_text="Anomaly Score")
    return fig


def main() -> None:
    args = parse_args()

    adapter = get_adapter(args.dataset_key)
    adapter.ensure("plotting")
    data_dir = args.data_dir or adapter.get_default_data_dir()
    if data_dir is None:
        raise ValueError(
            f"Dataset adapter '{args.dataset_key}' does not define a default data directory; "
            "specify one with --data-dir."
        )

    print(f"Dataset adapter : {adapter.key}")
    print(f"Data directory  : {data_dir}")

    measurement_vars = list(adapter.measurement_vars)
    if not measurement_vars:
        print(
            f"Warning: dataset adapter '{adapter.key}' does not expose measurement variables. "
            "Plotting of reconstructed sensor values will be unavailable."
        )

    include_values = not args.anomaly_only
    if include_values:
        if not measurement_vars:
            raise ValueError(
                f"Dataset '{adapter.key}' does not define measurement variables; "
                "use --anomaly-only instead."
            )
        sensor_name = args.sensor or measurement_vars[0]
        if sensor_name not in measurement_vars:
            preview = ", ".join(measurement_vars[:5]) + ("..." if len(measurement_vars) > 5 else "")
            raise ValueError(
                f"Sensor '{sensor_name}' not found in dataset '{adapter.key}'. "
                f"Available examples: {preview}"
            )
        sensor_index = measurement_vars.index(sensor_name)
    else:
        sensor_name = args.sensor
        sensor_index = None
        if args.denormalize:
            print("Warning: --denormalize has no effect when --anomaly-only is set.")

    dataset_keys: List[str] = []
    if args.include_all_faults:
        dataset_keys.append("baseline")
        fault_keys = adapter.list_fault_keys()
        if not fault_keys:
            print(f"Warning: dataset '{adapter.key}' does not define fault splits; skipping include-all-faults.")
        dataset_keys.extend(fault_keys)
    if args.datasets:
        dataset_keys.extend(args.datasets)
    if not dataset_keys:
        dataset_keys.append(args.dataset)

    seen = set()
    ordered_keys: List[str] = []
    for key in dataset_keys:
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)
    dataset_keys = ordered_keys

    if args.device in ("auto", "cuda") and args.cuda_device is None:
        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    device, _ = resolve_devices(args.device, args.cuda_device, None)
    amp_enabled = args.use_amp and device.type == "cuda"

    control_var_names = adapter.get_control_variables(data_dir)
    model = init_model(
        device,
        args.window_size,
        len(control_var_names),
        adapter.measurement_count(),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    missing = model.load_state_dict(checkpoint, strict=False)
    if missing.missing_keys:
        print(f"Missing keys when loading checkpoint: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {missing.unexpected_keys}")

    multi_dataset = len(dataset_keys) > 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"{args.dataset_key}_{Path(args.checkpoint).stem}_{timestamp}"
    run_root = Path(args.output_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Output run dir   : {run_root}")

    if args.output_html:
        html_candidate = Path(args.output_html)
        if html_candidate.suffix and multi_dataset:
            raise ValueError("--output-html must be a directory when processing multiple datasets.")
        if html_candidate.suffix:
            html_override = html_candidate
            base_html_dir = html_candidate.parent
        else:
            html_override = None
            base_html_dir = html_candidate
    else:
        html_override = None
        base_html_dir = run_root

    if args.output_csv:
        csv_candidate = Path(args.output_csv)
        if csv_candidate.suffix and multi_dataset:
            raise ValueError("--output-csv must be a directory when processing multiple datasets.")
        if csv_candidate.suffix:
            csv_override = csv_candidate
            base_csv_dir = csv_candidate.parent
        else:
            csv_override = None
            base_csv_dir = csv_candidate
    else:
        csv_override = None
        base_csv_dir = run_root

    label_for_naming = sensor_name if include_values else "anomaly_score"
    safe_sensor = label_for_naming.replace("/", "_").replace(" ", "_")

    file_suffix = "reconstruction" if include_values else "anomaly"

    for dataset_key in dataset_keys:
        print(f"\n=== Processing dataset: {dataset_key} ===")
        dataset = build_dataset(adapter, dataset_key, args, data_dir)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        print("Collecting reconstruction time series...")
        raw_df = gather_time_series(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
            amp_enabled=amp_enabled,
            sensor_index=sensor_index,
            max_windows=args.max_windows,
            include_values=include_values,
        )

        timeseries = aggregate_by_timestamp(
            raw_df,
            denormalize=args.denormalize,
            dataset=dataset,
            sensor_index=sensor_index,
            include_values=include_values,
        )

        if html_override and not multi_dataset:
            html_path = html_override
            html_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            html_dir = base_html_dir / dataset_key if html_override is None or multi_dataset else base_html_dir
            html_dir.mkdir(parents=True, exist_ok=True)
            safe_dataset = dataset_key.replace("/", "_").replace(" ", "_")
            html_path = html_dir / f"{safe_dataset}_{safe_sensor}_{file_suffix}.html"

        fig = build_plot(timeseries, label_for_naming, dataset_key, include_values)
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved Plotly visualization to {html_path}")

        if csv_override and not multi_dataset:
            csv_path = csv_override
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            csv_dir = base_csv_dir / dataset_key if csv_override is None or multi_dataset else base_csv_dir
            csv_dir.mkdir(parents=True, exist_ok=True)
            safe_dataset = dataset_key.replace("/", "_").replace(" ", "_")
            csv_path = csv_dir / f"{safe_dataset}_{safe_sensor}_{file_suffix}.csv"

        timeseries.to_csv(csv_path, index=False)
        print(f"Wrote aggregated time-series data to {csv_path}")


if __name__ == "__main__":
    main()
