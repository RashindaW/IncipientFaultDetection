#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dyedgegat.src.data.ashrae_column_config import (
    BASELINE_FILES,
    FAULT_FILES,
    BENCHMARK_DIR,
    BASELINE_FAULT_CODE_WHITELIST,
    BASELINE_UNIT_STATUS_WHITELIST,
)
from dyedgegat.src.data.ashrae_dataset import ASHRAEDataset, ASHRAEFaultDataset

try:
    from scipy.stats import ks_2samp

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


DEFAULT_WINDOW_SIZES = [64, 128, 256, 512, 1024]
DEFAULT_BANDS = [
    ("band_0_0.05", 0.0, 0.05),
    ("band_0.05_0.15", 0.05, 0.15),
    ("band_0.15_0.3", 0.15, 0.3),
    ("band_0.3_0.5", 0.3, 0.5),
]
DEFAULT_ROLLOFFS = [0.85, 0.95]
EPS = 1e-8


def ensure_numeric_time(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        base = series.iloc[0]
        return series.view("int64") / 1e9 - base.to_datetime64().view("int64") / 1e9
    values = pd.to_numeric(series, errors="coerce")
    values = values.ffill().bfill()
    base = values.iloc[0]
    return values - base


def estimate_dt_seconds(time_seconds: pd.Series) -> float:
    diffs = np.diff(time_seconds.to_numpy(dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def sanitize(name: str) -> str:
    return (
        name.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(",", "_")
    )


def parse_window_sizes(value: Optional[str]) -> List[int]:
    if not value:
        return list(DEFAULT_WINDOW_SIZES)
    sizes = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        sizes.append(int(raw))
    if not sizes:
        raise ValueError("No valid window sizes provided.")
    return sizes


def parse_fault_keys(value: str) -> List[str]:
    if value.strip().lower() == "all":
        return list(FAULT_FILES.keys())
    return [item.strip() for item in value.split(",") if item.strip()]


def sample_window_indices(dataset: ASHRAEDataset, max_windows: Optional[int], rng: np.random.Generator) -> List[int]:
    total = len(dataset)
    if total == 0:
        return []
    if not max_windows or max_windows <= 0 or total <= max_windows:
        return list(range(total))
    indices = rng.choice(total, size=max_windows, replace=False)
    return sorted(int(idx) for idx in indices)


def build_band_masks(freqs: np.ndarray, bands: Sequence[Tuple[str, float, float]]) -> List[Tuple[str, np.ndarray]]:
    if freqs.size == 0:
        return [(name, np.zeros_like(freqs, dtype=bool)) for name, _, _ in bands]
    nyquist = freqs[-1]
    masks = []
    for name, low_frac, high_frac in bands:
        low = low_frac * nyquist
        high = high_frac * nyquist
        mask = (freqs >= low) & (freqs < high)
        masks.append((name, mask))
    return masks


def _metrics_from_magnitude(
    mag: np.ndarray,
    freqs: np.ndarray,
    band_masks: Sequence[Tuple[str, np.ndarray]],
    rolloff_fracs: Sequence[float],
    window_size: int,
    prefix: str = "",
) -> Dict[str, np.ndarray]:
    n_features = mag.shape[0]
    power = (mag ** 2) / max(window_size, 1)
    total_power = power.sum(axis=-1)
    p = power / (total_power[:, None] + EPS)

    centroid = (power * freqs).sum(axis=-1) / (total_power + EPS)
    diff = freqs - centroid[:, None]
    bandwidth = np.sqrt((diff ** 2 * power).sum(axis=-1) / (total_power + EPS))

    flatness = np.exp(np.mean(np.log(power + EPS), axis=-1)) / (np.mean(power, axis=-1) + EPS)
    entropy = -(p * np.log(p + EPS)).sum(axis=-1) / np.log(p.shape[-1] + EPS)

    peak_idx = np.argmax(power, axis=-1)
    peak_freq = freqs[peak_idx]

    m2 = (diff ** 2 * p).sum(axis=-1)
    m3 = (diff ** 3 * p).sum(axis=-1)
    m4 = (diff ** 4 * p).sum(axis=-1)
    skewness = m3 / np.power(m2 + EPS, 1.5)
    kurtosis = m4 / np.power(m2 + EPS, 2)

    freqs_nonzero = freqs[1:]
    if freqs_nonzero.size >= 2:
        log_power = np.log(power[:, 1:] + EPS)
        f_centered = freqs_nonzero - freqs_nonzero.mean()
        denom = np.sum(f_centered ** 2) + EPS
        slope = (log_power * f_centered).sum(axis=-1) / denom
    else:
        slope = np.zeros(n_features)

    cumulative = np.cumsum(power, axis=-1)
    rolloff_values = {}
    for frac in rolloff_fracs:
        target = (total_power * frac)[:, None]
        idx = (cumulative >= target).argmax(axis=-1)
        rolloff_values[f"rolloff_{int(frac * 100)}"] = freqs[idx]

    metrics: Dict[str, np.ndarray] = {
        f"{prefix}total_power": total_power,
        f"{prefix}centroid": centroid,
        f"{prefix}bandwidth": bandwidth,
        f"{prefix}flatness": flatness,
        f"{prefix}entropy": entropy,
        f"{prefix}peak_freq": peak_freq,
        f"{prefix}slope": slope,
        f"{prefix}skewness": skewness,
        f"{prefix}kurtosis": kurtosis,
    }
    metrics.update({f"{prefix}{name}": value for name, value in rolloff_values.items()})

    for name, mask in band_masks:
        if mask.any():
            band_power = power[:, mask].sum(axis=-1)
        else:
            band_power = np.zeros(n_features)
        metrics[f"{prefix}bandpower_{name}"] = band_power
        metrics[f"{prefix}bandpower_ratio_{name}"] = band_power / (total_power + EPS)

    return metrics


def compute_metrics_for_window(
    values: np.ndarray,
    freqs: np.ndarray,
    band_masks: Sequence[Tuple[str, np.ndarray]],
    rolloff_fracs: Sequence[float],
    include_log: bool,
) -> Dict[str, np.ndarray]:
    n_features, window_size = values.shape
    if window_size <= 0:
        return {}

    fft_vals = np.fft.rfft(values, axis=-1)
    mag = np.abs(fft_vals)
    metrics = _metrics_from_magnitude(mag, freqs, band_masks, rolloff_fracs, window_size)
    if include_log:
        log_mag = np.log1p(mag)
        metrics.update(
            _metrics_from_magnitude(
                log_mag,
                freqs,
                band_masks,
                rolloff_fracs,
                window_size,
                prefix="log_",
            )
        )
    return metrics


def collect_metrics(
    dataset: ASHRAEDataset,
    window_indices: Sequence[int],
    dt_seconds: float,
    bands: Sequence[Tuple[str, float, float]],
    rolloff_fracs: Sequence[float],
    include_log: bool,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if not window_indices:
        return {}, []

    freqs = np.fft.rfftfreq(dataset.window_size, d=dt_seconds)
    band_masks = build_band_masks(freqs, bands)

    metric_store: Optional[Dict[str, np.ndarray]] = None
    metric_names: List[str] = []

    for idx_pos, idx in enumerate(window_indices):
        data = dataset.get(idx).x.numpy()
        data = data - data.mean(axis=-1, keepdims=True)
        window_metrics = compute_metrics_for_window(data, freqs, band_masks, rolloff_fracs, include_log)
        if metric_store is None:
            metric_names = list(window_metrics.keys())
            metric_store = {
                name: np.zeros((len(window_indices), data.shape[0]), dtype=np.float64)
                for name in metric_names
            }
        for name, values in window_metrics.items():
            metric_store[name][idx_pos] = values

    return metric_store or {}, metric_names


def summarize_metrics(metrics_store: Dict[str, np.ndarray], feature_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric, values in metrics_store.items():
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        median = np.median(values, axis=0)
        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "mean": mean[idx],
                    "std": std[idx],
                    "median": median[idx],
                    "n_windows": values.shape[0],
                }
            )
    return pd.DataFrame(rows)


def summarize_variability(metrics_store: Dict[str, np.ndarray], feature_names: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric, values in metrics_store.items():
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        cv = std / (np.abs(mean) + EPS)
        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "cv": cv[idx],
                    "mean": mean[idx],
                    "std": std[idx],
                    "n_windows": values.shape[0],
                }
            )
    return pd.DataFrame(rows)


def compare_metrics(
    baseline: Dict[str, np.ndarray],
    fault: Dict[str, np.ndarray],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for metric in baseline.keys():
        base_vals = baseline[metric]
        fault_vals = fault[metric]
        base_mean = base_vals.mean(axis=0)
        fault_mean = fault_vals.mean(axis=0)
        base_std = base_vals.std(axis=0, ddof=1)
        fault_std = fault_vals.std(axis=0, ddof=1)
        base_median = np.median(base_vals, axis=0)
        fault_median = np.median(fault_vals, axis=0)
        pooled = np.sqrt((base_std ** 2 + fault_std ** 2) / 2.0)
        cohen_d = (fault_mean - base_mean) / (pooled + EPS)
        pct_change = (fault_mean - base_mean) / (np.abs(base_mean) + EPS)

        if HAS_SCIPY:
            ks_stat = np.zeros(len(feature_names))
            ks_p = np.zeros(len(feature_names))
            for idx in range(len(feature_names)):
                try:
                    stat, pval = ks_2samp(base_vals[:, idx], fault_vals[:, idx], mode="auto")
                except Exception:
                    stat, pval = np.nan, np.nan
                ks_stat[idx] = stat
                ks_p[idx] = pval
        else:
            ks_stat = np.full(len(feature_names), np.nan)
            ks_p = np.full(len(feature_names), np.nan)

        for idx, feature in enumerate(feature_names):
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "baseline_mean": base_mean[idx],
                    "fault_mean": fault_mean[idx],
                    "baseline_std": base_std[idx],
                    "fault_std": fault_std[idx],
                    "baseline_median": base_median[idx],
                    "fault_median": fault_median[idx],
                    "mean_diff": fault_mean[idx] - base_mean[idx],
                    "pct_change": pct_change[idx],
                    "cohen_d": cohen_d[idx],
                    "ks_stat": ks_stat[idx],
                    "ks_p": ks_p[idx],
                    "n_baseline": base_vals.shape[0],
                    "n_fault": fault_vals.shape[0],
                }
            )
    return pd.DataFrame(rows)


def build_baseline_file_list(baseline_from: str) -> List[str]:
    train_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES["train"]]
    val_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES["val"]]
    return val_files if baseline_from == "val" else train_files


def build_train_stats(
    data_dir: str,
    feature_option: str,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES["train"]]
    train_dataset = ASHRAEDataset(
        data_files=train_files,
        window_size=window_size,
        stride=stride,
        data_dir=data_dir,
        normalize=True,
        feature_option=feature_option,
        fault_code_whitelist=BASELINE_FAULT_CODE_WHITELIST,
        unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
    )
    return train_dataset.get_normalization_stats()


def load_baseline_dataset(
    data_dir: str,
    feature_option: str,
    window_size: int,
    stride: int,
    norm_stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    baseline_from: str,
) -> ASHRAEDataset:
    baseline_files = build_baseline_file_list(baseline_from)
    return ASHRAEDataset(
        data_files=baseline_files,
        window_size=window_size,
        stride=stride,
        data_dir=data_dir,
        normalize=True,
        normalization_stats=norm_stats,
        feature_option=feature_option,
        fault_code_whitelist=BASELINE_FAULT_CODE_WHITELIST,
        unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
    )


def load_fault_datasets(
    data_dir: str,
    feature_option: str,
    window_size: int,
    stride: int,
    norm_stats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fault_keys: Iterable[str],
) -> Dict[str, ASHRAEDataset]:
    datasets = {}
    for idx, fault_name in enumerate(fault_keys, start=1):
        entry = FAULT_FILES.get(fault_name)
        if entry is None:
            print(f"[skip] Unknown fault key: {fault_name}")
            continue
        subdir, fault_file = entry
        rel_path = os.path.join(subdir, fault_file)
        full_path = Path(data_dir) / rel_path
        if not full_path.exists():
            print(f"[skip] Missing fault file: {full_path}")
            continue
        datasets[fault_name] = ASHRAEFaultDataset(
            data_files=[rel_path],
            fault_label=idx,
            window_size=window_size,
            stride=stride,
            data_dir=data_dir,
            normalize=True,
            normalization_stats=norm_stats,
            feature_option=feature_option,
            unit_status_whitelist=BASELINE_UNIT_STATUS_WHITELIST,
        )
    return datasets


def compute_dt_seconds(dataset: ASHRAEDataset) -> float:
    if "Timestamp" not in dataset.data.columns:
        return 1.0
    time_series = ensure_numeric_time(dataset.data["Timestamp"])
    dt = estimate_dt_seconds(time_series)
    return dt if dt > 0 else 1.0


def summarize_separation(compare_df: pd.DataFrame, window_size: int, fault_name: str) -> Dict[str, float]:
    abs_d = np.abs(compare_df["cohen_d"].to_numpy(dtype=float))
    abs_d = abs_d[np.isfinite(abs_d)]
    if abs_d.size == 0:
        median_abs_d = np.nan
        strong_rate = np.nan
        strong_count = 0
    else:
        median_abs_d = float(np.median(abs_d))
        strong_rate = float(np.mean(abs_d >= 0.8))
        strong_count = int((abs_d >= 0.8).sum())
    return {
        "window_size": window_size,
        "fault": fault_name,
        "median_abs_cohen_d": median_abs_d,
        "strong_rate": strong_rate,
        "strong_count": strong_count,
        "total_pairs": int(abs_d.size),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute spectral metrics across window sizes for ASHRAE baseline vs faults."
    )
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "ASHRAE_1043_RP"))
    parser.add_argument("--feature-option", type=str, choices=["a", "b"], default="a")
    parser.add_argument("--baseline-from", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--faults", type=str, default="all", help="Comma list or 'all'.")
    parser.add_argument("--window-sizes", type=str, default=None, help="Comma list, e.g. 64,128,256,512,1024")
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding windows.")
    parser.add_argument("--max-windows", type=int, default=500, help="Max windows per dataset per size.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--include-log",
        action="store_true",
        help="Also compute metrics on log1p(|rFFT|) with 'log_' prefixes.",
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join("freq_analysis", "metrics", "ashrae"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    window_sizes = parse_window_sizes(args.window_sizes)
    fault_keys = parse_fault_keys(args.faults)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (ROOT_DIR / data_dir).resolve()
    args.data_dir = data_dir.as_posix()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_window_size = max(window_sizes)
    stride_for_stats = args.stride if args.stride is not None else max(1, max_window_size // 2)
    print(f"Computing training normalization stats (window_size={max_window_size}, stride={stride_for_stats})")
    norm_stats = build_train_stats(
        data_dir=args.data_dir,
        feature_option=args.feature_option,
        window_size=max_window_size,
        stride=stride_for_stats,
    )

    separation_rows = []

    for window_size in window_sizes:
        stride = args.stride if args.stride is not None else max(1, window_size // 2)
        print(f"\n=== Window size {window_size} (stride {stride}) ===")
        window_dir = out_dir / f"window_{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)

        baseline_dataset = load_baseline_dataset(
            data_dir=args.data_dir,
            feature_option=args.feature_option,
            window_size=window_size,
            stride=stride,
            norm_stats=norm_stats,
            baseline_from=args.baseline_from,
        )
        feature_names = baseline_dataset.measurement_vars
        print(f"Baseline windows: {len(baseline_dataset)} | Features: {len(feature_names)}")

        baseline_indices = sample_window_indices(baseline_dataset, args.max_windows, rng)
        if not baseline_indices:
            print("[skip] No baseline windows available for this size.")
            continue

        dt_seconds = compute_dt_seconds(baseline_dataset)
        baseline_metrics, metric_names = collect_metrics(
            baseline_dataset,
            baseline_indices,
            dt_seconds,
            DEFAULT_BANDS,
            DEFAULT_ROLLOFFS,
            args.include_log,
        )
        if not baseline_metrics:
            print("[skip] Failed to compute baseline metrics.")
            continue

        baseline_summary = summarize_metrics(baseline_metrics, feature_names)
        baseline_summary.to_csv(window_dir / "baseline_summary.csv", index=False)
        baseline_variability = summarize_variability(baseline_metrics, feature_names)
        baseline_variability.to_csv(window_dir / "baseline_variability.csv", index=False)

        fault_datasets = load_fault_datasets(
            data_dir=args.data_dir,
            feature_option=args.feature_option,
            window_size=window_size,
            stride=stride,
            norm_stats=norm_stats,
            fault_keys=fault_keys,
        )
        if not fault_datasets:
            print("[skip] No fault datasets available for this size.")
            continue

        for fault_name, fault_dataset in fault_datasets.items():
            print(f"  Fault {fault_name}: windows={len(fault_dataset)}")
            fault_indices = sample_window_indices(fault_dataset, args.max_windows, rng)
            if not fault_indices:
                print(f"  [skip] No windows for fault {fault_name}")
                continue
            fault_dt_seconds = compute_dt_seconds(fault_dataset)
            fault_metrics, _ = collect_metrics(
                fault_dataset,
                fault_indices,
                fault_dt_seconds,
                DEFAULT_BANDS,
                DEFAULT_ROLLOFFS,
                args.include_log,
            )
            if not fault_metrics:
                print(f"  [skip] Failed metrics for fault {fault_name}")
                continue

            compare_df = compare_metrics(baseline_metrics, fault_metrics, feature_names)
            compare_path = window_dir / f"compare_baseline_{sanitize(fault_name)}.csv"
            compare_df.to_csv(compare_path, index=False)
            separation_rows.append(summarize_separation(compare_df, window_size, fault_name))

    if separation_rows:
        summary_df = pd.DataFrame(separation_rows)
        summary_df.to_csv(out_dir / "window_size_fault_summary.csv", index=False)
        overall = (
            summary_df.groupby("window_size")
            .agg(
                mean_median_abs_cohen_d=("median_abs_cohen_d", "mean"),
                mean_strong_rate=("strong_rate", "mean"),
            )
            .reset_index()
        )
        overall.to_csv(out_dir / "window_size_overall.csv", index=False)

    print(f"\nDone. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
