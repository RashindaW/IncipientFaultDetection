#!/usr/bin/env python
from __future__ import annotations

"""
Generate frequency-domain visualizations organized by dataset/split/feature.

Folder layout (under freq_analysis/plots):
  <dataset>/<healthy|faulty>/<feature>/window_XX.png

For each dataset split, the script:
  1) Picks the top-K active features by standard deviation.
  2) Draws N random windows for each feature.
  3) Plots the windowed signal and its rFFT magnitude.
"""

import runpy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import scipy.io
import xlrd

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(13)
TOP_K_FEATURES = 5
WINDOWS_PER_FEATURE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_numeric_time(series: pd.Series) -> pd.Series:
    """Convert timestamps or numeric-like columns to float seconds."""
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


def compute_top_features(df: pd.DataFrame, measurement_cols: Sequence[str], k: int, max_rows: int) -> List[str]:
    if df.empty:
        return []
    stats_slice = df[measurement_cols].head(max_rows)
    stds = stats_slice.std().sort_values(ascending=False)
    return list(stds.index[: min(k, len(stds))])


def plot_single_feature_windows(
    dataset_name: str,
    split_name: str,
    feature: str,
    df: pd.DataFrame,
    window_size: int,
    num_windows: int,
    feature_dir: Path,
    dt_override: Optional[float] = None,
) -> None:
    total = len(df)
    if total < window_size:
        print(f"[skip] {dataset_name}/{split_name}/{feature}: not enough samples ({total} < {window_size})")
        return

    time_seconds = df["time_seconds"].to_numpy(dtype=float)
    dt = dt_override if dt_override is not None else estimate_dt_seconds(pd.Series(time_seconds))
    dt = dt if dt > 0 else 1.0

    series = df[feature].to_numpy(dtype=float)
    feature_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_windows):
        start = int(RNG.integers(0, total - window_size + 1))
        end = start + window_size
        window_time = time_seconds[start:end]
        window_vals = series[start:end]
        centered = window_vals - window_vals.mean()

        fft_vals = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(len(centered), d=dt)
        mags = np.abs(fft_vals)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)

        axes[0].plot(window_time - window_time[0], centered, color="steelblue")
        axes[0].set_title(f"{dataset_name} | {split_name} | {feature} | window {i+1}")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Centered value")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freqs, mags, color="darkorange")
        axes[1].set_xlim(left=0)
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("|rFFT| magnitude")
        axes[1].grid(True, alpha=0.3)

        out_path = feature_dir / f"window_{i+1:02d}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)


def process_dataset(
    dataset_name: str,
    splits: Dict[str, pd.DataFrame],
    measurement_cols: Sequence[str],
    window_size: int,
    max_rows_for_stats: int,
    dt_override: Optional[float] = None,
) -> None:
    base_dir = PLOTS_DIR / dataset_name
    for split_name, df in splits.items():
        if df is None or df.empty:
            print(f"[skip] {dataset_name}/{split_name}: no data")
            continue

        split_dir = base_dir / split_name
        top_feats = compute_top_features(df, measurement_cols, TOP_K_FEATURES, max_rows_for_stats)
        if not top_feats:
            print(f"[skip] {dataset_name}/{split_name}: no features selected")
            continue

        for feat in top_feats:
            feature_dir = split_dir / sanitize(feat)
            plot_single_feature_windows(
                dataset_name=dataset_name,
                split_name=split_name,
                feature=feat,
                df=df,
                window_size=window_size,
                num_windows=WINDOWS_PER_FEATURE,
                feature_dir=feature_dir,
                dt_override=dt_override,
            )
        print(f"[ok] {dataset_name}/{split_name}: saved {WINDOWS_PER_FEATURE} windows for {len(top_feats)} features")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_co2(dataset_name: str, base_dir: Path, window_size: int) -> Dict:
    cfg = runpy.run_path(BASE_DIR.parent / "dystgat" / "src" / "data" / "column_config.py")
    meas = cfg["MEASUREMENT_VARS"]
    healthy_path = base_dir / "BaselineTestA.csv"
    fault_path = base_dir / "Fault1_DisplayCaseDoorOpen.csv"

    healthy_df = pd.read_csv(healthy_path, parse_dates=["Timestamp"])
    healthy_df["time_seconds"] = ensure_numeric_time(healthy_df["Timestamp"])
    healthy_df = healthy_df[["time_seconds", *meas]].dropna()

    faulty_df = pd.read_csv(fault_path, parse_dates=["Timestamp"])
    faulty_df["time_seconds"] = ensure_numeric_time(faulty_df["Timestamp"])
    faulty_df = faulty_df[["time_seconds", *meas]].dropna()
    return {
        "dataset_name": dataset_name,
        "splits": {"healthy": healthy_df, "faulty": faulty_df},
        "measurement_cols": meas,
        "window_size": window_size,
        "max_rows_for_stats": 15000,
    }


def load_ashrae(window_size: int = 512) -> Dict:
    cfg = runpy.run_path(BASE_DIR.parent / "dystgat" / "src" / "data" / "ashrae_column_config.py")
    meas = cfg["get_measurement_vars"](None)
    file_path = BASE_DIR.parent / "data" / "ASHRAE_1043_RP" / "Benchmark Tests" / "normal1.xls"

    book = xlrd.open_workbook(file_path)
    sheet = book.sheet_by_index(0)
    rows, cols = sheet.nrows, sheet.ncols
    headers = [sheet.cell_value(0, c) for c in range(cols)]
    data = [[sheet.cell_value(r, c) for c in range(cols)] for r in range(1, rows)]
    healthy_df = pd.DataFrame(data, columns=headers)
    time_col = "Time" if "Time" in healthy_df.columns else healthy_df.columns[0]
    healthy_df["time_seconds"] = ensure_numeric_time(pd.Series(healthy_df[time_col]))
    meas_available = [c for c in meas if c in healthy_df.columns]
    healthy_df = healthy_df[["time_seconds", *meas_available]].dropna()

    fault_path = BASE_DIR.parent / "data" / "ASHRAE_1043_RP" / "Refrigerant leak" / "rl10.xls"
    if fault_path.exists():
        fbook = xlrd.open_workbook(fault_path)
        fsheet = fbook.sheet_by_index(0)
        frows, fcols = fsheet.nrows, fsheet.ncols
        fheaders = [fsheet.cell_value(0, c) for c in range(fcols)]
        fdata = [[fsheet.cell_value(r, c) for c in range(fcols)] for r in range(1, frows)]
        faulty_df = pd.DataFrame(fdata, columns=fheaders)
        f_time_col = "Time" if "Time" in faulty_df.columns else faulty_df.columns[0]
        faulty_df["time_seconds"] = ensure_numeric_time(pd.Series(faulty_df[f_time_col]))
        meas_fault = [c for c in meas if c in faulty_df.columns]
        faulty_df = faulty_df[["time_seconds", *meas_fault]].dropna()
    else:
        faulty_df = pd.DataFrame(columns=["time_seconds", *meas_available])

    return {
        "dataset_name": "ashrae",
        "splits": {"healthy": healthy_df, "faulty": faulty_df},
        "measurement_cols": meas_available,
        "window_size": window_size,
        "max_rows_for_stats": 5000,
    }


def load_pronto(window_size: int = 512) -> Dict:
    cfg = runpy.run_path(BASE_DIR.parent / "dystgat" / "src" / "data" / "pronto_column_config.py")
    all_vars = cfg["ALL_VARS"]
    meas_indices = cfg["MEASUREMENT_INDICES"]
    meas = [all_vars[i] for i in meas_indices]

    mat_path = BASE_DIR.parent / "data" / "pronto" / "pronto_benchmark" / "Pre-processed data" / "Process data" / "HealthySet.mat"
    mat = scipy.io.loadmat(mat_path)
    segment = mat["HealthySet"][0][0]
    healthy_df = pd.DataFrame(segment, columns=all_vars)
    healthy_df["time_seconds"] = np.arange(len(healthy_df), dtype=float)
    healthy_df = healthy_df[["time_seconds", *meas]]

    fault_path = BASE_DIR.parent / "data" / "pronto" / "pronto_benchmark" / "Pre-processed data" / "Process data" / "Blockage_120air_01water.mat"
    if fault_path.exists():
        fault_mat = scipy.io.loadmat(fault_path)
        content = fault_mat["Blockage_120air_01water"][0, 0]
        data_arr = content[1]  # shape (N, 19)
        keep_indices = [i for i in range(19) if i not in [7, 18]]
        data_arr = data_arr[:, keep_indices]
        faulty_df = pd.DataFrame(data_arr, columns=all_vars)
        faulty_df["time_seconds"] = np.arange(len(faulty_df), dtype=float)
        faulty_df = faulty_df[["time_seconds", *meas]]
    else:
        faulty_df = pd.DataFrame(columns=["time_seconds", *meas])

    return {
        "dataset_name": "pronto",
        "splits": {"healthy": healthy_df, "faulty": faulty_df},
        "measurement_cols": meas,
        "window_size": window_size,
        "max_rows_for_stats": 5000,
    }


def load_tep(window_size: int = 512) -> Dict:
    cfg = runpy.run_path(BASE_DIR.parent / "dystgat" / "src" / "data" / "tep_column_config.py")
    meas = cfg["MEASUREMENT_VARS"]
    control = cfg["CONTROL_VARS"]
    fault_col = cfg["FAULT_LABEL_COL"]
    run_col = cfg["RUN_COL"]
    sample_col = cfg["SAMPLE_COL"]
    rdata_path = BASE_DIR.parent / "data" / "tep" / "raw" / "TEP_FaultFree_Training.RData"

    def canonical(name: str) -> str:
        return name.lower().replace(".", "_")

    def load_rdata(path: Path) -> pd.DataFrame:
        result = pyreadr.read_r(path)
        df_raw = next(iter(result.values()))
        canon_map = {canonical(c): c for c in df_raw.columns}

        def resolve(col: str) -> str:
            key = canonical(col)
            if key in canon_map:
                return canon_map[key]
            key_no = key.replace("_", "")
            for canon, orig in canon_map.items():
                if canon.replace("_", "") == key_no:
                    return orig
            raise KeyError(f"Column {col} not found in {path}")

        selected = {col: resolve(col) for col in meas + control + [fault_col, run_col, sample_col]}
        df = df_raw[list(selected.values())].rename(columns={v: k for k, v in selected.items()})
        df = df.head(200000)  # trim for speed/plotting
        df["time_seconds"] = df[sample_col].astype(float)
        return df[["time_seconds", *meas]]

    healthy_df = load_rdata(rdata_path)
    faulty_path = BASE_DIR.parent / "data" / "tep" / "raw" / "TEP_Faulty_Training.RData"
    faulty_df = load_rdata(faulty_path) if faulty_path.exists() else pd.DataFrame(columns=["time_seconds", *meas])
    return {
        "dataset_name": "tep",
        "splits": {"healthy": healthy_df, "faulty": faulty_df},
        "measurement_cols": meas,
        "window_size": window_size,
        "max_rows_for_stats": 20000,
    }


def load_ims(window_size: int = 2048, sample_rate_hz: float = 20000.0) -> Dict:
    healthy_file = sorted((BASE_DIR.parent / "data" / "IMS_Bearing" / "1st_test").glob("*"))[0]
    healthy_df = pd.read_csv(healthy_file, sep=r"\s+", header=None, engine="python", nrows=10000)
    healthy_df = healthy_df.apply(pd.to_numeric, errors="coerce")
    healthy_meas = [f"ch{i+1}" for i in range(healthy_df.shape[1])]
    healthy_df.columns = healthy_meas
    dt = 1.0 / sample_rate_hz
    healthy_df["time_seconds"] = np.arange(len(healthy_df), dtype=float) * dt
    healthy_df = healthy_df[["time_seconds", *healthy_meas]]

    faulty_df = pd.DataFrame(columns=["time_seconds", *healthy_meas])
    faulty_candidates = [p for p in (BASE_DIR.parent / "data" / "IMS_Bearing" / "3rd_test").rglob("*") if p.is_file()]
    measurement_cols = healthy_meas
    if faulty_candidates:
        try:
            candidate_df = pd.read_csv(faulty_candidates[0], sep=r"\s+", header=None, engine="python", nrows=10000)
            candidate_df = candidate_df.apply(pd.to_numeric, errors="coerce")
            faulty_meas = [f"ch{i+1}" for i in range(candidate_df.shape[1])]
            candidate_df.columns = faulty_meas
            candidate_df["time_seconds"] = np.arange(len(candidate_df), dtype=float) * dt

            common_len = min(len(healthy_meas), len(faulty_meas))
            measurement_cols = [f"ch{i+1}" for i in range(common_len)]
            healthy_df = healthy_df[["time_seconds", *measurement_cols]]
            faulty_df = candidate_df[["time_seconds", *measurement_cols]]
        except Exception:
            faulty_df = pd.DataFrame(columns=["time_seconds", *healthy_meas])
            measurement_cols = healthy_meas

    return {
        "dataset_name": "ims_bearing",
        "splits": {"healthy": healthy_df, "faulty": faulty_df},
        "measurement_cols": measurement_cols,
        "window_size": window_size,
        "max_rows_for_stats": 20000,
        "dt_override": dt,
    }


DATASET_LOADERS = [
    lambda: load_co2("co2_1min", BASE_DIR.parent / "data" / "co2" / "1min", window_size=512),
    lambda: load_co2("co2_raw", BASE_DIR.parent / "data" / "co2" / "raw", window_size=512),
    load_ashrae,
    load_pronto,
    load_tep,
    load_ims,
]


if __name__ == "__main__":
    print(f"Saving plots to {PLOTS_DIR}")
    for loader in DATASET_LOADERS:
        try:
            info: Dict = loader()
            process_dataset(
                dataset_name=info["dataset_name"],
                splits=info["splits"],
                measurement_cols=info["measurement_cols"],
                window_size=info["window_size"],
                max_rows_for_stats=info["max_rows_for_stats"],
                dt_override=info.get("dt_override"),
            )
        except FileNotFoundError as exc:
            print(f"[skip] {loader.__name__}: missing file - {exc}")
        except Exception as exc:
            print(f"[skip] {loader.__name__}: {exc}")
