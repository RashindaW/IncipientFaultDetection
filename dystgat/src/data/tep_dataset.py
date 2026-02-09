from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from .tep_column_config import (
    ALL_DATA_COLUMNS,
    CONTROL_VARS,
    FAULT_LABEL_COL,
    MEASUREMENT_VARS,
    RUN_COL,
    SAMPLE_COL,
)


def _canonical(name: str) -> str:
    """Lowercase and replace dots with underscores for robust column matching."""
    return name.lower().replace(".", "_")


# Global cache to prevent reloading the huge RData file multiple times
# Key: absolute file path, Value: DataFrame
_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def _read_rdata_frame(path: str) -> pd.DataFrame:
    """Load the first DataFrame from an RData file (cached)."""
    abs_path = os.path.abspath(path)
    if abs_path in _DATA_CACHE:
        print(f"  [Cache Hit] Using pre-loaded data for {os.path.basename(path)}")
        return _DATA_CACHE[abs_path]

    try:
        import pyreadr  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "pyreadr is required to load TEP RData files. Install with `pip install pyreadr`."
        ) from exc

    print(f"  Loading {path} from disk...")
    result = pyreadr.read_r(path)
    if not result:
        raise ValueError(f"No objects found inside RData file: {path}")
    
    df = next(iter(result.values()))
    _DATA_CACHE[abs_path] = df
    return df


class TEPDataset(Dataset):
    """
    Dataset for the Tennessee Eastman Process (TEP) RData time series.

    Creates sliding-window samples, preserving run boundaries so windows do
    not cross simulation runs. Labels are derived from the `faultNumber`
    column (0 for normal, 1..20 for fault classes).
    """

    def __init__(
        self,
        data_files: List[str],
        window_size: int = 60,
        stride: int = 1,
        data_dir: str = "data/tep/raw",
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        fault_filter: Optional[Iterable[int]] = None,
        pred_horizon: int = 0,
        run_filter: Optional[Iterable[int]] = None,
        fault_onset_sample: int = 0,
    ):
        super().__init__()
        self.data_files = data_files
        self.window_size = window_size
        self.stride = max(1, stride)
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.fault_filter = set(int(f) for f in fault_filter) if fault_filter is not None else None
        self.pred_horizon = max(0, int(pred_horizon))
        self.run_filter = set(int(r) for r in run_filter) if run_filter is not None else None
        self.fault_onset_sample = fault_onset_sample

        self.n_measurement_vars = len(MEASUREMENT_VARS)
        self.n_control_vars = len(CONTROL_VARS)

        self.data = self._load_and_preprocess()
        self.run_ranges: List[Tuple[int, int]] = self._compute_run_ranges()
        self.windows = self._create_windows()
        print(f"Dataset created: {len(self.windows)} samples from {len(self.data_files)} file(s)")

    def _load_and_preprocess(self) -> pd.DataFrame:
        dfs: List[pd.DataFrame] = []
        
        for filename in self.data_files:
            path = os.path.join(self.data_dir, filename)
            # This uses the cached loader
            raw_df = _read_rdata_frame(path)

            # Normalize column names for robust selection
            canonical_map = {_canonical(col): col for col in raw_df.columns}

            def resolve(col: str) -> str:
                key = _canonical(col)
                if key in canonical_map:
                    return canonical_map[key]
                # Also try dropping underscores for safety (e.g., xmeas1 vs xmeas_1)
                key_no_underscore = key.replace("_", "")
                for canon_key, orig in canonical_map.items():
                    if canon_key.replace("_", "") == key_no_underscore:
                        return orig
                raise KeyError(f"Column '{col}' not found in {path}. Available: {list(raw_df.columns)[:5]}...")

            selected_cols = {col: resolve(col) for col in ALL_DATA_COLUMNS}
            df = raw_df[list(selected_cols.values())].copy()
            df = df.rename(columns={v: k for k, v in selected_cols.items()})

            # Apply fault filtering
            if self.fault_filter is not None:
                # Check if the column exists (some TEP variants differ)
                if FAULT_LABEL_COL in df.columns:
                    df = df[df[FAULT_LABEL_COL].astype(int).isin(self.fault_filter)]

            # Apply run filtering (for splitting val/test by simulation runs)
            if self.run_filter is not None:
                if RUN_COL in df.columns:
                    df = df[df[RUN_COL].astype(int).isin(self.run_filter)]

            # Drop pre-fault samples: in TEP faulty runs the process operates
            # normally before the fault injection point.  These samples carry the
            # run's faultNumber but are not truly faulty.
            if self.fault_onset_sample > 0 and FAULT_LABEL_COL in df.columns and SAMPLE_COL in df.columns:
                pre_fault_mask = (df[FAULT_LABEL_COL].astype(int) > 0) & (df[SAMPLE_COL].astype(int) < self.fault_onset_sample)
                n_dropped = pre_fault_mask.sum()
                if n_dropped > 0:
                    df = df[~pre_fault_mask]
                    print(f"  Dropped {n_dropped} pre-fault samples (onset at sample {self.fault_onset_sample})")

            if df.empty:
                print(f"  Skipping {filename}: no rows match the requested filters (fault: {self.fault_filter}, run: {self.run_filter}).")
                continue

            # Ensure numeric dtype
            for col in MEASUREMENT_VARS + CONTROL_VARS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            if FAULT_LABEL_COL in df.columns:
                df[FAULT_LABEL_COL] = pd.to_numeric(df[FAULT_LABEL_COL], errors="coerce").fillna(0).astype(int)
            else:
                df[FAULT_LABEL_COL] = 0
                
            if RUN_COL in df.columns:
                df[RUN_COL] = pd.to_numeric(df[RUN_COL], errors="coerce").fillna(0).astype(int)
            else:
                df[RUN_COL] = 0
            
            if SAMPLE_COL in df.columns:
                df[SAMPLE_COL] = pd.to_numeric(df[SAMPLE_COL], errors="coerce").fillna(0).astype(int)
            else:
                df[SAMPLE_COL] = np.arange(len(df))

            # Fill missing measurement/control values
            feature_cols = MEASUREMENT_VARS + CONTROL_VARS
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

            # Create a monotonically increasing timestamp for plotting
            # Sort to ensure time continuity
            df = df.sort_values([RUN_COL, SAMPLE_COL]).reset_index(drop=True)
            
            # Mock timestamp: Run ID * 1M seconds + Sample ID seconds
            # This separates runs in time plots by ~11.5 days
            df["Timestamp"] = pd.to_timedelta(df[RUN_COL] * 1_000_000 + df[SAMPLE_COL], unit="s")

            dfs.append(df)
            # print(f"  Processed {filename}: kept {len(df)} rows.")

        if not dfs:
            # Special case: if we filtered out everything, return empty DF structure
            # to allow graceful failure or empty dataset handling
            print("Warning: No data loaded (filters removed all rows).")
            return pd.DataFrame(columns=ALL_DATA_COLUMNS + ["Timestamp"])

        combined = pd.concat(dfs, ignore_index=True)

        # Compute or apply normalization statistics
        if self.normalize:
            if self.normalization_stats is None:
                meas = combined[MEASUREMENT_VARS].values.astype(np.float32)
                ctrl = combined[CONTROL_VARS].values.astype(np.float32)
                self.measurement_mean = meas.mean(axis=0)
                self.measurement_std = np.clip(meas.std(axis=0), 1e-6, None)
                self.control_mean = ctrl.mean(axis=0)
                self.control_std = np.clip(ctrl.std(axis=0), 1e-6, None)
            else:
                (
                    self.measurement_mean,
                    self.measurement_std,
                    self.control_mean,
                    self.control_std,
                ) = self.normalization_stats

            combined[MEASUREMENT_VARS] = (combined[MEASUREMENT_VARS] - self.measurement_mean) / self.measurement_std
            combined[CONTROL_VARS] = (combined[CONTROL_VARS] - self.control_mean) / self.control_std
        else:
            zero = np.zeros(self.n_measurement_vars, dtype=np.float32)
            ones = np.ones(self.n_measurement_vars, dtype=np.float32)
            self.measurement_mean = zero
            self.measurement_std = ones
            self.control_mean = np.zeros(self.n_control_vars, dtype=np.float32)
            self.control_std = np.ones(self.n_control_vars, dtype=np.float32)

        return combined

    def _compute_run_ranges(self) -> List[Tuple[int, int]]:
        """Compute (start, end) indices per simulation run to prevent window crossing."""
        ranges: List[Tuple[int, int]] = []
        if self.data.empty:
            return ranges
            
        if RUN_COL not in self.data.columns:
            ranges.append((0, len(self.data)))
            return ranges

        grouped = self.data.groupby(RUN_COL, sort=True)
        cursor = 0
        # Since self.data is already sorted by RUN_COL, we can iterate linearly
        # But groupby is safer if there are gaps
        
        # Faster approach for large DFs: use change points
        # However, standard groupby is robust enough for 9M rows if RAM allows
        for _, frame in grouped:
            length = len(frame)
            start = frame.index[0]
            end = start + length
            # Verify continuity (optional, but assumes reset_index was done)
            ranges.append((start, end))
            
        return ranges

    def _create_windows(self) -> List[Tuple[int, int]]:
        windows: List[Tuple[int, int]] = []
        for start, end in self.run_ranges:
            run_length = end - start
            total_len = self.window_size + self.pred_horizon
            if run_length < total_len:
                continue
            for offset in range(0, run_length - total_len + 1, self.stride):
                win_start = start + offset
                win_end = win_start + total_len
                windows.append((win_start, win_end))
        return windows

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.measurement_mean,
            self.measurement_std,
            self.control_mean,
            self.control_std,
        )

    def len(self) -> int:
        return len(self.windows)

    def get(self, idx: int) -> Data:
        start, end = self.windows[idx]
        history_end = start + self.window_size
        # Use .iloc for integer positioning
        window_data = self.data.iloc[start:history_end]

        measurements = torch.tensor(
            window_data[MEASUREMENT_VARS].values.T, dtype=torch.float32
        )  # [n_measurement_vars, window_size]
        controls = torch.tensor(
            window_data[CONTROL_VARS].values.T, dtype=torch.float32
        )  # [n_control_vars, window_size]

        edge_index = self._create_fully_connected_graph(self.n_measurement_vars)
        data = Data(x=measurements, edge_index=edge_index, c=controls)
        if self.pred_horizon > 0:
            future_data = self.data.iloc[history_end:end]
            future_measurements = torch.tensor(
                future_data[MEASUREMENT_VARS].values.T, dtype=torch.float32
            )
            data.y_future = future_measurements

        # Take the label from the last timestep (standard for detection)
        # or mode? TEP usually has constant fault per run.
        label = int(window_data[FAULT_LABEL_COL].iloc[-1])
        data.y = torch.tensor([label], dtype=torch.long)
        return data

    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        src = []
        dst = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)
