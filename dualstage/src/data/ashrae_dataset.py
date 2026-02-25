"""
PyTorch Geometric Dataset for ASHRAE refrigeration system data.

This module handles loading, preprocessing, and creating temporal graph data
for the DualSTAGE model using cleaned CSV files.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional, Iterable
from .ashrae_column_config import (
    LABEL_COLUMNS,
    TIME_COLUMN,
    get_all_selected_columns,
    get_control_vars,
    get_measurement_vars,
)

COLUMN_ALIASES = {
    "Heat Balance": "Heat Balance (kW)",
}


class ASHRAEDataset(Dataset):
    """
    Dataset for ASHRAE 1043-RP refrigeration system time series data.
    
    Creates sliding window samples for temporal graph learning with DualSTAGE.
    Handles CSV files.
    
    Args:
        data_files: List of CSV file paths to load (relative to data_dir)
        window_size: Length of sliding window for temporal sequences
        stride: Step size for sliding window (default: 1)
        data_dir: Directory containing the CSV files
        normalize: Whether to normalize the data (default: True)
        normalization_stats: Optional pre-computed (measurement_mean, measurement_std,
            control_mean, control_std) tuple for normalization
    """
    
    def __init__(
        self,
        data_files: List[str],
        window_size: int = 15,
        stride: int = 1,
        data_dir: str = "data/ASHRAE_csv",
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        fault_code_whitelist: Optional[Iterable[float]] = None,
        unit_status_whitelist: Optional[Iterable[float]] = None,
        feature_option: str | None = None,
        pred_horizon: int = 0,
        max_time_gap: float = 12.0,
    ):
        super().__init__()
        
        self.data_files = data_files
        self.window_size = window_size
        self.stride = stride
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.feature_option = feature_option
        self.pred_horizon = max(0, int(pred_horizon))
        self.max_time_gap = max_time_gap
        self.fault_code_whitelist = (
            set(float(code) for code in fault_code_whitelist)
            if fault_code_whitelist is not None
            else None
        )
        self.unit_status_whitelist = (
            set(float(status) for status in unit_status_whitelist)
            if unit_status_whitelist is not None
            else None
        )

        # Feature configuration
        self.measurement_vars = get_measurement_vars()
        self.control_vars = get_control_vars()
        self.label_columns = LABEL_COLUMNS
        self.all_selected_columns = get_all_selected_columns()

        # Dimensions
        self.n_measurement_vars = len(self.measurement_vars)
        self.n_control_vars = len(self.control_vars)
        
        # Load and preprocess data
        self.data = self._load_and_preprocess()
        self.file_row_ranges: List[Tuple[int, int]] = getattr(self, "file_row_ranges", [])
        
        # Create sliding window indices
        self.windows = self._create_windows()
        
        print(f"Dataset created: {len(self.windows)} samples from {len(data_files)} files")
    
    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess all CSV data files."""
        dfs: List[pd.DataFrame] = []
        file_lengths: List[int] = []
        
        for filename in self.data_files:
            filepath = os.path.join(self.data_dir, filename)
            
            # Load CSV file
            print(f"  Loading {filepath}...")
            df = pd.read_csv(filepath)

            # Normalize the time column so the rest of the pipeline can rely on a single name
            timestamp_col = None
            for candidate in (TIME_COLUMN, "Timestamp", "Time (minutes)"):
                if candidate in df.columns:
                    timestamp_col = candidate
                    break
            if timestamp_col is None:
                raise ValueError(
                    f"File {filename} does not contain a recognised time column. "
                    f"Expected one of {TIME_COLUMN}, Timestamp, or Time (minutes)."
                )
            if timestamp_col != TIME_COLUMN:
                df = df.rename(columns={timestamp_col: TIME_COLUMN})
            
            # Normalize legacy column names before enforcing schema
            for legacy, canonical in COLUMN_ALIASES.items():
                if legacy in df.columns and canonical not in df.columns:
                    df = df.rename(columns={legacy: canonical})

            # Verify all required columns are present
            missing_cols = [col for col in self.all_selected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing columns in {filename}: {missing_cols}\n"
                    f"Available columns: {list(df.columns)}"
                )
            
            # Select only the columns we need
            df = df[self.all_selected_columns].copy()

            # Ensure controls are numeric for filtering/cleaning
            df[self.control_vars] = df[self.control_vars].apply(pd.to_numeric, errors="coerce")

            original_rows = len(df)

            # Filter by allowed fault codes (if labels exist)
            if self.fault_code_whitelist is not None and "Active Fault" in df.columns:
                df = df[df["Active Fault"].isin(self.fault_code_whitelist)]

            # Filter by allowed Unit Status(es) if provided and present
            if self.unit_status_whitelist is not None and "Unit Status" in df.columns:
                df = df[df["Unit Status"].isin(self.unit_status_whitelist)]

            # Ensure temporal ordering before forward/back fill
            df[TIME_COLUMN] = pd.to_numeric(df[TIME_COLUMN], errors="coerce")
            df = df[df[TIME_COLUMN].notna()]
            df = df.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)

            if df.empty:
                print(f"  Skipping {filename}: no rows remained after filtering criteria.")
                continue
            
            # Handle potential missing values
            feature_columns = self.measurement_vars + self.control_vars
            
            # Ensure numerical dtype before cleaning
            df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
            
            # Replace NaN and inf values
            for col in feature_columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values (assumes temporal continuity)
            df[feature_columns] = df[feature_columns].ffill()
            # Backward fill any remaining NaNs at the start
            df[feature_columns] = df[feature_columns].bfill()

            # Fill any remaining NaNs with 0
            if df[feature_columns].isna().any().any():
                missing_cols = df[feature_columns].isna().any()
                missing_list = missing_cols[missing_cols].index.tolist()
                preview = ", ".join(missing_list[:5])
                suffix = " ..." if len(missing_list) > 5 else ""
                print(
                    f"  Warning: {filename} still has NaNs in columns: {preview}{suffix}. Filling with zeros."
                )
                df[missing_list] = df[missing_list].fillna(0.0)
            
            # Reorder columns to ensure consistency
            ordered_columns = [TIME_COLUMN] + self.measurement_vars + self.control_vars + self.label_columns
            df = df.reindex(columns=ordered_columns)
            dfs.append(df)
            file_lengths.append(len(df))
            print(f"  Loaded {filename}: kept {len(df)} of {original_rows} rows after filtering")
        
        # Concatenate all dataframes
        if not dfs:
            raise ValueError(
                "No data remained after applying filtering criteria. "
                "Check fault/unit-status filters or input files."
            )
        combined_df = pd.concat(dfs, ignore_index=True)

        # Track per-file row ranges to prevent windows from crossing file boundaries
        self.file_row_ranges = []
        cursor = 0
        for length in file_lengths:
            start_idx = cursor
            end_idx = cursor + length
            self.file_row_ranges.append((start_idx, end_idx))
            cursor = end_idx
        
        # Normalize if requested
        if self.normalize:
            if self.normalization_stats is None:
                # Compute normalization statistics from the data
                measurement_data = combined_df[self.measurement_vars].values
                control_data = combined_df[self.control_vars].values

                with np.errstate(invalid="ignore"):
                    measurement_mean = np.nanmean(measurement_data, axis=0)
                    measurement_std = np.nanstd(measurement_data, axis=0)
                    control_mean = np.nanmean(control_data, axis=0)
                    control_std = np.nanstd(control_data, axis=0)

                self.measurement_mean = measurement_mean
                self.measurement_std = measurement_std + 1e-8
                self.control_mean = control_mean
                self.control_std = control_std + 1e-8

                # Handle NaN values in statistics
                meas_nan_mask = np.isnan(self.measurement_mean) | np.isnan(self.measurement_std)
                if np.any(meas_nan_mask):
                    cols = [self.measurement_vars[i] for i in np.flatnonzero(meas_nan_mask)]
                    preview = ", ".join(cols[:5])
                    suffix = " ..." if len(cols) > 5 else ""
                    print(
                        f"Warning: no valid measurements found for columns: {preview}{suffix}. Using zeros."
                    )
                    self.measurement_mean[meas_nan_mask] = 0.0
                    self.measurement_std[meas_nan_mask] = 1.0

                ctrl_nan_mask = np.isnan(self.control_mean) | np.isnan(self.control_std)
                if np.any(ctrl_nan_mask):
                    cols = [self.control_vars[i] for i in np.flatnonzero(ctrl_nan_mask)]
                    preview = ", ".join(cols[:5])
                    suffix = " ..." if len(cols) > 5 else ""
                    print(
                        f"Warning: no valid control readings found for columns: {preview}{suffix}. Using zeros."
                    )
                    self.control_mean[ctrl_nan_mask] = 0.0
                    self.control_std[ctrl_nan_mask] = 1.0
            else:
                # Use provided normalization statistics
                (
                    self.measurement_mean,
                    self.measurement_std,
                    self.control_mean,
                    self.control_std,
                ) = self.normalization_stats
            
            # Apply normalization
            combined_df[self.measurement_vars] = (
                combined_df[self.measurement_vars].values - self.measurement_mean
            ) / self.measurement_std
            combined_df[self.control_vars] = (
                combined_df[self.control_vars].values - self.control_mean
            ) / self.control_std
        else:
            # Even without normalization, store identity transforms
            self.measurement_mean = np.zeros(self.n_measurement_vars)
            self.measurement_std = np.ones(self.n_measurement_vars)
            self.control_mean = np.zeros(self.n_control_vars)
            self.control_std = np.ones(self.n_control_vars)
        
        return combined_df
    
    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding window start/end indices with gap awareness."""
        windows: List[Tuple[int, int]] = []
        if not hasattr(self, "file_row_ranges"):
            self.file_row_ranges = [(0, len(self.data))]

        total_len = self.window_size + self.pred_horizon
        time_values = self.data[TIME_COLUMN].to_numpy(dtype=float, copy=False)

        for start_idx, end_idx in self.file_row_ranges:
            file_length = end_idx - start_idx
            if file_length < total_len:
                continue
            if self.max_time_gap is None or self.max_time_gap <= 0:
                max_idx = end_idx - total_len + 1
                for window_start in range(start_idx, max_idx, self.stride):
                    window_end = window_start + total_len
                    windows.append((window_start, window_end))
                continue

            times = time_values[start_idx:end_idx]
            diffs = np.diff(times)
            gap_mask = (~np.isfinite(diffs)) | (diffs <= 0) | (diffs > self.max_time_gap)
            seg_start = 0
            for i, is_gap in enumerate(gap_mask):
                if not is_gap:
                    continue
                seg_end = i + 1
                seg_len = seg_end - seg_start
                if seg_len >= total_len:
                    max_idx = seg_end - total_len + 1
                    for window_start in range(seg_start, max_idx, self.stride):
                        window_end = window_start + total_len
                        windows.append((start_idx + window_start, start_idx + window_end))
                seg_start = seg_end

            seg_end = len(times)
            seg_len = seg_end - seg_start
            if seg_len >= total_len:
                max_idx = seg_end - total_len + 1
                for window_start in range(seg_start, max_idx, self.stride):
                    window_end = window_start + total_len
                    windows.append((start_idx + window_start, start_idx + window_end))
        
        return windows
    
    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.windows)
    
    def get(self, idx: int) -> Data:
        """
        Get a single temporal graph sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Data: PyTorch Geometric Data object containing:
                - x: Node features [window_size, n_nodes, n_features] = [T, N, 1]
                  (For autoencoding, each measurement at each timestep is a node with value as feature)
                - c: Control variables [window_size, n_control_vars]
                - edge_index: Fully connected graph edges [2, num_edges]
        """
        window_start, window_end = self.windows[idx]
        
        # Extract window data (history + optional future)
        history_end = window_start + self.window_size
        window_data = self.data.iloc[window_start:history_end]
        
        # Measurement variables: [window_size, n_measurement_vars] -> [n_measurement_vars, window_size]
        measurements = torch.tensor(
            window_data[self.measurement_vars].values.T,
            dtype=torch.float32,
        )
        # Control variables: [window_size, n_control_vars] -> [n_control_vars, window_size]
        controls = torch.tensor(
            window_data[self.control_vars].values.T,
            dtype=torch.float32,
        )
        
        edge_index = self._create_fully_connected_graph(self.n_measurement_vars)
        
        # Create PyG Data object
        data = Data(x=measurements, edge_index=edge_index, c=controls)
        if self.pred_horizon > 0:
            future_data = self.data.iloc[history_end:window_end]
            future_measurements = torch.tensor(
                future_data[self.measurement_vars].values.T,
                dtype=torch.float32,
            )
            data.y_future = future_measurements

        return data

    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge indices with self-loops."""
        src = []
        dst = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (mean, std) pairs for measurements and controls."""
        return (
            self.measurement_mean,
            self.measurement_std,
            self.control_mean,
            self.control_std,
        )
    
    def denormalize_measurements(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize measurement data back to original scale."""
        return normalized_data * self.measurement_std + self.measurement_mean
    
    def denormalize_controls(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize control data back to original scale."""
        return normalized_data * self.control_std + self.control_mean


class ASHRAEFaultDataset(ASHRAEDataset):
    """
    Extended dataset that includes fault labels for supervised scenarios.
    Inherits from ASHRAEDataset and adds fault_label to each sample.
    """
    
    def __init__(
        self,
        data_files: List[str],
        fault_label: int,
        window_size: int = 15,
        stride: int = 1,
        data_dir: str = "data/ASHRAE_csv",
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fault_code_whitelist: Optional[Iterable[float]] = None,
        unit_status_whitelist: Optional[Iterable[float]] = None,
        feature_option: str | None = None,
        pred_horizon: int = 0,
        max_time_gap: float = 12.0,
    ):
        self.fault_label = fault_label
        super().__init__(
            data_files=data_files,
            window_size=window_size,
            stride=stride,
            data_dir=data_dir,
            normalize=normalize,
            normalization_stats=normalization_stats,
            fault_code_whitelist=fault_code_whitelist,
            unit_status_whitelist=unit_status_whitelist,
            feature_option=feature_option,
            pred_horizon=pred_horizon,
            max_time_gap=max_time_gap,
        )
    
    def get(self, idx: int) -> Data:
        """Get a sample with fault label."""
        data = super().get(idx)
        data.y = torch.LongTensor([self.fault_label])
        return data


def get_ashrae_control_variable_names(data_dir: str, feature_option: str | None = None) -> List[str]:
    """Return the control-variable column names for the ASHRAE dataset."""
    # ASHRAE dataset doesn't have time-of-day features like the 1-min CO2 dataset
    _ = data_dir  # unused but kept for signature compatibility
    return get_control_vars(feature_option)
