"""
PyTorch Geometric Dataset for refrigeration system data.

This module handles loading, preprocessing, and creating temporal graph data
for the DualSTAGE model.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional
from .column_config import (
    MEASUREMENT_VARS,
    CONTROL_VARS,
    ALL_SELECTED_COLUMNS,
    SENTINEL_VALUES,
    SENTINEL_THRESHOLD,
)


TIME_FEATURE_COLUMNS = [
    "Time_HourOfDay_sin",
    "Time_HourOfDay_cos",
    "Time_MinuteOfDay_sin",
    "Time_MinuteOfDay_cos",
]


def should_add_temporal_features(data_dir: str) -> bool:
    """
    Determine whether to augment control variables with time-of-day features.

    For now this targets the 1-minute aggregated dataset, which places all CSVs
    inside a directory containing the token "1min".
    """
    base = os.path.basename(os.path.normpath(data_dir)).lower()
    return "1min" in base


def get_control_variable_names(data_dir: str, feature_option: Optional[str] = None) -> List[str]:
    """Return the control-variable column names for the requested dataset."""
    _ = feature_option  # Unused; kept for interface compatibility
    controls = CONTROL_VARS.copy()
    if should_add_temporal_features(data_dir):
        controls += TIME_FEATURE_COLUMNS
    return controls


class RefrigerationDataset(Dataset):
    """
    Dataset for refrigeration system time series data.
    
    Creates sliding window samples for temporal graph learning with DualSTAGE.
    
    Args:
        data_files: List of CSV file paths to load
        window_size: Length of sliding window for temporal sequences
        stride: Step size for sliding window (default: 1)
        data_dir: Directory containing the CSV files
        normalize: Whether to normalize the data (default: True)
        normalization_stats: Optional pre-computed (mean, std) for normalization
    """
    
    def __init__(
        self,
        data_files: List[str],
        window_size: int = 15,
        stride: int = 1,
        data_dir: str = 'Dataset',
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pred_horizon: int = 0,
    ):
        super().__init__()
        
        self.data_files = data_files
        self.window_size = window_size
        self.stride = stride
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.pred_horizon = max(0, int(pred_horizon))

        self.base_control_vars = CONTROL_VARS.copy()
        self.using_time_features = should_add_temporal_features(self.data_dir)
        self.time_feature_columns = TIME_FEATURE_COLUMNS if self.using_time_features else []
        self.control_vars = self.base_control_vars + self.time_feature_columns

        # Dimensions
        self.n_measurement_vars = len(MEASUREMENT_VARS)
        self.n_control_vars = len(self.control_vars)
        
        # Load and preprocess data
        self.data = self._load_and_preprocess()
        
        # Create sliding window indices
        self.windows = self._create_windows()
        
        print(f"Dataset created: {len(self.windows)} samples from {len(data_files)} files")
    
    def _load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess all data files."""
        dfs = []
        
        for filename in self.data_files:
            filepath = os.path.join(self.data_dir, filename)
            
            # Load CSV
            df = pd.read_csv(filepath)
            
            # Fix BaselineTestB column name issue if present (legacy datasets only)
            if 'BaselineTestB' in filename and 'T-MT_BPHX_C02_EXIT' not in df.columns:
                # Older dumps used an incorrect label at position 158; guard for length before renaming
                if len(df.columns) > 158:
                    df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'
                else:
                    # Fallback: try to find a close variant and rename it
                    for col in df.columns:
                        if col.replace("CO2", "C02") == 'T-MT_BPHX_C02_EXIT':
                            df = df.rename(columns={col: 'T-MT_BPHX_C02_EXIT'})
                            break
            
            # Select only the columns we need
            df = df[ALL_SELECTED_COLUMNS]
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Handle sentinel values (missing data indicators)
            base_feature_columns = MEASUREMENT_VARS + self.base_control_vars
            for col in base_feature_columns:
                # Replace known sentinel values
                df[col] = df[col].replace(SENTINEL_VALUES, np.nan)
                # Replace any remaining large negative values
                df.loc[df[col] < SENTINEL_THRESHOLD, col] = np.nan
            
            # Forward fill missing values (assumes temporal continuity)
            df[base_feature_columns] = df[base_feature_columns].ffill()
            # Backward fill any remaining NaNs at the start
            df[base_feature_columns] = df[base_feature_columns].bfill()

            if df[base_feature_columns].isna().any().any():
                missing_cols = df[base_feature_columns].isna().any()
                missing_list = missing_cols[missing_cols].index.tolist()
                preview = ", ".join(missing_list[:5])
                suffix = " ..." if len(missing_list) > 5 else ""
                print(
                    f"  Warning: {filename} still has NaNs in columns: {preview}{suffix}. Filling with zeros."
                )
                df[missing_list] = df[missing_list].fillna(0.0)

            if self.using_time_features:
                df = self._add_temporal_features(df)
            
            ordered_columns = ['Timestamp'] + MEASUREMENT_VARS + self.control_vars
            df = df.reindex(columns=ordered_columns)
            
            dfs.append(df)
            print(f"  Loaded {filename}: {len(df)} rows")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Normalize if requested
        if self.normalize:
            if self.normalization_stats is None:
                # Compute normalization statistics from the data
                measurement_data = combined_df[MEASUREMENT_VARS].values
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

                meas_nan_mask = np.isnan(self.measurement_mean) | np.isnan(self.measurement_std)
                if np.any(meas_nan_mask):
                    cols = [MEASUREMENT_VARS[i] for i in np.flatnonzero(meas_nan_mask)]
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
                (self.measurement_mean, self.measurement_std,
                 self.control_mean, self.control_std) = self.normalization_stats

            fill_measure = pd.Series(self.measurement_mean, index=MEASUREMENT_VARS)
            fill_control = pd.Series(self.control_mean, index=self.control_vars)
            combined_df[MEASUREMENT_VARS] = combined_df[MEASUREMENT_VARS].fillna(fill_measure)
            combined_df[self.control_vars] = combined_df[self.control_vars].fillna(fill_control)

            combined_df[MEASUREMENT_VARS] = (
                (combined_df[MEASUREMENT_VARS] - self.measurement_mean) / self.measurement_std
            )
            combined_df[self.control_vars] = (
                (combined_df[self.control_vars] - self.control_mean) / self.control_std
            )

            combined_df[MEASUREMENT_VARS] = combined_df[MEASUREMENT_VARS].replace([np.inf, -np.inf], 0.0)
            combined_df[self.control_vars] = combined_df[self.control_vars].replace([np.inf, -np.inf], 0.0)

        return combined_df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sine/cosine encodings for hour and minute of day."""
        timestamps = df["Timestamp"]
        hour = timestamps.dt.hour.astype(float)
        minute = timestamps.dt.minute.astype(float)
        minute_of_day = hour * 60.0 + minute

        hour_angle = 2.0 * np.pi * hour / 24.0
        minute_angle = 2.0 * np.pi * minute_of_day / 1440.0

        df["Time_HourOfDay_sin"] = np.sin(hour_angle)
        df["Time_HourOfDay_cos"] = np.cos(hour_angle)
        df["Time_MinuteOfDay_sin"] = np.sin(minute_angle)
        df["Time_MinuteOfDay_cos"] = np.cos(minute_angle)
        return df
    
    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding window indices."""
        windows = []
        total_len = self.window_size + self.pred_horizon
        max_idx = len(self.data) - total_len + 1
        
        for start_idx in range(0, max_idx, self.stride):
            end_idx = start_idx + total_len
            windows.append((start_idx, end_idx))
        
        return windows
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get normalization statistics for use with other datasets."""
        if not self.normalize:
            raise ValueError("Dataset was created without normalization")
        return (self.measurement_mean, self.measurement_std,
                self.control_mean, self.control_std)
    
    def len(self) -> int:
        """Return the number of samples."""
        return len(self.windows)
    
    def get(self, idx: int) -> Data:
        """
        Get a single sample as PyTorch Geometric Data object.
        
        Returns:
            Data object with:
                - x: Measurement variables [n_nodes, window_size]
                - c: Control variables [n_control_vars, window_size]
                - edge_index: Fully connected graph edges [2, n_edges]
                - batch: Batch assignment (all zeros for single sample)
        """
        start_idx, end_idx = self.windows[idx]
        
        # Extract window data (history + optional future)
        history_end = start_idx + self.window_size
        window_data = self.data.iloc[start_idx:history_end]
        
        # Measurement variables: [window_size, n_measurement_vars] -> [n_measurement_vars, window_size]
        measurements = torch.tensor(
            window_data[MEASUREMENT_VARS].values.T,
            dtype=torch.float32
        )
        
        # Control variables: [window_size, n_control_vars] -> [n_control_vars, window_size]
        controls = torch.tensor(
            window_data[self.control_vars].values.T,
            dtype=torch.float32
        )
        
        # Create fully connected graph (all sensors connected to each other)
        n_nodes = self.n_measurement_vars
        edge_index = self._create_fully_connected_graph(n_nodes)
        
        # Create Data object
        data = Data(
            x=measurements,  # [n_nodes, window_size]
            c=controls,      # [n_control_vars, window_size]
            edge_index=edge_index,  # [2, n_edges]
        )
        if self.pred_horizon > 0:
            future_data = self.data.iloc[history_end:end_idx]
            future_measurements = torch.tensor(
                future_data[MEASUREMENT_VARS].values.T,
                dtype=torch.float32,
            )
            data.y_future = future_measurements
        
        return data
    
    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge indices."""
        # Create all possible edges (including self-loops)
        src = []
        dst = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index


class FaultDataset(RefrigerationDataset):
    """
    Dataset for fault data (extends RefrigerationDataset).
    
    Same as RefrigerationDataset but also returns fault labels.
    
    Args:
        Same as RefrigerationDataset, plus:
        fault_label: Integer label for this fault type (0=normal, 1-6=fault types)
    """
    
    def __init__(
        self,
        data_files: List[str],
        fault_label: int = 1,
        window_size: int = 15,
        stride: int = 1,
        data_dir: str = 'Dataset',
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pred_horizon: int = 0,
    ):
        self.fault_label = fault_label
        super().__init__(
            data_files=data_files,
            window_size=window_size,
            stride=stride,
            data_dir=data_dir,
            normalize=normalize,
            normalization_stats=normalization_stats,
            pred_horizon=pred_horizon,
        )
    
    def get(self, idx: int) -> Data:
        """Get sample with fault label."""
        data = super().get(idx)
        data.y = torch.tensor([self.fault_label], dtype=torch.long)
        return data


if __name__ == "__main__":
    # Test the dataset
    from column_config import BASELINE_FILES
    
    print("Testing RefrigerationDataset...")
    print()
    
    # Create training dataset
    train_dataset = RefrigerationDataset(
        data_files=BASELINE_FILES['train'],
        window_size=15,
        stride=5,  # Use stride for faster testing
        data_dir='../../Dataset',
    )
    
    print(f"\nDataset size: {len(train_dataset)} samples")
    print(f"Measurement vars: {train_dataset.n_measurement_vars}")
    print(f"Control vars: {train_dataset.n_control_vars}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"\nSample data shapes:")
    print(f"  x (measurements): {sample.x.shape}")
    print(f"  c (controls): {sample.c.shape}")
    print(f"  edge_index: {sample.edge_index.shape}")
    
    # Get normalization stats
    stats = train_dataset.get_normalization_stats()
    print(f"\nNormalization stats computed:")
    print(f"  Measurement mean shape: {stats[0].shape}")
    print(f"  Measurement std shape: {stats[1].shape}")
    print(f"  Control mean shape: {stats[2].shape}")
    print(f"  Control std shape: {stats[3].shape}")
