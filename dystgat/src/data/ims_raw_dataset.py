"""
PyTorch Geometric Dataset for IMS Bearing raw accelerometer data.

This module handles loading raw 20 kHz vibration signals from the IMS Bearing
dataset without feature engineering. The raw signals are ideal for spectral
analysis, where DySTGAT's spectral encoder can learn frequency patterns directly
from the time-domain data via FFT.

Key design choices:
- Window-within-file approach: Each 20,480-sample file is divided into multiple
  overlapping windows (e.g., 1024 samples each) for fine-grained temporal dynamics.
- Multi-file sequences: Consecutive files are grouped to capture degradation
  progression over time.
- Direct accelerometer values: No feature extraction - raw signals preserve
  frequency information for the spectral encoder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from .ims_raw_column_config import (
    MEASUREMENT_VARS,
    CONTROL_VARS,
    SAMPLE_RATE,
    SAMPLES_PER_FILE,
    CHANNEL_CONFIG,
    FAILURE_INFO,
    DEFAULT_WINDOW_CONFIG,
)


def load_raw_file(filepath: str) -> np.ndarray:
    """
    Load a single IMS bearing raw data file.

    Args:
        filepath: Path to the data file (tab-separated accelerometer values).

    Returns:
        2D array of shape [n_samples, n_channels] where n_samples is typically
        20,480 and n_channels is 4 or 8 depending on the dataset set.
    """
    try:
        data = np.loadtxt(filepath, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception:
        # Silently return empty array for corrupted files
        # (known issue: file 2004.02.12.17.42.39 in 2nd_test is truncated)
        return np.array([], dtype=np.float32)


def get_sorted_file_list(data_dir: str, dataset_set: str = "2nd_test") -> List[str]:
    """
    Get chronologically sorted list of data files for a dataset set.

    Args:
        data_dir: Base data directory containing test set folders.
        dataset_set: One of "1st_test", "2nd_test", "3rd_test".

    Returns:
        Sorted list of full file paths.
    """
    set_dir = os.path.join(data_dir, dataset_set)
    if not os.path.exists(set_dir):
        return []

    files = []
    for f in os.listdir(set_dir):
        fpath = os.path.join(set_dir, f)
        if os.path.isfile(fpath) and not f.startswith('.'):
            files.append(fpath)

    # Sort by timestamp filename (format: YYYY.MM.DD.HH.MM.SS)
    files.sort()
    return files


def assign_degradation_labels(
    n_files: int,
    healthy_ratio: float = 0.7,
    degraded_ratio: float = 0.2,
) -> np.ndarray:
    """
    Assign degradation labels based on position in run-to-failure sequence.

    The IMS dataset is run-to-failure, so we assign labels based on temporal
    position:
    - First healthy_ratio: healthy (0)
    - Next degraded_ratio: degraded (1)
    - Remaining: faulty (2)

    Args:
        n_files: Total number of files in the sequence.
        healthy_ratio: Fraction of files considered healthy.
        degraded_ratio: Fraction considered degraded (after healthy).

    Returns:
        Array of integer labels [0, 1, 2] for each file.
    """
    labels = np.zeros(n_files, dtype=np.int64)

    healthy_end = int(n_files * healthy_ratio)
    degraded_end = int(n_files * (healthy_ratio + degraded_ratio))

    labels[healthy_end:degraded_end] = 1  # Degraded
    labels[degraded_end:] = 2  # Faulty

    return labels


class IMSRawDataset(Dataset):
    """
    Dataset for IMS Bearing raw accelerometer data.

    Each sample consists of raw vibration signals suitable for spectral analysis.
    The spectral encoder can apply FFT to extract frequency features directly.

    Two windowing approaches are supported:
    1. sample_window: Create overlapping windows within each file (sub-second resolution)
    2. file_window: Group consecutive files for temporal context

    Attributes:
        data_dir: Base directory containing 1st_test, 2nd_test, 3rd_test folders.
        dataset_set: Which test set to use.
        sample_window: Number of samples per window within a file (e.g., 1024).
        sample_stride: Stride between sample windows within a file.
        file_window: Number of consecutive files per sample.
        file_stride: Stride across files.
        normalize: Whether to apply z-score normalization.
        healthy_ratio: Fraction of data considered healthy for label assignment.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_set: str = "2nd_test",
        sample_window: int = 1024,
        sample_stride: int = 512,
        file_window: int = 1,
        file_stride: int = 1,
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        healthy_ratio: float = 0.7,
        fault_filter: Optional[List[int]] = None,
        require_stats: bool = False,
        max_files: Optional[int] = None,
        use_time_control: bool = False,
        downsample_factor: int = 1,
    ):
        """
        Initialize the raw IMS dataset.

        Args:
            data_dir: Base directory containing test set folders.
            dataset_set: One of "1st_test", "2nd_test", "3rd_test".
            sample_window: Number of samples per window within each file.
            sample_stride: Stride between sample windows within a file.
            file_window: Number of consecutive files to group per sample.
            file_stride: Stride across file groups.
            normalize: Whether to apply z-score normalization per bearing.
            normalization_stats: Pre-computed (mean, std) for normalization.
            healthy_ratio: Fraction of data labeled as healthy.
            fault_filter: If set, only include samples with these labels.
            require_stats: Raise error if normalization_stats not provided.
            max_files: Limit number of files to load (for debugging).
            use_time_control: Add time-based control variables.
            downsample_factor: Factor to downsample raw signals (1 = no downsampling).
        """
        super().__init__()

        self.data_dir = data_dir
        self.dataset_set = dataset_set
        self.sample_window = sample_window
        self.sample_stride = max(1, sample_stride)
        self.file_window = max(1, file_window)
        self.file_stride = max(1, file_stride)
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.healthy_ratio = healthy_ratio
        self.fault_filter = set(fault_filter) if fault_filter is not None else None
        self.require_stats = require_stats
        self.max_files = max_files
        self.use_time_control = use_time_control
        self.downsample_factor = max(1, downsample_factor)

        # Get channel configuration
        self.channel_config = CHANNEL_CONFIG.get(dataset_set, CHANNEL_CONFIG["2nd_test"])
        self.n_bearings = self.channel_config["n_bearings"]
        self.n_channels = self.channel_config["n_channels"]
        self.n_measurement_vars = self.n_bearings
        self.n_control_vars = 2 if use_time_control else 0

        # Effective samples per window after downsampling
        self.effective_window = sample_window // self.downsample_factor

        # Load data and create windows
        self._load_file_list()
        self._create_window_indices()

        print(f"IMS Raw Dataset ({dataset_set}): {len(self.windows)} samples, "
              f"{len(self.file_list)} files, {self.n_bearings} bearings, "
              f"window={self.sample_window} samples")

    def _load_file_list(self):
        """Load and filter the list of data files."""
        self.file_list = get_sorted_file_list(self.data_dir, self.dataset_set)

        if not self.file_list:
            raise ValueError(
                f"No files found in {self.data_dir}/{self.dataset_set}. "
                "Please ensure the data is extracted correctly."
            )

        if self.max_files is not None:
            self.file_list = self.file_list[:self.max_files]

        n_files = len(self.file_list)
        self.file_labels = assign_degradation_labels(n_files, self.healthy_ratio)

        # Time indices for control variables
        self.time_indices = np.arange(n_files, dtype=np.float32)
        self.normalized_time = self.time_indices / max(n_files - 1, 1)

    def _create_window_indices(self):
        """
        Create indices for all valid windows.

        Each window is defined by:
        - file_idx: Starting file index
        - sample_start: Starting sample index within the file(s)

        Windows span file_window consecutive files and sample_window samples.
        """
        self.windows = []

        n_files = len(self.file_list)
        samples_per_file = SAMPLES_PER_FILE // self.downsample_factor

        # Number of sample windows per file
        n_sample_windows = (samples_per_file - self.effective_window) // (self.sample_stride // self.downsample_factor) + 1
        n_sample_windows = max(1, n_sample_windows)

        # Iterate over file groups
        for file_start in range(0, n_files - self.file_window + 1, self.file_stride):
            file_end = file_start + self.file_window

            # Label is based on the last file in the group
            label = self.file_labels[file_end - 1]

            # Apply fault filter
            if self.fault_filter is not None and label not in self.fault_filter:
                continue

            # Create sample windows within this file group
            for sample_idx in range(n_sample_windows):
                sample_start = sample_idx * (self.sample_stride // self.downsample_factor)

                self.windows.append({
                    "file_start": file_start,
                    "file_end": file_end,
                    "sample_start": sample_start,
                    "label": label,
                })

    def _load_file_data(self, file_idx: int) -> np.ndarray:
        """
        Load raw accelerometer data from a single file.

        Args:
            file_idx: Index into self.file_list.

        Returns:
            Array of shape [n_samples, n_bearings] with accelerometer values.
            For 1st_test (8 channels), x/y channels are averaged per bearing.
        """
        filepath = self.file_list[file_idx]
        raw_data = load_raw_file(filepath)

        if raw_data.size == 0:
            # Track corrupted files (for one-time reporting)
            if not hasattr(self, '_corrupted_files'):
                self._corrupted_files = set()
            self._corrupted_files.add(os.path.basename(filepath))
            # Return zeros if file couldn't be loaded
            return np.zeros((SAMPLES_PER_FILE, self.n_bearings), dtype=np.float32)

        # Handle different channel configurations
        if self.dataset_set == "1st_test" and raw_data.shape[1] == 8:
            # Average x and y channels for each bearing
            bearing_data = np.zeros((raw_data.shape[0], 4), dtype=np.float32)
            for b in range(4):
                bearing_data[:, b] = (raw_data[:, b * 2] + raw_data[:, b * 2 + 1]) / 2
            return bearing_data
        else:
            # Already in [n_samples, n_bearings] format
            return raw_data[:, :self.n_bearings]

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization from all files."""
        if self.normalization_stats is not None:
            self.mean, self.std = self.normalization_stats
            return

        if self.require_stats:
            raise ValueError("Normalization stats required but not provided")

        print("Computing normalization statistics from raw data...")

        # Sample a subset of files for efficiency
        n_sample = min(100, len(self.file_list))
        sample_indices = np.linspace(0, len(self.file_list) - 1, n_sample, dtype=int)

        all_data = []
        for idx in sample_indices:
            data = self._load_file_data(idx)
            if data.size > 0:
                all_data.append(data)

        if not all_data:
            self.mean = np.zeros(self.n_bearings, dtype=np.float32)
            self.std = np.ones(self.n_bearings, dtype=np.float32)
            return

        combined = np.concatenate(all_data, axis=0)  # [total_samples, n_bearings]
        self.mean = combined.mean(axis=0).astype(np.float32)
        self.std = np.clip(combined.std(axis=0), 1e-6, None).astype(np.float32)

        print(f"Normalization stats - Mean: {self.mean}, Std: {self.std}")

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalization statistics for use in test/val datasets."""
        if not hasattr(self, 'mean') or not hasattr(self, 'std'):
            self._compute_normalization_stats()
        return self.mean.copy(), self.std.copy()

    def len(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.windows)

    def get(self, idx: int) -> Data:
        """
        Get a single sample as a PyTorch Geometric Data object.

        Args:
            idx: Sample index.

        Returns:
            Data object with:
            - x: Raw signal tensor [n_bearings, sample_window]
            - c: Control variables [n_control, sample_window] (if use_time_control)
            - edge_index: Fully connected graph edges
            - y: Degradation label
        """
        window = self.windows[idx]
        file_start = window["file_start"]
        file_end = window["file_end"]
        sample_start = window["sample_start"]
        label = window["label"]

        # Load data from all files in the window
        file_data_list = []
        for file_idx in range(file_start, file_end):
            file_data = self._load_file_data(file_idx)

            # Apply downsampling if specified
            if self.downsample_factor > 1:
                file_data = file_data[::self.downsample_factor]

            file_data_list.append(file_data)

        # Concatenate across files (for multi-file windows)
        if len(file_data_list) > 1:
            combined_data = np.concatenate(file_data_list, axis=0)
        else:
            combined_data = file_data_list[0]

        # Extract the sample window
        sample_end = sample_start + self.effective_window
        if sample_end > combined_data.shape[0]:
            # Pad with zeros if necessary
            window_data = np.zeros((self.effective_window, self.n_bearings), dtype=np.float32)
            available = combined_data.shape[0] - sample_start
            if available > 0:
                window_data[:available] = combined_data[sample_start:sample_start + available]
        else:
            window_data = combined_data[sample_start:sample_end]

        # Compute normalization stats on first access if needed
        if self.normalize and not hasattr(self, 'mean'):
            self._compute_normalization_stats()

        # Normalize
        if self.normalize:
            window_data = (window_data - self.mean) / self.std
            window_data = np.nan_to_num(window_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Transpose to [n_bearings, sample_window] for DySTGAT
        x = torch.tensor(window_data.T, dtype=torch.float32)

        # Control variables
        if self.use_time_control:
            file_idx = file_end - 1
            time_idx = self.time_indices[file_idx]
            norm_time = self.normalized_time[file_idx]
            # Replicate time values across the window
            c = torch.tensor([
                [time_idx] * self.effective_window,
                [norm_time] * self.effective_window,
            ], dtype=torch.float32)
        else:
            c = torch.zeros(0, self.effective_window, dtype=torch.float32)

        # Fully connected graph
        edge_index = self._create_fully_connected_graph(self.n_bearings)

        # Create data object
        data = Data(x=x, edge_index=edge_index, c=c)
        data.y = torch.tensor([label], dtype=torch.long)

        # Store metadata for debugging
        data.file_idx = torch.tensor([file_start], dtype=torch.long)
        data.sample_idx = torch.tensor([sample_start], dtype=torch.long)

        return data

    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge index including self-loops."""
        src, dst = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)


def get_ims_raw_control_variables(use_time: bool = False) -> List[str]:
    """Get control variable names for raw IMS dataset."""
    if use_time:
        return ["time_index", "normalized_time"]
    return []


def get_ims_raw_measurement_variables() -> List[str]:
    """Get measurement variable names (bearing names)."""
    return MEASUREMENT_VARS.copy()
