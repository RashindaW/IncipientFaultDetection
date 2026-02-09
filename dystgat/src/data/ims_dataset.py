"""
PyTorch Geometric Dataset for IMS Bearing vibration data.

This module handles loading, feature extraction, and creating temporal graph data
for the DySTGAT model using raw accelerometer signals from the IMS Bearing dataset.

The key challenge is converting 20 kHz vibration signals into meaningful features
that capture bearing health degradation over time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings

import numpy as np
import torch
from scipy import stats as scipy_stats
from scipy.fft import rfft, rfftfreq
from torch_geometric.data import Data, Dataset

from .ims_column_config import (
    MEASUREMENT_VARS,
    CONTROL_VARS,
    ALL_BEARING_FEATURES,
    N_FEATURES_PER_BEARING,
    SAMPLE_RATE,
    SAMPLES_PER_FILE,
    FAILURE_INFO,
)


def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract time-domain features from a vibration signal.

    Args:
        signal: 1D array of vibration samples

    Returns:
        Dictionary of feature name -> value
    """
    signal = signal.astype(np.float64)

    # Handle edge cases
    if len(signal) == 0 or np.all(signal == 0):
        return {
            "rms": 0.0,
            "peak": 0.0,
            "crest_factor": 0.0,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "std": 0.0,
            "peak_to_peak": 0.0,
            "shape_factor": 0.0,
        }

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    std = np.std(signal)

    # Avoid division by zero
    eps = 1e-10
    crest_factor = peak / (rms + eps)

    # Scipy kurtosis and skewness (Fisher's definition)
    kurtosis = scipy_stats.kurtosis(signal, fisher=True)
    skewness = scipy_stats.skew(signal)

    peak_to_peak = np.max(signal) - np.min(signal)

    mean_abs = np.mean(np.abs(signal))
    shape_factor = rms / (mean_abs + eps)

    return {
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(crest_factor),
        "kurtosis": float(kurtosis),
        "skewness": float(skewness),
        "std": float(std),
        "peak_to_peak": float(peak_to_peak),
        "shape_factor": float(shape_factor),
    }


def extract_spectral_features(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract frequency-domain features from a vibration signal.

    Args:
        signal: 1D array of vibration samples
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary of feature name -> value
    """
    signal = signal.astype(np.float64)

    if len(signal) == 0 or np.all(signal == 0):
        return {
            "spectral_centroid": 0.0,
            "spectral_spread": 0.0,
            "spectral_rolloff": 0.0,
            "spectral_flatness": 0.0,
            "band_power_low": 0.0,
            "band_power_mid": 0.0,
            "band_power_high": 0.0,
            "dominant_freq": 0.0,
        }

    # Compute FFT
    n = len(signal)
    fft_vals = rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = rfftfreq(n, 1.0 / sample_rate)

    # Power spectrum (squared magnitude)
    power = fft_mag ** 2
    total_power = np.sum(power) + 1e-10

    # Spectral centroid (center of mass)
    spectral_centroid = np.sum(freqs * power) / total_power

    # Spectral spread (standard deviation around centroid)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / total_power)

    # Spectral rolloff (frequency below which 85% of energy exists)
    cumsum_power = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum_power, 0.85 * total_power)
    spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

    # Spectral flatness (geometric mean / arithmetic mean)
    # Indicates how "noise-like" vs "tonal" the signal is
    log_power = np.log(power + 1e-10)
    geometric_mean = np.exp(np.mean(log_power))
    arithmetic_mean = np.mean(power)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    # Band powers
    # Low: 0-2 kHz (bearing defect frequencies are typically here)
    # Mid: 2-5 kHz (resonance frequencies)
    # High: 5-10 kHz (high-frequency components)
    low_mask = freqs < 2000
    mid_mask = (freqs >= 2000) & (freqs < 5000)
    high_mask = (freqs >= 5000) & (freqs < 10000)

    band_power_low = np.sum(power[low_mask]) / total_power
    band_power_mid = np.sum(power[mid_mask]) / total_power
    band_power_high = np.sum(power[high_mask]) / total_power

    # Dominant frequency
    dominant_idx = np.argmax(power)
    dominant_freq = freqs[dominant_idx]

    return {
        "spectral_centroid": float(spectral_centroid),
        "spectral_spread": float(spectral_spread),
        "spectral_rolloff": float(spectral_rolloff),
        "spectral_flatness": float(spectral_flatness),
        "band_power_low": float(band_power_low),
        "band_power_mid": float(band_power_mid),
        "band_power_high": float(band_power_high),
        "dominant_freq": float(dominant_freq),
    }


def extract_all_features(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract all features from a vibration signal.

    Args:
        signal: 1D array of vibration samples
        sample_rate: Sampling rate in Hz

    Returns:
        1D array of feature values in order of ALL_BEARING_FEATURES
    """
    time_feats = extract_time_features(signal)
    spectral_feats = extract_spectral_features(signal, sample_rate)

    # Combine in the order defined by ALL_BEARING_FEATURES
    features = []
    for feat_name in ALL_BEARING_FEATURES:
        if feat_name in time_feats:
            features.append(time_feats[feat_name])
        elif feat_name in spectral_feats:
            features.append(spectral_feats[feat_name])
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def load_ims_file(filepath: str) -> np.ndarray:
    """
    Load a single IMS bearing data file.

    Args:
        filepath: Path to the data file

    Returns:
        2D array of shape [n_samples, n_channels]
    """
    try:
        data = np.loadtxt(filepath, dtype=np.float32)
        return data
    except Exception as e:
        warnings.warn(f"Error loading {filepath}: {e}")
        return np.array([])


def get_file_list(data_dir: str, dataset_set: str = "2nd_test") -> List[str]:
    """
    Get sorted list of data files for a dataset set.

    Args:
        data_dir: Base data directory
        dataset_set: One of "1st_test", "2nd_test", "3rd_test"

    Returns:
        Sorted list of file paths
    """
    set_dir = os.path.join(data_dir, dataset_set)
    if not os.path.exists(set_dir):
        return []

    files = []
    for f in os.listdir(set_dir):
        fpath = os.path.join(set_dir, f)
        if os.path.isfile(fpath):
            files.append(fpath)

    # Sort by timestamp (filename is timestamp)
    files.sort()
    return files


def assign_labels(n_files: int, dataset_set: str, healthy_ratio: float = 0.7) -> np.ndarray:
    """
    Assign degradation labels to files based on position in run-to-failure.

    Since IMS is a run-to-failure dataset:
    - First ~70% of files: healthy (0)
    - Next ~20% of files: degraded (1)
    - Last ~10% of files: faulty (2)

    Args:
        n_files: Total number of files
        dataset_set: Dataset identifier
        healthy_ratio: Fraction considered healthy

    Returns:
        Array of labels [0, 1, 2] for each file
    """
    labels = np.zeros(n_files, dtype=np.int64)

    healthy_end = int(n_files * healthy_ratio)
    degraded_end = int(n_files * 0.9)

    labels[healthy_end:degraded_end] = 1  # Degraded
    labels[degraded_end:] = 2  # Faulty

    return labels


class IMSBearingDataset(Dataset):
    """
    Dataset for IMS Bearing vibration data.

    Extracts features from raw accelerometer signals and creates sliding windows
    for temporal graph learning.

    Attributes:
        data_dir: Base directory containing 1st_test, 2nd_test, 3rd_test folders
        dataset_set: Which test set to use ("1st_test", "2nd_test", "3rd_test")
        window_size: Number of consecutive snapshots per sample
        stride: Step between windows
        feature_agg: How to aggregate features across window ("last", "mean", "all")
    """

    def __init__(
        self,
        data_dir: str,
        dataset_set: str = "2nd_test",
        window_size: int = 15,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        healthy_ratio: float = 0.7,
        fault_filter: Optional[List[int]] = None,
        require_stats: bool = False,
        max_files: Optional[int] = None,
        use_time_control: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_set = dataset_set
        self.window_size = window_size
        self.stride = max(1, stride)
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.healthy_ratio = healthy_ratio
        self.fault_filter = set(fault_filter) if fault_filter is not None else None
        self.require_stats = require_stats
        self.max_files = max_files
        self.use_time_control = use_time_control

        self.n_bearings = 4
        self.n_features = N_FEATURES_PER_BEARING
        self.n_measurement_vars = self.n_bearings  # Each bearing is a node
        self.n_control_vars = 2 if use_time_control else 0

        # Load and preprocess data
        self._load_data()
        self._create_windows()

        print(f"IMS Dataset ({dataset_set}): {len(self.windows)} samples, "
              f"{self.features.shape[0]} snapshots, {self.n_bearings} bearings")

    def _load_data(self):
        """Load all files and extract features."""
        files = get_file_list(self.data_dir, self.dataset_set)

        if not files:
            raise ValueError(f"No files found in {self.data_dir}/{self.dataset_set}")

        if self.max_files is not None:
            files = files[:self.max_files]

        n_files = len(files)
        n_channels = 8 if self.dataset_set == "1st_test" else 4

        # Preallocate feature array: [n_files, n_bearings, n_features]
        # For set 1, we average x and y channels per bearing
        all_features = []

        print(f"Extracting features from {n_files} files...")
        for i, fpath in enumerate(files):
            if i % 100 == 0:
                print(f"  Processing file {i}/{n_files}...")

            raw_data = load_ims_file(fpath)

            if raw_data.size == 0:
                # Fill with zeros if file couldn't be loaded
                bearing_features = np.zeros((self.n_bearings, self.n_features), dtype=np.float32)
            else:
                bearing_features = []

                if self.dataset_set == "1st_test":
                    # 8 channels: pairs of (x, y) for each bearing
                    # Average features from x and y channels
                    for b in range(4):
                        ch_x = raw_data[:, b * 2] if raw_data.shape[1] > b * 2 else np.zeros(SAMPLES_PER_FILE)
                        ch_y = raw_data[:, b * 2 + 1] if raw_data.shape[1] > b * 2 + 1 else np.zeros(SAMPLES_PER_FILE)

                        feats_x = extract_all_features(ch_x)
                        feats_y = extract_all_features(ch_y)
                        feats_avg = (feats_x + feats_y) / 2
                        bearing_features.append(feats_avg)
                else:
                    # 4 channels: one per bearing
                    for b in range(4):
                        ch = raw_data[:, b] if raw_data.shape[1] > b else np.zeros(SAMPLES_PER_FILE)
                        feats = extract_all_features(ch)
                        bearing_features.append(feats)

                bearing_features = np.stack(bearing_features, axis=0)

            all_features.append(bearing_features)

        self.features = np.stack(all_features, axis=0)  # [n_files, n_bearings, n_features]
        self.labels = assign_labels(n_files, self.dataset_set, self.healthy_ratio)

        # Create time indices for control variables
        self.time_indices = np.arange(n_files, dtype=np.float32)
        self.normalized_time = self.time_indices / max(n_files - 1, 1)

        # Normalize features
        if self.normalize:
            self._normalize_features()

    def _normalize_features(self):
        """Normalize features to zero mean and unit variance."""
        # Reshape to [n_files * n_bearings, n_features] for normalization
        n_files, n_bearings, n_features = self.features.shape
        flat_features = self.features.reshape(-1, n_features)

        if self.normalization_stats is None:
            if self.require_stats:
                raise ValueError("Normalization stats required but not provided")

            self.feature_mean = flat_features.mean(axis=0)
            self.feature_std = np.clip(flat_features.std(axis=0), 1e-6, None)
        else:
            self.feature_mean, self.feature_std = self.normalization_stats

        # Normalize
        flat_features = (flat_features - self.feature_mean) / self.feature_std

        # Handle NaN/Inf
        flat_features = np.nan_to_num(flat_features, nan=0.0, posinf=0.0, neginf=0.0)

        self.features = flat_features.reshape(n_files, n_bearings, n_features)

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalization statistics for use in test/val datasets."""
        return self.feature_mean, self.feature_std

    def _create_windows(self):
        """Create sliding window indices."""
        n_files = self.features.shape[0]
        self.windows = []

        for start in range(0, n_files - self.window_size + 1, self.stride):
            end = start + self.window_size

            # Get label for this window (use last timestamp's label)
            label = self.labels[end - 1]

            # Apply fault filter if specified
            if self.fault_filter is not None and label not in self.fault_filter:
                continue

            self.windows.append((start, end, label))

    def len(self) -> int:
        return len(self.windows)

    def get(self, idx: int) -> Data:
        start, end, label = self.windows[idx]

        # Get features for this window: [window_size, n_bearings, n_features]
        window_features = self.features[start:end]

        # For DySTGAT, we need x as [n_nodes, window_size] where each node is a bearing
        # We use a single feature (e.g., RMS) as the time series, or aggregate features

        # Option 1: Use RMS as the primary signal (index 0)
        # x = window_features[:, :, 0].T  # [n_bearings, window_size]

        # Option 2: Use mean of all features as the signal
        # x = window_features.mean(axis=2).T  # [n_bearings, window_size]

        # Option 3: Use all features concatenated (increases node feature dim)
        # For now, use RMS (most common vibration indicator)
        x = window_features[:, :, 0].T  # [n_bearings, window_size]

        x = torch.tensor(x, dtype=torch.float32)

        # Control variables (time-based)
        if self.use_time_control:
            time_idx = self.time_indices[start:end]
            norm_time = self.normalized_time[start:end]
            c = torch.tensor(
                np.stack([time_idx, norm_time], axis=0),
                dtype=torch.float32
            )  # [2, window_size]
        else:
            c = torch.zeros(0, self.window_size, dtype=torch.float32)

        # Edge index (fully connected graph)
        edge_index = self._create_fully_connected_graph(self.n_bearings)

        # Create data object
        data = Data(x=x, edge_index=edge_index, c=c)
        data.y = torch.tensor([label], dtype=torch.long)

        # Store additional features for potential use
        data.all_features = torch.tensor(window_features, dtype=torch.float32)

        return data

    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge index."""
        src, dst = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)


def get_ims_control_variables(use_time: bool = False) -> List[str]:
    """Get control variable names."""
    if use_time:
        return ["time_index", "normalized_time"]
    return []


def get_ims_measurement_variables() -> List[str]:
    """Get measurement variable names (bearing names)."""
    return MEASUREMENT_VARS.copy()
