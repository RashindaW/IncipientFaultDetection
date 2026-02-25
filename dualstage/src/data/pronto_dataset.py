"""
PRONTO Dataset for PyTorch Geometric

Loads PRONTO benchmark data from raw CSV files, ensuring consistent column order
matching DyEdgeGAT configuration. Handles windowing, normalization, and proper
per-run data cleaning to avoid leakage.

Supports three splitting modes:
- 'temporal': Traditional temporal split (maintains time ordering across splits)
- 'window_shuffle': Creates windows from all data, shuffles windows, then splits.
  This ensures train/val/test have identical distributions while preserving
  temporal order within each window.
- 'segment_shuffle': Divides normal data into N equal segments and assigns
  segments to train/val/test. Addresses temporal drift by ensuring each split
  samples from diverse operating conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Tuple, Dict, Any

from .pronto_raw_loader import (
    load_split_data,
    load_all_normal_data,
    DYEDGEGAT_COLUMNS,
    MEASUREMENT_COLUMNS,
    CONDITIONING_COLUMNS,
    get_measurement_indices,
    get_conditioning_indices,
)
from .pronto_column_config import (
    FAULT_LABEL_COL,
    RUN_COL,
    SAMPLE_COL,
    ALL_VARS,
    MEASUREMENT_VARS,
    CONDITIONING_VARS,
)


class PRONTODataset(Dataset):
    """
    Dataset for the PRONTO benchmark loaded from raw CSV files.

    Loads data directly from raw CSV files to ensure consistent column order.
    Handles 17 variables: splits into Measurements (X) and Conditioning (C).

    Variable Classification (per DyEdgeGAT paper):
    - Conditioning variables (6): control flows + external temperatures
    - Measurement variables (11): sensor readings and valve positions

    Key features:
    - Per-run NaN/Inf filling to avoid cross-run leakage
    - Proper column validation at load time
    - Support for train/val/test splits with proper data isolation

    Split Modes:
    - 'temporal': Traditional temporal split (default for backward compatibility)
    - 'window_shuffle': Creates windows from all normal data, shuffles windows,
      then splits into train/val/test. Ensures identical distributions across
      splits while preserving temporal order within each window.
    - 'segment_shuffle': Divides normal data into N segments, assigns segments
      to splits. Windows created within each segment to maintain temporal coherence.
    """

    # Class-level cache for window-shuffle mode
    # Stores: {cache_key: {'windows': [...], 'data': df, 'norm_stats': ...}}
    _window_shuffle_cache: Dict[str, Any] = {}

    # Class-level cache for segment-shuffle mode
    _segment_shuffle_cache: Dict[str, Any] = {}

    def __init__(
        self,
        data_dir: str,
        split: str,
        window_size: int = 15,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        require_stats: bool = False,
        pred_horizon: int = 0,
        split_mode: str = 'temporal',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_seed: int = 42,
        n_segments: int = 10,
        train_segments: Optional[List[int]] = None,
        val_segments: Optional[List[int]] = None,
        test_segments: Optional[List[int]] = None,
    ):
        """
        Initialize PRONTO dataset.

        Args:
            data_dir: Path to PRONTO benchmark data directory
            split: Data split name ('train', 'val', 'test_baseline', 'test_slugging',
                   'test_blockage', 'test_leakage', 'test_diverted')
            window_size: Number of timesteps per window
            stride: Step size between consecutive windows
            normalize: Whether to normalize data
            normalization_stats: Pre-computed (mean, std) stats for normalization.
                                 If None, stats are computed from this dataset.
            require_stats: If True, raise error if normalization_stats is None
            pred_horizon: Number of future timesteps for prediction (0 = no prediction)
            split_mode: How to split data:
                - 'temporal': Traditional temporal split (default)
                - 'window_shuffle': Create windows, shuffle, then split
                - 'segment_shuffle': Divide into segments, assign to splits
            split_ratios: Ratios for train/val/test_baseline when using window_shuffle
                         (default: 70% train, 15% val, 15% test_baseline)
            random_seed: Random seed for window shuffling (for reproducibility)
            n_segments: Number of segments for segment_shuffle mode (default: 10)
            train_segments: Explicit segment indices for training (overrides split_ratios)
            val_segments: Explicit segment indices for validation (overrides split_ratios)
            test_segments: Explicit segment indices for testing (overrides split_ratios)
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.window_size = window_size
        self.stride = max(1, stride)
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.require_stats = require_stats
        self.pred_horizon = max(0, int(pred_horizon))
        self.split_mode = split_mode
        self.split_ratios = split_ratios
        self.random_seed = random_seed
        self.n_segments = n_segments
        self.train_segments = train_segments
        self.val_segments = val_segments
        self.test_segments = test_segments

        # Get column indices
        self.measurement_indices = np.array(get_measurement_indices())
        self.conditioning_indices = np.array(get_conditioning_indices())
        self.n_measurement_vars = len(self.measurement_indices)
        self.n_conditioning_vars = len(self.conditioning_indices)

        # Load and preprocess data based on split mode
        if split_mode == 'segment_shuffle' and split in ('train', 'val', 'test_baseline'):
            self._load_segment_shuffle_mode()
        elif split_mode == 'window_shuffle' and split in ('train', 'val', 'test_baseline'):
            self._load_window_shuffle_mode()
        else:
            # Traditional temporal split or test sets (slugging, blockage, etc.)
            self.data, self.metadata = self._load_and_preprocess()
            self.run_ranges: List[Tuple[int, int]] = self._compute_run_ranges()
            self.windows = self._create_windows()

        print(f"PRONTO Dataset [{split}] (mode={split_mode}): {len(self.windows)} samples")

    def _load_window_shuffle_mode(self) -> None:
        """
        Load data using two-stage split for balanced train/val with temporal test.

        Two-Stage Split Approach:
        =========================

        Stage 1: Temporal Hold-out (85% / 15%)
        - First 85% of data → used for train/val (shuffled windows)
        - Last 15% of data → test_baseline (strict temporal, no shuffle)

        Stage 2: Window Shuffle within the 85%
        - Create windows from the first 85%
        - Shuffle windows
        - Split: train = 70% of total, val = 15% of total
          (which is ~82.4% and ~17.6% of the 85% trainval portion)

        Benefits:
        - Zero leakage to test_baseline (strict temporal separation)
        - Matched distributions between train and val (both shuffled from same pool)
        - Test evaluates true generalization to future data

        Final Distribution:
        - Train: 70% of 85% = 59.5% of total data (shuffled windows)
        - Val: 15% of 85% = 12.75% of total data (shuffled windows)
        - Test_baseline: 15% of total data (temporal, last portion)

        Uses class-level caching to ensure all splits use the same data/windows.
        """
        # Create cache key based on parameters that affect window creation
        cache_key = f"{self.data_dir}_{self.window_size}_{self.stride}_{self.pred_horizon}_{self.random_seed}"

        # Check if we have cached data
        if cache_key not in PRONTODataset._window_shuffle_cache:
            print(f"  [Two-Stage Split] Loading all normal data...")

            # Load ALL normal data (Test9 + Test11)
            all_data, metadata = load_all_normal_data(self.data_dir)

            if all_data.empty:
                raise ValueError("No normal data found for window-shuffle mode")

            # Clean data per-run (before any splitting)
            all_data = self._clean_data_per_run(all_data)

            # ================================================================
            # STAGE 1: Temporal Hold-out (85% for trainval, 15% for test)
            # ================================================================
            n_total_samples = len(all_data)
            trainval_ratio = 1.0 - self.split_ratios[2]  # 1 - 0.15 = 0.85
            n_trainval = int(n_total_samples * trainval_ratio)

            trainval_data = all_data.iloc[:n_trainval].copy()  # First 85%
            test_data = all_data.iloc[n_trainval:].copy()       # Last 15%

            print(f"  [Stage 1] Temporal split: trainval={len(trainval_data)} ({trainval_ratio*100:.1f}%), "
                  f"test_baseline={len(test_data)} ({self.split_ratios[2]*100:.1f}%)")

            # ================================================================
            # Compute normalization stats from trainval_data ONLY (no leakage)
            # ================================================================
            meas_cols = list(MEASUREMENT_COLUMNS)
            cond_cols = list(CONDITIONING_COLUMNS)

            meas_trainval = trainval_data[meas_cols].values.astype(np.float32)
            cond_trainval = trainval_data[cond_cols].values.astype(np.float32)

            norm_stats = (
                meas_trainval.mean(axis=0),
                np.clip(meas_trainval.std(axis=0), 1e-6, None),
                cond_trainval.mean(axis=0),
                np.clip(cond_trainval.std(axis=0), 1e-6, None),
            )

            # Normalize trainval_data
            trainval_data[meas_cols] = (trainval_data[meas_cols] - norm_stats[0]) / norm_stats[1]
            trainval_data[cond_cols] = (trainval_data[cond_cols] - norm_stats[2]) / norm_stats[3]

            # Normalize test_data using trainval stats (no leakage)
            test_data[meas_cols] = (test_data[meas_cols] - norm_stats[0]) / norm_stats[1]
            test_data[cond_cols] = (test_data[cond_cols] - norm_stats[2]) / norm_stats[3]

            # Reset indices for proper window indexing
            trainval_data = trainval_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)

            # ================================================================
            # Create windows from trainval_data
            # ================================================================
            trainval_run_ranges: List[Tuple[int, int]] = []
            grouped_trainval = trainval_data.groupby(RUN_COL, sort=True)
            for _, frame in grouped_trainval:
                start = frame.index[0]
                end = start + len(frame)
                trainval_run_ranges.append((start, end))

            trainval_windows: List[Tuple[int, int]] = []
            total_len = self.window_size + self.pred_horizon

            for start, end in trainval_run_ranges:
                run_length = end - start
                if run_length < total_len:
                    continue
                for offset in range(0, run_length - total_len + 1, self.stride):
                    win_start = start + offset
                    win_end = win_start + total_len
                    trainval_windows.append((win_start, win_end))

            print(f"  [Stage 2] Created {len(trainval_windows)} windows from trainval data "
                  f"({len(trainval_run_ranges)} runs)")

            # ================================================================
            # Shuffle trainval windows
            # ================================================================
            rng = np.random.RandomState(self.random_seed)
            shuffled_indices = rng.permutation(len(trainval_windows))
            shuffled_trainval_windows = [trainval_windows[i] for i in shuffled_indices]

            # ================================================================
            # STAGE 2: Split shuffled windows into train/val
            # train_ratio = 0.70 / 0.85 ≈ 0.824 of trainval
            # val_ratio = 0.15 / 0.85 ≈ 0.176 of trainval
            # ================================================================
            train_fraction_of_trainval = self.split_ratios[0] / trainval_ratio
            n_train = int(len(shuffled_trainval_windows) * train_fraction_of_trainval)

            train_windows = shuffled_trainval_windows[:n_train]
            val_windows = shuffled_trainval_windows[n_train:]

            print(f"  [Stage 2] Shuffled split: train={len(train_windows)}, val={len(val_windows)}")

            # ================================================================
            # Create windows from test_data (temporal, NO shuffle)
            # ================================================================
            test_run_ranges: List[Tuple[int, int]] = []
            grouped_test = test_data.groupby(RUN_COL, sort=True)
            for _, frame in grouped_test:
                start = frame.index[0]
                end = start + len(frame)
                test_run_ranges.append((start, end))

            test_baseline_windows: List[Tuple[int, int]] = []
            for start, end in test_run_ranges:
                run_length = end - start
                if run_length < total_len:
                    continue
                for offset in range(0, run_length - total_len + 1, self.stride):
                    win_start = start + offset
                    win_end = win_start + total_len
                    test_baseline_windows.append((win_start, win_end))

            print(f"  [Two-Stage Split] test_baseline={len(test_baseline_windows)} windows "
                  f"(temporal, from last {self.split_ratios[2]*100:.0f}%)")

            # Final summary
            total_windows = len(train_windows) + len(val_windows) + len(test_baseline_windows)
            print(f"  [Two-Stage Split] Final: train={len(train_windows)} ({len(train_windows)/total_windows*100:.1f}%), "
                  f"val={len(val_windows)} ({len(val_windows)/total_windows*100:.1f}%), "
                  f"test_baseline={len(test_baseline_windows)} ({len(test_baseline_windows)/total_windows*100:.1f}%)")

            # Cache the results
            PRONTODataset._window_shuffle_cache[cache_key] = {
                'trainval_data': trainval_data,
                'test_data': test_data,
                'norm_stats': norm_stats,
                'train_windows': train_windows,
                'val_windows': val_windows,
                'test_baseline_windows': test_baseline_windows,
                'metadata': metadata,
            }

        # Retrieve from cache
        cache = PRONTODataset._window_shuffle_cache[cache_key]

        # Set normalization stats
        (
            self.measurement_mean,
            self.measurement_std,
            self.conditioning_mean,
            self.conditioning_std,
        ) = cache['norm_stats']

        # Select data and windows for this split
        if self.split == 'train':
            self.data = cache['trainval_data']
            self.windows = cache['train_windows']
            self.metadata = cache['metadata']
        elif self.split == 'val':
            self.data = cache['trainval_data']
            self.windows = cache['val_windows']
            self.metadata = cache['metadata']
        elif self.split == 'test_baseline':
            self.data = cache['test_data']
            self.windows = cache['test_baseline_windows']
            self.metadata = cache['metadata']
        else:
            raise ValueError(f"Unknown split for window_shuffle mode: {self.split}")

        # Store run_ranges for compatibility (though not used directly in this mode)
        self.run_ranges = []

    def _load_segment_shuffle_mode(self) -> None:
        """
        Load data using segment shuffle mode for balanced operating conditions.

        Divides all normal data into N equal segments, assigns segments to
        train/val/test splits (either explicitly or by ratio), and creates
        windows within assigned segments only. Normalization uses train segment
        data only to avoid leakage.

        Uses class-level caching so that train/val/test_baseline splits share
        the same underlying data and segment assignments.
        """
        cache_key = (
            f"seg_{self.data_dir}_{self.window_size}_{self.stride}"
            f"_{self.pred_horizon}_{self.random_seed}_{self.n_segments}"
        )

        if cache_key not in PRONTODataset._segment_shuffle_cache:
            print(f"  [Segment Shuffle] Loading all normal data...")

            # Load ALL normal data (Test9 + Test11) via existing loader
            all_data, metadata = load_all_normal_data(self.data_dir)

            if all_data.empty:
                raise ValueError("No normal data found for segment-shuffle mode")

            # Clean data per-run (before any splitting)
            all_data = self._clean_data_per_run(all_data)

            n_rows = len(all_data)

            # Step 1: Create equal segments
            segment_size = n_rows // self.n_segments
            segment_ranges = []
            for i in range(self.n_segments):
                start = i * segment_size
                end = (i + 1) * segment_size if i < self.n_segments - 1 else n_rows
                segment_ranges.append((start, end))

            # Step 2: Assign segments to splits
            if (self.train_segments is not None
                    and self.val_segments is not None
                    and self.test_segments is not None):
                train_seg_indices = np.array(self.train_segments)
                val_seg_indices = np.array(self.val_segments)
                test_seg_indices = np.array(self.test_segments)

                all_assigned = set(train_seg_indices) | set(val_seg_indices) | set(test_seg_indices)
                all_segments = set(range(self.n_segments))
                extra = all_assigned - all_segments
                if extra:
                    raise ValueError(
                        f"Segment indices out of range: {extra}. "
                        f"Valid segments are 0-{self.n_segments-1}"
                    )
                overlap_tv = set(train_seg_indices) & set(val_seg_indices)
                overlap_tt = set(train_seg_indices) & set(test_seg_indices)
                overlap_vt = set(val_seg_indices) & set(test_seg_indices)
                if overlap_tv or overlap_tt or overlap_vt:
                    raise ValueError(
                        f"Segment overlap detected. "
                        f"Train∩Val: {overlap_tv}, Train∩Test: {overlap_tt}, Val∩Test: {overlap_vt}"
                    )
                unused = all_segments - all_assigned
                if unused:
                    print(f"  Note: segments {sorted(unused)} not assigned (excluded from CV)")
                print(f"  Segment shuffle (explicit, n_segments={self.n_segments}):")
            else:
                rng = np.random.RandomState(self.random_seed)
                shuffled = rng.permutation(self.n_segments)
                n_train = int(self.n_segments * self.split_ratios[0])
                n_val = max(1, int(self.n_segments * self.split_ratios[1]))
                train_seg_indices = shuffled[:n_train]
                val_seg_indices = shuffled[n_train:n_train + n_val]
                test_seg_indices = shuffled[n_train + n_val:]
                print(f"  Segment shuffle (seed={self.random_seed}, n_segments={self.n_segments}):")

            print(f"    Train segments ({len(train_seg_indices)}): {sorted(train_seg_indices.tolist())}")
            print(f"    Val segments ({len(val_seg_indices)}): {sorted(val_seg_indices.tolist())}")
            print(f"    Test segments ({len(test_seg_indices)}): {sorted(test_seg_indices.tolist())}")

            # Step 3: Compute normalization stats from TRAIN segments only
            meas_cols = list(MEASUREMENT_COLUMNS)
            cond_cols = list(CONDITIONING_COLUMNS)

            train_dfs = [all_data.iloc[segment_ranges[i][0]:segment_ranges[i][1]]
                         for i in train_seg_indices]
            train_data_for_stats = pd.concat(train_dfs, ignore_index=True)

            meas_train = train_data_for_stats[meas_cols].values.astype(np.float32)
            cond_train = train_data_for_stats[cond_cols].values.astype(np.float32)

            norm_stats = (
                meas_train.mean(axis=0),
                np.clip(meas_train.std(axis=0), 1e-6, None),
                cond_train.mean(axis=0),
                np.clip(cond_train.std(axis=0), 1e-6, None),
            )

            # Step 4: Normalize ALL data using train stats
            all_data[meas_cols] = (all_data[meas_cols] - norm_stats[0]) / norm_stats[1]
            all_data[cond_cols] = (all_data[cond_cols] - norm_stats[2]) / norm_stats[3]

            # Step 5: Create windows within each split's segments
            total_len = self.window_size + self.pred_horizon

            def _make_segment_windows(seg_indices):
                windows = []
                for si in sorted(seg_indices):
                    seg_start, seg_end = segment_ranges[si]
                    seg_length = seg_end - seg_start
                    if seg_length < total_len:
                        print(f"  Warning: Segment {si} [{seg_start}:{seg_end}] too short "
                              f"({seg_length} < {total_len}), skipping")
                        continue
                    for offset in range(0, seg_length - total_len + 1, self.stride):
                        win_start = seg_start + offset
                        win_end = win_start + total_len
                        windows.append((win_start, win_end))
                return windows

            train_windows = _make_segment_windows(train_seg_indices)
            val_windows = _make_segment_windows(val_seg_indices)
            test_windows = _make_segment_windows(test_seg_indices)

            print(f"  [Segment Shuffle] Windows: train={len(train_windows)}, "
                  f"val={len(val_windows)}, test_baseline={len(test_windows)}")

            PRONTODataset._segment_shuffle_cache[cache_key] = {
                'data': all_data,
                'norm_stats': norm_stats,
                'train_windows': train_windows,
                'val_windows': val_windows,
                'test_baseline_windows': test_windows,
                'metadata': metadata,
                'segment_ranges': segment_ranges,
                'train_seg_indices': train_seg_indices,
                'val_seg_indices': val_seg_indices,
                'test_seg_indices': test_seg_indices,
            }

        # Retrieve from cache
        cache = PRONTODataset._segment_shuffle_cache[cache_key]

        self.data = cache['data']
        self.metadata = cache['metadata']
        self.run_ranges = []

        (
            self.measurement_mean,
            self.measurement_std,
            self.conditioning_mean,
            self.conditioning_std,
        ) = cache['norm_stats']

        if self.split == 'train':
            self.windows = cache['train_windows']
        elif self.split == 'val':
            self.windows = cache['val_windows']
        elif self.split == 'test_baseline':
            self.windows = cache['test_baseline_windows']
        else:
            raise ValueError(f"Unknown split for segment_shuffle mode: {self.split}")

    def _load_and_preprocess(self) -> Tuple[pd.DataFrame, dict]:
        """Load data from raw CSV and preprocess."""
        # Load data using raw CSV loader
        data, metadata = load_split_data(self.data_dir, self.split)

        if data.empty:
            return data, metadata

        # Clean data per-run to avoid cross-run leakage (fixes Issue #2)
        data = self._clean_data_per_run(data)

        # Normalize
        if self.normalize:
            data = self._normalize_data(data)

        return data, metadata

    def _clean_data_per_run(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NaN/Inf values per-run to avoid cross-run leakage.

        Instead of filling across the entire concatenated DataFrame,
        this fills NaN values within each run independently.
        """
        value_cols = list(DYEDGEGAT_COLUMNS)

        for run_id in data[RUN_COL].unique():
            mask = data[RUN_COL] == run_id
            run_data = data.loc[mask, value_cols]

            # Replace inf with nan
            run_data = run_data.replace([np.inf, -np.inf], np.nan)

            # Check if any NaN values exist
            if run_data.isna().values.any():
                # Forward fill, then backward fill, then fill with 0
                run_data = run_data.ffill().bfill().fillna(0.0)
                data.loc[mask, value_cols] = run_data

        return data

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize measurement and conditioning variables."""
        meas_cols = list(MEASUREMENT_COLUMNS)
        cond_cols = list(CONDITIONING_COLUMNS)

        if self.normalization_stats is None:
            if self.require_stats:
                raise ValueError(
                    "Normalization stats are required for this dataset split "
                    "(e.g., test/val) to avoid data leakage."
                )

            # Compute stats from this data
            meas = data[meas_cols].values.astype(np.float32)
            cond = data[cond_cols].values.astype(np.float32)
            self.measurement_mean = meas.mean(axis=0)
            self.measurement_std = np.clip(meas.std(axis=0), 1e-6, None)
            self.conditioning_mean = cond.mean(axis=0)
            self.conditioning_std = np.clip(cond.std(axis=0), 1e-6, None)
        else:
            (
                self.measurement_mean,
                self.measurement_std,
                self.conditioning_mean,
                self.conditioning_std,
            ) = self.normalization_stats

        # Apply normalization
        data[meas_cols] = (data[meas_cols] - self.measurement_mean) / self.measurement_std
        data[cond_cols] = (data[cond_cols] - self.conditioning_mean) / self.conditioning_std

        return data

    def _compute_run_ranges(self) -> List[Tuple[int, int]]:
        """Compute start/end indices for each run."""
        ranges: List[Tuple[int, int]] = []
        if self.data.empty:
            return ranges

        # Group by run_id and get index ranges
        # The data should have contiguous indices after reset_index
        grouped = self.data.groupby(RUN_COL, sort=True)
        for _, frame in grouped:
            start = frame.index[0]
            end = start + len(frame)
            ranges.append((start, end))

        return ranges

    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows respecting run boundaries."""
        windows: List[Tuple[int, int]] = []
        total_len = self.window_size + self.pred_horizon

        for start, end in self.run_ranges:
            run_length = end - start
            if run_length < total_len:
                continue

            for offset in range(0, run_length - total_len + 1, self.stride):
                win_start = start + offset
                win_end = win_start + total_len
                windows.append((win_start, win_end))

        return windows

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return normalization statistics for reuse in other splits."""
        return (
            self.measurement_mean,
            self.measurement_std,
            self.conditioning_mean,
            self.conditioning_std,
        )

    def len(self) -> int:
        """Return number of windows."""
        return len(self.windows)

    def get(self, idx: int) -> Data:
        """Get a single data sample."""
        start, end = self.windows[idx]
        history_end = start + self.window_size
        window_data = self.data.iloc[start:history_end]

        # Select X (Measurements) and C (Conditioning variables)
        meas_cols = list(MEASUREMENT_COLUMNS)
        cond_cols = list(CONDITIONING_COLUMNS)

        measurements = torch.tensor(
            window_data[meas_cols].values.T, dtype=torch.float32
        )  # [n_meas, window]

        conditioning = torch.tensor(
            window_data[cond_cols].values.T, dtype=torch.float32
        )  # [n_cond, window]

        edge_index = self._create_fully_connected_graph(self.n_measurement_vars)

        data = Data(x=measurements, edge_index=edge_index, c=conditioning)

        # Add future prediction targets if requested
        if self.pred_horizon > 0:
            future_data = self.data.iloc[history_end:end]
            future_measurements = torch.tensor(
                future_data[meas_cols].values.T, dtype=torch.float32
            )
            data.y_future = future_measurements

        # Add fault label
        label = int(window_data[FAULT_LABEL_COL].iloc[-1])
        data.y = torch.tensor([label], dtype=torch.long)

        return data

    @staticmethod
    def _create_fully_connected_graph(n_nodes: int) -> torch.Tensor:
        """Create fully connected graph edge index."""
        src = []
        dst = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                src.append(i)
                dst.append(j)
        return torch.tensor([src, dst], dtype=torch.long)


class PRONTODatasetLegacy(Dataset):
    """
    Legacy PRONTO dataset that loads from .mat files.

    This class is kept for backward compatibility with existing code that
    relies on the old interface. For new code, use PRONTODataset instead.

    NOTE: This loader has known issues with column order that may cause
    incorrect variable mapping. Use PRONTODataset for reliable results.
    """

    def __init__(
        self,
        data_files: List[str],
        window_size: int = 15,
        stride: int = 1,
        data_dir: str = "",
        normalize: bool = True,
        normalization_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        fault_filter: Optional[List[int]] = None,
        segments_to_load: Optional[List[int]] = None,
        require_stats: bool = False,
        severity_range: Optional[Tuple[int, int]] = None,
        pred_horizon: int = 0,
    ):
        """Legacy constructor for backward compatibility."""
        import warnings
        import os
        import scipy.io

        warnings.warn(
            "PRONTODatasetLegacy uses .mat files which have known column order issues. "
            "Consider using PRONTODataset with raw CSV loading instead.",
            DeprecationWarning,
            stacklevel=2
        )

        super().__init__()
        self.data_files = data_files
        self.window_size = window_size
        self.stride = max(1, stride)
        self.data_dir = data_dir
        self.normalize = normalize
        self.normalization_stats = normalization_stats
        self.fault_filter = set(fault_filter) if fault_filter is not None else None
        self.segments_to_load = set(segments_to_load) if segments_to_load is not None else None
        self.require_stats = require_stats
        self.severity_range = severity_range
        self.pred_horizon = max(0, int(pred_horizon))

        from .pronto_column_config import MEASUREMENT_INDICES, CONTROL_INDICES, SLUGGING_LABEL

        self.measurement_indices = np.array(MEASUREMENT_INDICES)
        self.control_indices = np.array(CONTROL_INDICES)
        self.n_measurement_vars = len(self.measurement_indices)
        self.n_control_vars = len(self.control_indices)

        self.data = self._load_and_preprocess_legacy()
        self.run_ranges: List[Tuple[int, int]] = self._compute_run_ranges()
        self.windows = self._create_windows()

        print(f"PRONTO Legacy Dataset: {len(self.windows)} samples from {len(self.data_files)} file(s)")

    def _load_and_preprocess_legacy(self) -> pd.DataFrame:
        """Load from .mat files (legacy method)."""
        import os
        import scipy.io

        from .pronto_column_config import SLUGGING_LABEL

        dfs: List[pd.DataFrame] = []
        run_counter = 0

        for filename in self.data_files:
            path = os.path.join(self.data_dir, filename)
            try:
                mat = scipy.io.loadmat(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            base_name = os.path.basename(filename)
            key = os.path.splitext(base_name)[0]

            segments = []
            labels = []

            if "HealthySet" in key:
                raw_segments = mat['HealthySet'][0]
                for i, seg in enumerate(raw_segments):
                    if self.segments_to_load is not None and i not in self.segments_to_load:
                        continue
                    segments.append(seg)
                    labels.append(0)
            elif "SlugSet" in key:
                raw_segments = mat['SlugSet'][0]
                for i, seg in enumerate(raw_segments):
                    if self.segments_to_load is not None and i not in self.segments_to_load:
                        continue
                    segments.append(seg)
                    labels.append(SLUGGING_LABEL)
            elif "Blockage" in key or "Leakage" in key or "Diverted" in key:
                content = mat[key][0, 0]
                lab_arr = content[0].flatten()
                data_arr = content[1]

                if self.severity_range is not None:
                    min_sev, max_sev = self.severity_range
                    severity_mask = (lab_arr >= min_sev) & (lab_arr <= max_sev)
                    data_arr = data_arr[severity_mask]
                    lab_arr = lab_arr[severity_mask]

                    if len(data_arr) == 0:
                        continue

                keep_indices = [i for i in range(19) if i not in [7, 18]]
                data_arr = data_arr[:, keep_indices]

                if self.segments_to_load is not None and 0 not in self.segments_to_load:
                    continue

                segments.append(data_arr)

                if "Blockage" in key:
                    l = 1
                elif "Leakage" in key:
                    l = 2
                elif "Diverted" in key:
                    l = 3
                else:
                    l = 1
                labels.append(l)
            else:
                continue

            for i, seg in enumerate(segments):
                df = pd.DataFrame(seg, columns=ALL_VARS)
                df[RUN_COL] = run_counter
                df[SAMPLE_COL] = np.arange(len(df))
                df[FAULT_LABEL_COL] = labels[i]
                dfs.append(df)
                run_counter += 1

        if not dfs:
            return pd.DataFrame(columns=ALL_VARS + [RUN_COL, SAMPLE_COL, FAULT_LABEL_COL])

        combined = pd.concat(dfs, ignore_index=True)

        if self.fault_filter is not None:
            combined = combined[combined[FAULT_LABEL_COL].isin(self.fault_filter)]
            if combined.empty:
                return pd.DataFrame(columns=ALL_VARS + [RUN_COL, SAMPLE_COL, FAULT_LABEL_COL])
            combined = combined.reset_index(drop=True)

        if self.normalize:
            meas_cols = [ALL_VARS[i] for i in self.measurement_indices]
            ctrl_cols = [ALL_VARS[i] for i in self.control_indices]

            if self.normalization_stats is None:
                if self.require_stats:
                    raise ValueError("Normalization stats are required.")

                meas = combined[meas_cols].values.astype(np.float32)
                ctrl = combined[ctrl_cols].values.astype(np.float32)
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

            combined[meas_cols] = (combined[meas_cols] - self.measurement_mean) / self.measurement_std
            combined[ctrl_cols] = (combined[ctrl_cols] - self.control_mean) / self.control_std
        else:
            self.measurement_mean = np.zeros(self.n_measurement_vars)
            self.measurement_std = np.ones(self.n_measurement_vars)
            self.control_mean = np.zeros(self.n_control_vars)
            self.control_std = np.ones(self.n_control_vars)

        value_cols = [ALL_VARS[i] for i in self.measurement_indices] + [ALL_VARS[i] for i in self.control_indices]
        combined[value_cols] = combined[value_cols].replace([np.inf, -np.inf], np.nan)
        if combined[value_cols].isna().values.any():
            combined[value_cols] = combined[value_cols].ffill().bfill().fillna(0.0)

        return combined

    def _compute_run_ranges(self) -> List[Tuple[int, int]]:
        """Compute run ranges."""
        ranges: List[Tuple[int, int]] = []
        if self.data.empty:
            return ranges

        grouped = self.data.groupby(RUN_COL, sort=True)
        for _, frame in grouped:
            start = frame.index[0]
            end = start + len(frame)
            ranges.append((start, end))
        return ranges

    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create windows."""
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
        """Return normalization stats."""
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
        window_data = self.data.iloc[start:history_end]

        meas_cols = [ALL_VARS[i] for i in self.measurement_indices]
        ctrl_cols = [ALL_VARS[i] for i in self.control_indices]

        measurements = torch.tensor(
            window_data[meas_cols].values.T, dtype=torch.float32
        )
        controls = torch.tensor(
            window_data[ctrl_cols].values.T, dtype=torch.float32
        )

        edge_index = self._create_fully_connected_graph(self.n_measurement_vars)
        data = Data(x=measurements, edge_index=edge_index, c=controls)

        if self.pred_horizon > 0:
            future_data = self.data.iloc[history_end:end]
            future_measurements = torch.tensor(
                future_data[meas_cols].values.T, dtype=torch.float32
            )
            data.y_future = future_measurements

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
