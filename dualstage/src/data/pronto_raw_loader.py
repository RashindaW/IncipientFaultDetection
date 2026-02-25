"""
PRONTO Raw CSV Loader

Loads PRONTO data directly from raw CSV files to ensure consistent column order
matching DyEdgeGAT configuration. Fixes issues with pre-processed .mat files
having inconsistent column ordering.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# =============================================================================
# DyEdgeGAT Column Configuration (17 variables)
# =============================================================================

# Exact 17 variables from DyEdgeGAT Table III (excludes Water Density & tank level)
# Classification per DyEdgeGAT paper:
#   - Conditioning (6): control flows (4) + external temperatures (2)
#   - Measurement (11): all other sensor/valve measurements
DYEDGEGAT_COLUMNS = [
    'Air In1',           # 0 - Conditioning (control) - FT305
    'Air In2',           # 1 - Conditioning (control) - FT302
    'Air T',             # 2 - Conditioning (external) - FT305/AI2
    'Air P',             # 3 - Measurement - PT312
    'Water In1',         # 4 - Conditioning (control) - FT102
    'Water In2',         # 5 - Conditioning (control) - FT104
    'Water T',           # 6 - Conditioning (external) - FT102/AI3
    'Mixture zone P',    # 7 - Measurement - PT417
    'riser outlet P',    # 8 - Measurement - PT408
    'P topsep',          # 9 - Measurement - PT403
    'FR topsep gas',     # 10 - Measurement - FT404
    'FR topsep liquid',  # 11 - Measurement - FT406
    'P_3phase',          # 12 - Measurement - PT501
    'Air Valve',         # 13 - Measurement (valve position) - PIC501
    'Water level',       # 14 - Measurement - LI502
    'Water coalescer',   # 15 - Measurement - LI503
    'Water level valve', # 16 - Measurement (valve position) - LVC502-SR
]

# Variables excluded to match DyEdgeGAT (present in raw CSV but not used)
EXCLUDED_COLUMNS = [
    'Water Density',     # FT102/AI2 - Column 8 in raw CSV
    'water tank level',  # LI101 - Column 19 in raw CSV
]

# Measurement variables (11 total) - system response variables (sensors and valve positions)
MEASUREMENT_COLUMNS = [
    'Air P',
    'Mixture zone P',
    'riser outlet P',
    'P topsep',
    'FR topsep gas',
    'FR topsep liquid',
    'P_3phase',
    'Air Valve',
    'Water level',
    'Water coalescer',
    'Water level valve',
]

# Conditioning variables (6 total) - control flows + external temperatures
# Per DyEdgeGAT paper: "input air flow rate (FT305/302) and input water flow rate (FT102/104)
# as control variables, as well as input air temperature (FT305-T) and input water temperature
# (FT102-T) as external variables influencing the operating conditions"
CONDITIONING_COLUMNS = [
    'Air In1',    # Control flow rate
    'Air In2',    # Control flow rate
    'Water In1',  # Control flow rate
    'Water In2',  # Control flow rate
    'Air T',      # External variable
    'Water T',    # External variable
]

# Legacy alias for backward compatibility
CONTROL_COLUMNS = CONDITIONING_COLUMNS

# =============================================================================
# Data Split Configuration
# =============================================================================

# CSV file mapping for each scenario/test combination
# Maps (scenario, test) -> relative path to Process Data CSV
CSV_FILE_MAPPING = {
    # Normal/Slugging conditions
    ('C0 Normal and Slugging conditions', 'Test9'): 'C0 Normal and Slugging conditions/Test9/Process Data/0912Testday4.csv',
    ('C0 Normal and Slugging conditions', 'Test10'): 'C0 Normal and Slugging conditions/Test10/Process Data/0912Testday4.csv',
    ('C0 Normal and Slugging conditions', 'Test11'): 'C0 Normal and Slugging conditions/Test11/Process Data/0626Testday5.csv',

    # Air Blockage
    ('C1 Air Blockage', 'Test2'): 'C1 Air Blockage/Test2/Process Data/0907Testday2.csv',
    ('C1 Air Blockage', 'Test3'): 'C1 Air Blockage/Test3/Process Data/0907Testday2.csv',

    # Air Leakage
    ('C2 Air Leakage', 'Test4'): 'C2 Air Leakage/Test4/Process Data/0907Testday2.csv',
    ('C2 Air Leakage', 'Test5'): 'C2 Air Leakage/Test5/Process Data/0911Testday3.csv',
    ('C2 Air Leakage', 'Test6'): 'C2 Air Leakage/Test6/Process Data/0911Testday3.csv',

    # Diverted flow
    ('C3 Diverted flow', 'Test7'): 'C3 Diverted flow/Test7/Process Data/0911Testday3.csv',
    ('C3 Diverted flow', 'Test8'): 'C3 Diverted flow/Test8/Process Data/0911Testday3.csv',
}

# Data splits following DyEdgeGAT methodology
# Format: list of (scenario, test, condition, start_ratio, end_ratio)
# condition: 'all', 'normal_only', 'normal_filtered', 'slugging_only', 'blockage', 'leakage', 'diverted'
#
# NOTE: We use TEMPORAL splits to maintain time series contiguity.
# The 'normal_filtered' condition removes shutdown phases (Air In1 < 10) BEFORE splitting,
# which makes the remaining data more homogeneous for train/val.
DATA_SPLITS = {
    # Training from Test11 (pure Normal from different day - 0626)
    # Filter out shutdown phase first, then take first 70% of filtered data
    'train': [
        ('C0 Normal and Slugging conditions', 'Test11', 'normal_filtered', 0.0, 0.7),
    ],

    # Validation from Test11 (last 30% of filtered data)
    'val': [
        ('C0 Normal and Slugging conditions', 'Test11', 'normal_filtered', 0.7, 1.0),
    ],

    # Test sets
    'test_baseline': [
        ('C0 Normal and Slugging conditions', 'Test9', 'normal_only', 0.0, 1.0),
    ],
    'test_slugging': [
        ('C0 Normal and Slugging conditions', 'Test9', 'slugging_only', 0.0, 1.0),
    ],
    'test_blockage': [
        ('C1 Air Blockage', 'Test2', 'blockage', 0.0, 1.0),
        ('C1 Air Blockage', 'Test3', 'blockage', 0.0, 1.0),
    ],
    'test_leakage': [
        ('C2 Air Leakage', 'Test4', 'leakage', 0.0, 1.0),
        ('C2 Air Leakage', 'Test5', 'leakage', 0.0, 1.0),
        ('C2 Air Leakage', 'Test6', 'leakage', 0.0, 1.0),
    ],
    'test_diverted': [
        ('C3 Diverted flow', 'Test7', 'diverted', 0.0, 1.0),
        ('C3 Diverted flow', 'Test8', 'diverted', 0.0, 1.0),
    ],
}

# Minimum air flow threshold to filter out shutdown phases
MIN_AIR_FLOW_THRESHOLD = 10.0

# Operating regime thresholds for Test11
# The data has two main operating regimes:
# - Low-flow regime: Air In1 ~100-130 (stable, used for training)
# - High-flow regime: Air In1 ~140-160 (different operating point)
# Using only the stable regime ensures train/val have similar distributions
STABLE_REGIME_AIR_IN1_MAX = 135.0

# Fault labels
LABEL_NORMAL = 0
LABEL_BLOCKAGE = 1
LABEL_LEAKAGE = 2
LABEL_DIVERTED = 3
LABEL_SLUGGING = 4

CONDITION_TO_LABEL = {
    'all': LABEL_NORMAL,
    'normal_only': LABEL_NORMAL,
    'normal_filtered': LABEL_NORMAL,  # Same as normal but with shutdown filtering
    'slugging_only': LABEL_SLUGGING,
    'blockage': LABEL_BLOCKAGE,
    'leakage': LABEL_LEAKAGE,
    'diverted': LABEL_DIVERTED,
}

# =============================================================================
# Consolidated CSV Data Splits
# =============================================================================
# For use with consolidated CSV files (normal.csv, slugging.csv, etc.)
# Format: (csv_file, label, source_test_filter, start_ratio, end_ratio)
# source_test_filter: None = all, or specific test name like 'Test9', 'Test11'

CONSOLIDATED_CSV_FILES = {
    'normal': 'normal.csv',
    'slugging': 'slugging.csv',
    'blockage': 'blockage.csv',
    'leakage': 'leakage.csv',
    'diverted': 'diverted.csv',
}

# Format: (csv_file, label, source_test_filter, start_ratio, end_ratio, split_mode)
# split_mode: 'temporal' (simple split) or 'stratified' (per source_test)
#
# Strategy: Train/Val from Test9 (same distribution), Test from Test11 (different day)
# This ensures:
# 1. Train and Val have matching distributions (both from Test9)
# 2. Test evaluates generalization to data from a different experimental session
# 3. Temporal order is preserved within each split
CONSOLIDATED_DATA_SPLITS = {
    # Training: 80% of Test9 normal data (~8,070 samples)
    'train': [
        ('normal.csv', LABEL_NORMAL, 'Test9', 0.0, 0.8, 'temporal'),
    ],

    # Validation: 20% of Test9 normal data (~2,018 samples)
    'val': [
        ('normal.csv', LABEL_NORMAL, 'Test9', 0.8, 1.0, 'temporal'),
    ],

    # Test baseline: ALL Test11 normal data (~3,212 samples)
    # Tests generalization to different day/session
    'test_baseline': [
        ('normal.csv', LABEL_NORMAL, 'Test11', 0.0, 1.0, 'temporal'),
    ],

    # Slugging: ALL slugging data (Test9 + Test11 = 4,702 samples)
    'test_slugging': [
        ('slugging.csv', LABEL_SLUGGING, None, 0.0, 1.0, 'temporal'),
    ],

    # Fault tests: ALL data from each fault type
    'test_blockage': [
        ('blockage.csv', LABEL_BLOCKAGE, None, 0.0, 1.0, 'temporal'),
    ],
    'test_leakage': [
        ('leakage.csv', LABEL_LEAKAGE, None, 0.0, 1.0, 'temporal'),
    ],
    'test_diverted': [
        ('diverted.csv', LABEL_DIVERTED, None, 0.0, 1.0, 'temporal'),
    ],
}


def classify_flow_regime(
    air_in1: float, air_in2: float, water_in1: float, water_in2: float
) -> str:
    """
    Classify flow regime based on TOTAL air and water flow (PRONTO paper Table 5).

    PRONTO has two air inlet valves (Air In1/FT305 and Air In2/FT302) and two
    water inlet valves (Water In1/FT102 and Water In2/FT104). The regime must
    be determined from the TOTAL flow, not individual valve readings.

    Args:
        air_in1: Air flow rate from Air In1 in sm³/h
        air_in2: Air flow rate from Air In2 in sm³/h
        water_in1: Water flow rate from Water In1 in kg/s
        water_in2: Water flow rate from Water In2 in kg/s

    Returns:
        'slugging' or 'normal'

    Based on PRONTO paper Table 5:
    - Slugging typically occurs at low total air flow (≤50 sm³/h) with low water flow
    - Normal flow at higher air flow rates (≥100 sm³/h)
    """
    total_air = air_in1 + air_in2
    total_water = water_in1 + water_in2

    # Slugging conditions based on Table 5
    if total_air <= 50 and total_water <= 2.0:
        return 'slugging'
    # Normal conditions
    return 'normal'


def load_raw_csv(csv_path: str, validate_columns: bool = True) -> pd.DataFrame:
    """
    Load raw CSV file and select DyEdgeGAT columns.

    Args:
        csv_path: Path to the raw CSV file
        validate_columns: Whether to validate that all required columns exist

    Returns:
        DataFrame with TIMESTAMP + 17 DyEdgeGAT columns

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Skip first 2 rows (metadata and sensor tags), row 3 is column names
    df = pd.read_csv(csv_path, skiprows=2)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    if validate_columns:
        # Check for required columns
        missing = set(DYEDGEGAT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        # Verify excluded columns are not accidentally included
        for excluded in EXCLUDED_COLUMNS:
            if excluded in df.columns:
                # This is expected - we'll exclude them below
                pass

    # Select TIMESTAMP + DyEdgeGAT columns only
    columns_to_keep = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS
    available = [c for c in columns_to_keep if c in df.columns]

    if 'TIMESTAMP' not in df.columns:
        # Some files might not have TIMESTAMP, use index
        df['TIMESTAMP'] = df.index

    df = df[available].copy()

    return df


def load_scenario_data(
    base_path: str,
    scenario: str,
    test: str,
    condition: str = 'all',
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Load data for a specific scenario/test combination.

    Args:
        base_path: Base path to PRONTO benchmark data
        scenario: Scenario name (e.g., 'C0 Normal and Slugging conditions')
        test: Test name (e.g., 'Test11')
        condition: Filter condition ('all', 'normal_only', 'normal_filtered',
                   'slugging_only', 'blockage', 'leakage', 'diverted')
        start_ratio: Start ratio for data subset (0.0 to 1.0)
        end_ratio: End ratio for data subset (0.0 to 1.0)

    Returns:
        DataFrame with filtered and subsetted data

    Note:
        For 'normal_filtered', shutdown phases are removed BEFORE the temporal
        split is applied. This ensures:
        1. Time series contiguity is maintained within train and val
        2. The shutdown phase (Air In1 < 10) is excluded from both
        3. Train/val have similar distributions since the anomalous
           shutdown period is removed before splitting
    """
    # Get CSV path from mapping
    key = (scenario, test)
    if key not in CSV_FILE_MAPPING:
        raise ValueError(f"Unknown scenario/test combination: {key}")

    csv_rel_path = CSV_FILE_MAPPING[key]
    csv_path = os.path.join(base_path, csv_rel_path)

    # Load raw data
    df = load_raw_csv(csv_path)

    # Apply condition filtering BEFORE temporal split
    if condition == 'normal_filtered' and 'Air In1' in df.columns:
        # Calculate total air flow from both inlet valves
        # Air In2 may not exist in all datasets, default to 0
        air_in2 = df['Air In2'] if 'Air In2' in df.columns else 0
        total_air = df['Air In1'] + air_in2

        # Filter to stable operating regime only:
        # 1. Remove shutdown phases (Total Air < MIN_AIR_FLOW_THRESHOLD)
        # 2. Remove high-flow regime (Total Air > STABLE_REGIME_AIR_IN1_MAX)
        # 3. Keep only the largest contiguous segment to maintain time series integrity
        mask = (total_air >= MIN_AIR_FLOW_THRESHOLD) & \
               (total_air <= STABLE_REGIME_AIR_IN1_MAX)
        df_filtered = df[mask].reset_index(drop=True)

        # Find the largest contiguous segment (gaps > 5 seconds indicate segment boundaries)
        if len(df_filtered) > 1 and 'TIMESTAMP' in df_filtered.columns:
            timestamps = df_filtered['TIMESTAMP'].values
            time_diffs = np.diff(timestamps) * 86400  # Convert to seconds

            # Find segment boundaries (gaps > 5 seconds)
            gap_indices = np.where(time_diffs > 5)[0]

            if len(gap_indices) > 0:
                # Build list of segment (start, end) tuples
                segment_starts = [0] + [idx + 1 for idx in gap_indices]
                segment_ends = [idx + 1 for idx in gap_indices] + [len(df_filtered)]

                # Find the largest segment
                segments = [(s, e, e - s) for s, e in zip(segment_starts, segment_ends)]
                largest_segment = max(segments, key=lambda x: x[2])

                # Keep only the largest contiguous segment
                df = df_filtered.iloc[largest_segment[0]:largest_segment[1]].reset_index(drop=True)
            else:
                df = df_filtered
        else:
            df = df_filtered

    elif condition in ('normal_only', 'slugging_only') and 'Air In1' in df.columns and 'Water In1' in df.columns:
        # Classify each row based on TOTAL air and water flow conditions
        # Handle missing Air In2 / Water In2 columns (default to 0)
        regimes = df.apply(
            lambda row: classify_flow_regime(
                row['Air In1'],
                row.get('Air In2', 0) if 'Air In2' in df.columns else 0,
                row['Water In1'],
                row.get('Water In2', 0) if 'Water In2' in df.columns else 0,
            ),
            axis=1
        )

        target = 'normal' if condition == 'normal_only' else 'slugging'
        mask = (regimes == target).values

        # Find contiguous segments to prevent windows from crossing time gaps
        # This is important because filtering creates non-adjacent rows
        segments = []
        in_segment = False
        start = 0
        for i, is_target in enumerate(mask):
            if is_target and not in_segment:
                start = i
                in_segment = True
            elif not is_target and in_segment:
                if i - start >= 10:  # Minimum 10 samples per segment
                    segments.append((start, i))
                in_segment = False
        if in_segment and len(mask) - start >= 10:
            segments.append((start, len(mask)))

        # Create segment_id for each contiguous block
        dfs = []
        for seg_id, (s, e) in enumerate(segments):
            seg = df.iloc[s:e].copy()
            seg['segment_id'] = seg_id
            dfs.append(seg)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()

    # Apply temporal split (maintains time series contiguity)
    n_samples = len(df)
    start_idx = int(n_samples * start_ratio)
    end_idx = int(n_samples * end_ratio)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Add label based on condition
    label = CONDITION_TO_LABEL.get(condition, LABEL_NORMAL)
    df['fault_label'] = label

    return df


def load_consolidated_csv(
    csv_path: str,
    label: int,
    source_test_filter: Optional[str] = None,
    start_ratio: float = 0.0,
    end_ratio: float = 1.0,
    split_mode: str = 'temporal',
) -> pd.DataFrame:
    """
    Load data from a consolidated CSV file.

    Args:
        csv_path: Path to the consolidated CSV file
        label: Fault label to assign
        source_test_filter: If provided, only load data from this test (e.g., 'Test9')
        start_ratio: Start ratio for split (0.0 to 1.0)
        end_ratio: End ratio for split (0.0 to 1.0)
        split_mode: How to apply the split:
            - 'temporal': Simple temporal split on the entire data
            - 'stratified': Apply split ratio to each source_test separately,
                           ensuring balanced representation across train/val/test.
                           Maintains temporal order within each source_test.

    Returns:
        DataFrame with data ready for dataset creation

    Note:
        Data is NEVER shuffled here to preserve temporal order for time series windows.
        Window shuffling happens in the DataLoader during training.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Consolidated CSV not found: {csv_path}")

    # Load CSV (no skiprows needed for consolidated files)
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Filter by source_test if specified
    if source_test_filter is not None and 'source_test' in df.columns:
        df = df[df['source_test'] == source_test_filter].reset_index(drop=True)

    if len(df) == 0:
        return df

    # Apply split based on mode
    if split_mode == 'stratified' and 'source_test' in df.columns:
        # Stratified split: apply ratio to each source_test separately
        # This ensures train/val/test have similar distributions
        dfs = []
        for test_name in df['source_test'].unique():
            test_df = df[df['source_test'] == test_name].reset_index(drop=True)
            n = len(test_df)
            start_idx = int(n * start_ratio)
            end_idx = int(n * end_ratio)
            dfs.append(test_df.iloc[start_idx:end_idx])
        df = pd.concat(dfs, ignore_index=True)
    else:
        # Simple temporal split
        n_samples = len(df)
        start_idx = int(n_samples * start_ratio)
        end_idx = int(n_samples * end_ratio)
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # Ensure TIMESTAMP column exists
    if 'TIMESTAMP' not in df.columns:
        df['TIMESTAMP'] = df.index

    # Add fault_label column
    df['fault_label'] = label

    # Select only required columns (drop 'label' and 'source_test' if present)
    columns_to_keep = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label']
    available = [c for c in columns_to_keep if c in df.columns]
    df = df[available].copy()

    return df


def is_consolidated_format(base_path: str) -> bool:
    """
    Check if base_path contains consolidated CSV files.

    Args:
        base_path: Path to check

    Returns:
        True if consolidated CSV files are found
    """
    # Check for presence of consolidated CSV files
    expected_files = ['normal.csv', 'slugging.csv', 'blockage.csv', 'leakage.csv', 'diverted.csv']
    found_count = sum(1 for f in expected_files if os.path.exists(os.path.join(base_path, f)))
    return found_count >= 3  # At least 3 files present


def load_split_data(
    base_path: str,
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load all data for a given split (train, val, test_*).

    Automatically detects whether to use consolidated CSV format or raw CSV format
    based on the directory structure.

    Args:
        base_path: Base path to PRONTO data directory
        split: Split name (e.g., 'train', 'val', 'test_baseline', 'test_blockage')

    Returns:
        Tuple of (combined DataFrame, metadata dict)

    Raises:
        ValueError: If split name is unknown
    """
    # Check if using consolidated CSV format
    if is_consolidated_format(base_path):
        return load_split_data_consolidated(base_path, split)
    else:
        return load_split_data_raw(base_path, split)


def load_split_data_consolidated(
    base_path: str,
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load data from consolidated CSV files.

    Args:
        base_path: Base path containing consolidated CSV files
        split: Split name

    Returns:
        Tuple of (combined DataFrame, metadata dict)
    """
    if split not in CONSOLIDATED_DATA_SPLITS:
        raise ValueError(f"Unknown split: {split}. Available: {list(CONSOLIDATED_DATA_SPLITS.keys())}")

    split_config = CONSOLIDATED_DATA_SPLITS[split]
    dfs = []
    run_counter = 0
    metadata = {
        'split': split,
        'format': 'consolidated',
        'sources': [],
        'n_runs': 0,
    }

    for entry in split_config:
        # Support both old format (5 elements) and new format (6 elements with split_mode)
        if len(entry) == 6:
            csv_file, label, source_test_filter, start_ratio, end_ratio, split_mode = entry
        else:
            csv_file, label, source_test_filter, start_ratio, end_ratio = entry
            split_mode = 'temporal'

        csv_path = os.path.join(base_path, csv_file)

        try:
            df = load_consolidated_csv(
                csv_path=csv_path,
                label=label,
                source_test_filter=source_test_filter,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
                split_mode=split_mode,
            )

            if len(df) == 0:
                print(f"Warning: No data loaded from {csv_file} (filter={source_test_filter})")
                continue

            # Assign run_id (single run per source)
            df['run_id'] = run_counter
            df['sample_id'] = np.arange(len(df))
            run_counter += 1

            dfs.append(df)
            metadata['sources'].append({
                'csv_file': csv_file,
                'source_test_filter': source_test_filter,
                'label': label,
                'n_samples': len(df),
                'split_mode': split_mode,
            })

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

    if not dfs:
        columns = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label', 'run_id', 'sample_id']
        return pd.DataFrame(columns=columns), metadata

    combined = pd.concat(dfs, ignore_index=True)
    metadata['n_runs'] = run_counter
    metadata['total_samples'] = len(combined)

    return combined, metadata


def load_split_data_raw(
    base_path: str,
    split: str,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load all data for a given split from raw CSV files (original format).

    Args:
        base_path: Base path to PRONTO benchmark data
        split: Split name (e.g., 'train', 'val', 'test_baseline', 'test_blockage')

    Returns:
        Tuple of (combined DataFrame, metadata dict)

    Raises:
        ValueError: If split name is unknown
    """
    if split not in DATA_SPLITS:
        raise ValueError(f"Unknown split: {split}. Available: {list(DATA_SPLITS.keys())}")

    split_config = DATA_SPLITS[split]
    dfs = []
    run_counter = 0
    metadata = {
        'split': split,
        'format': 'raw',
        'sources': [],
        'n_runs': 0,
    }

    for scenario, test, condition, start_ratio, end_ratio in split_config:
        try:
            df = load_scenario_data(
                base_path=base_path,
                scenario=scenario,
                test=test,
                condition=condition,
                start_ratio=start_ratio,
                end_ratio=end_ratio,
            )

            if len(df) == 0:
                print(f"Warning: No data loaded from {scenario}/{test} with condition={condition}")
                continue

            # Handle run_id assignment based on whether segment_id exists
            # segment_id is set when normal_only/slugging_only creates contiguous blocks
            if 'segment_id' in df.columns:
                # Offset segment_ids to create unique run_ids across scenarios
                n_segments = df['segment_id'].nunique()
                df['run_id'] = df['segment_id'] + run_counter
                run_id_start = run_counter
                run_counter += n_segments
            else:
                # Single run for this scenario
                n_segments = 1
                df['run_id'] = run_counter
                run_id_start = run_counter
                run_counter += 1

            df['sample_id'] = np.arange(len(df))

            dfs.append(df)
            metadata['sources'].append({
                'scenario': scenario,
                'test': test,
                'condition': condition,
                'n_samples': len(df),
                'n_segments': n_segments,
                'run_id_start': run_id_start,
            })

        except Exception as e:
            print(f"Error loading {scenario}/{test}: {e}")
            continue

    if not dfs:
        # Return empty DataFrame with correct columns
        columns = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label', 'run_id', 'sample_id']
        return pd.DataFrame(columns=columns), metadata

    combined = pd.concat(dfs, ignore_index=True)
    metadata['n_runs'] = run_counter
    metadata['total_samples'] = len(combined)

    return combined, metadata


def load_all_normal_data(
    base_path: str,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load ALL normal data from consolidated CSV for window-shuffle mode.

    This function loads all normal data without any splitting, intended for use
    with the window-shuffle approach where windows are created first, shuffled,
    and then split into train/val/test.

    Args:
        base_path: Base path to PRONTO data directory

    Returns:
        Tuple of (combined DataFrame with all normal data, metadata dict)
    """
    metadata = {
        'split': 'all_normal',
        'format': 'consolidated',
        'sources': [],
        'n_runs': 0,
    }

    # Check if consolidated format exists
    normal_csv = os.path.join(base_path, 'normal.csv')
    if not os.path.exists(normal_csv):
        # Fall back to raw format - load both Test9 and Test11 normal data
        print(f"  [load_all_normal_data] Consolidated normal.csv not found, using raw format")
        return _load_all_normal_data_raw(base_path)

    # Load all normal data from consolidated CSV
    print(f"  [load_all_normal_data] Loading all normal data from {normal_csv}")
    df = pd.read_csv(normal_csv)
    df.columns = [c.strip() for c in df.columns]

    if len(df) == 0:
        columns = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label', 'run_id', 'sample_id']
        return pd.DataFrame(columns=columns), metadata

    # Ensure TIMESTAMP column exists
    if 'TIMESTAMP' not in df.columns:
        df['TIMESTAMP'] = df.index

    # Add fault_label column (all normal = 0)
    df['fault_label'] = LABEL_NORMAL

    # Create run_id based on source_test if available
    if 'source_test' in df.columns:
        # Assign unique run_id to each source_test
        unique_tests = df['source_test'].unique()
        test_to_run = {test: i for i, test in enumerate(unique_tests)}
        df['run_id'] = df['source_test'].map(test_to_run)
        metadata['n_runs'] = len(unique_tests)
        metadata['sources'] = [
            {'source_test': test, 'n_samples': (df['source_test'] == test).sum()}
            for test in unique_tests
        ]
    else:
        df['run_id'] = 0
        metadata['n_runs'] = 1

    df['sample_id'] = np.arange(len(df))

    # Select only required columns
    columns_to_keep = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label', 'run_id', 'sample_id']
    available = [c for c in columns_to_keep if c in df.columns]
    df = df[available].copy()

    metadata['total_samples'] = len(df)
    print(f"  [load_all_normal_data] Loaded {len(df)} samples from {metadata['n_runs']} runs")

    return df, metadata


def _load_all_normal_data_raw(
    base_path: str,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Load ALL normal data from raw CSV files (fallback when consolidated not available).

    Loads Test9 and Test11 normal data and combines them.
    """
    metadata = {
        'split': 'all_normal',
        'format': 'raw',
        'sources': [],
        'n_runs': 0,
    }

    dfs = []
    run_counter = 0

    # Load Test9 and Test11 normal data
    tests_to_load = [
        ('C0 Normal and Slugging conditions', 'Test9'),
        ('C0 Normal and Slugging conditions', 'Test11'),
    ]

    for scenario, test in tests_to_load:
        try:
            df = load_scenario_data(
                base_path=base_path,
                scenario=scenario,
                test=test,
                condition='normal_only',
                start_ratio=0.0,
                end_ratio=1.0,
            )

            if len(df) == 0:
                print(f"  Warning: No normal data from {scenario}/{test}")
                continue

            # Handle segment_id if present
            if 'segment_id' in df.columns:
                n_segments = df['segment_id'].nunique()
                df['run_id'] = df['segment_id'] + run_counter
                run_counter += n_segments
            else:
                df['run_id'] = run_counter
                run_counter += 1

            df['sample_id'] = np.arange(len(df))
            dfs.append(df)

            metadata['sources'].append({
                'scenario': scenario,
                'test': test,
                'n_samples': len(df),
            })

        except Exception as e:
            print(f"  Error loading {scenario}/{test}: {e}")
            continue

    if not dfs:
        columns = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS + ['fault_label', 'run_id', 'sample_id']
        return pd.DataFrame(columns=columns), metadata

    combined = pd.concat(dfs, ignore_index=True)
    metadata['n_runs'] = run_counter
    metadata['total_samples'] = len(combined)

    print(f"  [load_all_normal_data_raw] Loaded {len(combined)} samples from {run_counter} runs")

    return combined, metadata


def validate_column_order(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has correct column order matching DyEdgeGAT.

    Args:
        df: DataFrame to validate

    Returns:
        True if columns are in correct order

    Raises:
        ValueError: If columns are missing or in wrong order
    """
    expected = ['TIMESTAMP'] + DYEDGEGAT_COLUMNS
    actual = [c for c in df.columns if c in expected]

    if actual != expected[:len(actual)]:
        raise ValueError(
            f"Column order mismatch. Expected: {expected[:len(actual)]}, Got: {actual}"
        )

    # Verify Water Density is NOT present
    if 'Water Density' in df.columns:
        raise ValueError("Water Density column found but should be excluded")
    if 'water tank level' in df.columns:
        raise ValueError("water tank level column found but should be excluded")

    return True


def get_measurement_columns() -> List[str]:
    """Return list of measurement variable names."""
    return MEASUREMENT_COLUMNS.copy()


def get_conditioning_columns() -> List[str]:
    """
    Return list of conditioning variable names.

    Conditioning variables include both control variables (flow rates) and
    external variables (temperatures) as specified in DyEdgeGAT paper.
    """
    return CONDITIONING_COLUMNS.copy()


def get_control_columns() -> List[str]:
    """
    Legacy alias for get_conditioning_columns().

    Returns list of conditioning variable names (control + external variables).
    """
    return CONDITIONING_COLUMNS.copy()


def get_all_columns() -> List[str]:
    """Return list of all DyEdgeGAT variable names in order."""
    return DYEDGEGAT_COLUMNS.copy()


def get_measurement_indices() -> List[int]:
    """Return indices of measurement variables in DYEDGEGAT_COLUMNS."""
    return [DYEDGEGAT_COLUMNS.index(c) for c in MEASUREMENT_COLUMNS]


def get_conditioning_indices() -> List[int]:
    """
    Return indices of conditioning variables in DYEDGEGAT_COLUMNS.

    Conditioning variables include both control variables (flow rates) and
    external variables (temperatures) as specified in DyEdgeGAT paper.
    """
    return [DYEDGEGAT_COLUMNS.index(c) for c in CONDITIONING_COLUMNS]


def get_control_indices() -> List[int]:
    """
    Legacy alias for get_conditioning_indices().

    Returns indices of conditioning variables (control + external variables).
    """
    return get_conditioning_indices()
