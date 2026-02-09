"""
IMS Bearing Raw Accelerometer Dataset Configuration.

This configuration is for loading raw 20 kHz accelerometer signals from the IMS
Bearing dataset, as opposed to pre-computed features. Raw signals are suitable
for spectral analysis using DySTGAT's spectral encoder, which can learn frequency
patterns directly from the time-domain data via FFT.

Dataset Structure:
- Set 1: 8 channels (2 per bearing: x and y axes), 2156 files, ~35 days
- Set 2: 4 channels (1 per bearing), 984 files, ~7 days
- Set 3: 4 channels (1 per bearing), 4448 files, ~31 days

Each file contains 20,480 samples (1 second at 20 kHz sampling rate).
Files are recorded every 10 minutes.

Failure Modes:
- Set 1: Inner race defect (bearing 3), roller element defect (bearing 4)
- Set 2: Outer race failure (bearing 1)
- Set 3: Outer race failure (bearing 3)
"""

# Sampling characteristics
SAMPLE_RATE = 20480  # Hz (approximately 20 kHz)
SAMPLES_PER_FILE = 20480  # 1 second of data per file
RECORDING_INTERVAL_MINUTES = 10  # Time between consecutive files

# Channel configuration per dataset set
CHANNEL_CONFIG = {
    "1st_test": {
        "n_channels": 8,
        "n_bearings": 4,
        "layout": "xy",  # 2 channels per bearing (x, y axes)
    },
    "2nd_test": {
        "n_channels": 4,
        "n_bearings": 4,
        "layout": "single",  # 1 channel per bearing
    },
    "3rd_test": {
        "n_channels": 4,
        "n_bearings": 4,
        "layout": "single",  # 1 channel per bearing
    },
}

# Measurement variables - each bearing is a node in the graph
MEASUREMENT_VARS = ["bearing1", "bearing2", "bearing3", "bearing4"]

# Control variables - none for raw data (constant operating conditions at 2000 RPM)
# Optional time-based controls can be added during dataset creation
CONTROL_VARS = []

# Fault labels for degradation state
FAULT_LABELS = {
    "healthy": 0,
    "degraded": 1,
    "faulty": 2,
}

# Failure information per dataset set
FAILURE_INFO = {
    "1st_test": {
        "failed_bearings": [3, 4],
        "failure_types": ["inner_race", "roller_element"],
        "duration_days": 35,
        "n_files": 2156,
    },
    "2nd_test": {
        "failed_bearings": [1],
        "failure_types": ["outer_race"],
        "duration_days": 7,
        "n_files": 984,
    },
    "3rd_test": {
        "failed_bearings": [3],
        "failure_types": ["outer_race"],
        "duration_days": 31,
        "n_files": 4448,
    },
}

# Bearing defect frequencies at 2000 RPM (approximate)
# These are the characteristic frequencies for defect detection
BEARING_DEFECT_FREQUENCIES = {
    "bpfo": 107.0,  # Ball Pass Frequency Outer Race
    "bpfi": 162.0,  # Ball Pass Frequency Inner Race
    "bsf": 70.0,   # Ball Spin Frequency
    "ftf": 11.6,   # Fundamental Train Frequency (cage)
}

# Default windowing parameters for raw signal processing
DEFAULT_WINDOW_CONFIG = {
    "sample_window": 1024,   # Samples per window within file (~50ms at 20kHz)
    "sample_stride": 512,    # 50% overlap between windows
    "file_window": 10,       # Number of consecutive files per sample
    "file_stride": 5,        # Stride across files
}


def get_measurement_vars(dataset_set: str = "2nd_test") -> list:
    """Get measurement variable names (bearing names)."""
    return MEASUREMENT_VARS.copy()


def get_control_vars(use_time: bool = False) -> list:
    """Get control variable names."""
    if use_time:
        return ["time_index", "normalized_time"]
    return []


def get_channel_count(dataset_set: str) -> int:
    """Get the number of channels for a dataset set."""
    return CHANNEL_CONFIG.get(dataset_set, CHANNEL_CONFIG["2nd_test"])["n_channels"]


def get_bearing_count(dataset_set: str = "2nd_test") -> int:
    """Get the number of bearings (always 4)."""
    return 4
