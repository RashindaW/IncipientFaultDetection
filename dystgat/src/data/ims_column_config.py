"""
IMS Bearing Dataset Column Configuration.

The IMS (Intelligent Maintenance Systems) Bearing dataset contains high-frequency
vibration data from 4 bearings running until failure. Since the raw data is 20 kHz
accelerometer signals, we extract statistical and spectral features to create a
time series suitable for graph-based anomaly detection.

Dataset Structure:
- Set 1: 8 channels (2 per bearing: x and y axes), 2156 files, ~35 days
- Set 2: 4 channels (1 per bearing), 984 files, ~7 days
- Set 3: 4 channels (1 per bearing), 4448 files, ~31 days

Failure Modes:
- Set 1: Inner race defect (bearing 3), roller element defect (bearing 4)
- Set 2: Outer race failure (bearing 1)
- Set 3: Outer race failure (bearing 3)

Since rotation speed is constant (2000 RPM), there are no traditional control
variables. We use extracted features as measurement variables (graph nodes).
"""

# Feature names extracted from each bearing's vibration signal
BEARING_FEATURES = [
    "rms",              # Root mean square - overall vibration level
    "peak",             # Maximum absolute value
    "crest_factor",     # Peak / RMS - indicator of impulsiveness
    "kurtosis",         # 4th moment - sensitive to spikes/impacts
    "skewness",         # 3rd moment - asymmetry indicator
    "std",              # Standard deviation
    "peak_to_peak",     # Max - Min range
    "shape_factor",     # RMS / mean(|x|)
]

# Spectral features
SPECTRAL_FEATURES = [
    "spectral_centroid",   # Center of mass of spectrum
    "spectral_spread",     # Spread around centroid
    "spectral_rolloff",    # Frequency below 85% energy
    "spectral_flatness",   # Geometric/arithmetic mean ratio
    "band_power_low",      # 0-2 kHz (bearing defect frequencies)
    "band_power_mid",      # 2-5 kHz (resonance frequencies)
    "band_power_high",     # 5-10 kHz (high-frequency components)
    "dominant_freq",       # Frequency with max power
]

# All features per bearing
ALL_BEARING_FEATURES = BEARING_FEATURES + SPECTRAL_FEATURES

# Number of features per bearing
N_FEATURES_PER_BEARING = len(ALL_BEARING_FEATURES)

# Bearing names for each dataset
BEARINGS_SET1 = ["bearing1_x", "bearing1_y", "bearing2_x", "bearing2_y",
                 "bearing3_x", "bearing3_y", "bearing4_x", "bearing4_y"]
BEARINGS_SET2 = ["bearing1", "bearing2", "bearing3", "bearing4"]
BEARINGS_SET3 = ["bearing1", "bearing2", "bearing3", "bearing4"]

def get_measurement_vars(dataset_set: str = "2nd_test") -> list:
    """
    Get measurement variable names for a given dataset set.

    Each bearing becomes multiple nodes (one per feature), OR
    each bearing is one node with multiple features over time.

    For DySTGAT, we use: each bearing = 1 node, features are the time series values.
    So measurement vars = bearing names.
    """
    if dataset_set == "1st_test":
        # 8 channels (can aggregate x,y or keep separate)
        return ["bearing1", "bearing2", "bearing3", "bearing4"]
    else:
        return ["bearing1", "bearing2", "bearing3", "bearing4"]

def get_feature_names() -> list:
    """Get the list of features extracted per bearing."""
    return ALL_BEARING_FEATURES.copy()

# Measurement variables (graph nodes) - one per bearing
# Each node's time series is the concatenation of extracted features
MEASUREMENT_VARS = ["bearing1", "bearing2", "bearing3", "bearing4"]

# Control variables - none for IMS (constant operating conditions)
# However, we can use normalized time/degradation index as implicit control
CONTROL_VARS = []  # Empty - no control variables

# Alternatively, use time-based control
CONTROL_VARS_WITH_TIME = ["time_index", "normalized_time"]

# Fault labels
FAULT_LABELS = {
    "healthy": 0,
    "degraded": 1,  # Early degradation
    "faulty": 2,    # Near failure
}

# Dataset paths
DATASET_PATHS = {
    "1st_test": "1st_test",
    "2nd_test": "2nd_test",
    "3rd_test": "3rd_test",
}

# Sampling rate
SAMPLE_RATE = 20480  # Hz (20 kHz, 20480 samples per second)
SAMPLES_PER_FILE = 20480  # 1 second of data per file

# Recording interval (time between snapshots)
RECORDING_INTERVAL_MINUTES = 10

# Failure information
FAILURE_INFO = {
    "1st_test": {
        "failed_bearings": [3, 4],
        "failure_types": ["inner_race", "roller_element"],
        "duration_days": 35,
    },
    "2nd_test": {
        "failed_bearings": [1],
        "failure_types": ["outer_race"],
        "duration_days": 7,
    },
    "3rd_test": {
        "failed_bearings": [3],
        "failure_types": ["outer_race"],
        "duration_days": 31,
    },
}
