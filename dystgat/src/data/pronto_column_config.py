"""
PRONTO Dataset Column Configuration

Column order matches DyEdgeGAT paper specification (17 measurement + 6 control variables).
The .mat files do NOT contain column names - only numeric arrays in this order.

Reference: DyEdgeGAT paper, Table 2 (PRONTO dataset variables)
"""
from typing import List

# ============================================================================
# MEASUREMENT VARIABLES (11 nodes in the graph)
# These are the sensor measurements and valve positions used for anomaly detection
# Per DyEdgeGAT paper: system response variables (not control inputs or external variables)
# ============================================================================
MEASUREMENT_VARS: List[str] = [
    'Air P',           # Air Pressure
    'Mixture zone P',  # Mixture Zone Pressure
    'riser outlet P',  # Riser Outlet Pressure
    'P topsep',        # Top Separator Pressure
    'FR topsep gas',   # Top Separator Gas Flow Rate
    'FR topsep liquid',# Top Separator Liquid Flow Rate
    'P_3phase',        # 3-Phase Separator Pressure
    'Air Valve',       # Air Valve Position (measured output, not control input)
    'Water level',     # Water Level
    'Water coalescer', # Water Coalescer Level
    'Water level valve', # Water Level Valve Position (measured output)
]

# ============================================================================
# CONDITIONING VARIABLES (Operating Conditions - OC)
# These define the operating conditions and are used for OC-aware encoding
# Per DyEdgeGAT paper: "input air flow rate (FT305/302) and input water flow rate
# (FT102/104) as control variables, as well as input air temperature (FT305-T) and
# input water temperature (FT102-T) as external variables influencing the operating conditions"
# ============================================================================
CONDITIONING_VARS: List[str] = [
    'Air In1',         # Air Inlet Flow Rate 1 (FT305) - control variable
    'Air In2',         # Air Inlet Flow Rate 2 (FT302) - control variable
    'Water In1',       # Water Inlet Flow Rate 1 (FT102) - control variable
    'Water In2',       # Water Inlet Flow Rate 2 (FT104) - control variable
    'Air T',           # Air Temperature (FT305-T) - external variable
    'Water T',         # Water Temperature (FT102-T) - external variable
]

# ============================================================================
# ALL 17 DYEDGEGAT COLUMNS (in order as they appear in raw CSV files)
# This is the order used in raw Process Data CSV files
# ============================================================================
DYEDGEGAT_COLUMNS: List[str] = [
    # First 4: Air inputs
    'Air In1',         # Column 0 - Conditioning (control flow rate)
    'Air In2',         # Column 1 - Conditioning (control flow rate)
    'Air T',           # Column 2 - Conditioning (external variable)
    'Air P',           # Column 3 - Measurement
    # Next 3: Water inputs
    'Water In1',       # Column 4 - Conditioning (control flow rate)
    'Water In2',       # Column 5 - Conditioning (control flow rate)
    'Water T',         # Column 6 - Conditioning (external variable)
    # Process measurements
    'Mixture zone P',  # Column 7 - Measurement
    'riser outlet P',  # Column 8 - Measurement
    'P topsep',        # Column 9 - Measurement
    'FR topsep gas',   # Column 10 - Measurement
    'FR topsep liquid',# Column 11 - Measurement
    'P_3phase',        # Column 12 - Measurement
    # Valve positions (measured outputs)
    'Air Valve',       # Column 13 - Measurement
    'Water level',     # Column 14 - Measurement
    'Water coalescer', # Column 15 - Measurement
    'Water level valve', # Column 16 - Measurement
]

# Column indices for quick lookup
MEASUREMENT_INDICES = [3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 11 measurement variables
CONDITIONING_INDICES = [0, 1, 4, 5, 2, 6]  # 6 conditioning variables (4 control + 2 external)

# Legacy aliases for backward compatibility
CONTROL_INDICES = CONDITIONING_INDICES
CONTROL_VARS = CONDITIONING_VARS

# Verify counts
assert len(MEASUREMENT_VARS) == 11, f"Expected 11 measurement vars, got {len(MEASUREMENT_VARS)}"
assert len(CONDITIONING_VARS) == 6, f"Expected 6 conditioning vars, got {len(CONDITIONING_VARS)}"
assert len(DYEDGEGAT_COLUMNS) == 17, f"Expected 17 total columns, got {len(DYEDGEGAT_COLUMNS)}"
assert len(MEASUREMENT_INDICES) == 11
assert len(CONDITIONING_INDICES) == 6

# ============================================================================
# Additional constants for dataset processing
# ============================================================================
ALL_VARS = DYEDGEGAT_COLUMNS.copy()
FAULT_LABEL_COL = 'fault_label'
RUN_COL = 'run_id'
SAMPLE_COL = 'sample_id'

# Label constants
SLUGGING_LABEL = 4
