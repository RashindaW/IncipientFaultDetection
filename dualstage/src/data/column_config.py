"""
Column configuration for the cleaned CO₂ refrigeration system dataset.

This module defines which columns act as measurement variables (fault-prone sensors)
and which serve as operating condition variables (control/external factors).
The current dataset keeps 142 measurement channels and 10 operating condition
variables (control inputs + external factors) to provide operating context.
"""

# ============================================================================
# MEASUREMENT VARIABLES (System-dependent sensors that can fault)
# These form the nodes in the temporal graph for DualSTAGE
# ============================================================================

MEASUREMENT_VARS = [
    'W_MT-COMP1',
    'W_MT-COMP2',
    'W_MT-COMP3',
    'W_LT-COMP1',
    'W_LT-COMP2',
    'W-CONDENSOR',
    'M-MTcooler',
    'M-LTcooler',
    'M-CompRack',
    'W-LT-BPHX',
    'W-MT-BPHX',
    'P-LT-BPHX',
    'P-MT-BPHX',
    'P-MTcase-SUC-inside',
    'P-MTcase-SUC',
    'P-MTcase-LIQ',
    'P-LTcase-SUC-inside',
    'P-LTcase-SUC',
    'P-LTcase-LIQ',
    'P-FlashTank',
    'P-GC-IN',
    'P-MT_Dis-OilSepIn',
    'P-LT-SUC',
    'P-MT_SUC',
    'T-MT-COMP1-SUC',
    'T-MT-COMP1-DIS',
    'T-MT-COMP2-SUC',
    'T-MT-COMP2-DIS',
    'T-MT-COMP3-SUC',
    'T-MT-COMP3-DIS',
    'T-LT-COMP1-SUC',
    'T-LT-COMP1-DIS',
    'T-LT-COMP2-SUC',
    'T-LT-COMP2-DIS',
    'T-GC-SUC',
    'T-GC-DIS',
    'T-BP-EEVout',
    'T-BP-srf',
    'T-MTrack-LIQ',
    'T-MTcase-LIQ',
    'T-LTcase-LIQ-srf',
    'T-LTcase-SUC-srf',
    'T-LTcase-EEVout',
    'T-LTcase-Sup',
    'T-LTcase-Ret',
    'T-LTcase-Liq',
    'T-LTcase-Suc',
    'T-GC-Fan2-Out',
    'T-GC-Fan1-Out',
    'T-MTRack-Suc-srf',
    'T-LT-Ret',
    'T-LT-Suc',
    'T-LT-Dis',
    'T-MT-Suc',
    'T-MT-Dis',
    'T-MTCase-Liq-Srf',
    'T-MTCase-Suc-Srf',
    'T-MTCase-EEVOut',
    'T-MTCase-Suc',
    'T-101',
    'T-102',
    'T-103',
    'T-104',
    'T-105',
    'T-106',
    'T-107',
    'T-108',
    'T-109',
    'T-110',
    'T-111',
    'T-112',
    'T-113',
    'T-114',
    'T-115',
    'T-116',
    'T-201',
    'T-202',
    'T-203',
    'T-204',
    'T-205',
    'T-206',
    'T-207',
    'T-208',
    'T-209',
    'T-210',
    'T-211',
    'T-212',
    'T-213',
    'T-214',
    'T-215',
    'T-216',
    'T-301',
    'T-302',
    'T-303',
    'T-304',
    'T-305',
    'T-306',
    'T-307',
    'T-308',
    'T-309',
    'T-310',
    'T-311',
    'T-312',
    'T-313',
    'T-314',
    'T-315',
    'T-316',
    'T-401',
    'T-402',
    'T-403',
    'T-404',
    'T-405',
    'T-406',
    'T-407',
    'T-408',
    'T-409',
    'T-410',
    'T-411',
    'T-412',
    'T-413',
    'T-414',
    'T-415',
    'T-416',
    'T-501',
    'T-502',
    'T-503',
    'T-504',
    'T-505',
    'T-506',
    'T-507',
    'T-508',
    'T-509',
    'T-510',
    'T-511',
    'T-512',
    'T-513',
    'T-514',
    'T-515',
    'T-516',
    'T-MT_BPHX_H20_OUTLET',
    'T-LT_BPHX_C02_EXIT',
    'T-MT_BPHX_C02_EXIT',
]

# Total measurement variables in cleaned dataset
assert len(MEASUREMENT_VARS) == 142, f"Expected 142 measurement vars, got {len(MEASUREMENT_VARS)}"


# ============================================================================
# OPERATING CONDITION VARIABLES (Control/External factors)
# These provide context about the operating state but don't fault themselves
# ============================================================================

CONTROL_VARS = [
    # Ambient conditions
    'T-GC-In',
    'T-GC-Fan2-In',
    # External cooling (water circuit)
    'T-LT_BPHX_H20_INLET',
    'T-MT_BPHX_H20_INLET',
    'T-LT_BPHX_H20_OUTLET',
    # System state indicators
    'SupHCompSuc',
    'SupHCompDisc',
    # Operating setpoints and load indicators
    'T-MTCase-Sup',
    'T-MTCase-Ret',
    # Test/experimental conditions
    'T-FalseLoad',
]

# Total operating condition variables supplying context to the model
assert len(CONTROL_VARS) == 10, f"Expected 10 control vars, got {len(CONTROL_VARS)}"


# ============================================================================
# COMBINED CONFIGURATION
# ============================================================================

ALL_SELECTED_COLUMNS = ['Timestamp'] + MEASUREMENT_VARS + CONTROL_VARS

# Metadata
N_MEASUREMENT_VARS = len(MEASUREMENT_VARS)
N_CONTROL_VARS = len(CONTROL_VARS)
N_TOTAL_COLUMNS = len(ALL_SELECTED_COLUMNS)

# Print configuration summary
def print_config_summary():
    """Print a summary of the selected variables."""
    power_mass = sum(var.startswith(('W_', 'M-')) for var in MEASUREMENT_VARS)
    pressure = sum(var.startswith('P-') for var in MEASUREMENT_VARS)
    temperature = sum(var.startswith('T-') for var in MEASUREMENT_VARS)
    flow = sum(var.startswith('F-') for var in MEASUREMENT_VARS)
    derived_other = N_MEASUREMENT_VARS - (power_mass + pressure + temperature + flow)

    print("=" * 70)
    print("REFRIGERATION SYSTEM - VARIABLE CONFIGURATION")
    print("=" * 70)
    print(f"Measurement Variables (Sensors): {N_MEASUREMENT_VARS}")
    print(f"  - Power / Mass-Flow: {power_mass}")
    print(f"  - Pressure:          {pressure}")
    print(f"  - Temperature:       {temperature}")
    print(f"  - Flow:              {flow}")
    print(f"  - Derived metrics:   {derived_other}")
    print()
    print(f"Operating Condition Variables: {N_CONTROL_VARS}")
    print(f"  - Set-points / thermostats: {N_CONTROL_VARS}")
    print()
    print(f"Total Selected Columns: {N_TOTAL_COLUMNS} (including Timestamp)")
    print("=" * 70)


# ============================================================================
# SENTINEL VALUES (Invalid/Missing Data Indicators)
# ============================================================================

# Large negative values in the dataset that indicate missing/invalid readings
SENTINEL_VALUES = [
    -98509.069,
    -98628.695,
    -98654.277,
    -98631.232,
]

# Threshold for detecting additional sentinel values
SENTINEL_THRESHOLD = -1000.0  # Any value below this is likely invalid


# ============================================================================
# DATASET FILE CONFIGURATION
# ============================================================================

BASELINE_FILES = {
    'train': [
        'BaselineTestB.csv',
        'BaselineTestC.csv',
        'BaselineTestD.csv',
        'BaselineTestE.csv',
    ],
    'val': [
        'BaselineTestA.csv',
    ],
}

FAULT_FILES = {
    'Fault1_DisplayCaseDoorOpen': 'Fault1_DisplayCaseDoorOpen.csv',
    'Fault2_IceAccumulation': 'Fault2_IceAccumulation.csv',
    'Fault3_EvapValveFailure': 'Fault3_EvapValveFailure.csv',
    'Fault4_MTEvapFanFailure': 'Fault4_MTEvapFanFailure.csv',
    'Fault5_CondAPBlock': 'Fault5_CondAPBlock.csv',
    'Fault6_MTEvapAPBlock': 'Fault6_MTEvapAPBlock.csv',
}


if __name__ == "__main__":
    # Print configuration when run as script
    print_config_summary()
    
    print("\nMeasurement Variables:")
    for i, var in enumerate(MEASUREMENT_VARS, 1):
        print(f"  {i:2d}. {var}")
    
    print("\nControl Variables:")
    for i, var in enumerate(CONTROL_VARS, 1):
        print(f"  {i}. {var}")
