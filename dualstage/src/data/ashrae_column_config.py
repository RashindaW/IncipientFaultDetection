"""
Column configuration for the ASHRAE dataset using a fixed variable split.

System-independent variables (controls + external factors) are treated as control vars.
System-dependent variables are treated as measurement vars.
Active Fault and Unit Status columns are not present in the CSVs.
Time is used as the sequence index.
"""

from typing import List

TIME_COLUMN = "Time"

# System-independent variables: controls + external factors.
CONTROL_VARS: List[str] = [
    # Control Variables (CVs)
    "TWE_set",
    "RLA%",
    "FWC",
    "FWE",
    "VSS",
    "VSL",
    "VH",
    "VM",
    "VC",
    "VE",
    "VW",
    # External Factors (EFs)
    "TEI",
    "TWEI",
    "TCI",
    "TWCI",
    "TWI",
    "THI",
]

# System-dependent variables: measurement nodes.
MEASUREMENT_VARS: List[str] = [
    # Evaporator & chilled water measurements
    "TEO",
    "TWEO",
    "PRE",
    "TRE",
    "TEA",
    "TWED",
    "Evap Tons",
    "Shared Evap Tons",
    "Evap Energy Balance",
    "Building Tons",
    "TBI",
    "TBO",
    # Condenser & cooling water measurements
    "TCO",
    "TWCO",
    "PRC",
    "TRC",
    "TCA",
    "TRC_sub",
    "TWCD",
    "Cond Tons",
    "Cooling Tons",
    "Shared Cond Tons",
    "Cond Energy Balance",
    "TSI",
    "TSO",
    "TWO",
    "THO",
    # Compressor & refrigerant metrics
    "kW",
    "COP",
    "kW/Ton",
    "T_suc",
    "TR_dis",
    "Tsh_suc",
    "Tsh_dis",
    "P_lift",
    "Amps",
    # Lubrication system metrics
    "TO_sump",
    "TO_feed",
    "PO_feed",
    "PO_net",
    # Calculated flow & balance metrics
    "FWW",
    "FWH",
    "FWB",
    "Heat Balance (kW)",
    "Heat Balance%",
    "Tolerance%",
]

LABEL_COLUMNS: List[str] = []


def get_measurement_vars(_feature_option: str | None = None) -> List[str]:
    return list(MEASUREMENT_VARS)


def get_control_vars(_feature_option: str | None = None) -> List[str]:
    return list(CONTROL_VARS)


def get_all_selected_columns(_feature_option: str | None = None) -> List[str]:
    """
    Ordered list of columns to load from each CSV file, including labels.
    Time is always first to simplify downstream processing.
    """
    cols = [TIME_COLUMN] + get_measurement_vars() + get_control_vars() + LABEL_COLUMNS
    # Remove any accidental duplicates while preserving order
    seen = set()
    ordered: List[str] = []
    for col in cols:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


# Baseline filtering configuration
# Disabled because Active Fault and Unit Status columns are removed.
BASELINE_FAULT_CODE_WHITELIST = None
BASELINE_UNIT_STATUS_WHITELIST = None

# Dataset file configuration
BASELINE_FILES = {
    "train": [
        "normal1.csv",
        "normal.csv",
        "normal2.csv",
        "normal r.csv",
        "normal r1.csv",
        "normal dpv.csv",
        "normal cf.csv",
        "normal eo.csv",
    ],
    "val": [
        "normal nc.csv",
        "near normal2.csv",
    ],
}

# Fault files across all fault categories.
# Map: friendly_key -> (subdirectory, filename)
FAULT_FILES = {
    # Condenser fouling (existing default)
    "Condenser_Fouling_06": ("Condenser fouling", "cf6.csv"),
    "Condenser_Fouling_12": ("Condenser fouling", "cf12.csv"),
    "Condenser_Fouling_20": ("Condenser fouling", "cf20.csv"),
    "Condenser_Fouling_30": ("Condenser fouling", "cf30.csv"),
    "Condenser_Fouling_45": ("Condenser fouling", "cf45.csv"),

    # Refrigerant leak
    "Refrigerant_Leak_10": ("Refrigerant leak", "rl10.csv"),
    "Refrigerant_Leak_20": ("Refrigerant leak", "rl20.csv"),
    "Refrigerant_Leak_30": ("Refrigerant leak", "rl30.csv"),
    "Refrigerant_Leak_40": ("Refrigerant leak", "rl40.csv"),
    "Refrigerant_Leak_40_alt": ("Refrigerant leak", "rl40 alt--many unsteady tests.csv"),

    # Refrigerant overcharge
    "Refrigerant_Overcharge_10": ("Refrigerant overcharge", "ro10.csv"),
    "Refrigerant_Overcharge_20": ("Refrigerant overcharge", "ro20.csv"),
    "Refrigerant_Overcharge_30": ("Refrigerant overcharge", "ro30.csv"),
    "Refrigerant_Overcharge_40": ("Refrigerant overcharge", "ro40.csv"),

    # Reduced evaporator water flow
    "Reduced_Evaporator_Water_Flow_10": ("Reduced evaporator water flow", "fwe10.csv"),
    "Reduced_Evaporator_Water_Flow_20": ("Reduced evaporator water flow", "fwe20.csv"),
    "Reduced_Evaporator_Water_Flow_20_alt": (
        "Reduced evaporator water flow",
        "fwe20 alt--unsteady test5.csv",
    ),
    "Reduced_Evaporator_Water_Flow_30": ("Reduced evaporator water flow", "fwe30.csv"),
    "Reduced_Evaporator_Water_Flow_40": ("Reduced evaporator water flow", "fwe40.csv"),
    "Reduced_Evaporator_Water_Flow_40_alt": (
        "Reduced evaporator water flow",
        "fwe40 alt--unsteady test8.csv",
    ),

    # Reduced condenser water flow
    "Reduced_Condenser_Water_Flow_10": ("Reduced condenser water flow", "fwc10.csv"),
    "Reduced_Condenser_Water_Flow_20": ("Reduced condenser water flow", "fwc20.csv"),
    "Reduced_Condenser_Water_Flow_20_alt": ("Reduced condenser water flow", "fwc20 alt.csv"),
    "Reduced_Condenser_Water_Flow_30": ("Reduced condenser water flow", "fwc30.csv"),
    "Reduced_Condenser_Water_Flow_40": ("Reduced condenser water flow", "fwc40.csv"),

    # Non-condensables in refrigerant
    "Non_Condensables_Modified": ("Non-condensables in refrigerant", "modified nc.csv"),
    "Non_Condensables_Aborted": ("Non-condensables in refrigerant", "aborted nc.csv"),
    "Non_Condensables_1": ("Non-condensables in refrigerant", "nc1.csv"),
    "Non_Condensables_2": ("Non-condensables in refrigerant", "nc2.csv"),
    "Non_Condensables_3": ("Non-condensables in refrigerant", "nc3.csv"),
    "Non_Condensables_5": ("Non-condensables in refrigerant", "nc5.csv"),
    "Non_Condensables_Trace": ("Non-condensables in refrigerant", "nc trace.csv"),
    "Non_Condensables_Trace2": ("Non-condensables in refrigerant", "nc trace2.csv"),

    # Excess oil
    "Excess_Oil_14": ("Excess oil", "eo14.csv"),
    "Excess_Oil_32": ("Excess oil", "eo32.csv"),
    "Excess_Oil_50": ("Excess oil", "eo50.csv"),
    "Excess_Oil_68": ("Excess oil", "eo68--unsteady test1.csv"),
    "Excess_Oil_73_alt": ("Excess oil", "eo73 alt--unsteady test1.csv"),
    "Excess_Oil_Aborted": ("Excess oil", "aborted eo86.csv"),

    # Other available faults (optional)
    "Defective_Pilot_Valve": ("Defective Pilot Valve", "defective pilot valve.csv"),
    "Multiple_Faults_FWE20_FWC20": ("Multiple faults", "fwe20fwc20.csv"),
}

# Subdirectories for different data types
BENCHMARK_DIR = "Benchmark Tests"

MEASUREMENT_VARS = get_measurement_vars()
CONTROL_VARS = get_control_vars()
ALL_SELECTED_COLUMNS = get_all_selected_columns()
N_MEASUREMENT_VARS = len(MEASUREMENT_VARS)
N_CONTROL_VARS = len(CONTROL_VARS)
N_TOTAL_COLUMNS = len(ALL_SELECTED_COLUMNS)


def print_config_summary(_feature_option: str | None = None) -> None:
    """Print a summary of the selected variables."""
    mv = get_measurement_vars()
    cv = get_control_vars()
    print("=" * 70)
    print("ASHRAE 1043-RP VARIABLE CONFIGURATION (FIXED SCHEMA)")
    print("=" * 70)
    print(f"Measurement Variables (Sensors): {len(mv)}")
    print(f"Control Variables: {len(cv)}")
    print(f"Label Columns (excluded from inputs): {', '.join(LABEL_COLUMNS) or 'None'}")
    print(f"Total Selected Columns (including Time/labels): {len(get_all_selected_columns())}")
    print("=" * 70)


if __name__ == "__main__":
    print_config_summary()
