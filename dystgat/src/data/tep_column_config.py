"""
Column configuration for the Tennessee Eastman Process (TEP) dataset.

The RData files expose 41 process measurements (xmeas_*), 11 manipulated
variables (xmv_*), and three metadata columns: faultNumber, simulationRun,
and sample.
"""

# Measurement variables (nodes of the graph)
MEASUREMENT_VARS = [f"xmeas_{i}" for i in range(1, 42)]  # 1..41

# Control / actuator variables (treated as operating-condition channels)
CONTROL_VARS = [f"xmv_{i}" for i in range(1, 12)]  # 1..11

# Metadata columns provided in the RData files
FAULT_LABEL_COL = "faultNumber"
RUN_COL = "simulationRun"
SAMPLE_COL = "sample"

# Combined columns we expect to read (additional "Timestamp" is added later)
ALL_DATA_COLUMNS = MEASUREMENT_VARS + CONTROL_VARS + [FAULT_LABEL_COL, RUN_COL, SAMPLE_COL]

# Distinct fault labels present in the faulty testing set
FAULT_LABELS = list(range(1, 21))  # 1..20

# Fault injection points (1-indexed sample number).
# Before this sample the process runs under normal conditions even in faulty runs.
# Rieth et al. (2017): training onset at sample 21, testing onset at sample 161.
FAULT_ONSET_SAMPLE_TRAIN = 21
FAULT_ONSET_SAMPLE_TEST = 161

# File names inside data/tep/raw
FAULT_FREE_TRAIN_FILE = "TEP_FaultFree_Training.RData"
FAULT_FREE_TEST_FILE = "TEP_FaultFree_Testing.RData"
FAULTY_TRAIN_FILE = "TEP_Faulty_Training.RData"
FAULTY_TEST_FILE = "TEP_Faulty_Testing.RData"

N_MEASUREMENT_VARS = len(MEASUREMENT_VARS)
N_CONTROL_VARS = len(CONTROL_VARS)

