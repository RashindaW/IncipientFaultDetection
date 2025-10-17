# Dataset Consistency Report

**Date**: Generated automatically  
**Dataset Location**: `/mnt/datassd3/rashinda/DyEdge/Dataset/`

---

## Executive Summary

✅ **Overall Status**: Dataset is mostly consistent with ONE critical column naming issue  
⚠️ **Issue Found**: BaselineTestB.csv has a duplicate column name  
✅ **All files have**: 185 columns  
✅ **Total files**: 11 (5 baseline, 6 fault)

---

## File Inventory

### Baseline Files (Healthy Data)
| File | Rows | Columns | Status |
|------|------|---------|--------|
| BaselineTestA.csv | 81,205 | 185 | ✅ OK |
| BaselineTestB.csv | 81,342 | 185 | ⚠️ Column naming issue |
| BaselineTestC.csv | 81,275 | 185 | ✅ OK |
| BaselineTestD.csv | 81,364 | 185 | ✅ OK |
| BaselineTestE.csv | 81,250 | 185 | ✅ OK |

### Fault Files
| File | Fault Type | Rows | Columns | Status |
|------|-----------|------|---------|--------|
| Fault1_DisplayCaseDoorOpen.csv | Display Case Door Open | 81,235 | 185 | ✅ OK |
| Fault2_IceAccumulation.csv | Ice Accumulation | 26,396 | 185 | ✅ OK |
| Fault3_EvapValveFailure.csv | Evaporator Valve Failure | 81,281 | 185 | ✅ OK |
| Fault4_MTEvapFanFailure.csv | MT Evaporator Fan Failure | 81,239 | 185 | ✅ OK |
| Fault5_CondAPBlock.csv | Condenser Air Path Blockage | 81,095 | 185 | ✅ OK |
| Fault6_MTEvapAPBlock.csv | MT Evap Air Path Blockage | 81,238 | 185 | ✅ OK |

**Note**: Fault2_IceAccumulation.csv has significantly fewer rows (~26K vs ~81K for others). This appears intentional and may represent a shorter experiment duration.

---

## 🔴 Critical Issue: Column Name Inconsistency

### Problem
**File**: `BaselineTestB.csv`  
**Column 159**: Should be `T-MT_BPHX_C02_EXIT` but is actually `T-LT_BPHX_C02_EXIT` (duplicate of column 158)

### Details
```
BaselineTestA.csv (CORRECT):
  Column 158: T-LT_BPHX_C02_EXIT
  Column 159: T-MT_BPHX_C02_EXIT  ✅

BaselineTestB.csv (INCORRECT):
  Column 158: T-LT_BPHX_C02_EXIT
  Column 159: T-LT_BPHX_C02_EXIT  ❌ DUPLICATE!
```

### Impact
- **Data loading**: May cause confusion when loading data
- **Feature mapping**: Column 159 has duplicate name, affecting feature identification
- **Model training**: Could lead to incorrect feature learning if not handled

### Recommendation
**Option 1** (Recommended): Fix the header in BaselineTestB.csv
```bash
# Backup first
cp BaselineTestB.csv BaselineTestB.csv.backup

# Fix the header (replace duplicate with correct name)
sed -i '1s/T-LT_BPHX_C02_EXIT,F-LT-BPHX/T-MT_BPHX_C02_EXIT,F-LT-BPHX/' BaselineTestB.csv
```

**Option 2**: Exclude BaselineTestB.csv from analysis if unsure about data validity

---

## Column Structure

### Total Columns: 185

#### Column Categories:
1. **Timestamp**: 1 column
2. **Power/Energy (W, M)**: 12 columns
   - Examples: W_MT-COMP1, W_MT-COMP2, M-MTcooler, etc.
3. **Pressure (P)**: 13 columns
   - Examples: P-LT-BPHX, P-MT-BPHX, P-MTcase-SUC, etc.
4. **Temperature (T)**: 140+ columns
   - T-101 through T-516 (case/sensor temperatures)
   - Named temperatures: T-MT-COMP1-SUC, T-GC-In, etc.
5. **Flow (F)**: 2 columns
   - F-LT-BPHX, F-MT-BPHX
6. **Calculated/Derived**: ~17 columns
   - SupHCompSuc, SupHCompDisc, SubClComCond, etc.
   - CapaAirside, CapaRefrside, EnergyBalance, EERA, EER

### Empty Column
- **Column 162**: Appears to be intentionally blank (placeholder)

---

## Data Quality Checks

### ✅ Passed Checks:
- All files have exactly 185 columns
- No empty/missing column headers (except one intentional blank column)
- All files are in CSV format with comma delimiters
- Data appears to be numeric (spot-checked)
- Timestamps follow consistent format: `MM/DD/YYYY HH:MM:SS`

### ⚠️ Observations:
1. **Large negative values**: Some columns contain values like `-98509.069`, `-98628.695`
   - These appear in columns: T-spare-13B, T-spare-16C, T-Spare-2D, etc.
   - **Likely**: Sentinel values indicating sensor disconnection or invalid readings
   - **Action**: Should be handled as NaN/missing data during preprocessing

2. **Variable row counts**: Different files have different lengths
   - **Reason**: Different experiment durations or data collection periods
   - **Not an issue**: This is normal for real-world experiments

---

## Recommendations for Data Preprocessing

### 1. **Fix Column Name Issue** (CRITICAL)
```python
# When loading BaselineTestB.csv, rename column 159
import pandas as pd

df = pd.read_csv('BaselineTestB.csv')
df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'  # Fix column 159 (0-indexed 158)
```

### 2. **Handle Sentinel Values**
```python
# Replace large negative values with NaN
df = df.replace({-98509.069: np.nan, -98628.695: np.nan, -98654.277: np.nan, -98631.232: np.nan})

# Or use a threshold
df[df < -1000] = np.nan
```

### 3. **Column Selection** for DyEdgeGAT
According to the paper (Pronto dataset), they used:
- **17 process variables** (measurement sensors)
- **4 operating condition variables** (control inputs)

**Suggested columns for process variables** (based on refrigeration system):
- Power: W_MT-COMP1, W_MT-COMP2, W_MT-COMP3, W_LT-COMP1, W_LT-COMP2
- Pressure: P-MTcase-SUC, P-MTcase-LIQ, P-LTcase-SUC, P-LTcase-LIQ, P-GC-IN
- Temperature: T-MT-Suc, T-MT-Dis, T-LT-Suc, T-LT-Dis, T-GC-In, T-GC-Out
- Flow: F-LT-BPHX, F-MT-BPHX

**Suggested operating condition variables** (control/external factors):
- Derived metrics that indicate operating state
- Or: ambient conditions, setpoints, etc.
- Need domain expert input to identify actual control variables

### 4. **Data Loading Template**
```python
import pandas as pd
import numpy as np

def load_refrigeration_data(filepath, fix_baseline_b=False):
    """Load and preprocess refrigeration dataset."""
    df = pd.read_csv(filepath)
    
    # Fix BaselineTestB column name issue
    if fix_baseline_b and 'BaselineTestB' in filepath:
        df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Replace sentinel values with NaN
    df = df.replace({v: np.nan for v in [-98509.069, -98628.695, -98654.277, -98631.232]})
    
    # Handle remaining large negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.where(x > -1000, np.nan))
    
    return df
```

---

## Next Steps

### Immediate Actions:
1. ✅ **Fix BaselineTestB.csv** column name issue
2. ⚠️ **Identify control variables** - Need domain expertise to select appropriate operating condition variables
3. ⚠️ **Handle missing data** - Decide on imputation strategy for sentinel values

### For DyEdgeGAT Implementation:
1. **Select 17 measurement variables** (sensors to monitor)
2. **Select 4 operating condition variables** (control inputs / external factors)
3. **Define normal vs fault conditions**:
   - Baseline files = normal operation
   - Fault files = anomalous operation
4. **Create train/test split**:
   - Train on baseline data only
   - Test on both baseline and fault data

---

## Summary Checklist

- [x] All files have same number of columns (185)
- [ ] **CRITICAL**: Fix BaselineTestB column name duplicate
- [x] All files readable and in CSV format
- [x] Baseline data available (5 files, ~81K rows each)
- [x] Fault data available (6 types, varying rows)
- [ ] Identify and select measurement variables (17 recommended)
- [ ] Identify and select control variables (4 recommended)
- [ ] Decide on handling strategy for sentinel values
- [ ] Create data loading pipeline

---

## Files Summary

```
Total dataset size: ~893K rows across 11 files
Baseline data: ~406K rows (5 files)
Fault data: ~487K rows (6 files)
Columns per file: 185
Measurement types: Power (W), Mass flow (M), Pressure (P), Temperature (T), Flow (F)
```

**Status**: ✅ Dataset is usable with minor preprocessing required

---

## Quick Fix Script

Run this to fix the BaselineTestB.csv issue:

```bash
cd /mnt/datassd3/rashinda/DyEdge/Dataset

# Backup
cp BaselineTestB.csv BaselineTestB.csv.backup

# Fix the header using Python
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('BaselineTestB.csv')
df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'
df.to_csv('BaselineTestB.csv', index=False)
print("✅ BaselineTestB.csv header fixed!")
EOF
```


