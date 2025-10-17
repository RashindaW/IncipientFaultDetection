# Dataset Quick Summary

## 🚨 Action Required

**ONE CRITICAL ISSUE FOUND**: `BaselineTestB.csv` has a duplicate column name at position 159.

### Quick Fix (Run this now):
```bash
cd /mnt/datassd3/rashinda/DyEdge/Dataset
cp BaselineTestB.csv BaselineTestB.csv.backup
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('BaselineTestB.csv')
df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'  # Fix the duplicate
df.to_csv('BaselineTestB.csv', index=False)
print("✅ Fixed!")
EOF
```

---

## ✅ What's Good

| Aspect | Status |
|--------|--------|
| All files have 185 columns | ✅ |
| All CSV files readable | ✅ |
| Baseline data (5 files) | ✅ 406K rows |
| Fault data (6 types) | ✅ 487K rows |
| Data format consistent | ✅ |

---

## 📊 Your Dataset

### Baseline Files (Normal Operation - for Training)
- BaselineTestA.csv: 81,205 rows ✅
- BaselineTestB.csv: 81,342 rows ⚠️ (needs header fix)
- BaselineTestC.csv: 81,275 rows ✅
- BaselineTestD.csv: 81,364 rows ✅
- BaselineTestE.csv: 81,250 rows ✅

### Fault Files (Anomalies - for Testing)
1. **Fault1_DisplayCaseDoorOpen.csv** (81,235 rows)
2. **Fault2_IceAccumulation.csv** (26,396 rows) - shorter duration
3. **Fault3_EvapValveFailure.csv** (81,281 rows)
4. **Fault4_MTEvapFanFailure.csv** (81,239 rows)
5. **Fault5_CondAPBlock.csv** (81,095 rows)
6. **Fault6_MTEvapAPBlock.csv** (81,238 rows)

---

## 📋 Column Types (185 total)

### Measurements:
- **Power (W_)**: 6 columns - Compressor and system power
- **Mass Flow (M-)**: 3 columns - Cooler mass flow rates
- **Pressure (P-)**: 13 columns - System pressures
- **Temperature (T-)**: 157 columns - Sensors throughout system
- **Flow Rate (F-)**: 2 columns - Heat exchanger flows

### Calculated Features:
- Superheat, Subcooling, Heat Transfer
- Capacity (Airside, Refrigerant side)
- Energy metrics (EER, EERA)

### Special:
- **Timestamp**: Date/time of measurement
- **1 blank column**: Placeholder (column 162)

---

## ⚠️ Data Quality Notes

### Large Negative Values
Some columns contain values like `-98509.069` - these are **sentinel values** indicating:
- Sensor disconnected
- Invalid reading
- Missing data

**Affected columns**: T-spare-13B, T-spare-16C, T-Spare-2D, etc.

**Solution**: Replace with NaN during preprocessing:
```python
df[df < -1000] = np.nan
```

---

## 🎯 Next Steps for DyEdgeGAT

### 1. Fix the Column Name Issue ✅ (Do this first!)

### 2. Select Variables
According to the DyEdgeGAT paper (Pronto dataset example):
- Need: **17 measurement variables** (sensors)
- Need: **4 operating condition variables** (control inputs)

**Current dataset has 185 columns** - you need to select which ones to use.

### 3. Recommendations for Variable Selection

#### Option A: Use domain knowledge
Select the 17 most critical sensors for refrigeration system monitoring.

#### Option B: Data-driven selection
Use variance/correlation analysis to select informative variables.

#### Suggested starters (based on refrigeration systems):
**Measurement Variables (17)**:
1. W_MT-COMP1 (MT Compressor 1 power)
2. W_MT-COMP2 (MT Compressor 2 power)
3. W_MT-COMP3 (MT Compressor 3 power)
4. W_LT-COMP1 (LT Compressor 1 power)
5. W_LT-COMP2 (LT Compressor 2 power)
6. P-MTcase-SUC (MT case suction pressure)
7. P-MTcase-LIQ (MT case liquid pressure)
8. P-LTcase-SUC (LT case suction pressure)
9. P-LTcase-LIQ (LT case liquid pressure)
10. P-GC-IN (Gas cooler inlet pressure)
11. T-MT-Suc (MT suction temperature)
12. T-MT-Dis (MT discharge temperature)
13. T-LT-Suc (LT suction temperature)
14. T-LT-Dis (LT discharge temperature)
15. T-GC-In (Gas cooler inlet temp)
16. T-GC-Out (Gas cooler outlet temp)
17. F-MT-BPHX (MT heat exchanger flow)

**Operating Condition Variables (4)**:
- **Option 1**: Use calculated fields (Tsetpt, RHsetpt, etc.)
- **Option 2**: Derive from ambient/external conditions
- **Need**: Domain expert input to identify actual control variables

---

## 📁 Files Created for You

1. **DATASET_CONSISTENCY_REPORT.md** - Detailed analysis
2. **DATASET_QUICK_SUMMARY.md** - This file
3. **column_names.txt** - All 185 column names (one per line)

---

## 🔧 Ready-to-Use Data Loader

```python
import pandas as pd
import numpy as np

def load_and_preprocess(filepath):
    """Load refrigeration data with preprocessing."""
    df = pd.read_csv(filepath)
    
    # Fix BaselineTestB if needed
    if 'BaselineTestB' in filepath:
        df.columns.values[158] = 'T-MT_BPHX_C02_EXIT'
    
    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Handle sentinel values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.where(x > -1000, np.nan))
    
    return df

# Usage
baseline_data = load_and_preprocess('Dataset/BaselineTestA.csv')
fault_data = load_and_preprocess('Dataset/Fault1_DisplayCaseDoorOpen.csv')
```

---

## ✅ Checklist Before Running DyEdgeGAT

- [ ] Fix BaselineTestB.csv column name (run the quick fix above)
- [ ] Select 17 measurement variables
- [ ] Identify 4 operating condition variables  
- [ ] Decide on handling missing/sentinel values
- [ ] Set window size (e.g., 15 timesteps as in paper)
- [ ] Configure model (n_nodes=17, window_size=15, ocvar_dim=4)

---

## 💡 Tips

1. **Start small**: Try with a subset of variables first
2. **Baseline for training**: Use only baseline files for training
3. **All data for testing**: Test on both baseline and fault data
4. **Temporal window**: Paper uses window_size=15, you can experiment
5. **Normalization**: Will be needed - standardize each sensor

---

**Questions?** Check `DATASET_CONSISTENCY_REPORT.md` for detailed information.

**Ready to proceed?** Fix the BaselineTestB issue first, then we can create a data loader for DyEdgeGAT!

