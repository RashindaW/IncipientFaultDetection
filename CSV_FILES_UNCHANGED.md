# CSV Files Are UNCHANGED (Except One Header Fix)

## Summary

✅ **Your original CSV files are safe and intact!**

All data preprocessing happens **in memory during loading**, not by modifying the CSV files.

---

## What Was Done to CSV Files

### ✅ ONE CSV Header Fix (BaselineTestB.csv)
- **Fixed**: Column 159 duplicate name
  - Before: `T-LT_BPHX_C02_EXIT` (duplicate of column 158)
  - After: `T-MT_BPHX_C02_EXIT` (correct name)
- **Backup created**: `BaselineTestB.csv.backup`
- **No data modified**: Only the header row was changed

### ❌ NO Other CSV Modifications
- All 11 CSV files still have **185 columns**
- All sentinel values (like `-98xxx`) are **still there**
- Flow columns (`F-LT-BPHX`, `F-MT-BPHX`) are **still in the CSV**
- Empty columns (like `Tsetpt`, `RHsetpt`) are **still in the CSV**
- **No data was deleted or modified**

---

## What Happens During Data Loading (In Code)

All data cleaning happens in `dyedgegat/src/data/dataset.py` when loading:

### 1. **Column Selection** (`column_config.py`)
```python
# We only SELECT 28 measurement + 6 control variables
# The CSV still has all 185 columns!

MEASUREMENT_VARS = [
    'W_MT-COMP1', 'W_MT-COMP2', ...  # 28 selected
    # F-LT-BPHX NOT selected (has no valid data)
    # F-MT-BPHX NOT selected (has no valid data)
]

CONTROL_VARS = [
    'M-MTcooler', 'M-LTcooler', ...  # 6 selected
    # Tsetpt NOT selected (all NaN)
    # RHsetpt NOT selected (all NaN)
]
```

### 2. **During Loading** (`dataset.py` lines 66-101):
```python
# Load CSV
df = pd.read_csv(filepath)  # All 185 columns loaded

# SELECT only the columns we want
df = df[ALL_SELECTED_COLUMNS]  # Now only 28+6+1 columns

# Handle sentinel values IN MEMORY
df[col] = df[col].replace(SENTINEL_VALUES, np.nan)
df.loc[df[col] < -1000, col] = np.nan

# Fill NaN values IN MEMORY
df = df.ffill()  # Forward fill
df = df.bfill()  # Backward fill

# Normalize IN MEMORY
df = (df - mean) / std
```

**Key Point**: The CSV file is **read**, then **transformed in memory**. The original file is never modified!

---

## Verification

You can verify this yourself:

### Check CSV files are unchanged:
```bash
cd Dataset

# All files still have 185 columns
for f in *.csv; do 
    echo "$f: $(head -n 1 $f | awk -F',' '{print NF}') columns"
done

# Flow columns are still there
head -n 1 BaselineTestA.csv | tr ',' '\n' | grep -n "F-.*-BPHX"

# Sentinel values are still there
head -n 100 BaselineTestA.csv | grep -o '\-98[0-9]*\.[0-9]*' | head -5
```

### Restore BaselineTestB if needed:
```bash
# If you want the original header back:
cd Dataset
cp BaselineTestB.csv.backup BaselineTestB.csv
```

---

## Why We Don't Modify CSVs

### ✅ Benefits of In-Memory Processing:

1. **Original data preserved** - You can always go back
2. **Reproducibility** - Same results every time
3. **Flexibility** - Easy to change preprocessing without touching data
4. **Safety** - No risk of corrupting original data
5. **Version control** - Code changes are tracked, not data changes

### ❌ Problems with Modifying CSVs:

1. **Data loss** - Hard to undo
2. **Debugging** - Can't compare to original
3. **Collaboration** - Others might need different preprocessing
4. **Irreversible** - Mistakes are permanent

---

## File Status Summary

| File | Original Columns | Modified? | Backup? |
|------|-----------------|-----------|---------|
| BaselineTestA.csv | 185 | ❌ No | N/A |
| **BaselineTestB.csv** | 185 | ✅ Header only | ✅ Yes (.backup) |
| BaselineTestC.csv | 185 | ❌ No | N/A |
| BaselineTestD.csv | 185 | ❌ No | N/A |
| BaselineTestE.csv | 185 | ❌ No | N/A |
| Fault1_DisplayCaseDoorOpen.csv | 185 | ❌ No | N/A |
| Fault2_IceAccumulation.csv | 185 | ❌ No | N/A |
| Fault3_EvapValveFailure.csv | 185 | ❌ No | N/A |
| Fault4_MTEvapFanFailure.csv | 185 | ❌ No | N/A |
| Fault5_CondAPBlock.csv | 185 | ❌ No | N/A |
| Fault6_MTEvapAPBlock.csv | 185 | ❌ No | N/A |

---

## What Actually Changed

### In Code (dyedgegat/src/):

1. **column_config.py**: Defines which 28+6 columns to SELECT
2. **dataset.py**: How to preprocess data IN MEMORY
3. **config.py**: Model configuration (28 nodes, 6 controls)

### In CSV Files:

1. **BaselineTestB.csv**: Header row column 159 name fixed
   - Backup available at `BaselineTestB.csv.backup`

---

## Conclusion

✅ **Your data is safe!**
- All preprocessing is done in Python code
- CSVs are read but not modified (except one header fix with backup)
- You can change preprocessing anytime without touching data
- Original data is always preserved for reproducibility

If you ever want to revert BaselineTestB to the original:
```bash
cd Dataset
cp BaselineTestB.csv.backup BaselineTestB.csv
```

---

**Bottom Line**: We follow best practices by keeping original data unchanged and doing all transformations in code!

