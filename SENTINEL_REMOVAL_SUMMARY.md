# Sentinel Value Removal Summary

**Date:** October 24, 2025  
**Action:** Removed rows containing sentinel values (missing data markers) from all datasets

---

## What Are Sentinel Values?

Sentinel values are special marker values used by the data acquisition system to indicate missing or invalid sensor readings:

```python
SENTINEL_VALUES = [
    -98509.069,
    -98628.695,
    -98654.277,
    -98631.232,
]

SENTINEL_THRESHOLD = -1000.0  # Any value below this
```

These large negative numbers are physically impossible for refrigeration system temperatures and pressures, making them easy to detect.

---

## Analysis Results

### Files Affected

Only **3 out of 11 files** contained sentinel values:

| File | Type | Rows Before | Rows Removed | Rows After | Impact |
|------|------|-------------|--------------|------------|--------|
| **BaselineTestB.csv** | Baseline | 81,341 | 1 | 81,340 | 0.0012% |
| **BaselineTestE.csv** | Baseline | 81,249 | 1 | 81,248 | 0.0012% |
| **Fault1_DisplayCaseDoorOpen.csv** | Fault | 81,234 | 2 | 81,232 | 0.0025% |

### Overall Statistics

**Baseline Files:**
- Total rows before: 406,431
- Rows removed: 2
- Impact: **0.000492%**

**Fault Files:**
- Total rows before: 432,478
- Rows removed: 2
- Impact: **0.000462%**

**Combined:**
- Total rows processed: 838,909
- Total rows removed: **4**
- Overall impact: **0.000477%**

---

## Why Remove Entire Rows?

Instead of filling/interpolating sentinel values, rows were completely removed because:

1. **Minimal Impact**: Only 4 rows out of 838,909 (0.0005%)
2. **Clean Training**: Ensures model trains on pristine data
3. **No Artifacts**: Avoids introducing interpolation artifacts
4. **Temporal Integrity**: Sliding windows will naturally skip over missing timesteps

---

## Files Modified

### Changed Files (3):
1. `Dataset/BaselineTestB.csv`: 81,341 → 81,340 rows
2. `Dataset/BaselineTestE.csv`: 81,249 → 81,248 rows
3. `Dataset/Fault1_DisplayCaseDoorOpen.csv`: 81,234 → 81,232 rows

### Unchanged Files (8):
- `BaselineTestA.csv` ✅
- `BaselineTestC.csv` ✅
- `BaselineTestD.csv` ✅
- `Fault2_IceAccumulation.csv` ✅
- `Fault3_EvapValveFailure.csv` ✅
- `Fault4_MTEvapFanFailure.csv` ✅
- `Fault5_CondAPBlock.csv` ✅
- `Fault6_MTEvapAPBlock.csv` ✅

---

## Backup Information

**Backup Location:** `Dataset_backup_before_sentinel_removal/`

Original files with sentinel values are preserved in this directory:
- `BaselineTestB.csv` (original)
- `BaselineTestE.csv` (original)
- `Fault1_DisplayCaseDoorOpen.csv` (original)

---

## Verification

All 11 files verified clean after processing:

✅ **Zero sentinel values remaining**  
✅ **Total: 838,905 clean rows**  
✅ **All datasets ready for training**

---

## Impact on Model Training

### Before Cleaning:
- Preprocessing code had to handle sentinel values
- Forward/backward fill could introduce artifacts
- Edge cases at file boundaries

### After Cleaning:
- No sentinel value handling needed
- Clean, continuous data
- Simpler preprocessing pipeline
- More reliable model training

### Expected Benefits:
1. **Faster data loading** (no sentinel detection/filling)
2. **Cleaner gradients** (no interpolation artifacts)
3. **More reliable baselines** (100% authentic data)
4. **Simplified code** (can remove sentinel handling)

---

## Code Updates Needed

The following code in `dataset.py` can now be **simplified** (sentinel handling is redundant):

```python
# This section can be removed or commented out:
# Handle sentinel values (missing data indicators)
for col in MEASUREMENT_VARS + CONTROL_VARS:
    # Replace known sentinel values
    df[col] = df[col].replace(SENTINEL_VALUES, np.nan)
    # Replace any remaining large negative values
    df.loc[df[col] < SENTINEL_THRESHOLD, col] = np.nan
```

However, keeping this code won't hurt - it will simply do nothing since no sentinel values remain.

---

## Rollback Instructions

If you need to restore the original files:

```bash
# Copy backup files back to Dataset directory
cp Dataset_backup_before_sentinel_removal/BaselineTestB.csv Dataset/
cp Dataset_backup_before_sentinel_removal/BaselineTestE.csv Dataset/
cp Dataset_backup_before_sentinel_removal/Fault1_DisplayCaseDoorOpen.csv Dataset/
```

---

## Summary

✅ **Minimal data loss** (0.0005% of rows)  
✅ **All files now 100% clean**  
✅ **Backups safely stored**  
✅ **Ready for model training**  
✅ **No code changes required** (existing preprocessing still works)

The datasets are now in optimal condition for training the DyEdgeGAT model with no missing data markers or artifacts.

---

**Status: ✅ COMPLETED - All datasets cleaned and verified**

