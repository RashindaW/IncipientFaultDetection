# Train/Validation Split Update

**Date:** October 24, 2025  
**Action:** Changed train/validation split to use BaselineTestA for validation

---

## Changes Made

### Previous Configuration:
```python
BASELINE_FILES = {
    'train': ['BaselineTestA.csv', 'BaselineTestC.csv', 'BaselineTestD.csv', 'BaselineTestE.csv'],
    'val': ['BaselineTestB.csv'],
}
```

**Statistics:**
- Training: 325,089 rows (80.0%)
- Validation: 81,340 rows (20.0%)

---

### New Configuration:
```python
BASELINE_FILES = {
    'train': ['BaselineTestB.csv', 'BaselineTestC.csv', 'BaselineTestD.csv', 'BaselineTestE.csv'],
    'val': ['BaselineTestA.csv'],
}
```

**Statistics:**
- Training: 325,225 rows (80.0%)
- Validation: 81,204 rows (20.0%)

---

## Rationale

Using BaselineTestA for validation provides:
1. **Held-out validation set** - A has not been used for training
2. **Slightly more training data** - 136 additional rows
3. **Maintains 80/20 split** - Industry standard ratio preserved

---

## Impact

### Files Modified:
- `dyedgegat/src/data/column_config.py` - BASELINE_FILES dictionary updated

### Automatic Propagation:
All components that read from `column_config.py` will automatically use the new split:
- ✅ `dataloader.py` - Uses `BASELINE_FILES['train']` and `BASELINE_FILES['val']`
- ✅ `train_dyedgegat.py` - Calls `create_dataloaders()` which reads config
- ✅ `fast_train_dyedgegat.py` - Same automatic handling
- ✅ `test_dyedgegat_model.py` - Same automatic handling

### No Code Changes Required:
The dataloader implementation already reads dynamically from the config:

```python
# dyedgegat/src/data/dataloader.py
train_dataset = RefrigerationDataset(
    data_files=BASELINE_FILES['train'],  # Automatically uses B,C,D,E now
    ...
)

val_dataset = RefrigerationDataset(
    data_files=BASELINE_FILES['val'],  # Automatically uses A now
    ...
)
```

---

## Verification

Confirmed working:
- ✅ Configuration loads without errors
- ✅ All files exist and are accessible
- ✅ Row counts calculated correctly
- ✅ 80/20 split ratio maintained
- ✅ No linter errors

---

## Next Steps

The updated split is ready to use immediately:

```bash
# Train with new split
python train_dyedgegat.py --epochs 50 --window-size 60 --save-model models/dyedgegat_new_split.pth
```

The model will automatically:
1. Train on BaselineTestB, C, D, E
2. Validate on BaselineTestA
3. Use the same 10 control variables and 142 measurement variables

---

**Status: ✅ COMPLETE - Ready for training with new split**

