# Quick Start Guide - Fault Detection Testing

## 🚀 Three Simple Steps

### Step 1: Test All Faults (Run This First)
```bash
python run_all_fault_tests.py
```

**What it does:**
- Tests all 6 fault datasets automatically
- Generates anomaly scores for each timestep
- Creates visualizations (heatmaps + line plots)
- Saves detailed CSV files

**Expected runtime:** 5-15 minutes (depending on hardware)

**Output location:** `outputs/anomaly_scores/`

---

### Step 2: View the Results

#### Look at Line Plots (Best for Incipient Faults!)
```bash
# View line plots to see fault progression over time
ls outputs/anomaly_scores/*_line.png

# Example files:
# - Fault1_DisplayCaseDoorOpen_line.png
# - Fault2_IceAccumulation_line.png
# - Fault3_EvapValveFailure_line.png
# - etc.
```

#### Look at Heatmaps (Pattern Overview)
```bash
ls outputs/anomaly_scores/*_heatmap.png
```

---

### Step 3: Analyze Results Quantitatively
```bash
python analyze_fault_results.py
```

**What it does:**
- Computes statistics for each fault (mean, max, std, percentiles)
- Finds first detection time for each fault
- Ranks faults by detectability
- Creates comparative visualizations

**Outputs:**
- `summary_statistics.csv` - Overall stats per fault
- `detection_times.csv` - When each fault was first detected
- `comparative_analysis.png` - Side-by-side comparison plots

---

## 📊 Understanding the Results

### Anomaly Score Interpretation

| Score Range | Meaning | Action |
|-------------|---------|--------|
| 0.0 - 1.5 | Normal operation | No concern |
| 1.5 - 2.5 | Slight deviation | Monitor closely |
| 2.5 - 4.0 | Incipient fault | Early warning! |
| 4.0+ | Fault present | Take action |

*(These ranges may vary - calibrate based on your baseline tests)*

---

## 🔍 Key Files to Review

### For Visual Analysis:
1. `*_line.png` - Shows temporal progression (BEST for incipient faults)
2. `*_heatmap.png` - Shows patterns across multiple windows

### For Data Analysis:
1. `*_anomaly_scores.csv` - Detailed per-timestep scores
2. `*_timestamp_mean.csv` - Averaged scores over time
3. `summary_statistics.csv` - Comparative statistics
4. `detection_times.csv` - First detection timestamps

---

## 💡 What to Look For (Incipient Faults)

### In Line Plots:
- ✓ **Gradual upward trend** - Early fault development
- ✓ **Increasing variability** - System becoming unstable
- ✓ **Small but consistent elevation** - Deviation from baseline

### In CSV Files:
```python
import pandas as pd

# Load results
df = pd.read_csv('outputs/anomaly_scores/Fault1_DisplayCaseDoorOpen_anomaly_scores.csv')

# Find when scores exceed threshold
threshold = 2.0
early_detections = df[df['anomaly_score'] > threshold]

if len(early_detections) > 0:
    first = early_detections.iloc[0]
    print(f"Fault detected at: {first['timestamp']}")
    print(f"Score: {first['anomaly_score']:.2f}")
```

---

## 🛠️ Troubleshooting

**Problem:** Script fails with "checkpoint not found"
```bash
# Check if checkpoint exists
ls -lh checkpoints/
```

**Problem:** Out of memory
```bash
# Edit run_all_fault_tests.py and reduce BATCH_SIZE
# Change: BATCH_SIZE = 64
# To:     BATCH_SIZE = 16
```

**Problem:** Want to test just one fault
```bash
python plot_anomaly_scores.py \
    --checkpoint checkpoints/dyedgegat_stride10.pt \
    --dataset Fault1_DisplayCaseDoorOpen \
    --stride 1 \
    --line-plot
```

---

## 📚 More Information

- **Full guide:** See `FAULT_TESTING_GUIDE.md`
- **Script details:** Run `python plot_anomaly_scores.py --help`
- **Code:** Review `plot_anomaly_scores.py` for implementation

---

## ✅ Expected Workflow

1. ✅ Run `python run_all_fault_tests.py` → Generate all results
2. ✅ Review line plots → Visual inspection of fault progression
3. ✅ Run `python analyze_fault_results.py` → Get statistics
4. ✅ Examine CSV files → Detailed analysis
5. ✅ Compare across faults → Determine best detection capabilities

---

## 🎯 Your 6 Fault Datasets

1. **Fault1_DisplayCaseDoorOpen** - Display case door left open
2. **Fault2_IceAccumulation** - Ice buildup on heat exchanger
3. **Fault3_EvapValveFailure** - Evaporator valve malfunction
4. **Fault4_MTEvapFanFailure** - MT evaporator fan failure
5. **Fault5_CondAPBlock** - Condenser air path blockage
6. **Fault6_MTEvapAPBlock** - MT evaporator air path blockage

Each represents a different failure mode in the refrigeration system.

---

**Ready to start? Run:**
```bash
python run_all_fault_tests.py
```

