# Fault Detection Testing Guide

## Overview
This guide explains how to test your trained DyEdgeGAT model on the 6 fault datasets and visualize anomaly scores to detect incipient faults.

## Available Fault Datasets (6 Total)

1. **Fault1_DisplayCaseDoorOpen** - Display case door left open
2. **Fault2_IceAccumulation** - Ice buildup on heat exchanger
3. **Fault3_EvapValveFailure** - Evaporator valve malfunction
4. **Fault4_MTEvapFanFailure** - Medium temperature evaporator fan failure
5. **Fault5_CondAPBlock** - Condenser air path blockage
6. **Fault6_MTEvapAPBlock** - Medium temperature evaporator air path blockage

## Quick Start - Test All Faults at Once

### Option 1: Using Python Script (Recommended)
```bash
python run_all_fault_tests.py
```

### Option 2: Using Bash Script
```bash
bash run_all_fault_tests.sh
```

This will automatically:
- Test all 6 fault datasets
- Generate anomaly scores for each timestep
- Create heatmaps and line plots
- Save CSV files with detailed results

## Manual Testing - Single Fault Dataset

### Basic Command
```bash
python plot_anomaly_scores.py \
    --checkpoint checkpoints/dyedgegat_stride10.pt \
    --dataset Fault1_DisplayCaseDoorOpen \
    --stride 1 \
    --batch-size 64 \
    --output-dir outputs/anomaly_scores \
    --line-plot
```

### Advanced Options

#### Test with specific sample indices
```bash
python plot_anomaly_scores.py \
    --checkpoint checkpoints/dyedgegat_stride10.pt \
    --dataset Fault2_IceAccumulation \
    --stride 1 \
    --sample-indices 0 10 20 30 40 50 \
    --line-plot
```

#### Increase number of samples in visualization
```bash
python plot_anomaly_scores.py \
    --checkpoint checkpoints/dyedgegat_stride10.pt \
    --dataset Fault3_EvapValveFailure \
    --stride 1 \
    --max-samples 24 \
    --line-plot
```

#### Use different stride (faster but less detailed)
```bash
python plot_anomaly_scores.py \
    --checkpoint checkpoints/dyedgegat_stride10.pt \
    --dataset Fault4_MTEvapFanFailure \
    --stride 5 \
    --line-plot
```

## Understanding the Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--checkpoint` | Path to trained model | `checkpoints/dyedgegat_stride10.pt` |
| `--dataset` | Fault dataset name | See list above |
| `--stride` | Window sampling density | `1` (dense, best for incipient faults) |
| `--batch-size` | Inference batch size | `64` |
| `--num-workers` | DataLoader workers | `4` |
| `--max-samples` | Sequences to plot | `12` (default) |
| `--line-plot` | Generate line plots | Include this flag |
| `--use-amp` | Mixed precision (faster) | Include for CUDA |

## Output Files

For each fault dataset, you'll get:

### 1. CSV Files
- **`<fault>_anomaly_scores.csv`**
  - Detailed per-timestep anomaly scores
  - Columns: `sample_id`, `timestep_index`, `timestamp`, `relative_seconds`, `anomaly_score`
  - Use this for detailed analysis

- **`<fault>_timestamp_mean.csv`**
  - Anomaly scores averaged across all windows at each timestamp
  - Columns: `timestamp`, `anomaly_score`
  - Use this to see overall system health over time

### 2. Visualization Files
- **`<fault>_heatmap.png`**
  - Heatmap showing anomaly scores for multiple sequences
  - X-axis: Timestep within window
  - Y-axis: Sample ID (window)
  - Color: Anomaly score intensity

- **`<fault>_line.png`** (if `--line-plot` is used)
  - Line plot showing temporal progression
  - X-axis: Timestamp
  - Y-axis: Anomaly score
  - Multiple lines for different sample windows
  - **Best for visualizing incipient faults!**

## Interpreting Results for Incipient Fault Detection

### What to Look For:

1. **Baseline Behavior**
   - Normal operation should show low, stable anomaly scores
   - Typical range: 0-2 (depends on your training)

2. **Incipient Fault Onset**
   - Gradual increase in anomaly scores over time
   - Small deviations appearing before major failure
   - Look for upward trends in the line plots

3. **Fault Progression**
   - Steadily rising anomaly scores
   - Increasing variability
   - Consistent elevation above baseline

4. **Full Fault Manifestation**
   - Sharp spikes in anomaly scores
   - Sustained high values
   - Clear separation from normal range

### Example Analysis Workflow:

1. **Run all tests:**
   ```bash
   python run_all_fault_tests.py
   ```

2. **Check the line plots** (`*_line.png`)
   - Look for gradual increases (incipient phase)
   - Note the time when scores start rising
   - Compare early vs late fault behavior

3. **Examine CSV files** for quantitative analysis
   ```python
   import pandas as pd
   df = pd.read_csv('outputs/anomaly_scores/Fault1_DisplayCaseDoorOpen_anomaly_scores.csv')
   
   # Find when anomaly score exceeds threshold
   threshold = 3.0
   first_detection = df[df['anomaly_score'] > threshold].iloc[0]
   print(f"Fault first detected at: {first_detection['timestamp']}")
   ```

4. **Compare across faults**
   - Which faults are easiest to detect early?
   - Which show clearest incipient patterns?
   - Detection time differences

## Troubleshooting

### Issue: "No anomaly scores were collected"
- Check that dataset files exist in `Dataset/` directory
- Verify stride and window-size settings
- Ensure checkpoint matches window size (default: 60)

### Issue: Out of memory
- Reduce `--batch-size` (try 32, 16)
- Reduce `--num-workers` (try 2 or 0)
- Use `--stride` > 1 for less dense sampling

### Issue: Plots look empty or strange
- Try `--max-samples 20` for more sequences
- Use `--sample-indices` to select specific windows
- Check CSV files to verify data exists

## Advanced: Comparing Model Checkpoints

To test multiple model checkpoints:

```bash
for ckpt in checkpoints/*.pt; do
    echo "Testing: $ckpt"
    python plot_anomaly_scores.py \
        --checkpoint "$ckpt" \
        --dataset Fault1_DisplayCaseDoorOpen \
        --output-dir "outputs/comparison_$(basename $ckpt .pt)"
done
```

## Questions?

- Review the `plot_anomaly_scores.py` docstring for all options
- Check individual fault CSV files for data quality
- Experiment with `--stride` and `--max-samples` for different views

