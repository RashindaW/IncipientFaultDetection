# ASHRAE 1043-RP Dataset Integration Guide

This document describes the integration of the ASHRAE 1043-RP water-cooled chiller dataset with the DyEdgeGAT framework for anomaly detection.

## Overview

The ASHRAE 1043-RP dataset contains experimental data from a water-cooled chiller system operating under normal and faulty conditions. This integration focuses on:

- **Training**: Benchmark test data (normal operation)
- **Testing**: Refrigerant leak fault scenarios

## Dataset Structure

### Directory Layout

```
data/ASHRAE_1043_RP/
├── Benchmark Tests/          # Normal operation data (18 files)
│   ├── normal.xls
│   ├── normal1.xls
│   ├── normal2.xls
│   ├── near normal1.xls
│   ├── near normal2.xls
│   ├── near normal3.xls
│   └── ...
└── Refrigerant leak/         # Fault data (5 files)
    ├── rl10.xls              # 10% refrigerant leak
    ├── rl20.xls              # 20% refrigerant leak
    ├── rl30.xls              # 30% refrigerant leak
    ├── rl40.xls              # 40% refrigerant leak
    └── rl40 alt--many unsteady tests.xls
```

### Data Characteristics

- **File Format**: XLS (Excel 97-2003)
- **Columns**: 66 variables per file
- **Sampling**: Time-series data with ~5191 rows per file
- **Variables**:
  - 62 measurement variables (sensors)
  - 3 control/operating condition variables
  - 1 time column

### Variable Categories

#### Measurement Variables (62 total)

1. **Temperature Sensors (29)**
   - Evaporator circuit: TEI, TWEI, TEO, TWEO, TEA
   - Condenser circuit: TCI, TWCI, TCO, TWCO, TCA
   - Refrigerant: TRE, TRC, TRC_sub, T_suc, TR_dis
   - Superheat: Tsh_suc, Tsh_dis
   - Other system points: TSI, TSO, TBI, TBO, TO_sump, TO_feed, TWI, TWO, THI, THO
   - Delta T: TWCD, TWED

2. **Pressure Sensors (5)**
   - PRE, PRC, PO_feed, PO_net, P_lift

3. **Flow Sensors (5)**
   - FWC, FWE, FWW, FWH, FWB

4. **Performance Metrics (5)**
   - kW (power), Amps (current), RLA% (rated load amps), COP (coefficient of performance), kW/Ton

5. **Capacity Metrics (6)**
   - Cond Tons, Cooling Tons, Shared Cond Tons, Evap Tons, Shared Evap Tons, Building Tons

6. **Energy Balance (5)**
   - Cond Energy Balance, Evap Energy Balance, Heat Balance (kW), Heat Balance%, Tolerance%

7. **Virtual Sensors (7)**
   - VSS, VSL, VH, VM, VC, VE, VW

#### Control Variables (3)

- TWE_set: Temperature Water Evaporator Setpoint
- Unit Status: Operational status indicator
- Active Fault: Fault indicator

## Usage

### 1. Quick Test

Test the integration to ensure everything is set up correctly:

```bash
python test_ashrae_integration.py --quick
```

This will verify:
- ✓ Adapter registration
- ✓ Column configuration
- ✓ Data file availability
- ✓ Dataset creation
- ✓ DataLoader creation

### 2. Training on Benchmark Tests

Train the DyEdgeGAT model using normal operation data:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python train_dyedgegat.py \
    --dataset-key ashrae \
    --epochs 20 \
    --batch-size 32 \
    --train-stride 5 \
    --val-stride 10 \
    --use-amp

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    train_dyedgegat.py \
    --dataset-key ashrae \
    --epochs 20 \
    --batch-size 64 \
    --train-stride 5 \
    --val-stride 10 \
    --num-workers 4 \
    --use-amp
```

The training uses:
- **Training files**: 15 benchmark test files (normal operation)
- **Validation files**: 3 near-normal test files
- **Output**: Checkpoints saved to `checkpoints/ashrae/`

### 3. Testing on Refrigerant Leak

After training, evaluate the model on refrigerant leak fault scenarios:

```bash
python test_dyedgegat_model.py --dataset-key ashrae
```

The test set includes:
- Baseline (near-normal operation)
- Refrigerant_Leak_10 (10% leak)
- Refrigerant_Leak_20 (20% leak)
- Refrigerant_Leak_30 (30% leak)
- Refrigerant_Leak_40 (40% leak)
- Refrigerant_Leak_40_alt (alternative 40% leak scenario)

### 4. Visualization

Generate interactive plots for reconstruction and anomaly scores:

```bash
# Single fault type
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/ashrae/dyedgegat_ashrae_best.pt \
    --dataset-key ashrae \
    --dataset Refrigerant_Leak_20 \
    --sensor TEO \
    --denormalize

# All fault types
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/ashrae/dyedgegat_ashrae_best.pt \
    --dataset-key ashrae \
    --include-all-faults \
    --sensor TEO \
    --denormalize

# Anomaly scores only (no sensor selection needed)
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/ashrae/dyedgegat_ashrae_best.pt \
    --dataset-key ashrae \
    --include-all-faults \
    --anomaly-only
```

## Implementation Details

### Dataset Adapter

The ASHRAE adapter is registered in `datasets/ashrae.py` and provides:

- **Key**: `ashrae`
- **Description**: ASHRAE 1043-RP water-cooled chiller dataset
- **Default directory**: `data/ASHRAE_1043_RP`
- **Measurement variables**: 62 sensors
- **Control variables**: 3 operating conditions

### Custom Dataset Class

`ASHRAEDataset` (in `dyedgegat/src/data/ashrae_dataset.py`) extends the base functionality to handle:

- **XLS file loading** instead of CSV
- **Custom column configuration** for ASHRAE variables
- **Normalization** based on training statistics
- **Temporal graph construction** for DyEdgeGAT

### Column Configuration

The column configuration is defined in `dyedgegat/src/data/ashrae_column_config.py`:

- Defines all 62 measurement variables and their categories
- Defines 3 control variables
- Maps benchmark test files for training
- Maps refrigerant leak files for testing

## Differences from CO₂ Refrigeration Dataset

| Aspect | CO₂ Dataset | ASHRAE Dataset |
|--------|-------------|----------------|
| File Format | CSV | XLS |
| Measurement Variables | 142 | 62 |
| Control Variables | 10 | 3 |
| System Type | CO₂ supermarket refrigeration | Water-cooled chiller |
| Fault Types | 6 different faults | 5 refrigerant leak scenarios |
| Time Features | Sinusoidal time encoding (1-min) | Not used |
| Data Source | Supermarket system | Laboratory chiller |

## Key Sensor Recommendations for Visualization

Good sensors for visualizing refrigerant leak effects:

1. **TEO** (Temperature Evaporator Outlet): Shows impact on evaporator performance
2. **TRE** (Temperature Refrigerant Evaporator): Direct refrigerant temperature
3. **PRE** (Pressure Refrigerant Evaporator): Evaporator pressure drop
4. **PRC** (Pressure Refrigerant Condenser): Condenser pressure changes
5. **COP** (Coefficient of Performance): Overall efficiency metric
6. **kW** (Power): Power consumption changes
7. **Cooling Tons**: Cooling capacity degradation

## Troubleshooting

### Missing Dependencies

If you see "No module named 'xlrd'" errors:

```bash
pip install xlrd openpyxl
```

### Data File Not Found

Verify the data directory structure:

```bash
ls -R data/ASHRAE_1043_RP/
```

Ensure files are in the correct subdirectories:
- Benchmark tests → `data/ASHRAE_1043_RP/Benchmark Tests/`
- Refrigerant leak → `data/ASHRAE_1043_RP/Refrigerant leak/`

### Memory Issues

If you encounter memory errors with large datasets:

- Increase `train_stride` and `val_stride` to reduce sample count
- Reduce `batch_size`
- Use fewer training files initially for testing

## Next Steps

After successful integration of refrigerant leak detection, you can extend to other fault types:

1. **Condenser fouling** (`data/ASHRAE_1043_RP/Condenser fouling/`)
2. **Reduced condenser water flow** (`data/ASHRAE_1043_RP/Reduced condenser water flow/`)
3. **Reduced evaporator water flow** (`data/ASHRAE_1043_RP/Reduced evaporator water flow/`)
4. **Excess oil** (`data/ASHRAE_1043_RP/Excess oil/`)
5. **Multiple faults** (`data/ASHRAE_1043_RP/Multiple faults/`)

To add new fault types, update `FAULT_FILES` in `dyedgegat/src/data/ashrae_column_config.py`.

## References

- DyEdgeGAT Paper: [DOI: 10.1109/JIOT.2024.3381002](https://doi.org/10.1109/JIOT.2024.3381002)
- ASHRAE 1043-RP: Standard dataset for HVAC fault detection and diagnostics

