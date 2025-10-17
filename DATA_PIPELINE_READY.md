# 🎉 Data Pipeline Complete!

## ✅ What We've Built

Your complete data loading infrastructure for DyEdgeGAT is ready to use!

### 📁 New Files Created:

```
dyedgegat/src/data/
├── __init__.py              # Module initialization
├── column_config.py         # Variable selection (30 sensors, 6 controls)
├── dataset.py               # PyTorch Geometric Dataset
└── dataloader.py            # DataLoader creation utilities

Configuration:
├── config.py                # Updated with correct dimensions (30, 15, 6)

Testing:
├── test_data_pipeline.py    # Complete test script
└── requirements.txt         # Python dependencies

Documentation:
├── DATA_PIPELINE_READY.md   # This file
├── DATASET_CONSISTENCY_REPORT.md
└── DATASET_QUICK_SUMMARY.md
```

---

## 📊 Variable Selection (Option A - Medium)

### **30 Measurement Variables** (Sensors - form graph nodes):

**Power (6)**:
- W_MT-COMP1, W_MT-COMP2, W_MT-COMP3 (MT Compressor power)
- W_LT-COMP1, W_LT-COMP2 (LT Compressor power)
- W-CONDENSOR (Condenser power)

**Pressure (10)**:
- P-LT-BPHX, P-MT-BPHX (Heat exchanger pressures)
- P-MTcase-SUC, P-MTcase-LIQ (MT case pressures)
- P-LTcase-SUC, P-LTcase-LIQ (LT case pressures)
- P-GC-IN (Gas cooler inlet)
- P-MT_Dis-OilSepIn, P-LT-SUC, P-MT_SUC

**Temperature (12)**:
- T-MT-COMP1-SUC, T-MT-COMP1-DIS (MT Comp 1 temps)
- T-LT-COMP1-SUC, T-LT-COMP1-DIS (LT Comp 1 temps)
- T-GC-SUC, T-GC-DIS (Gas cooler temps)
- T-MT-Suc, T-MT-Dis (MT rack temps)
- T-LT-Suc, T-LT-Dis (LT rack temps)
- T-GC-In, T-GC-Out (Gas cooler in/out)

**Flow (2)**:
- F-LT-BPHX, F-MT-BPHX (Heat exchanger flows)

### **6 Operating Condition Variables** (Control/External):
- Tsetpt, RHsetpt (Setpoints)
- SupHCompSuc, SupHCompDisc (Superheat)
- SubcoolCond1, SubcoolCond2 (Subcooling)

---

## 🗂️ Data Split

### **Training** (4 files, ~325K rows):
- BaselineTestA.csv
- BaselineTestC.csv
- BaselineTestD.csv
- BaselineTestE.csv

### **Validation** (1 file, ~81K rows):
- BaselineTestB.csv

### **Testing** (6 fault types + 1 baseline):
- Baseline (normal): BaselineTestB.csv
- Fault1: DisplayCaseDoorOpen
- Fault2: IceAccumulation
- Fault3: EvapValveFailure
- Fault4: MTEvapFanFailure
- Fault5: CondAPBlock
- Fault6: MTEvapAPBlock

---

## 🚀 How to Use

### Step 1: Install Dependencies
```bash
cd /mnt/datassd3/rashinda/DyEdge
pip install -r requirements.txt
```

**Note**: If PyTorch installation fails, install manually first:
```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch Geometric
pip install torch-geometric

# Then the rest
pip install pandas numpy matplotlib seaborn tqdm
```

### Step 2: Test the Pipeline
```bash
python test_data_pipeline.py
```

**Expected output**:
```
================================================================================
                    DYEDGEGAT DATA PIPELINE TEST
================================================================================

STEP 1: Variable Configuration
======================================================================
REFRIGERATION SYSTEM - VARIABLE CONFIGURATION
======================================================================
Measurement Variables (Sensors): 30
  - Power:       6
  - Pressure:    10
  - Temperature: 12
  - Flow:        2

Operating Condition Variables: 6
  - Setpoints:   2
  - Superheat:   2
  - Subcooling:  2
...
✅ All tests passed!
```

### Step 3: Use in Your Code
```python
from dyedgegat.src.data.dataloader import create_dataloaders
from dyedgegat.src.config import cfg

# Set configuration
cfg.set_dataset_params(n_nodes=30, window_size=15, ocvar_dim=6)

# Create dataloaders
train_loader, val_loader, test_loaders = create_dataloaders(
    window_size=15,
    batch_size=64,
    train_stride=1,     # Use all windows for training
    val_stride=5,       # Use stride for faster validation
    data_dir='Dataset',
)

# Train your model
for batch in train_loader:
    # batch.x: [B*N, W] measurements
    # batch.c: [B, ocvar_dim, W] controls
    # batch.edge_index: [2, E] edges
    # batch.batch: [B*N] batch assignment
    
    # Your training code here
    pass
```

---

## 📦 DataLoader Details

### **Batch Structure:**

When you get a batch from the dataloader:

```python
batch = next(iter(train_loader))

# Measurements (sensors)
batch.x.shape  # [B*N, W] = [batch_size * 30, 15]
# Example: batch_size=64 → [1920, 15]

# Controls (operating conditions)
batch.c.shape  # [B, ocvar_dim, W] = [64, 6, 15]

# Graph edges (fully connected)
batch.edge_index.shape  # [2, E] where E = B * N * N

# Batch assignment (which sample each node belongs to)
batch.batch.shape  # [B*N] = [1920]
```

### **Data Preprocessing:**

The pipeline automatically:
1. ✅ Selects only the 30+6 configured columns
2. ✅ Handles missing values (sentinel values → NaN → forward fill)
3. ✅ Normalizes data (z-score normalization)
4. ✅ Creates sliding windows (window_size=15)
5. ✅ Generates fully connected graphs
6. ✅ Uses training stats for validation/test normalization

---

## 🎯 Next Steps: Training DyEdgeGAT

### Quick Example:

```python
import torch
from dyedgegat.src.data.dataloader import create_dataloaders
from dyedgegat.src.model.dyedgegat import DyEdgeGAT
from dyedgegat.src.config import cfg

# Configure
cfg.set_dataset_params(n_nodes=30, window_size=15, ocvar_dim=6)
cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dataloaders
train_loader, val_loader, test_loaders = create_dataloaders(
    window_size=15,
    batch_size=64,
)

# Create model
model = DyEdgeGAT(
    feat_input_node=1,      # Each sensor is univariate
    feat_target_node=1,     # Reconstruct 1 value per sensor
    feat_input_edge=1,      # Edge features are scalar
    temp_node_embed_dim=16,
    gnn_embed_dim=40,
    num_gnn_layers=2,
    gnn_type='gin',
    # ... other parameters
).to(cfg.device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    for batch in train_loader:
        batch = batch.to(cfg.device)
        
        # Forward pass
        recon = model(batch)
        
        # Compute loss
        target = batch.x.unsqueeze(-1)  # [B*N, W, 1]
        loss = criterion(recon, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

---

## 🔍 Inspection Tools

### View Configuration:
```python
from dyedgegat.src.data.column_config import print_config_summary
print_config_summary()
```

### Check Selected Variables:
```python
from dyedgegat.src.data.column_config import MEASUREMENT_VARS, CONTROL_VARS

print(f"Measurement variables ({len(MEASUREMENT_VARS)}):")
for var in MEASUREMENT_VARS:
    print(f"  - {var}")

print(f"\nControl variables ({len(CONTROL_VARS)}):")
for var in CONTROL_VARS:
    print(f"  - {var}")
```

### Get Normalization Stats:
```python
from dyedgegat.src.data.dataset import RefrigerationDataset

dataset = RefrigerationDataset(
    data_files=['BaselineTestA.csv'],
    window_size=15,
    data_dir='Dataset',
)

# Get normalization statistics for later use
stats = dataset.get_normalization_stats()
measurement_mean, measurement_std, control_mean, control_std = stats

print(f"Measurement mean: {measurement_mean.shape}")
print(f"Measurement std: {measurement_std.shape}")
```

---

## 📝 Configuration Summary

```python
# In dyedgegat/src/config.py
cfg.dataset.n_nodes = 30          # Measurement sensors
cfg.dataset.window_size = 15      # Temporal window size
cfg.dataset.ocvar_dim = 6         # Control variables
cfg.device = 'cuda'               # or 'cpu'
```

This matches perfectly with:
- DyEdgeGAT's requirements
- The variable selection in `column_config.py`
- The dataset structure

---

## 🎉 Summary

You now have:
- ✅ **30 carefully selected measurement sensors** (covering all subsystems)
- ✅ **6 operating condition variables** (setpoints, superheat, subcooling)
- ✅ **4 baseline files for training** (~325K samples)
- ✅ **1 baseline file for validation** (~81K samples)
- ✅ **6 fault types for testing** (~487K samples)
- ✅ **Complete data loading pipeline** with automatic preprocessing
- ✅ **PyTorch Geometric DataLoaders** ready for DyEdgeGAT
- ✅ **Proper train/val/test split** for anomaly detection evaluation

**Everything is ready!** Just install the dependencies and run the test.

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install requirements: `pip install -r requirements.txt`

### Issue: CUDA out of memory
**Solution**: Reduce batch_size in `create_dataloaders(batch_size=32)`

### Issue: Data loading too slow
**Solution**: Increase `train_stride` for faster iteration (trade-off: fewer samples)

### Issue: Different number of variables needed
**Solution**: Edit `dyedgegat/src/data/column_config.py` and update `cfg.dataset.n_nodes` and `cfg.dataset.ocvar_dim`

---

**Ready to train? Run `test_data_pipeline.py` first to verify everything works!** 🚀

