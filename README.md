# DyEdgeGAT Refrigeration Toolkit

End-to-end utilities for training, evaluating, and visualising DyEdgeGAT anomaly detection models on the CO₂ supermarket refrigeration benchmark.

This repository now focuses on three entrypoints:

- `train_dyedgegat.py` / `train_dyedgegat_1min.py` – model training (single or multi‑GPU).
- `test_dyedgegat_model.py` – quick functional test of the pipeline.
- `plot_reconstruction_plotly.py` – interactive Plotly plots for reconstructions and anomaly scores.

The DyEdgeGAT library itself lives under `dyedgegat/`, and datasets are stored in `Dataset/` (original cadence) and `Dataset_1min/` (1‑minute aggregated variant).

---

## 1. Environment Setup

1. Create/activate a Python environment that matches your CUDA toolchain (tested with Python 3.11, PyTorch ≥ 2.2).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The file pins PyTorch Geometric and friends; make sure your `pip` install uses a wheel that matches your CUDA version.
3. Expected directory layout:
   ```
   DyEdge/
   ├── Dataset/            # 5 baseline CSVs + 6 fault CSVs (original cadence)
   ├── Dataset_1min/       # 1-minute aggregated CSVs (same naming without suffix)
   ├── checkpoints/        # Saved checkpoints (best model, per-epoch, etc.)
   └── dyedgegat/          # Source code for dataset/model utilities
   ```

> **Torch Scatter/Sparse warnings**  
> If you see warnings about `torch_scatter` or `torch_sparse` when running scripts, they simply fall back to CPU kernels inside PyG. No action is needed for inference or plotting.

---

## 2. Training

`train_dyedgegat.py` drives the full training loop with support for single-GPU and DistributedDataParallel (DDP) multi-GPU runs.

### 2.1 Common Flags
- `--epochs` – number of training epochs (default 10).
- `--batch-size` – per-GPU batch size.
- `--train-stride`, `--val-stride`, `--test-stride` – sliding-window strides.
- `--data-dir` – root directory with the CSVs (defaults to `Dataset/`).
- `--use-amp` – enable automatic mixed precision on CUDA.
- `--checkpoint` – optional weight file to resume from or evaluate.
- `--checkpoint-dir` – directory for per-epoch checkpoints + metrics.

### 2.2 Single-GPU Example
```bash
CUDA_VISIBLE_DEVICES=0 python train_dyedgegat.py \
    --epochs 20 \
    --batch-size 64 \
    --data-dir Dataset \
    --train-stride 1 \
    --val-stride 5 \
    --use-amp
```
`--cuda-device` can also be used instead of `CUDA_VISIBLE_DEVICES`.

### 2.3 Multi-GPU (DDP) Example
Launch with `torchrun`; each process controls one GPU and DataLoader shard.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    train_dyedgegat.py \
    --epochs 20 \
    --batch-size 128 \
    --data-dir Dataset \
    --train-stride 1 \
    --val-stride 5 \
    --num-workers 4 \
    --use-amp
```
Notes:
- Batch size is per process; total effective batch size = `batch_size * nproc`.
- `--dist-backend` defaults to `nccl` (recommended for CUDA). Switch to `gloo` only for CPU experiments.
- Avoid `--cuda-devices` when using DDP; rely on `CUDA_VISIBLE_DEVICES`.

### 2.4 1-minute Aggregated Dataset
`train_dyedgegat_1min.py` wraps the base trainer and injects `--data-dir Dataset_1min` when you do not supply one yourself. Usage mirrors the commands above:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    train_dyedgegat_1min.py \
    --epochs 15 \
    --batch-size 96 \
    --train-stride 2 \
    --val-stride 10 \
    --use-amp
```

---

## 3. Testing & Quick Validation

`test_dyedgegat_model.py` exercises the full pipeline (data loaders, forward pass, loss, backward step, anomaly scoring) using small strides so it finishes quickly.

Run it after training to sanity-check a checkpoint:
```bash
python test_dyedgegat_model.py
```
Key behaviour:
- Uses `Dataset/` by default; edit the script or set `data_dir` manually if you want `Dataset_1min`.
- Loads `cfg.dataset` with the measurement/control variable definitions from `dyedgegat/src/data/column_config.py`.
- Prints batch/tensor shapes, loss values, and checks for NaNs/Infs.

---

## 4. Plotting With Plotly

`plot_reconstruction_plotly.py` loads a trained checkpoint, runs inference, aggregates per‑timestamp statistics, and writes interactive Plotly HTML files plus CSV timeseries.

### 4.1 Basic Run (single dataset)
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/per_epoch/dyedgegat_20251027_050012/dyedgegat_20251027_epoch_100.pt \
    --data-dir /mnt/datassd3/rashinda/DyEdge/Dataset_1min \
    --dataset baseline \
    --sensor T-MTcase-LIQ \
    --denormalize
```
Outputs:
- `outputs/plotly/baseline_T-MTcase-LIQ_reconstruction.html`
- `outputs/plotly/baseline_T-MTcase-LIQ_reconstruction.csv`

### 4.2 Sweep All Faults + Baseline
Use `--include-all-faults` to process baseline plus every defined fault in `column_config.py`. The script saves per-dataset files under `outputs/plotly/<dataset>/`.
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/.../best.pt \
    --data-dir /mnt/datassd3/rashinda/DyEdge/Dataset_1min \
    --include-all-faults \
    --sensor T-MTcase-LIQ \
    --denormalize
```

### 4.3 Anomaly-Only Mode
If you only care about anomaly trajectories, skip the actual/reconstructed curves:
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/.../best.pt \
    --data-dir /mnt/datassd3/rashinda/DyEdge/Dataset_1min \
    --include-all-faults \
    --anomaly-only
```
- Sensor selection is not required.
- Each HTML/CSV file is suffixed with `_anomaly`.

### 4.4 Useful Flags
- `--datasets fault1 fault3` – process specific dataset keys.
- `--max-windows 12` – limit inference to the first N sliding windows for quick previews.
- `--output-html /path/to/dir` and/or `--output-csv /path/to/dir` – customise output directories (pass directories when using multiple datasets).

> The Plotly helper runs on a single GPU. Launch separate processes (with different `CUDA_VISIBLE_DEVICES` masks) if you want simultaneous plots on multiple GPUs.

---

## 5. Dataset & Feature Notes

- Both dataset directories contain five baseline CSVs (`BaselineTestA` … `BaselineTestE`) and six faults (`Fault1_…` … `Fault6_…`). Filenames **do not** include `_1min`; the script appends suffixes only when you explicitly pass `--dataset-suffix`.
- Measurement and control variables are defined in `dyedgegat/src/data/column_config.py`. Training scripts use `get_control_variable_names` to match the dataset and automatically include sinusoidal time encodings for the 1-minute aggregated data.
- Normalisation stats are derived from the training baseline split and reused for evaluation/plotting to keep scales consistent.

---

## 6. Checkpoints

- Drop trained weights under `checkpoints/`. Per-epoch checkpoints generated by `EpochCheckpointManager` live in timestamped subdirectories (`checkpoints/per_epoch/<run_id>/`).
- The testing and Plotly scripts accept the same `--checkpoint` path.

---

## 7. Troubleshooting & Tips

- **“Missing torch_scatter / torch_sparse” warnings:** Safe to ignore for inference. If you need GPU kernels, reinstall the matching wheels for your CUDA/PyTorch version.
- **“Sensor not found” in Plotly script:** Ensure the casing matches `MEASUREMENT_VARS`. For example, `T-MTcase-LIQ` (lowercase “case”).
- **DDP launch issues:** Confirm `torchrun --nproc_per_node` equals the number of visible GPUs, and avoid mixing `--cuda-device` with DDP.
- **Large CSV load times:** Increase `--train-stride` / `--val-stride` or use `--max-windows` in plotting to reduce memory usage during experiments.

---

Happy experimenting! Feel free to extend the scripts for additional diagnostics or integrate them into your automation pipeline. All core training, testing, and plotting functionality is now centralised in the files listed above.
