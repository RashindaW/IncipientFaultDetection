# DyEdgeGAT Refrigeration Toolkit

End-to-end utilities for training, evaluating, and visualising DyEdgeGAT anomaly detection models on the CO₂ supermarket refrigeration benchmark.

This repository now focuses on three entrypoints:

- `train_dyedgegat.py` / `train_dyedgegat_1min.py` – model training (single or multi‑GPU).
- `test_dyedgegat_model.py` – quick functional test of the pipeline.
- `plot_reconstruction_plotly.py` – interactive Plotly plots for reconstructions and anomaly scores.

The DyEdgeGAT library itself lives under `dyedgegat/`, and datasets are expected under `data/` (see [Dataset adapters](#11-dataset-adapters) for layout and defaults).

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
   ├── data/
   │   ├── co2/
   │   │   ├── raw/        # 5 baseline + 6 fault CSVs (original cadence, git-ignored)
   │   │   └── 1min/       # 1-minute aggregated CSVs (same filenames)
   │   └── tep/
   │       └── raw/        # Tennessee Eastman Process RData files
   ├── checkpoints/        # Saved checkpoints (best model, per-epoch, etc.)
   └── dyedgegat/          # Source code for dataset/model utilities
   ```

> **Torch Scatter/Sparse warnings**  
> If you see warnings about `torch_scatter` or `torch_sparse` when running scripts, they simply fall back to CPU kernels inside PyG. No action is needed for inference or plotting.

---

## 1.1 Dataset adapters

Dataset-specific defaults and metadata are defined under the tracked `datasets/` package. Each adapter exposes:

- Measurement and control variable lists.
- Default on-disk location (`data/<name>/...`).
- Split definitions (train/baseline/fault files).
- Dataloader construction hooks.

Built-in adapters:

| Key        | Description                                       | Default data dir      | Status            |
|------------|---------------------------------------------------|-----------------------|-------------------|
| `co2`      | CO₂ refrigeration benchmark (original cadence)    | `data/co2/raw`        | Training-ready    |
| `co2_1min` | CO₂ refrigeration benchmark (1-minute aggregation)| `data/co2/1min`       | Training-ready    |
| `ashrae`   | ASHRAE 1043-RP water-cooled chiller (XLS files)   | `data/ASHRAE_1043_RP` | Training-ready    |
| `tep`      | Tennessee Eastman Process (RData)                 | `data/tep/raw`        | Scaffolding only* |

`tep` is registered so you can extend it later, but all capabilities currently raise `NotImplementedError` until preprocessing and dataloaders are implemented.

> **ASHRAE dataset**: Training uses the “Benchmark Tests” XLS files (normal and near‑normal operation) while testing targets the “Refrigerant leak” scenarios. The adapter loads the native spreadsheets directly (requires `xlrd`) so no additional conversion step is needed.

Adapters are selected via the new `--dataset-key` argument across training, testing, and plotting scripts. You can override the default location with `--data-dir` when needed.

---

## 2. Training

`train_dyedgegat.py` drives the full training loop with support for single-GPU and DistributedDataParallel (DDP) multi-GPU runs.

### 2.1 Common Flags
- `--epochs` – number of training epochs (default 10).
- `--batch-size` – per-GPU batch size.
- `--train-stride`, `--val-stride`, `--test-stride` – sliding-window strides.
- `--dataset-key` – dataset adapter to use (`co2`, `co2_1min`, `ashrae`, `tep`).
- `--data-dir` – override the adapter's default data directory when needed.
- `--use-amp` – enable automatic mixed precision on CUDA.
- `--checkpoint` – optional weight file to resume from or evaluate.
- `--checkpoint-dir` – directory for per-epoch checkpoints + metrics.

Unless overridden, checkpoints are organised under `checkpoints/<dataset-key>/`, and the best-performing weights are written to `checkpoints/<dataset-key>/dyedgegat_<dataset-key>_best.pt`.

### 2.2 Single-GPU Example
```bash
CUDA_VISIBLE_DEVICES=0 python train_dyedgegat.py \
    --dataset-key co2 \
    --epochs 20 \
    --batch-size 64 \
    --train-stride 1 \
    --val-stride 5 \
    --use-amp
```
`--cuda-device` can also be used instead of `CUDA_VISIBLE_DEVICES`.

The `co2` adapter looks under `data/co2/raw` by default; override with `--data-dir` if your CSVs live elsewhere.

### 2.3 Multi-GPU (DDP) Example
Launch with the PyTorch DDP launcher; each process controls one GPU and DataLoader shard.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    train_dyedgegat.py \
    --dataset-key co2 \
    --epochs 20 \
    --batch-size 128 \
    --train-stride 1 \
    --val-stride 5 \
    --num-workers 4 \
    --use-amp
```
Notes:
- Batch size is per process; total effective batch size = `batch_size * nproc`.
- `--dist-backend` defaults to `nccl` (recommended for CUDA). Switch to `gloo` only for CPU experiments.
- Avoid `--cuda-devices` when using DDP; rely on `CUDA_VISIBLE_DEVICES`.
- You can substitute `torchrun` for `python -m torch.distributed.run` if it is available on your PATH.

### 2.4 1-minute Aggregated Dataset
`train_dyedgegat_1min.py` wraps the base trainer and injects `--dataset-key co2_1min` plus `--data-dir data/co2/1min` when you do not provide them. Usage mirrors the commands above:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 \
    train_dyedgegat_1min.py \
    --epochs 20 \
    --batch-size 32 \
    --train-stride 2 \
    --val-stride 10 \
    --num-workers 4 \
    --use-amp
```

---

## 3. Testing & Quick Validation

`test_dyedgegat_model.py` exercises the full pipeline (data loaders, forward pass, loss, backward step, anomaly scoring) using small strides so it finishes quickly.

Run it after training to sanity-check a checkpoint:
```bash
python test_dyedgegat_model.py --dataset-key co2
```
Optional flags: `--data-dir` to point at a non-default location, `--batch-size` and `--stride` to tweak runtime.
Key behaviour:
- Uses the `co2` adapter (`data/co2/raw`) by default; edit `DATASET_KEY` inside the script to target other adapters.
- Loads `cfg.dataset` with the measurement/control variable definitions from `dyedgegat/src/data/column_config.py`.
- Prints batch/tensor shapes, loss values, and checks for NaNs/Infs.

---

## 4. Plotting With Plotly

`plot_reconstruction_plotly.py` loads a trained checkpoint, runs inference, aggregates per‑timestamp statistics, and writes interactive Plotly HTML files plus CSV timeseries.

### 4.1 Basic Run (single dataset)
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/per_epoch/dyedgegat_20251027_050012/dyedgegat_20251027_epoch_100.pt \
    --dataset-key co2_1min \
    --data-dir data/co2/1min \
    --dataset baseline \
    --sensor T-MTcase-LIQ \
    --denormalize
```
Outputs are grouped under `outputs/plotly/<run>/<dataset>/...` where `<run>` defaults to a timestamped name derived from the dataset key and checkpoint. You can provide `--run-name` to customise it or `--output-root` to relocate the entire tree.

### 4.2 Sweep All Faults + Baseline
Use `--include-all-faults` to process baseline plus every defined fault in `column_config.py`. The script saves per-dataset files under `outputs/plotly/<dataset>/`.
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/.../best.pt \
    --dataset-key co2_1min \
    --data-dir data/co2/1min \
    --include-all-faults \
    --sensor T-MTcase-LIQ \
    --denormalize
```

### 4.3 Anomaly-Only Mode
If you only care about anomaly trajectories, skip the actual/reconstructed curves:
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/.../best.pt \
    --dataset-key co2_1min \
    --data-dir data/co2/1min \
    --include-all-faults \
    --anomaly-only
```
- Sensor selection is not required.
- Each HTML/CSV file is suffixed with `_anomaly`.

Useful options:
- `--run-name myexperiment` – fixed subdirectory name under `outputs/plotly/`.
- `--output-root /tmp/plots` – write the entire run elsewhere.
- `--output-html` / `--output-csv` – override specific paths for HTML or CSV artefacts.

### 4.4 Useful Flags
- `--datasets fault1 fault3` – process specific dataset keys.
- `--max-windows 12` – limit inference to the first N sliding windows for quick previews.
- `--output-html /path/to/dir` and/or `--output-csv /path/to/dir` – customise output directories (pass directories when using multiple datasets).

> The Plotly helper runs on a single GPU. Launch separate processes (with different `CUDA_VISIBLE_DEVICES` masks) if you want simultaneous plots on multiple GPUs.

---

## 5. Dataset & Feature Notes

- `data/co2/raw/` and `data/co2/1min/` each contain five baseline CSVs (`BaselineTestA` … `BaselineTestE`) plus six faults (`Fault1_…` … `Fault6_…`). Filenames **do not** include `_1min`; adapters honour the directory layout instead.
- Measurement and control variables are defined in `dyedgegat/src/data/column_config.py`. Training scripts use `get_control_variable_names` to match the dataset and automatically include sinusoidal time encodings for the 1-minute aggregated data.
- Normalisation stats are derived from the training baseline split and reused for evaluation/plotting to keep scales consistent.

---

## 6. Checkpoints

- By default, training runs write per-epoch checkpoints to `checkpoints/<dataset-key>/dyedgegat_<dataset-key>_<timestamp>/...`.
- The best-performing weights are also saved to `checkpoints/<dataset-key>/dyedgegat_<dataset-key>_best.pt` unless you override `--save-model`.
- Pass `--checkpoint-dir` to change the root directory, or `--save-model` to pick a different destination for the best weights.
- The testing and Plotly scripts accept any of these checkpoint paths via `--checkpoint`.

---

## 7. Troubleshooting & Tips

- **“Missing torch_scatter / torch_sparse” warnings:** Safe to ignore for inference. If you need GPU kernels, reinstall the matching wheels for your CUDA/PyTorch version.
- **“Sensor not found” in Plotly script:** Ensure the name matches the adapter's measurement list (e.g., `T-MTcase-LIQ` for the CO₂ dataset).
- **DDP launch issues:** Confirm `torchrun --nproc_per_node` equals the number of visible GPUs, and avoid mixing `--cuda-device` with DDP.
- **Large CSV load times:** Increase `--train-stride` / `--val-stride` or use `--max-windows` in plotting to reduce memory usage during experiments.

---

Happy experimenting! Feel free to extend the scripts for additional diagnostics or integrate them into your automation pipeline. All core training, testing, and plotting functionality is now centralised in the files listed above.
