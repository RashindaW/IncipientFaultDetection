# DualSTAGE: Dual-View Spectral-Temporal Graph Attention for Enhanced Fault Detection

A specialized framework for **incipient fault detection** in complex industrial systems using **Dynamic Graph Neural Networks**.

This repository implements **DualSTAGE** (Dynamic Spectral-Temporal Graph ATtention) with a novel **Dual-View (Spectral + Temporal) Architecture**. It learns two concurrent graph topologies:
1.  **Temporal Graph ($A_{time}$)**: Captures dynamic correlations (nodes moving together).
2.  **Spectral Graph ($A_{freq}$)**: Captures frequency-domain similarities (nodes resonating together).

By monitoring the **structural divergence** between these two graphs ($D_{div}$), the model detects incipient faults (wear, drift, fouling) *before* they manifest as gross reconstruction errors.

---

## 1. Key Features

*   **Dual-View Learning**: Simultaneous `GRUEncoder` (time) and `SpectralEncoder` (frequency) branches.
*   **Structural Divergence Score**: Explicitly measures mismatch between physical connectivity and spectral behavior (early warning signal).
*   **Topology-Aware Anomaly Scoring**: Penalizes errors on central nodes more heavily.
*   **Multi-Dataset Support**: Ready for TEP, CO2, PRONTO, ASHRAE, and IMS Bearing benchmarks.
*   **Interactive Visualization**: Plotly-based dashboards for reconstruction and anomaly score analysis.

---

## 2. Supported Datasets

Data adapters are defined in `datasets/` and accessed via `--dataset-key`.

| Dataset Key | Name | Type | Status | Source |
| :--- | :--- | :--- | :--- | :--- |
| `tep` | Tennessee Eastman Process | Chemical Process | **Ready** | [Open](http://brahms.scs.uiuc.edu) |
| `pronto` | PRONTO Benchmark | Multiphase Flow | **Ready** | [Zenodo](https://zenodo.org/records/1341583) |
| `ashrae` | ASHRAE 1043-RP | HVAC/Refrigeration | **Ready** | Research Project |
| `ims` | NASA IMS Bearing | Rotating Machinery | **Ready** | [NASA](https://data.nasa.gov/dataset/ims-bearings) |
| `swat` | SWaT (Secure Water Treatment) | ICS / Cyber-Physical | *Request* | [iTrust](https://itrust.sutd.edu.sg) |
| `co2` | CO2 Refrigeration | HVAC | **Ready** | Proprietary/Internal |

### 2.1 Data Setup
Place datasets in `data/`:
```bash
DualSTAGE/
├── data/
│   ├── tep/raw/           # TEP .RData files
│   ├── pronto/            # PRONTO benchmark files
│   ├── ASHRAE_1043_RP/    # ASHRAE CSV files
│   ├── IMS_Bearing/       # 1st_test, 2nd_test, etc.
│   └── swat/              # SWaT .csv files
```
See `data/README.md` for detailed download instructions.

---

## 3. Training (Dual-View Mode)

To train the **Dual-View Spectral-Temporal** model, use the `--use-spectral-view` flag.

### Example: Training on TEP
```bash
python train_dualstage.py \
    --dataset-key tep \
    --use-spectral-view \
    --freq-embed-dim 16 \
    --freq-band-mix mlp \
    --lambda-div 0.1 \
    --anomaly-weight 0.5 \
    --epochs 20 \
    --batch-size 64 \
    --use-amp
```

### Key Arguments
*   `--use-spectral-view`: Enables the spectral branch.
*   `--freq-embed-dim`: Dimension of frequency embeddings (default: 16).
*   `--freq-band-mix`: Method to mix frequency bins (`none`, `conv`, `mlp`).
*   `--lambda-div`: Weight for the **Structural Divergence Loss** (crucial for incipient detection).
*   `--anomaly-weight`: Weight for the Topology-Aware reconstruction penalty.

---

## 4. Evaluation & Inference

### 4.1 TEP Evaluation
Evaluates precision/recall/F1 across all 20 fault types.
```bash
python evaluate_tep_anomaly.py \
    --checkpoint checkpoints/tep/best_model.pt \
    --dataset-key tep \
    --use-spectral-view \
    --divergence-type js
```

### 4.2 Interactive Plotting
Visualize reconstruction and the **divergence score** trajectory.
```bash
python plot_reconstruction_plotly.py \
    --checkpoint checkpoints/tep/best_model.pt \
    --dataset-key tep \
    --dataset Fault_10 \
    --use-spectral-view \
    --include-all-faults
```
Outputs HTML plots to `outputs/plotly/`.

---

## 5. Repository Structure

*   `dualstage/src/model/dualstage.py`: Core model (Dual-View Architecture).
*   `train_dualstage.py`: Main training loop with divergence loss.
*   `evaluate_tep_anomaly.py`: Evaluation script for TEP.
*   `datasets/`: Data adapters for different benchmarks.
*   `checkpoints/`: Model artifacts.

## 6. Quick Start
1.  Install dependencies: `pip install -r requirements.txt`
2.  Download TEP or PRONTO data (see `data/README.md`).
3.  Run a test: `python test_dualstage_model.py --dataset-key tep`
