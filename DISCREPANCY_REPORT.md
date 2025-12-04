# Discrepancy Analysis: DyEdgeGAT Code vs. Algorithm Description

This document outlines the discrepancies found between the provided algorithm description ("Dual-View Context-Aware Spectral--Temporal Graph Learning") and the current codebase implementation (`dyedgegat/src/model/dyedgegat.py`).

## 1. Context Encoder is Disconnected
**Algorithm Section 2 & 9:**
The algorithm states that the context embedding $h^{ctx}$ is "broadcast to all nodes... and can be injected into downstream components, e.g., as part of node feature initialization or decoder initial state."

**Code Implementation (`DyEdgeGAT.forward`):**
- The context embedding is correctly computed and broadcasted (lines 726-732):
  ```python
  context = self.control_encoder(c_in)
  context_expanded = context.repeat_interleave(n_nodes, dim=0)
  ```
- **Discrepancy:** The `context_expanded` variable is **never used**. It is not passed to `self.node_encoder`, nor is it passed to `self.decoder`. The code explicitly contains a `pass` statement where injection should happen (line 748).

## 2. Spectral Encoder Missing Log-Transform
**Algorithm Section 4.1:**
The algorithm specifies a log-magnitude transformation for the frequency domain:
\[ m_{b,i,k} = \log(1 + |X^{(f)}_{b,i,k}|) \]

**Code Implementation (`SpectralEncoder.forward`):**
- The code computes the magnitude but skips the log-transform (lines 395-399):
  ```python
  mag = torch.abs(fft_out)
  if self.n_bins < mag.shape[-1]:
      mag = mag[..., :self.n_bins]
  # Missing: mag = torch.log(1 + mag)
  ```
- **Discrepancy:** The input to the band mixer is the raw magnitude, not the log-magnitude. This may make the model sensitive to scaling differences in frequency components.

## 3. Anomaly Scoring Logic (Inference)
**Algorithm Section 11.3:**
The final anomaly score is defined as a weighted combination:
\[ S(b) = S_{rec}(b) + \beta \, S_{div}(b) \]

**Code Implementation (`evaluate` in `train_dyedgegat.py` & `DyEdgeGAT`):**
- The evaluation loop calculates the score using `compute_topology_aware_anomaly_score` (which is $S_{rec\_topo}$).
- It calculates `div_loss` ($S_{div}$) separately.
- **Discrepancy:** The returned `anomaly_score` (used for AUC/F1 metrics) **does not include the divergence term**. It essentially sets $\beta = 0$ during inference, even though the algorithm implies $\beta$ influences the final detection score.

## 4. Decoder Context Conditioning
**Algorithm Section 9:**
"Decode to reconstruct windows... (optionally conditioned on $h^{ctx}$)"

**Code Implementation (`ReconstructionModel`):**
- The `ReconstructionModel` and its `reconstruct` method do not accept or utilize the context vector `c` (line 446).

## 5. Topology-Aware Loss Normalization
**Algorithm Section 10.2:**
The edge-level score is defined as:
\[ s_{b,ij} = \frac{1}{2} (\varepsilon_{b,i} + \varepsilon_{b,j}) \cdot |A_{b,ij}| \]
And the loss averages this:
\[ \mathcal{L}_{topo} = \frac{1}{B} \sum \frac{1}{|E|} \sum s_{b,ij} \]

**Code Implementation (`_topology_scores_per_graph`):**
- The code calculates `graph_edge_sum / graph_edge_cnt` (line 964), where `graph_edge_cnt` is the sum of weights (or count of edges).
- **Minor Check:** The code uses `graph_edge_cnt.index_add_(..., torch.ones_like(edge_scores))`. This computes the *mean* over the number of edges present. This matches the algorithm's $1/|E|$ term. This part seems correct, but worth noting that if $|E|$ varies significantly, the weighting per batch might fluctuate.

---

## Recommendations for Fixes

1.  **Connect Context:**
    - Modify `GRUEncoder` to accept an initial state `h_0` (the context).
    - Pass `context_expanded` as the initial state to `self.node_encoder` and potentially `self.decoder`.

2.  **Add Log-Transform:**
    - Add `mag = torch.log(1 + mag)` in `SpectralEncoder.forward`.

3.  **Update Anomaly Score:**
    - In `evaluate` (and `train_dyedgegat.py`), compute the final score as `score = recon_score + beta * div_score` before saving/plotting. Note that `beta` might need to be a new hyperparameter for inference (or reuse `lambda_div`).


