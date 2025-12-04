# Code vs Paper Discrepancy Analysis

**Date**: December 4, 2025  
**Model**: Dual-View Context-Aware Spectral-Temporal Graph Learning for Divergence-Enhanced Fault Detection

## Executive Summary

This document provides a detailed comparison between the mathematical formulation in your LaTeX paper and the actual implementation in the DyEdge codebase. The analysis identifies both aligned implementations and discrepancies.

---

## 1. Problem Setting ✅ (ALIGNED)

### Paper Specification
- Input: `X ∈ ℝ^{B × N × W}` (batch of windows)
- Context: `U ∈ ℝ^{B × C × W}` (optional control variables)

### Code Implementation
```python
# dyedgegat.py line 706
x, c, edge_index, batch = data.x, data.c, data.edge_index, data.batch
# x: [B*N, W] (flattened), c: [B*C, W] (control variables)
```

**Status**: ✅ Shapes match after reshaping

---

## 2. Context Encoder ⚠️ (PARTIAL MISMATCH)

### Paper Specification (Eq. 1)
```
h^ctx_b = GRU_ctx(U_b^T) ∈ ℝ^{d_ctx}
```
- Context should be encoded via GRU
- Should be used to initialize decoder or augment node features

### Code Implementation
```python
# Lines 711-748
if self.aug_control:
    context = self.control_encoder(c_in)  # [B, Dim]
    context_expanded = context.repeat_interleave(n_nodes, dim=0)  # [B*N, Dim]
    # BUT: context is computed but NOT used!
```

**❌ DISCREPANCY 1: Context Embedding Not Injected**
- The code computes `context` but doesn't inject it into:
  - Node encoder initial state
  - Decoder initial state  
  - Node features

The paper states: *"The context embedding is broadcast to all nodes within the same window and can be injected into downstream components, e.g., as part of node feature initialization or decoder initial state."*

**Recommendation**: 
```python
# Should modify GRUEncoder to accept h0:
h_temp = self.node_encoder(x_nodes, h0=context)
# OR concatenate with node embeddings
```

---

## 3. Temporal Node Encoder ✅ (ALIGNED)

### Paper Specification (Eq. 2-3)
```
H^time = Enc_time(X) ∈ ℝ^{B × N × d_time}
h^time_{b,i} = GRU_time(X_{b,i,:})
```

### Code Implementation
```python
# Lines 551-556, 750-757
self.node_encoder = GRUEncoder(
    in_channels=feat_input_node,
    out_channels=temp_node_embed_dim,
    mode='univariate',
)
# Forward:
x_nodes = x.view(b, n_nodes, -1)
h_temp = self.node_encoder(x_nodes)  # [B, N, Dim]
```

**Status**: ✅ Correct - GRU processes each node's window independently

---

## 4. Spectral Encoder ❌ (MAJOR DISCREPANCY)

### Paper Specification (Eq. 4-6)
```
X^(f)_{b,i,:} = rFFT(X_{b,i,:}) ∈ ℂ^F
m_{b,i,k} = log(1 + |X^(f)_{b,i,k}|)  ← LOG TRANSFORM
h^freq_{b,i} = φ_freq(m_{b,i,:})
```

### Code Implementation
```python
# Lines 389-395
fft_out = torch.fft.rfft(x, dim=-1)  # ✅ Correct
mag = torch.abs(fft_out)             # ✅ Correct magnitude
# ❌ MISSING: log(1 + mag)
```

**❌ DISCREPANCY 2: Missing Log Transform**

The paper explicitly states (Eq. 5):
```
m_{b,i,k} = log(1 + |X^(f)_{b,i,k}|)
```

But the code directly uses the magnitude without log transformation.

**Recommendation**:
```python
# Add after line 395:
mag = torch.log1p(mag)  # log(1 + mag)
```

---

## 5. Graph Construction ⚠️ (IMPLEMENTATION CONCERNS)

### Paper Specification (Eq. 7-12)
```
e_{b,ij} = a^T σ(W_l h_{b,i} + W_r h_{b,j})
N_{b,i} = arg topk_j e_{b,ij}
α_{b,ij} = exp(e_{b,ij}) / Σ_{j' ∈ N_{b,i}} exp(e_{b,ij'})
```

### Code Implementation
```python
# Lines 88-255 (FeatureGraph class)
x_cat = x_l.unsqueeze(2) + x_r.unsqueeze(1)  # ✅ Correct addition
x_cat = F.leaky_relu(x_cat, 0.2)             # ✅ Activation
alpha = (x_cat * self.att).sum(dim=-1)       # ✅ Attention scores
attention, indices = torch.topk(alpha, k, dim=-1)  # ✅ Top-k
attention = F.softmax(attention, dim=-1)     # ✅ Normalization
```

**Status**: ✅ Mostly correct

**⚠️ CONCERN**: The paper mentions using LeakyReLU (confirmed in code), but edge symmetrization via `to_undirected()` (line 252) may not match the paper's directed graph description.

---

## 6. Dual GNN Streams ⚠️ (IMPLEMENTATION DETAILS)

### Paper Specification (Eq. 13-15)
```
z^(ℓ)_time,b,i = MLP^(ℓ)((1 + ε^(ℓ)) z^(ℓ-1) + Σ A^time_{b,ij} z^(ℓ-1)_{b,j})
```
- GIN-style with learnable ε

### Code Implementation
```python
# Lines 594-627
if gnn_type == 'gin':
    mlp = nn.Sequential(
        nn.Linear(in_dim, gnn_embed_dim),
        nn.ReLU(),
        nn.Linear(gnn_embed_dim, gnn_embed_dim)
    )
    self.gnn_layers.append(GINEConv(mlp, edge_dim=1))  # ✅ GIN
```

**Status**: ✅ Using GINEConv (Graph Isomorphism Network with Edge attributes)

**⚠️ NOTE**: PyG's `GINEConv` includes edge attributes in aggregation, which aligns with using attention weights as edge features.

---

## 7. View Fusion ✅ (ALIGNED)

### Paper Specification (Eq. 16-19)
- Concatenation: `Z = W_fuse [Z_time || Z_freq]`
- Gated: `Z = g·Z_time + (1-g)·Z_freq`

### Code Implementation
```python
# Lines 840-848
if self.fuse_mode == "concat":
    z_cat = torch.cat([z_temp, z_freq], dim=-1)
    z_fused = self.fusion_layer(z_cat)
elif self.fuse_mode == "gated":
    z_cat = torch.cat([z_temp, z_freq], dim=-1)
    g = self.gate(z_cat)
    z_fused = g * z_temp + (1-g) * z_freq
```

**Status**: ✅ Correct implementation

---

## 8. Cross-View Structural Divergence ✅ (ALIGNED)

### Paper Specification (Eq. 20-23)
```
D_JS(p_b || q_b) = 0.5 D_KL(p_b || m_b) + 0.5 D_KL(q_b || m_b)
where m_b = 0.5(p_b + q_b)
```

### Code Implementation
```python
# Lines 903-919
def _js_divergence(self, temp_dense, freq_dense, eps=1e-8):
    P = temp_dense.clamp_min(eps)
    Q = freq_dense.clamp_min(eps)
    P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
    Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
    M = 0.5 * (P + Q)
    kl_PM = (P * (P / M).log()).sum(dim=1)
    kl_QM = (Q * (Q / M).log()).sum(dim=1)
    js = 0.5 * (kl_PM + kl_QM)
    return js
```

**Status**: ✅ Correct Jensen-Shannon divergence implementation

**✅ Dense Attention Conversion** (lines 885-901): Also correctly converts sparse attention to dense distributions and normalizes rows.

---

## 9. Reconstruction Decoder ⚠️ (IMPLEMENTATION CONCERN)

### Paper Specification (Eq. 24-25)
```
x̃_{b,i,:} = Dec(z_{b,i}) ∈ ℝ^W
X̂_{b,i,t} = x̃_{b,i, W - t + 1}  ← REVERSE ORDER
```
- Should decode from fused embedding
- Output in reverse chronological order

### Code Implementation
```python
# Lines 446-470 (ReconstructionModel)
def reconstruct(self, z, window_size):
    z_rep = z_flat.unsqueeze(1).repeat(1, window_size, 1)  # Repeat z
    out, _ = self.rnn(z_rep)
    recon = self.out(out)
    recon = recon.squeeze(-1)
    return torch.flip(recon, dims=[1])  # ✅ Reversed!
```

**Status**: ✅ Reversal is implemented

**⚠️ CONCERN**: The paper mentions context could initialize decoder hidden state, but this is not implemented:
```python
# Missing: context injection into decoder
# Should be: out, _ = self.rnn(z_rep, h0=context)
```

---

## 10. Training Objective ⚠️ (PARTIAL MISMATCH)

### Paper Specification (Eq. 27-30)
```
L_total = L_rec + λ_topo L_topo + λ_div L_div
```

Where:
- `L_rec`: MSE reconstruction loss (Eq. 27)
- `L_topo`: Topology-aware reconstruction (Eq. 29)
- `L_div`: JS divergence between graphs (Eq. 23)

### Code Implementation
```python
# train_dyedgegat.py lines 467-480
recon_loss = criterion(recon, target)  # ✅ L_rec
anom_score = base_model.compute_topology_aware_anomaly_score(...)  # ✅ L_topo
div_loss = aux.get("divergence_loss", ...)  # ✅ L_div
loss = recon_loss + cfg.anomaly_weight * anom_score + div_weight * div_loss
```

**Status**: ✅ All three components are present

**Variable Name Mismatch**:
- Code uses: `anomaly_weight` for λ_topo, `lambda_div` for λ_div
- Paper uses: `λ_topo`, `λ_div`

**Topology-Aware Loss Implementation** (lines 921-964):
```python
# Paper Eq. 28-29:
# s_{b,ij} = 0.5(ε_{b,i} + ε_{b,j}) · |A^time_{b,ij}|
node_err = ((x_true - x_recon) ** 2).mean(dim=-1)  # ✅ ε_{b,i}
edge_scores = 0.5 * (node_err[src] + node_err[dst]) * weights  # ✅ Correct
```

**Status**: ✅ Correct implementation

---

## 11. Inference and Anomaly Scoring ✅ (ALIGNED)

### Paper Specification (Eq. 31-33)
```
S_rec(b) = mean reconstruction error
S_div(b) = D_JS(p_b || q_b)
S(b) = S_rec(b) + β S_div(b)
```

### Code Implementation
```python
# Lines 980-1009 provide:
compute_anomaly_scores_per_sample()  # ✅ Returns topology-aware scores per graph
compute_anomaly_scores_per_timestep()  # ✅ Returns per-timestep MSE
# Forward returns divergence_score in aux dict  # ✅ S_div available
```

**Status**: ✅ Functionality is available (combination handled in evaluation)

---

## Summary of Discrepancies

### ❌ Critical Issues

1. **Missing Log Transform in Spectral Encoder** (Section 4)
   - Paper Eq. 5: `m = log(1 + |FFT|)`
   - Code: Uses raw magnitude
   - **Impact**: Significant - log transform stabilizes learning and emphasizes smaller frequency components

2. **Context Embedding Not Injected** (Section 2)
   - Paper: Context should initialize GRU or augment features
   - Code: Context is computed but unused
   - **Impact**: Moderate - context information is lost

### ⚠️ Minor Concerns

3. **Context Not Passed to Decoder** (Section 9)
   - Paper mentions decoder can be conditioned on context
   - Code: Decoder doesn't accept context
   - **Impact**: Low - reconstruction may miss context-dependent patterns

4. **Graph Symmetrization** (Section 5)
   - Paper describes directed graphs
   - Code: `to_undirected()` is called conditionally
   - **Impact**: Low - may affect divergence interpretation

### ✅ Correctly Implemented

- Temporal GRU encoder
- Top-k graph construction with attention
- GIN layers with edge weights
- View fusion (concat/gated)
- JS divergence computation
- Dense attention conversion
- Topology-aware loss
- Output reversal in decoder
- Training loss combination

---

## Recommendations

### Priority 1 (Critical)
```python
# 1. Add log transform to SpectralEncoder (line 395)
mag = torch.abs(fft_out)
mag = torch.log1p(mag)  # Add this line
```

### Priority 2 (Moderate)
```python
# 2. Inject context into node encoder
# Modify GRUEncoder to accept h0 parameter
class GRUEncoder(nn.Module):
    def forward(self, x, h0=None):
        if self.mode == 'univariate':
            # ... reshape logic
            out, h = self.gru(x, h0)  # Pass h0
```

### Priority 3 (Nice to have)
```python
# 3. Pass context to decoder
# Modify ReconstructionModel.reconstruct()
def reconstruct(self, z, window_size, h0=None):
    # ... 
    out, _ = self.rnn(z_rep, h0)
```

---

## Conclusion

Your implementation is **largely faithful** to the paper, with the main concern being the **missing log transformation** in the spectral encoder. The context injection mechanism is also incomplete but has lower impact. The core components (dual-view learning, graph construction, divergence computation, topology-aware loss) are correctly implemented.

**Estimated Implementation Fidelity**: ~85%
- Core architecture: ✅ 95%
- Spectral processing: ❌ 60%
- Context handling: ⚠️ 40%
- Training/inference: ✅ 100%


