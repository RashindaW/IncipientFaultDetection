# Detailed Equation-to-Code Mapping

**Purpose**: Line-by-line mapping from paper equations to actual code implementation

---

## Section 1: Problem Setting

### Paper (Page 1)
```
X ∈ ℝ^{B × N × W}  # Batch of sliding windows
U ∈ ℝ^{B × C × W}  # Context/control variables
```

### Code Implementation
**File**: `dyedgegat.py` lines 705-756

```python
def forward(self, data, return_graph=False):
    x, c, edge_index, batch = data.x, data.c, data.edge_index, data.batch
    
    # x arrives as [B*N, W] (PyG batching convention)
    # c arrives as [B*C, W]
    
    n_nodes = cfg.dataset.n_nodes  # N
    b = x.shape[0] // n_nodes      # B
    x_nodes = x.view(b, n_nodes, -1)  # Reshape to [B, N, W]
```

**Status**: ✅ Correct - PyG uses flattened batching, code correctly reshapes

---

## Section 2: Context Encoder

### Paper Equation (1)
```
h^ctx_b = GRU_ctx(U_b^T) ∈ ℝ^{d_ctx}
```

### Code Implementation
**File**: `dyedgegat.py` lines 543-549, 711-748

```python
# Initialization (lines 543-549):
if self.aug_control:
    self.control_encoder = ENCODER_DICT[contr_encoder_type](
        in_channels=cfg.dataset.ocvar_dim,  # C
        out_channels=temp_node_embed_dim,   # d_ctx
        mode='multivariate',
    )

# Forward pass (lines 711-748):
if self.aug_control:
    # Reshape c from [B*C, W] to [B, C, W]
    c = c.view(b, n_ctrl, -1)
    c_in = c.transpose(1, 2)  # [B, W, C] for GRU
    context = self.control_encoder(c_in)  # [B, d_ctx] ✅ Matches paper
    
    # Expand to nodes
    context_expanded = context.repeat_interleave(n_nodes, dim=0)  # [B*N, d_ctx]
    
    # ❌ PROBLEM: context is computed but NEVER USED!
    pass  # <-- Does nothing with context
```

**Issue**: Context is encoded correctly but not injected into:
- Node encoder (should initialize GRU hidden state)
- Decoder (should initialize decoder hidden state)

**Paper says**: *"injected into downstream components, e.g., as part of node feature initialization or decoder initial state"*

---

## Section 3: Temporal Node Encoder

### Paper Equations (2-3)
```
H^time = Enc_time(X) ∈ ℝ^{B × N × d_time}
h^time_{b,i} = GRU_time(X_{b,i,:}) ∈ ℝ^{d_time}
```

### Code Implementation
**File**: `dyedgegat.py` lines 302-328, 551-556, 750-757

```python
# GRUEncoder class (lines 302-328):
class GRUEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, mode='univariate'):
        self.gru = nn.GRU(in_channels, out_channels, batch_first=True)
    
    def forward(self, x):
        # x: [batch, nodes, window] for univariate
        if self.mode == 'univariate':
            b, n, w = x.shape
            x = x.view(b * n, w, 1)  # [B*N, W, 1]
            out, h = self.gru(x)
            h = h.squeeze(0).view(b, n, -1)  # [B, N, d_time] ✅
            return h

# In DyEdgeGAT.__init__ (lines 551-556):
self.node_encoder = GRUEncoder(
    in_channels=feat_input_node,       # 1 (univariate)
    out_channels=temp_node_embed_dim,  # d_time
    mode='univariate',
)

# In forward (lines 750-757):
x_nodes = x.view(b, n_nodes, -1)     # [B, N, W] ✅
h_temp = self.node_encoder(x_nodes)  # [B, N, d_time] ✅
h_temp = h_temp.view(b * n_nodes, -1)  # Flatten for GNN
```

**Status**: ✅ Correct - Each node's time series processed independently by GRU

---

## Section 4: Spectral Encoder

### Paper Equations (4-6)
```
X^(f)_{b,i,:} = rFFT(X_{b,i,:}) ∈ ℂ^F               (Eq. 4)
m_{b,i,k} = log(1 + |X^(f)_{b,i,k}|)                (Eq. 5) ← KEY
h^freq_{b,i} = φ_freq(m_{b,i,:}) ∈ ℝ^{d_freq}      (Eq. 6)
```

### Code Implementation
**File**: `dyedgegat.py` lines 330-419

```python
class SpectralEncoder(nn.Module):
    def __init__(self, window_size, embed_dim, max_freq_bins, band_mixer):
        full_bins = (window_size // 2) + 1  # F = ⌊W/2⌋ + 1 ✅
        self.n_bins = min(max_freq_bins, full_bins) if max_freq_bins > 0 else full_bins
        
        # Band mixer φ_freq
        if band_mixer == "mlp":
            self.mixer = nn.Sequential(
                nn.Linear(self.n_bins, embed_dim * 2),
                nn.ReLU(),
                nn.Linear(embed_dim * 2, embed_dim)
            )
        else:
            self.mixer = nn.Linear(self.n_bins, embed_dim)
    
    def forward(self, x):
        # x: [B, N, W]
        b, n, w = x.shape
        
        # Step 1: Compute rFFT
        fft_out = torch.fft.rfft(x, dim=-1)  # [B, N, F] complex ✅
        
        # Step 2: Compute magnitude
        mag = torch.abs(fft_out)  # [B, N, F] ✅
        
        # ❌ MISSING STEP: Log transform!
        # Should be: mag = torch.log1p(mag)  # log(1 + mag)
        
        # Step 3: Truncate to F' bins
        if self.n_bins < mag.shape[-1]:
            mag = mag[..., :self.n_bins]  # ✅
        
        # Step 4: Mix bands
        mag_flat = mag.reshape(b * n, -1)  # [B*N, F']
        h_freq = self.mixer(mag_flat)       # [B*N, d_freq]
        h_freq = h_freq.reshape(b, n, -1)   # [B, N, d_freq] ✅
        
        return h_freq
```

**Status**: ❌ **CRITICAL ISSUE** - Missing `log(1 + magnitude)` transform on line 395

**Impact**: 
- Log transform stabilizes training by compressing large magnitude variations
- Emphasizes lower-magnitude frequency components
- Standard practice in frequency-domain processing

**Fix**:
```python
# Add after line 395:
mag = torch.log1p(mag)  # Eq. 5: m = log(1 + |FFT|)
```

---

## Section 5: Dual Graph Construction

### Paper Equations (7-12)

#### Step 1: Project embeddings (Eq. 7)
```
h̃^(l)_{b,i} = W_l h_{b,i},  h̃^(r)_{b,j} = W_r h_{b,j}
```

#### Step 2: Compute attention (Eq. 8)
```
e_{b,ij} = a^T σ(h̃^(l)_{b,i} + h̃^(r)_{b,j})
```

#### Step 3: Top-k selection (Eq. 9)
```
N_{b,i} = arg topk_j e_{b,ij}, |N_{b,i}| = k
```

#### Step 4: Normalize (Eq. 10)
```
α_{b,ij} = exp(e_{b,ij}) / Σ_{j'∈N_{b,i}} exp(e_{b,ij'})
```

### Code Implementation
**File**: `dyedgegat.py` lines 88-255 (FeatureGraph class)

```python
class FeatureGraph(nn.Module):
    def __init__(self, in_channels, embed_dim, n_nodes, topk):
        self.lin_l = nn.Linear(in_channels, embed_dim)  # W_l
        self.lin_r = nn.Linear(in_channels, embed_dim)  # W_r
        self.att = nn.Parameter(torch.Tensor(1, embed_dim))  # a
        self.topk = min(topk, n_nodes)
    
    def forward(self, x, edge_index, batch):
        # x: [B, N, in_channels]
        b, n, _ = x.shape
        
        # Eq. 7: Project
        x_l = self.lin_l(x).reshape(-1, self.out_channels)  # ✅
        x_r = self.lin_r(x).reshape(-1, self.out_channels)  # ✅
        x_l = x_l.view(b, n, -1)
        x_r = x_r.view(b, n, -1)
        
        # Eq. 8: Compute attention (broadcasting for all pairs)
        x_cat = x_l.unsqueeze(2) + x_r.unsqueeze(1)  # [B, N, N, d] ✅
        x_cat = F.leaky_relu(x_cat, 0.2)              # σ(·) ✅
        alpha = (x_cat * self.att).sum(dim=-1)        # a^T (...) ✅
        
        # Eq. 9: Top-k selection
        k = min(self.topk, n)
        attention, indices = torch.topk(alpha, k, dim=-1)  # ✅ [B, N, k]
        
        # Eq. 10: Softmax normalization
        attention = F.softmax(attention, dim=-1)  # ✅
        
        # Construct sparse edge_index and edge_attr
        # [Construction logic for PyG format...]
        
        return new_edge_index, attention
```

**Status**: ✅ Correct implementation of attention mechanism

**Minor Note**: Line 252 calls `to_undirected()` which may symmetrize the graph. The paper describes directed graphs, but symmetrization can improve stability.

---

## Section 6: Dual GNN Streams

### Paper Equations (13-14)
```
Z^(ℓ)_time = GNN^(ℓ)_time(Z^(ℓ-1)_time, A_time)    (Eq. 13)

z^(ℓ)_{time,b,i} = MLP^(ℓ)((1+ε^(ℓ))z^(ℓ-1) + Σ_j A^time_{b,ij} z^(ℓ-1)_{b,j})    (Eq. 14)
```

### Code Implementation
**File**: `dyedgegat.py` lines 594-627, 804-823

```python
# Initialization (lines 594-627):
self.gnn_layers = nn.ModuleList()
for i in range(num_gnn_layers):
    in_dim = temp_node_embed_dim if i == 0 else gnn_embed_dim
    if gnn_type == 'gin':
        mlp = nn.Sequential(
            nn.Linear(in_dim, gnn_embed_dim),
            nn.ReLU(),
            nn.Linear(gnn_embed_dim, gnn_embed_dim)
        )
        self.gnn_layers.append(GINEConv(mlp, edge_dim=1))  # ✅

# Forward pass (lines 804-823):
# Temporal GNN
z_temp = h_temp  # Z^(0) = H^time ✅
for conv in self.gnn_layers:
    z_temp = conv(z_temp, adj_temp, attn_temp)  # ✅ Eq. 14
    z_temp = F.relu(z_temp)

# Spectral GNN (similar)
z_freq = h_freq  # Z^(0) = H^freq ✅
for conv in self.gnn_layers_freq:
    z_freq = conv(z_freq, adj_freq, attn_freq)  # ✅
    z_freq = F.relu(z_freq)
```

**Status**: ✅ Correct - Using GINEConv (GIN with edge attributes)

**Note**: PyG's `GINEConv` implements:
```
h^(ℓ) = MLP^(ℓ)((1 + ε)·h^(ℓ-1) + Σ_j e_{ij}·h^(ℓ-1)_j)
```
where `e_{ij}` are edge features (attention weights). This matches Eq. 14.

---

## Section 7: View Fusion

### Paper Equations (16-19)

#### Concatenation (Eq. 16-17)
```
Z̃_{b,i} = [Z_time,b,i || Z_freq,b,i] ∈ ℝ^{2d_g}
Z_{b,i} = W_fuse Z̃_{b,i}
```

#### Gated (Eq. 18-19)
```
g_{b,i} = σ(w_g^T [Z_time,b,i || Z_freq,b,i])
Z_{b,i} = g_{b,i} Z_time,b,i + (1-g_{b,i}) Z_freq,b,i
```

### Code Implementation
**File**: `dyedgegat.py` lines 629-642, 840-848

```python
# Initialization (lines 629-642):
if self.use_spectral_view:
    if fuse_mode == "concat":
        fusion_dim = gnn_embed_dim * 2
        self.fusion_layer = nn.Linear(fusion_dim, gnn_embed_dim)  # W_fuse ✅
    elif fuse_mode == "gated":
        self.gate = nn.Sequential(
            nn.Linear(gnn_embed_dim * 2, 1),  # w_g ✅
            nn.Sigmoid()
        )

# Forward (lines 840-848):
if self.fuse_mode == "concat":
    z_cat = torch.cat([z_temp, z_freq], dim=-1)  # ✅ Eq. 16
    z_fused = self.fusion_layer(z_cat)           # ✅ Eq. 17
elif self.fuse_mode == "gated":
    z_cat = torch.cat([z_temp, z_freq], dim=-1)
    g = self.gate(z_cat)                         # ✅ Eq. 18
    z_fused = g * z_temp + (1-g) * z_freq       # ✅ Eq. 19
```

**Status**: ✅ Perfect match to paper

---

## Section 8: Cross-View Structural Divergence

### Paper Equations (20-23)

#### Dense distributions (Eq. 20-21)
```
P̂_b(i,j) = (P_b)_{ij} / Σ_j' (P_b)_{ij'}
p_b = vec(P̂_b), q_b = vec(Q̂_b)
```

#### Jensen-Shannon divergence (Eq. 22-23)
```
m_b = 1/2(p_b + q_b)
D_JS(p_b || q_b) = 1/2 D_KL(p_b || m_b) + 1/2 D_KL(q_b || m_b)
L_div = 1/B Σ_b D_JS(p_b || q_b)
```

### Code Implementation
**File**: `dyedgegat.py` lines 885-919

```python
# Dense attention conversion (lines 885-901):
def _dense_attn(self, edge_index, attn, num_graphs, n_nodes):
    attn = attn.view(-1)
    src = edge_index[0]
    dst = edge_index[1]
    g = src // n_nodes
    src_local = src % n_nodes
    dst_local = dst % n_nodes
    
    # Build dense matrix
    dense = torch.zeros((num_graphs, n_nodes, n_nodes), device=device)
    dense[g, src_local, dst_local] = attn  # Populate sparse → dense
    
    # Row-normalize (Eq. 20-21)
    row_sum = dense.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    dense = dense / row_sum  # ✅ P̂_b
    return dense

# JS divergence (lines 903-919):
def _js_divergence(self, temp_dense, freq_dense, eps=1e-8):
    P = temp_dense.clamp_min(eps)
    Q = freq_dense.clamp_min(eps)
    
    # Vectorize and normalize
    P = P.view(P.size(0), -1)  # vec(P̂_b) ✅
    Q = Q.view(Q.size(0), -1)  # vec(Q̂_b) ✅
    P = P / P.sum(dim=1, keepdim=True).clamp_min(eps)
    Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(eps)
    
    # Compute JS
    M = 0.5 * (P + Q)                        # ✅ Eq. 22
    kl_PM = (P * (P / M).log()).sum(dim=1)  # D_KL(P||M)
    kl_QM = (Q * (Q / M).log()).sum(dim=1)  # D_KL(Q||M)
    js = 0.5 * (kl_PM + kl_QM)               # ✅ Eq. 23
    return js

# In forward (lines 794-802):
dense_temp = self._dense_attn(adj_temp, attn_temp, b_graphs, n_nodes)
dense_freq = self._dense_attn(adj_freq, attn_freq, b_graphs, n_nodes)
js_per_graph = self._js_divergence(dense_temp, dense_freq)
div_loss = js_per_graph.mean()  # ✅ L_div = mean(JS) over batch
```

**Status**: ✅ Excellent implementation - matches paper exactly

---

## Section 9: Reconstruction Decoder

### Paper Equations (24-25)
```
x̃_{b,i,:} = Dec(z_{b,i}) ∈ ℝ^W           (Eq. 24)
X̂_{b,i,t} = x̃_{b,i, W-t+1}, t=1,...,W   (Eq. 25) ← REVERSE ORDER
```

### Code Implementation
**File**: `dyedgegat.py` lines 422-470, 850-857

```python
class ReconstructionModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_layers):
        self.rnn = nn.GRU(in_channels, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, out_channels)
    
    def reconstruct(self, z, window_size):
        # z: [B, N, d_z]
        b, n, d = z.shape
        z_flat = z.view(b * n, d)
        
        # Repeat embedding for each timestep
        z_rep = z_flat.unsqueeze(1).repeat(1, window_size, 1)  # [B*N, W, d]
        
        # GRU decoder
        out, _ = self.rnn(z_rep)  # [B*N, W, hidden]
        recon = self.out(out)     # [B*N, W, 1]
        recon = recon.squeeze(-1) # [B*N, W]
        
        # Eq. 25: Reverse order ✅
        return torch.flip(recon, dims=[1])

# In DyEdgeGAT.forward (lines 850-857):
z_fused_nodes = z_fused.view(b, n, -1)
recon = self.decoder.reconstruct(z_fused_nodes, cfg.dataset.window_size)
```

**Status**: ✅ Reversal is implemented

**Issue**: Paper mentions decoder could be initialized with context `h^ctx`, but this is not implemented:
```python
# Missing:
out, _ = self.rnn(z_rep, h0=context)
```

---

## Section 10: Training Objective

### Paper Equations (27-30)

#### Reconstruction loss (Eq. 27)
```
L_rec = 1/(BNW) Σ_b Σ_i Σ_t (X_{b,i,t} - X̂_{b,i,t})^2
```

#### Topology-aware loss (Eq. 28-29)
```
ε_{b,i} = 1/W Σ_t (X_{b,i,t} - X̂_{b,i,t})^2
s_{b,ij} = 1/2(ε_{b,i} + ε_{b,j})·|A^time_{b,ij}|
L_topo = 1/B Σ_b 1/|E^(b)| Σ_{(i,j)∈E^(b)} s_{b,ij}
```

#### Total loss (Eq. 30)
```
L_total = L_rec + λ_topo L_topo + λ_div L_div
```

### Code Implementation

#### Reconstruction loss
**File**: `train_dyedgegat.py` lines 467

```python
recon_loss = criterion(recon, target)  # MSE ✅ Eq. 27
```

#### Topology-aware loss
**File**: `dyedgegat.py` lines 921-964

```python
def _topology_scores_per_graph(self, x_true, x_recon, edge_index, edge_weight):
    # Eq. 28: Per-node error
    node_err = ((x_true - x_recon) ** 2).mean(dim=-1)  # ε_{b,i} ✅
    
    n = cfg.dataset.n_nodes
    b = max(int(node_err.numel() // n), 1)
    
    src = edge_index[0]
    dst = edge_index[1]
    weights = edge_weight.view(-1).abs()
    graph_ids = src // n
    
    # Eq. 29: Edge-level score
    edge_scores = 0.5 * (node_err[src] + node_err[dst]) * weights  # ✅
    
    # Average per graph
    graph_edge_sum = torch.zeros(b, device=device)
    graph_edge_cnt = torch.zeros(b, device=device)
    graph_edge_sum.index_add_(0, graph_ids, edge_scores)
    graph_edge_cnt.index_add_(0, graph_ids, torch.ones_like(edge_scores))
    
    return graph_edge_sum / graph_edge_cnt.clamp_min(1.0)  # ✅ L_topo per graph
```

#### Total loss
**File**: `train_dyedgegat.py` lines 467-480

```python
recon_loss = criterion(recon, target)  # L_rec
anom_score = base_model.compute_topology_aware_anomaly_score(...)  # L_topo
div_loss = aux.get("divergence_loss", ...)  # L_div

# Eq. 30:
loss = recon_loss + cfg.anomaly_weight * anom_score + div_weight * div_loss
#      ^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^            ^^^^^^^^^^^^^^^^^^^^^
#      L_rec         λ_topo * L_topo                 λ_div * L_div
```

**Status**: ✅ All components correctly implemented

**Note**: Variable names differ from paper:
- `cfg.anomaly_weight` ≡ λ_topo
- `div_weight` (from `cfg.lambda_div`) ≡ λ_div

---

## Section 11: Inference and Anomaly Scoring

### Paper Equations (31-33)
```
S_rec(b) = 1/(NW) Σ_i Σ_t (X - X̂)^2     (Eq. 31)
S_div(b) = D_JS(p_b || q_b)              (Eq. 32)
S(b) = S_rec(b) + β S_div(b)             (Eq. 33)
```

### Code Implementation
**File**: `dyedgegat.py` lines 966-1009

```python
# S_topo (topology-aware variant of S_rec)
def compute_topology_aware_anomaly_score(self, x_true, x_recon, edge_index, edge_weight):
    graph_scores = self._topology_scores_per_graph(...)
    return graph_scores.mean()

# S_rec per sample
def compute_anomaly_scores_per_sample(self, x_true, x_recon, edge_index, edge_weight):
    return self._topology_scores_per_graph(...)  # [batch] ✅

# S_rec per timestep
def compute_anomaly_scores_per_timestep(self, x_true, x_recon, edge_index, edge_weight):
    x_true, x_recon = self._align_target_and_recon(x_true, x_recon)
    n = cfg.dataset.n_nodes
    b = max(int(x_true.shape[0] // n), 1)
    x_true = x_true.view(b, n, -1)
    x_recon = x_recon.view(b, n, -1)
    per_timestep = ((x_true - x_recon) ** 2).mean(dim=1)  # [B, W] ✅
    return per_timestep
```

**Status**: ✅ Functionality available

**Note**: The combined score S(b) = S_rec + β·S_div is computed in evaluation/test scripts by combining:
- `compute_anomaly_scores_per_sample()` → S_rec or S_topo
- `aux["divergence_score"]` → S_div

---

## Summary of All Issues

### ❌ Issue 1: Missing Log Transform (CRITICAL)
**Location**: `dyedgegat.py` line 395  
**Paper**: Eq. 5 - `m = log(1 + |FFT|)`  
**Code**: Uses raw magnitude  
**Fix**:
```python
mag = torch.abs(fft_out)
mag = torch.log1p(mag)  # ADD THIS LINE
```

### ❌ Issue 2: Context Not Injected (MODERATE)
**Location**: `dyedgegat.py` lines 726-748  
**Paper**: Section 2 - Context should initialize GRU  
**Code**: Context computed but unused  
**Fix**: Modify `GRUEncoder` to accept `h0`:
```python
class GRUEncoder(nn.Module):
    def forward(self, x, h0=None):
        # ...
        out, h = self.gru(x, h0)
```

### ⚠️ Issue 3: Context Not in Decoder (MINOR)
**Location**: `dyedgegat.py` line 455  
**Paper**: Section 9 - Decoder can be conditioned on context  
**Code**: Decoder doesn't accept context  
**Fix**: Pass context to decoder RNN:
```python
def reconstruct(self, z, window_size, h0=None):
    # ...
    out, _ = self.rnn(z_rep, h0)
```

---

## Conclusion

**Implementation Accuracy**: ~85%

**Critical Path**:
1. ✅ Data format correct
2. ⚠️ Context encoder works but not used
3. ✅ Temporal encoder correct
4. ❌ **Spectral encoder missing log transform**
5. ✅ Graph construction correct
6. ✅ GNN layers correct
7. ✅ Fusion correct
8. ✅ Divergence computation correct
9. ✅ Decoder reversal correct (but missing context)
10. ✅ Loss formulation correct

**Most Important Fix**: Add log transform in spectral encoder (1 line of code, significant impact)


