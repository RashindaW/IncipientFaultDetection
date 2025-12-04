# Quick Fix Guide: Code vs Paper Discrepancies

## TL;DR - What Needs Fixing?

### 🔴 CRITICAL (Must Fix)
**Issue**: Missing log transform in spectral encoder  
**Impact**: Spectral branch learns poorly, divergence signal unreliable  
**Effort**: 1 line of code

### 🟡 MODERATE (Should Fix)
**Issue**: Context embedding not injected into node encoder  
**Impact**: Operating condition information lost  
**Effort**: ~10 lines of code

### 🟢 MINOR (Nice to Have)
**Issue**: Context not passed to decoder  
**Impact**: Minimal if context already in fused embeddings  
**Effort**: ~5 lines of code

---

## Fix #1: Add Log Transform to Spectral Encoder (CRITICAL)

### What the Paper Says (Eq. 5)
```
m_{b,i,k} = log(1 + |X^(f)_{b,i,k}|)
```

### Current Code
**File**: `dyedgegat/src/model/dyedgegat.py`  
**Lines**: 389-400

```python
def forward(self, x):
    b, n, w = x.shape
    
    # 1. Compute rFFT
    fft_out = torch.fft.rfft(x, dim=-1)
    
    # 2. Compute Magnitude
    mag = torch.abs(fft_out)
    
    # ❌ MISSING: Log transform should be here
    
    # 3. Truncate/Select bins
    if self.n_bins < mag.shape[-1]:
        mag = mag[..., :self.n_bins]
```

### ✅ Fixed Code
```python
def forward(self, x):
    b, n, w = x.shape
    
    # 1. Compute rFFT
    fft_out = torch.fft.rfft(x, dim=-1)
    
    # 2. Compute Magnitude
    mag = torch.abs(fft_out)
    
    # ✅ ADD THIS LINE: Log transform (Eq. 5)
    mag = torch.log1p(mag)  # log(1 + mag)
    
    # 3. Truncate/Select bins
    if self.n_bins < mag.shape[-1]:
        mag = mag[..., :self.n_bins]
```

### Why This Matters
1. **Numerical Stability**: Raw FFT magnitudes can vary by orders of magnitude
2. **Feature Scale**: Log compresses dynamic range, helps neural network learning
3. **Low-Frequency Emphasis**: Log transform gives more weight to smaller components
4. **Standard Practice**: Nearly all frequency-domain ML papers use log transform

### Testing the Fix
```python
# Before fix: mag ranges from 0 to 1000+
# After fix: mag ranges from 0 to ~7 (log scale)

# You should see:
# 1. Faster convergence during training
# 2. Lower divergence loss values
# 3. Better anomaly detection performance
```

---

## Fix #2: Inject Context into Node Encoder (MODERATE)

### What the Paper Says (Section 2)
> "The context embedding is broadcast to all nodes within the same window and can be injected into downstream components, e.g., as part of node feature initialization or **decoder initial state**."

### Current Code
**File**: `dyedgegat/src/model/dyedgegat.py`  
**Lines**: 726-748

```python
if self.aug_control:
    context = self.control_encoder(c_in)  # [B, Dim]
    context_expanded = context.repeat_interleave(n_nodes, dim=0)  # [B*N, Dim]
    
    # ❌ PROBLEM: context is computed but never used!
    pass

# Later:
h_temp = self.node_encoder(x_nodes)  # ❌ No context passed
```

### ✅ Fix Option A: Modify GRUEncoder to Accept h0

**Step 1**: Update `GRUEncoder` class (lines 302-328)

```python
class GRUEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func=None, mode='univariate'):
        super().__init__()
        self.mode = mode
        if mode == 'univariate':
            self.gru = nn.GRU(in_channels, out_channels, batch_first=True)
        elif mode == 'multivariate':
            self.gru = nn.GRU(in_channels, out_channels, batch_first=True)
        self.norm = norm_func(out_channels) if norm_func else nn.Identity()
    
    def forward(self, x, h0=None):  # ✅ Add h0 parameter
        if self.mode == 'univariate':
            b, n, w = x.shape
            x = x.view(b * n, w, 1)
            
            # ✅ Prepare h0 if provided
            if h0 is not None:
                h0 = h0.unsqueeze(0)  # [1, B*N, Dim] for GRU
            
            out, h = self.gru(x, h0)  # ✅ Pass h0
            h = h.squeeze(0).view(b, n, -1)
            return h
        elif self.mode == 'multivariate':
            if h0 is not None:
                h0 = h0.unsqueeze(0)
            out, h = self.gru(x, h0)
            return h.squeeze(0)
```

**Step 2**: Update forward pass to use context (lines 750-757)

```python
# 2. Encode Nodes (X) -> H_temp
x_nodes = x.view(b, n_nodes, -1)

# ✅ Pass context as initial hidden state
if self.aug_control and context_expanded is not None:
    h_temp = self.node_encoder(x_nodes, h0=context_expanded)
else:
    h_temp = self.node_encoder(x_nodes)

h_temp = torch.nan_to_num(h_temp, nan=0.0, posinf=0.0, neginf=0.0)
h_temp = h_temp.view(b * n_nodes, -1)
```

### ✅ Fix Option B: Concatenate Context with Node Embeddings (Simpler)

```python
# After computing h_temp (line 756):
h_temp = self.node_encoder(x_nodes)
h_temp = torch.nan_to_num(h_temp, nan=0.0, posinf=0.0, neginf=0.0)

# ✅ Add context via concatenation
if self.aug_control and context_expanded is not None:
    h_temp_flat = h_temp.view(b * n_nodes, -1)
    h_temp_flat = torch.cat([h_temp_flat, context_expanded], dim=-1)
    
    # Project back to original dimension
    h_temp = self.context_projection(h_temp_flat)  # Need to add this layer
else:
    h_temp = h_temp.view(b * n_nodes, -1)
```

**If using Option B, add in `__init__`:**
```python
if self.aug_control:
    self.context_projection = nn.Linear(
        temp_node_embed_dim + temp_node_embed_dim,  # Concat dim
        temp_node_embed_dim  # Output dim
    )
```

### Recommendation
- Use **Option A** (h0 initialization) - More aligned with paper
- Use **Option B** (concatenation) - Simpler, less code changes

---

## Fix #3: Pass Context to Decoder (MINOR)

### Current Code
**File**: `dyedgegat/src/model/dyedgegat.py`  
**Lines**: 446-470

```python
class ReconstructionModel(nn.Module):
    def reconstruct(self, z, window_size):
        b, n, d = z.shape
        z_flat = z.view(b * n, d)
        z_rep = z_flat.unsqueeze(1).repeat(1, window_size, 1)
        
        out, _ = self.rnn(z_rep)  # ❌ No h0 passed
        recon = self.out(out)
        recon = recon.squeeze(-1)
        return torch.flip(recon, dims=[1])
```

### ✅ Fixed Code

**Step 1**: Modify signature
```python
def reconstruct(self, z, window_size, h0=None):  # ✅ Add h0
    b, n, d = z.shape
    z_flat = z.view(b * n, d)
    z_rep = z_flat.unsqueeze(1).repeat(1, window_size, 1)
    
    # ✅ Prepare h0 if provided
    if h0 is not None:
        # h0 shape: [B*N, Dim]
        h0 = h0.unsqueeze(0).expand(self.rnn.num_layers, -1, -1)  # [num_layers, B*N, Dim]
    
    out, _ = self.rnn(z_rep, h0)  # ✅ Pass h0
    recon = self.out(out)
    recon = recon.squeeze(-1)
    return torch.flip(recon, dims=[1])
```

**Step 2**: Update call site (line 856)
```python
# Before:
recon = self.decoder.reconstruct(z_fused_nodes, cfg.dataset.window_size)

# After:
if self.aug_control and context_expanded is not None:
    recon = self.decoder.reconstruct(
        z_fused_nodes, 
        cfg.dataset.window_size,
        h0=context_expanded  # ✅ Pass context
    )
else:
    recon = self.decoder.reconstruct(z_fused_nodes, cfg.dataset.window_size)
```

---

## Implementation Priority

### Phase 1: Critical Fix (Do This First)
```bash
# Edit dyedgegat/src/model/dyedgegat.py
# Add 1 line at line ~396:
mag = torch.log1p(mag)
```

**Expected Results**:
- Training converges faster
- Divergence loss stabilizes
- Anomaly detection improves

### Phase 2: Context Injection (Do This Second)
Choose one approach:
- Option A: Modify `GRUEncoder` to accept `h0` (~20 lines)
- Option B: Concatenate context with embeddings (~10 lines)

**Expected Results**:
- Model learns context-dependent patterns
- Better performance under varying operating conditions

### Phase 3: Context in Decoder (Optional)
- Modify `ReconstructionModel.reconstruct()` (~10 lines)

**Expected Results**:
- Marginal improvement (context already in fused embeddings)

---

## Validation Checklist

After applying fixes:

### ✅ Code Correctness
- [ ] No syntax errors
- [ ] Shapes match (print tensor shapes)
- [ ] No NaN/Inf values during forward pass
- [ ] Backward pass works (loss.backward())

### ✅ Training Behavior
- [ ] Loss decreases smoothly
- [ ] Divergence loss < 1.0 after a few epochs
- [ ] No gradient explosions
- [ ] GPU memory usage reasonable

### ✅ Model Performance
- [ ] Reconstruction error on validation set decreases
- [ ] Anomaly scores higher for fault data than normal data
- [ ] Divergence scores spike during anomalies

### ✅ Ablation Study
Test configurations:
1. **Baseline**: No spectral view (`use_spectral_view=False`)
2. **With log**: Spectral view + log transform
3. **With context**: Log + context injection
4. **Full model**: All fixes

Expected ranking: Full model > With context > With log > Baseline

---

## Testing the Fixes

### Quick Test Script

```python
import torch
from dyedgegat.src.model.dyedgegat import SpectralEncoder

# Test spectral encoder with log transform
encoder = SpectralEncoder(
    window_size=15,
    embed_dim=16,
    max_freq_bins=8,
    band_mixer="mlp",
)

# Dummy input
x = torch.randn(4, 28, 15)  # [B=4, N=28, W=15]

# Forward pass
h_freq = encoder(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {h_freq.shape}")  # Should be [4, 28, 16]
print(f"Output range: [{h_freq.min():.3f}, {h_freq.max():.3f}]")
print(f"Contains NaN: {torch.isnan(h_freq).any()}")
```

**Expected output (after fix)**:
```
Input shape: torch.Size([4, 28, 15])
Output shape: torch.Size([4, 28, 16])
Output range: [-1.234, 2.456]
Contains NaN: False
```

---

## Common Issues After Fixing

### Issue: NaN in spectral embeddings
**Cause**: Log of very small magnitudes  
**Solution**: Already handled by `log1p()` which is stable at 0

### Issue: Context dimension mismatch
**Cause**: GRU hidden state expects [num_layers, batch, dim]  
**Solution**: Use `h0.unsqueeze(0)` to add layer dimension

### Issue: Decoder shape error
**Cause**: Context not properly broadcasted to all nodes  
**Solution**: Use `context.repeat_interleave(n_nodes, dim=0)`

---

## Summary

| Fix | Effort | Impact | Status |
|-----|--------|--------|--------|
| Log transform | 1 line | ⚠️ High | Required |
| Context injection | 20 lines | 🟡 Medium | Recommended |
| Context in decoder | 10 lines | 🟢 Low | Optional |

**Minimum viable fix**: Add log transform  
**Recommended fix**: Log transform + context injection  
**Complete fix**: All three


