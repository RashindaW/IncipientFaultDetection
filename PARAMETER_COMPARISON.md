# DyEdgeGAT Parameter and Metrics Comparison

## Executive Summary
This document compares your implementation with the DyEdgeGAT paper to ensure you're following the same evaluation metrics and parameters.

---

## 1. EVALUATION METRICS COMPARISON

### Your Implementation (from `train_dyedgegat.py`)
- **AUC-ROC**: ✅ Implemented (line 653)
- **Precision**: ✅ Implemented (line 662)
- **Recall**: ✅ Implemented (line 662)
- **F1 Score**: ✅ Implemented (line 662)
- **Threshold**: 99th percentile of baseline anomaly scores (line 659)

### Paper's Metrics (from TABLE VII, IX)
- **AUC (AUC†)**: Area under ROC curve ✅
- **F1 (F1†)**: F1 score at 95th percentile threshold
- **F1* (F1* †)**: Best F1 score (maximal from precision-recall curve)
- **Delay* (Delay*)**: Best detection delay at threshold of best F1
- **Ambiguity**: Model's ability to distinguish novel OCs from faults

### ⚠️ KEY DIFFERENCES IDENTIFIED:

#### 1. **Threshold Selection** ❌
- **Your Code**: Uses **99th percentile** (line 659 in train_dyedgegat.py)
  ```python
  threshold = np.percentile(baseline_scores, 99) # 1% false alarm rate target
  ```
- **Paper Standard**: Uses **95th percentile** for F1 calculation
  - From paper: "F1: Average of precision and recall, determined by a threshold at the 95th percentile of normal validation anomaly scores."

#### 2. **Missing Metrics** ⚠️
- **F1* (Best F1)**: NOT implemented
  - Paper: "textit{Best} F1 (F1*): The maximal F1 score obtained from the precision-recall curve."
  - You should calculate the precision-recall curve and find the best F1 score
  
- **Delay* (Best Detection Delay)**: NOT implemented
  - Paper: "Best Detection Delay (Delay*): Measures the time taken to identify faults after their occurrence using the threshold of best F1."
  
- **Ambiguity Metric**: NOT implemented
  - Paper: "Ambiguity Metric (Ambiguity): A novel metric defined as Ambiguity = 1 - 2 × |AUC - 0.5|"
  - This is ONLY for industrial dataset (Pronto) to evaluate novel OCs

---

## 2. MODEL PARAMETERS COMPARISON

### Your Implementation (from `train_dyedgegat.py` lines 305-348)

| Parameter | Your Value | Paper (Table VI - Pronto) | Status |
|-----------|------------|---------------------------|---------|
| **num_gnn_layers (L)** | 2 | 2 | ✅ |
| **gnn_embed_dim** | 40 | ? | ❓ |
| **temp_node_embed_dim** | 16 | temp. embed. 5 | ❌ |
| **time_dim** | 5 | - | ✅ |
| **temporal_window** | 5 | - | ✅ |
| **topk** | 20 | - | ❓ |
| **dropout** | 0.3 | - | ❓ |
| **recon_hidden_dim** | 16 | - | ❓ |
| **num_recon_layers** | 1 | - | ❓ |
| **encoder_norm_type** | "layer" | LN+BN | ⚠️ |
| **gnn_norm_type** | "layer" | LN+BN | ⚠️ |
| **decoder_norm_type** | "layer" | LN+BN | ⚠️ |
| **temp_edge_hid_dim** | 100 | - | ❓ |
| **temp_edge_embed_dim** | 1 | edge embed. 20 | ❌ |
| **feat_edge_hid_dim** | 128 | - | ❓ |
| **freq_node_embed_dim** | 16 (--freq-embed-dim) | - | ❓ |

### Training Hyperparameters

| Parameter | Your Value | Paper (Table VI) | Status |
|-----------|------------|------------------|---------|
| **window_size** | 15 | 15 | ✅ |
| **batch_size** | 64 | 256 (synthetic), ? (pronto) | ⚠️ |
| **epochs** | 30 | 300 (synthetic), ? (pronto) | ❌ |
| **learning_rate** | 1e-3 | 1e-3 | ✅ |
| **weight_decay** | 1e-5 | - | ❓ |
| **anomaly_weight** | 0.5 | - | ❓ |
| **lambda_div** | 0.1 | - | ❓ |

### ⚠️ CRITICAL DIFFERENCES:

1. **temp_node_embed_dim**: You use 16, paper shows "temp. embed. 5" in Table VI
2. **temp_edge_embed_dim**: You use 1, paper shows "edge embed. 20" in Table VI  
3. **epochs**: You trained for 30 epochs, paper used 300 for synthetic (not clear for Pronto)
4. **batch_size**: You use 64, paper used 256 for synthetic
5. **Normalization**: You use "layer" norm, paper explicitly states "LN+BN" (Layer Norm + Batch Norm combination)

---

## 3. YOUR RESULTS vs PAPER RESULTS

### Your Results (from terminal output):

| Fault Type | AUC | F1 | 
|------------|-----|-----|
| slugging | 0.5255 | 0.4136 |
| faults_all | 1.0000 | 0.9995 |
| Blockage_120air_01water | 1.0000 | 0.9969 |
| Blockage_150air_05water | 1.0000 | 0.9967 |
| Leakage_120air_01water | 1.0000 | 0.9955 |
| Leakage_150air_05water | 1.0000 | 0.9976 |
| Diverted_120air_01water | 1.0000 | 0.9970 |
| Diverted_150air_05water | 1.0000 | 0.9962 |

### Paper Results (TABLE IX - DyEdgeGAT row):

| Metric | Air leakage | Air blockage | Diverted flow | Average |
|--------|-------------|--------------|---------------|---------|
| **AUC†** | 0.73 ± 0.04 | 0.84 ± 0.00 | 0.74 ± 0.02 | **0.80 ± 0.05** |
| **F1†** | 0.40 ± 0.02 | 0.63 ± 0.05 | 0.70 ± 0.11 | **0.83 ± 0.02** |
| **F1* †** | 0.83 ± 0.02 | 0.83 ± 0.01 | 0.80 ± 0.02 | **0.86 ± 0.02** |
| **Delay*** | 905 ± 559 | 1644 ± 83 | 511 ± 4 | **61 ± 1** |

### 🎯 ANALYSIS:
- **Your fault detection is TOO PERFECT** (AUC = 1.0, F1 ≈ 0.99) compared to paper
- **Paper shows much lower scores**, especially for specific fault types
- **Possible reasons**:
  1. ❌ Wrong threshold (99th vs 95th percentile)
  2. ❌ Different evaluation protocol
  3. ❌ Model overfitting (only 30 epochs vs paper's longer training)
  4. ⚠️ Paper averages across multiple flow conditions, you're testing individual conditions
  5. ❌ Your slugging result (AUC=0.53, F1=0.41) is close to random, which IS consistent with paper's ambiguity findings

---

## 4. RECOMMENDATIONS

### 🔴 CRITICAL (Must Fix):

1. **Change threshold from 99th to 95th percentile**:
   ```python
   # In train_dyedgegat.py, line 659
   threshold = np.percentile(baseline_scores, 95)  # NOT 99
   ```

2. **Implement F1* (Best F1) calculation**:
   ```python
   from sklearn.metrics import precision_recall_curve
   precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
   f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
   best_f1_idx = np.argmax(f1_scores)
   best_f1 = f1_scores[best_f1_idx]
   best_threshold = thresholds[best_f1_idx]
   ```

3. **Implement Detection Delay (Delay*)**:
   - Measure time from fault onset to first detection at best F1 threshold
   - Need to track fault onset timestamps in test data

4. **Implement Ambiguity Metric** (for Pronto only):
   ```python
   ambiguity = 1 - 2 * abs(auc - 0.5)
   ```

### 🟡 IMPORTANT (Should Fix):

5. **Verify model parameter alignment**:
   - Check if `temp_node_embed_dim=16` should be `5` as in paper
   - Check if `temp_edge_embed_dim=1` should be `20` as in paper
   - Verify normalization strategy (LN+BN vs just layer norm)

6. **Extend training**:
   - Paper likely trained for more epochs
   - Your training plateaued after epoch 2 (validation loss didn't improve)

7. **Aggregate results properly**:
   - Paper shows results averaged across flow conditions
   - You should average Blockage_120 + Blockage_150 for "Air blockage"
   - Average Leakage_120 + Leakage_150 for "Air leakage"  
   - Average Diverted_120 + Diverted_150 for "Diverted flow"

### 📝 NICE TO HAVE:

8. **Add model parameter count verification**:
   - Paper shows 3921 parameters for Pronto
   - Calculate total parameters in your model

9. **Match batch size if possible**:
   - Paper used 256 for synthetic
   - Consider using larger batch size if memory allows

---

## 5. SUGGESTED CODE CHANGES

### File: `train_dyedgegat.py`

#### Change 1: Fix threshold (Line ~659)
```python
# BEFORE:
threshold = np.percentile(baseline_scores, 99) # 1% false alarm rate target

# AFTER:
threshold = np.percentile(baseline_scores, 95) # Following paper's protocol
```

#### Change 2: Add F1*, Delay*, and Ambiguity calculations (After line 662)
```python
# Calculate Best F1 (F1*)
precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_f1_idx]
best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else pr_thresholds[-1]

# Calculate Ambiguity (for novel OCs like slugging)
ambiguity = 1 - 2 * abs(auc - 0.5)

# Calculate Detection Delay at best F1 threshold
# (Requires tracking fault onset - implementation depends on your data structure)
y_pred_best = (y_scores > best_threshold).astype(int)
# ... delay calculation logic ...
```

#### Change 3: Update CSV output (Line 638-672)
```python
fieldnames = ['test_set', 'auc_roc', 'precision', 'recall', 'f1_score', 'best_f1', 'ambiguity', 'threshold', 'best_threshold']
```

---

## 6. SUMMARY CHECKLIST

- [ ] Threshold changed from 99th to 95th percentile
- [ ] F1* (Best F1) metric implemented
- [ ] Detection Delay* metric implemented  
- [ ] Ambiguity metric implemented
- [ ] Model parameters verified against Table VI
- [ ] Normalization strategy checked (LN+BN)
- [ ] Results aggregated across flow conditions
- [ ] Model parameter count calculated
- [ ] Training extended if needed
- [ ] Results properly averaged like paper's Table IX

---

**Generated**: December 3, 2025
**Based on**: DyEdgeGAT paper (IEEE IoT Journal, Vol. 11, No. 13, July 2024)

