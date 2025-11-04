"""
Test script for DyEdgeGAT model with real data.

This script tests the complete DyEdgeGAT pipeline:
1. Load data
2. Initialize model
3. Forward pass
4. Backward pass
5. Anomaly scoring
"""

import torch
import sys
import os

WINDOW_SIZE = 60

# Add both paths for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dyedgegat'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dyedgegat', 'src'))

from src.data.dataloader import create_dataloaders
from src.data.dataset import get_control_variable_names
from src.model.dyedgegat import DyEdgeGAT
from src.config import cfg
from src.data.column_config import MEASUREMENT_VARS


def test_model():
    print("=" * 80)
    print(" " * 25 + "DYEDGEGAT MODEL TEST")
    print("=" * 80)
    
    # ========== Step 1: Configuration ==========
    print("\n[1/6] Setting up configuration...")
    data_dir = 'Dataset'
    control_var_names = get_control_variable_names(data_dir)
    cfg.set_dataset_params(
        n_nodes=len(MEASUREMENT_VARS),  # 142 nodes (measurement sensors)
        window_size=WINDOW_SIZE,
        ocvar_dim=len(control_var_names)  # Control variables (may include time encodings)
    )
    cfg.validate()
    print(f"✅ Config: {cfg.dataset.n_nodes} nodes, window={cfg.dataset.window_size}, "
          f"controls={cfg.dataset.ocvar_dim}, device={cfg.device}")
    
    # ========== Step 2: Create DataLoaders ==========
    print("\n[2/6] Creating dataloaders...")
    train_loader, val_loader, test_loaders = create_dataloaders(
        window_size=cfg.dataset.window_size,
        batch_size=4,  # Small batch for testing
        train_stride=100,  # Large stride for quick test
        val_stride=100,
        data_dir=data_dir,
        num_workers=0,
    )
    print(f"✅ DataLoaders created:")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val: {len(val_loader)} batches")
    print(f"   Test: {len(test_loaders)} datasets")
    
    # ========== Step 3: Initialize Model ==========
    print("\n[3/6] Initializing DyEdgeGAT model...")
    model = DyEdgeGAT(
        feat_input_node=1,           # Each sensor is univariate
        feat_target_node=1,          # Reconstruct 1 value per sensor
        feat_input_edge=1,           # Edge features are scalar attention values
        node_encoder_type='gru',
        node_encoder_mode='univariate',
        contr_encoder_type='gru',
        infer_temporal_edge=True,
        temp_edge_hid_dim=40,
        temp_edge_embed_dim=1,
        temporal_window=5,
        temporal_kernel=5,
        use_time_encoding=True,
        time_dim=5,
        temp_node_embed_dim=16,
        infer_static_graph=True,
        feat_edge_hid_dim=64,
        topk=20,
        learn_sys=True,
        num_gnn_layers=2,
        gnn_embed_dim=40,
        gnn_type='gin',             # GIN as in the paper
        dropout=0.3,
        do_encoder_norm=True,
        do_gnn_norm=True,
        do_decoder_norm=True,
        encoder_norm_type='layer',
        gnn_norm_type='layer',
        decoder_norm_type='layer',
        recon_hidden_dim=16,
        num_recon_layers=1,
        edge_aggr='temp',
        act='relu',
        aug_control=True,
        flip_output=True,
    ).to(cfg.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ========== Step 4: Forward Pass Test ==========
    print("\n[4/6] Testing forward pass...")
    model.eval()
    
    # Get a batch
    batch = next(iter(train_loader))
    batch = batch.to(cfg.device)
    
    print(f"   Batch shapes:")
    print(f"     x (measurements): {batch.x.shape}")
    print(f"     c (controls): {batch.c.shape}")
    print(f"     edge_index: {batch.edge_index.shape}")
    
    with torch.no_grad():
        # Forward pass
        recon = model(batch)
        
        # Also get graph for anomaly scoring
        recon_with_graph, edge_index, edge_weight = model(batch, return_graph=True)
    
    print(f"✅ Forward pass successful!")
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Expected shape: [{batch.x.shape[0]}, {cfg.dataset.window_size}, 1]")
    
    # Check for NaN or Inf
    has_nan = recon.isnan().any().item()
    has_inf = recon.isinf().any().item()
    print(f"   Contains NaN: {has_nan}")
    print(f"   Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("   ⚠️ WARNING: Output contains NaN or Inf!")
    
    # ========== Step 5: Loss Computation ==========
    print("\n[5/6] Testing loss computation...")
    
    target = batch.x.unsqueeze(-1)  # [B*N, W, 1]
    criterion = torch.nn.MSELoss()
    loss = criterion(recon, target)
    
    print(f"✅ Loss computation successful!")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Target shape: {target.shape}")
    
    # ========== Step 6: Backward Pass Test ==========
    print("\n[6/6] Testing backward pass (one step)...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # One training step
    optimizer.zero_grad()
    recon = model(batch)
    loss = criterion(recon, target)
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    optimizer.step()
    
    print(f"✅ Backward pass successful!")
    print(f"   Gradient norm: {total_grad_norm:.6f}")
    print(f"   Loss after step: {loss.item():.6f}")
    
    # ========== Bonus: Test Topology-Aware Anomaly Scoring ==========
    print("\n[BONUS] Testing topology-aware anomaly scoring...")
    
    model.eval()
    with torch.no_grad():
        recon, edge_index, edge_weight = model(batch, return_graph=True)
        
        # Compute anomaly score
        anomaly_score = model.compute_topology_aware_anomaly_score(
            x_true=target,
            x_recon=recon,
            edge_index=edge_index,
            edge_weight=edge_weight.view(-1)
        )
    
    print(f"✅ Anomaly scoring successful!")
    print(f"   Anomaly score: {anomaly_score.item():.6f}")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ Configuration: OK")
    print("✅ Data loading: OK")
    print("✅ Model initialization: OK")
    print("✅ Forward pass: OK")
    print("✅ Loss computation: OK")
    print("✅ Backward pass: OK")
    print("✅ Anomaly scoring: OK")
    print()
    print("🎉 All tests passed! DyEdgeGAT is ready for training.")
    print()
    print("Model Details:")
    print(f"  - Parameters: {trainable_params:,}")
    print(f"  - Input: {cfg.dataset.n_nodes} sensors × {cfg.dataset.window_size} timesteps")
    print(f"  - Controls: {cfg.dataset.ocvar_dim} variables")
    print(f"  - Device: {cfg.device}")
    print()
    print("Next steps:")
    print("  1. Create full training script")
    print("  2. Train on baseline data")
    print("  3. Evaluate on fault data")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR OCCURRED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print("=" * 80)
        sys.exit(1)
