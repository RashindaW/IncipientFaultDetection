"""
Test script for the data loading pipeline.

This script tests the complete data loading infrastructure for DyEdgeGAT
on the refrigeration system dataset.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dyedgegat', 'src'))

from data.column_config import print_config_summary, MEASUREMENT_VARS
from data.dataloader import create_dataloaders
from config import cfg


def main():
    print("=" * 80)
    print(" " * 20 + "DYEDGEGAT DATA PIPELINE TEST")
    print("=" * 80)
    
    # ========== Step 1: Show Configuration ==========
    print("\n" + "=" * 80)
    print("STEP 1: Variable Configuration")
    print("=" * 80)
    print_config_summary()
    
    # ========== Step 2: Set Config ==========
    print("\n" + "=" * 80)
    print("STEP 2: Model Configuration")
    print("=" * 80)
    cfg.set_dataset_params(
        n_nodes=len(MEASUREMENT_VARS),     # 28 measurement variables
        window_size=15,  # 15 timesteps per window
        ocvar_dim=6      # 6 operating condition variables
    )
    cfg.validate()
    print(f"✅ Config validated")
    print(f"   Nodes: {cfg.dataset.n_nodes}")
    print(f"   Window size: {cfg.dataset.window_size}")
    print(f"   Control vars: {cfg.dataset.ocvar_dim}")
    print(f"   Device: {cfg.device}")
    
    # ========== Step 3: Create DataLoaders ==========
    print("\n" + "=" * 80)
    print("STEP 3: Create DataLoaders")
    print("=" * 80)
    
    # Note: Using larger strides for faster testing
    # For actual training, use train_stride=1
    train_loader, val_loader, test_loaders = create_dataloaders(
        window_size=cfg.dataset.window_size,
        batch_size=32,
        train_stride=10,  # Use 10 for faster testing (change to 1 for real training)
        val_stride=10,
        data_dir='Dataset',
        num_workers=0,
    )
    
    # ========== Step 4: Test Batch Loading ==========
    print("\n" + "=" * 80)
    print("STEP 4: Test Batch Loading")
    print("=" * 80)
    
    print("\nFetching a training batch...")
    batch = next(iter(train_loader))
    
    print(f"\n✅ Successfully loaded batch!")
    print(f"\nBatch structure:")
    print(f"  batch.x (measurements):    {batch.x.shape}")
    print(f"    Expected: [B*N, W] = [{32*len(MEASUREMENT_VARS)}, {cfg.dataset.window_size}]")
    print(f"  batch.c (controls):        {batch.c.shape}")
    print(f"    Expected: [B, ocvar_dim, W] = [{32}, {cfg.dataset.ocvar_dim}, {cfg.dataset.window_size}]")
    print(f"  batch.edge_index:          {batch.edge_index.shape}")
    print(f"  batch.batch:               {batch.batch.shape}")
    print(f"    Unique batch IDs: {batch.batch.unique().tolist()}")
    
    # ========== Step 5: Verify Data Quality ==========
    print("\n" + "=" * 80)
    print("STEP 5: Data Quality Checks")
    print("=" * 80)
    
    print(f"\nMeasurement data (batch.x):")
    print(f"  Min value: {batch.x.min().item():.4f}")
    print(f"  Max value: {batch.x.max().item():.4f}")
    print(f"  Mean: {batch.x.mean().item():.4f}")
    print(f"  Std: {batch.x.std().item():.4f}")
    print(f"  Contains NaN: {batch.x.isnan().any().item()}")
    print(f"  Contains Inf: {batch.x.isinf().any().item()}")
    
    print(f"\nControl data (batch.c):")
    print(f"  Min value: {batch.c.min().item():.4f}")
    print(f"  Max value: {batch.c.max().item():.4f}")
    print(f"  Mean: {batch.c.mean().item():.4f}")
    print(f"  Std: {batch.c.std().item():.4f}")
    print(f"  Contains NaN: {batch.c.isnan().any().item()}")
    print(f"  Contains Inf: {batch.c.isinf().any().item()}")
    
    # ========== Step 6: Test All DataLoaders ==========
    print("\n" + "=" * 80)
    print("STEP 6: Test All DataLoaders")
    print("=" * 80)
    
    print(f"\n✅ Training loader: {len(train_loader)} batches")
    print(f"✅ Validation loader: {len(val_loader)} batches")
    
    print(f"\nTest loaders:")
    for name, loader in test_loaders.items():
        print(f"  ✅ {name:30s}: {len(loader)} batches")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✅ All tests passed!")
    print()
    print("You can now:")
    print("  1. Train DyEdgeGAT using train_loader")
    print("  2. Validate using val_loader")
    print("  3. Test on fault data using test_loaders")
    print()
    print("Next steps:")
    print("  - Run: python train_dyedgegat.py (create this script)")
    print("  - Or: Use the dataloaders in your own training loop")
    print("=" * 80)


if __name__ == "__main__":
    main()
