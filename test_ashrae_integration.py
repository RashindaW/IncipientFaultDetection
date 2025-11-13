"""
Test script for ASHRAE 1043-RP dataset integration with DyEdgeGAT.

This script verifies that the ASHRAE dataset adapter is correctly registered
and can create dataloaders for training and testing.

Usage:
    python test_ashrae_integration.py --quick
    python test_ashrae_integration.py --full
"""

import argparse
import sys
import os


def test_adapter_registration():
    """Test that the ASHRAE adapter is registered."""
    print("=" * 70)
    print("TEST 1: Adapter Registration")
    print("=" * 70)
    
    try:
        from datasets import get_adapter, list_adapter_keys
        
        available = list_adapter_keys()
        print(f"Available adapters: {available}")
        
        if 'ashrae' not in available:
            print("❌ FAILED: 'ashrae' adapter not found in registry")
            return False
        
        ashrae = get_adapter('ashrae')
        print(f"\n✓ ASHRAE adapter registered successfully")
        print(f"  Key: {ashrae.key}")
        print(f"  Description: {ashrae.description}")
        print(f"  Default directory: {ashrae.default_data_dir}")
        print(f"  Measurement variables: {ashrae.measurement_count()}")
        print(f"  Fault keys: {ashrae.list_fault_keys()}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_column_config():
    """Test the ASHRAE column configuration."""
    print("\n" + "=" * 70)
    print("TEST 2: Column Configuration")
    print("=" * 70)
    
    try:
        from dyedgegat.src.data.ashrae_column_config import (
            MEASUREMENT_VARS,
            CONTROL_VARS,
            BASELINE_FILES,
            FAULT_FILES,
            print_config_summary,
        )
        
        print_config_summary()
        
        print(f"\n✓ Column configuration loaded successfully")
        print(f"  Training files: {len(BASELINE_FILES['train'])}")
        print(f"  Validation files: {len(BASELINE_FILES['val'])}")
        print(f"  Fault files: {len(FAULT_FILES)}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_availability():
    """Test that ASHRAE data files exist."""
    print("\n" + "=" * 70)
    print("TEST 3: Data File Availability")
    print("=" * 70)
    
    try:
        from dyedgegat.src.data.ashrae_column_config import (
            BASELINE_FILES,
            FAULT_FILES,
            BENCHMARK_DIR,
            REFRIGERANT_LEAK_DIR,
        )
        
        data_dir = "data/ASHRAE_1043_RP"
        
        # Check benchmark test files
        print(f"\nChecking training files in {BENCHMARK_DIR}:")
        missing_train = []
        for filename in BASELINE_FILES['train']:
            filepath = os.path.join(data_dir, BENCHMARK_DIR, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (NOT FOUND)")
                missing_train.append(filename)
        
        print(f"\nChecking validation files in {BENCHMARK_DIR}:")
        missing_val = []
        for filename in BASELINE_FILES['val']:
            filepath = os.path.join(data_dir, BENCHMARK_DIR, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ {filename} (NOT FOUND)")
                missing_val.append(filename)
        
        # Check refrigerant leak files
        print(f"\nChecking fault files in {REFRIGERANT_LEAK_DIR}:")
        missing_fault = []
        for fault_name, filename in FAULT_FILES.items():
            filepath = os.path.join(data_dir, REFRIGERANT_LEAK_DIR, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {fault_name}: {filename}")
            else:
                print(f"  ✗ {fault_name}: {filename} (NOT FOUND)")
                missing_fault.append(filename)
        
        if missing_train or missing_val or missing_fault:
            print(f"\n⚠ WARNING: Some data files are missing")
            print(f"  Missing training: {len(missing_train)}")
            print(f"  Missing validation: {len(missing_val)}")
            print(f"  Missing fault: {len(missing_fault)}")
            return False
        
        print(f"\n✓ All data files found")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation(quick=True):
    """Test creating ASHRAE dataset."""
    print("\n" + "=" * 70)
    print("TEST 4: Dataset Creation")
    print("=" * 70)
    
    try:
        from dyedgegat.src.data.ashrae_dataset import ASHRAEDataset
        from dyedgegat.src.data.ashrae_column_config import BASELINE_FILES, BENCHMARK_DIR
        
        data_dir = "data/ASHRAE_1043_RP"
        
        # Use only one file for quick test
        if quick:
            test_files = [os.path.join(BENCHMARK_DIR, BASELINE_FILES['val'][0])]
            print(f"Quick test with 1 file: {test_files[0]}")
        else:
            test_files = [os.path.join(BENCHMARK_DIR, f) for f in BASELINE_FILES['train'][:3]]
            print(f"Full test with {len(test_files)} files")
        
        print("\nCreating dataset...")
        dataset = ASHRAEDataset(
            data_files=test_files,
            window_size=15,
            stride=10,  # Use larger stride for quick testing
            data_dir=data_dir,
            normalize=True,
        )
        
        print(f"\n✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Measurement variables: {dataset.n_measurement_vars}")
        print(f"  Control variables: {dataset.n_control_vars}")
        
        # Test getting a sample
        print("\nTesting sample retrieval...")
        sample = dataset.get(0)
        print(f"  ✓ Sample retrieved")
        print(f"    x shape: {sample.x.shape}")
        print(f"    c shape: {sample.c.shape}")
        print(f"    edge_index shape: {sample.edge_index.shape}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_creation(quick=True):
    """Test creating dataloaders using the adapter."""
    print("\n" + "=" * 70)
    print("TEST 5: DataLoader Creation")
    print("=" * 70)
    
    try:
        from datasets import get_adapter
        
        ashrae = get_adapter('ashrae')
        data_dir = ashrae.default_data_dir
        
        print(f"Creating dataloaders from {data_dir}...")
        
        # Use smaller parameters for quick test
        if quick:
            window_size = 15
            batch_size = 8
            train_stride = 50
            val_stride = 50
        else:
            window_size = 15
            batch_size = 32
            train_stride = 10
            val_stride = 20
        
        train_loader, val_loader, test_loaders = ashrae.create_dataloaders(
            window_size=window_size,
            batch_size=batch_size,
            train_stride=train_stride,
            val_stride=val_stride,
            test_stride=None,
            data_dir=data_dir,
            num_workers=0,
            distributed=False,
        )
        
        print(f"\n✓ DataLoaders created successfully")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        print(f"  Test datasets: {len(test_loaders)}")
        
        # Test getting a batch
        print("\nTesting batch retrieval...")
        batch = next(iter(train_loader))
        print(f"  ✓ Batch retrieved from training loader")
        print(f"    Batch x shape: {batch.x.shape}")
        print(f"    Batch c shape: {batch.c.shape}")
        print(f"    Batch edge_index shape: {batch.edge_index.shape}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test ASHRAE dataset integration')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--full', action='store_true', help='Run full tests')
    args = parser.parse_args()
    
    quick = args.quick or not args.full
    
    print("\n" + "=" * 70)
    print("ASHRAE 1043-RP DATASET INTEGRATION TESTS")
    print("=" * 70)
    print(f"Mode: {'QUICK' if quick else 'FULL'}")
    print()
    
    results = []
    
    # Test 1: Adapter registration
    results.append(("Adapter Registration", test_adapter_registration()))
    
    # Test 2: Column configuration
    results.append(("Column Configuration", test_column_config()))
    
    # Test 3: Data availability
    results.append(("Data Availability", test_data_availability()))
    
    # Test 4: Dataset creation
    results.append(("Dataset Creation", test_dataset_creation(quick=quick)))
    
    # Test 5: DataLoader creation
    results.append(("DataLoader Creation", test_dataloader_creation(quick=quick)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:10s} {name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The ASHRAE dataset integration is ready.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

