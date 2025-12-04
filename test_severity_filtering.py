#!/usr/bin/env python3
"""
Test script to validate severity filtering implementation.
This verifies that the severity_range parameter works correctly.
"""

import os
import sys
import numpy as np

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from dyedgegat.src.data.pronto_dataset import PRONTODataset

def test_severity_filtering():
    """Test that severity filtering works correctly."""
    
    data_dir = "data/pronto/pronto_benchmark/Pre-processed data/Process data"
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print("="*70)
    print("TESTING SEVERITY FILTERING IMPLEMENTATION")
    print("="*70)
    
    test_file = "Blockage_120air_01water.mat"
    
    # Test 1: No filtering (all severities)
    print("\n[Test 1] Loading without severity filter (ALL severities)...")
    try:
        ds_all = PRONTODataset(
            data_files=[test_file],
            window_size=15,
            stride=5,
            data_dir=data_dir,
            normalize=False,
            severity_range=None
        )
        samples_all = len(ds_all)
        print(f"✅ Success: {samples_all} samples")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 2: Early faults only (10-20)
    print("\n[Test 2] Loading with severity_range=(10, 20) [EARLY faults]...")
    try:
        ds_early = PRONTODataset(
            data_files=[test_file],
            window_size=15,
            stride=5,
            data_dir=data_dir,
            normalize=False,
            severity_range=(10, 20)
        )
        samples_early = len(ds_early)
        print(f"✅ Success: {samples_early} samples")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 3: Moderate faults (30-40)
    print("\n[Test 3] Loading with severity_range=(30, 40) [MODERATE faults]...")
    try:
        ds_moderate = PRONTODataset(
            data_files=[test_file],
            window_size=15,
            stride=5,
            data_dir=data_dir,
            normalize=False,
            severity_range=(30, 40)
        )
        samples_moderate = len(ds_moderate)
        print(f"✅ Success: {samples_moderate} samples")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 4: Severe faults (60-80)
    print("\n[Test 4] Loading with severity_range=(60, 80) [SEVERE faults]...")
    try:
        ds_severe = PRONTODataset(
            data_files=[test_file],
            window_size=15,
            stride=5,
            data_dir=data_dir,
            normalize=False,
            severity_range=(60, 80)
        )
        samples_severe = len(ds_severe)
        print(f"✅ Success: {samples_severe} samples")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Test 5: Single severity level
    print("\n[Test 5] Loading with severity_range=(10, 10) [SINGLE severity]...")
    try:
        ds_single = PRONTODataset(
            data_files=[test_file],
            window_size=15,
            stride=5,
            data_dir=data_dir,
            normalize=False,
            severity_range=(10, 10)
        )
        samples_single = len(ds_single)
        print(f"✅ Success: {samples_single} samples")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    
    # Validation
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nSample counts:")
    print(f"  All severities (10-80):  {samples_all:4d} samples")
    print(f"  Early faults (10-20):    {samples_early:4d} samples ({samples_early/samples_all*100:.1f}%)")
    print(f"  Moderate faults (30-40): {samples_moderate:4d} samples ({samples_moderate/samples_all*100:.1f}%)")
    print(f"  Severe faults (60-80):   {samples_severe:4d} samples ({samples_severe/samples_all*100:.1f}%)")
    print(f"  Single severity (10):    {samples_single:4d} samples ({samples_single/samples_all*100:.1f}%)")
    
    # Verify filtering is working
    success = True
    
    if samples_early >= samples_all:
        print("\n❌ FAILED: Early faults should have FEWER samples than all severities!")
        success = False
    else:
        print(f"\n✅ PASS: Early faults ({samples_early}) < All severities ({samples_all})")
    
    if samples_early + samples_moderate + samples_severe > samples_all * 1.1:
        print("❌ FAILED: Sum of filtered samples should be ≤ total samples!")
        success = False
    else:
        print(f"✅ PASS: Filtered samples sum check passed")
    
    if samples_single > samples_early:
        print("❌ FAILED: Single severity (10) should have fewer samples than range (10-20)!")
        success = False
    else:
        print(f"✅ PASS: Single severity ({samples_single}) < Range (10-20) ({samples_early})")
    
    # Expected ranges (approximate, based on 15-window with stride 5)
    expected_early_pct = 0.25  # ~25% of samples are severity 10-20
    expected_severe_pct = 0.30  # ~30% are severity 60-80
    
    early_pct = samples_early / samples_all
    severe_pct = samples_severe / samples_all
    
    if 0.15 < early_pct < 0.35:  # Reasonable range
        print(f"✅ PASS: Early fault percentage ({early_pct*100:.1f}%) is reasonable")
    else:
        print(f"⚠️  WARNING: Early fault percentage ({early_pct*100:.1f}%) seems unusual (expected ~25%)")
    
    if 0.20 < severe_pct < 0.40:  # Reasonable range
        print(f"✅ PASS: Severe fault percentage ({severe_pct*100:.1f}%) is reasonable")
    else:
        print(f"⚠️  WARNING: Severe fault percentage ({severe_pct*100:.1f}%) seems unusual (expected ~30%)")
    
    print("\n" + "="*70)
    if success:
        print("✅✅✅ ALL TESTS PASSED! Severity filtering is working correctly! ✅✅✅")
        print("\nYou can now run training with:")
        print("  --severity-range 10,20  (for early fault detection, matching paper)")
        print("  --severity-range 30,40  (for moderate faults)")
        print("  --severity-range 60,80  (for severe faults)")
        print("  (no flag)               (for all severities, current behavior)")
    else:
        print("❌❌❌ SOME TESTS FAILED! Check implementation! ❌❌❌")
    print("="*70)
    
    return success

if __name__ == "__main__":
    success = test_severity_filtering()
    sys.exit(0 if success else 1)


