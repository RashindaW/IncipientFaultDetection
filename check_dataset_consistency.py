
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path to import config
sys.path.append(os.getcwd())

try:
    from dyedgegat.src.data.ashrae_column_config import BASELINE_FILES, FAULT_FILES, MEASUREMENT_VARS, CONTROL_VARS, ALL_SELECTED_COLUMNS
    
    BASE_DIR = 'data/ASHRAE_1043_RP'
    BENCHMARK_DIR = os.path.join(BASE_DIR, 'Benchmark Tests')
    FAULT_DIR = os.path.join(BASE_DIR, 'Condenser fouling')
    
    COLUMN_ALIASES = {
        "Heat Balance": "Heat Balance (kW)",
    }

    def check_file(filepath, type_label):
        if not os.path.exists(filepath):
            print(f"[MISSING] {type_label}: {os.path.basename(filepath)}")
            return False
            
        try:
            # Read just header
            df = pd.read_excel(filepath, nrows=1)
            
            # Rename
            renamed_cols = []
            for col in df.columns:
                if col in ["Timestamp", "Time", "Time (minutes)"]:
                    renamed_cols.append("Timestamp")
                elif col in COLUMN_ALIASES:
                    renamed_cols.append(COLUMN_ALIASES[col])
                else:
                    renamed_cols.append(col)
            
            current_file_cols = set(renamed_cols)
            required_cols = set(ALL_SELECTED_COLUMNS)
            
            missing = required_cols - current_file_cols
            
            if missing:
                print(f"[FAIL] {type_label}: {os.path.basename(filepath)} missing {len(missing)} columns")
                for m in list(missing)[:5]:
                    print(f"  - Missing: {m}")
                if len(missing) > 5: print("  ...")
                return False
            
            print(f"[OK] {type_label}: {os.path.basename(filepath)}")
            return True
            
        except Exception as e:
            print(f"[ERROR] {type_label}: {os.path.basename(filepath)} - {str(e)}")
            return False

    print("="*60)
    print("CHECKING DATASET CONSISTENCY (Condenser Fouling Setup)")
    print("="*60)
    print(f"Target Features: {len(MEASUREMENT_VARS)} measurements, {len(CONTROL_VARS)} controls")
    print("-" * 60)

    all_good = True

    # 1. Check Training Files
    print("\n--- Training Files (Benchmark Tests) ---")
    for fname in BASELINE_FILES['train']:
        path = os.path.join(BENCHMARK_DIR, fname)
        if not check_file(path, "TRAIN"): all_good = False

    # 2. Check Validation Files
    print("\n--- Validation Files (Benchmark Tests) ---")
    for fname in BASELINE_FILES['val']:
        path = os.path.join(BENCHMARK_DIR, fname)
        if not check_file(path, "VAL"): all_good = False

    # 3. Check Test Files (Condenser Fouling)
    print("\n--- Test Files (Condenser Fouling) ---")
    for fname in FAULT_FILES.values():
        path = os.path.join(FAULT_DIR, fname)
        if not check_file(path, "TEST"): all_good = False
        
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL FILES CONSISTENT. READY FOR TRAINING.")
    else:
        print("❌ DATASET INCONSISTENCIES FOUND.")
    print("="*60)

except Exception as e:
    print(f"Script Error: {e}")

