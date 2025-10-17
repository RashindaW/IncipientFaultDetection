"""Debug script to investigate data loading issues."""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dyedgegat', 'src'))

from data.column_config import MEASUREMENT_VARS, CONTROL_VARS

# Load a sample file
df = pd.read_csv('Dataset/BaselineTestA.csv')

print("=" * 80)
print("DATA LOADING DEBUG")
print("=" * 80)

print(f"\nActual CSV columns: {len(df.columns)}")
print(f"Expected measurement vars: {len(MEASUREMENT_VARS)}")
print(f"Expected control vars: {len(CONTROL_VARS)}")

print("\n" + "=" * 80)
print("Checking which columns are MISSING from CSV:")
print("=" * 80)

missing_measurement = []
missing_control = []

for var in MEASUREMENT_VARS:
    if var not in df.columns:
        missing_measurement.append(var)
        
for var in CONTROL_VARS:
    if var not in df.columns:
        missing_control.append(var)

if missing_measurement:
    print(f"\n❌ Missing {len(missing_measurement)} measurement variables:")
    for var in missing_measurement:
        print(f"  - {var}")
else:
    print("\n✅ All measurement variables found!")

if missing_control:
    print(f"\n❌ Missing {len(missing_control)} control variables:")
    for var in missing_control:
        print(f"  - {var}")
else:
    print("\n✅ All control variables found!")

print("\n" + "=" * 80)
print("Data sample for found columns:")
print("=" * 80)

# Show data for columns that exist
existing_measurement = [v for v in MEASUREMENT_VARS if v in df.columns]
existing_control = [v for v in CONTROL_VARS if v in df.columns]

if existing_measurement:
    print(f"\nMeasurement data sample (first 3 rows, first 5 columns):")
    print(df[existing_measurement[:5]].head(3))
    print(f"\nMeasurement data stats:")
    print(df[existing_measurement[:5]].describe())

if existing_control:
    print(f"\nControl data sample (first 3 rows):")
    print(df[existing_control].head(3))
    print(f"\nControl data stats:")
    print(df[existing_control].describe())

# Check for sentinel values
print("\n" + "=" * 80)
print("Checking for sentinel values:")
print("=" * 80)

if existing_measurement:
    for col in existing_measurement[:3]:  # Check first 3
        sentinel_count = (df[col] < -1000).sum()
        print(f"{col:30s}: {sentinel_count} sentinel values")

# Find columns with similar names (for typos)
if missing_measurement or missing_control:
    print("\n" + "=" * 80)
    print("Possible column name matches (for typos):")
    print("=" * 80)
    
    all_missing = missing_measurement + missing_control
    for missing in all_missing[:5]:  # Show first 5
        # Find similar column names
        similar = [col for col in df.columns if missing.lower() in col.lower() or col.lower() in missing.lower()]
        if similar:
            print(f"\n'{missing}' might be:")
            for s in similar:
                print(f"  -> '{s}'")

