#!/usr/bin/env python3
"""Extract PRONTO dataset from raw CSVs into properly segmented files.

Reads 4 raw CSVs from the PRONTO benchmark and segments them according
to the Operation Log into 5 output files:
  - normal.csv    (from 0912 + 0626)
  - slugging.csv  (from 0912)
  - blockage.csv  (from 0907, Test2+Test3)
  - leakage.csv   (from 0907 Test4 + 0911 Test5+Test6)
  - diverted.csv  (from 0911, Test7+Test8)
"""

import csv
import os
import sys
from datetime import datetime, timedelta

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = '/mnt/datassd3/rashinda/DySTGAT/data/pronto/pronto_benchmark'
OUT  = '/mnt/datassd3/rashinda/DySTGAT/data/Pronto_data'

RAW = {
    '0912': os.path.join(BASE, 'C0 Normal and Slugging conditions',
                         'Test10', 'Process Data', '0912Testday4.csv'),
    '0626': os.path.join(BASE, 'C0 Normal and Slugging conditions',
                         'Test11', 'Process Data', '0626Testday5.csv'),
    '0907': os.path.join(BASE, 'C1 Air Blockage',
                         'Test2', 'Process Data', '0907Testday2.csv'),
    '0911': os.path.join(BASE, 'C2 Air Leakage',
                         'Test5', 'Process Data', '0911Testday3.csv'),
}

# 17 target instruments in desired output column order
INSTRUMENTS = [
    'FT305/OUT.CV',           # Air In1
    'FT302/OUT.CV',           # Air In2
    'FT305/AI2/OUT.CV',       # Air T
    'PT312/OUT.CV',           # Air P
    'FT102/OUT.CV',           # Water In1
    'FT104/OUT.CV',           # Water In2
    'FT102/AI3/OUT.CV',       # Water T
    'PT417/OUT.CV',           # Mixture zone P
    'PT408/OUT.CV',           # riser outlet P
    'PT403/OUT.CV',           # P topsep
    'FT404/OUT.CV',           # FR topsep gas
    'FT406/OUT.CV',           # FR topsep liquid
    'PT501/OUT.CV',           # P_3phase
    'PIC501/PID1/OUT.CV',     # Air Valve
    'LI502/OUT.CV',           # Water level
    'LI503/OUT.CV',           # Water coalescer
    'LVC502-SR/PID1/OUT.CV',  # Water level valve
]

FEATURE_NAMES = [
    'Air In1', 'Air In2', 'Air T', 'Air P',
    'Water In1', 'Water In2', 'Water T',
    'Mixture zone P', 'riser outlet P', 'P topsep',
    'FR topsep gas', 'FR topsep liquid',
    'P_3phase', 'Air Valve', 'Water level',
    'Water coalescer', 'Water level valve',
]

HEADER = ['TIMESTAMP'] + FEATURE_NAMES + ['label']

# ── Segment definitions from Operation Log ─────────────────────────────────
# Each entry: (start_HH:MM, end_HH:MM, label)

# 0912Testday4: Normal + Slugging (Test9/10)
# Data starts 10:00; 10:00-10:33 discarded (startup)
SEG_0912 = [
    ('10:33', '10:41', 'Slugging'),     # severe slugging
    ('10:41', '10:48', 'Slugging'),     # severe slugging
    ('10:48', '10:56', 'Normal'),       # healthy/churn
    ('10:56', '11:05', 'Normal'),       # annular
    ('11:05', '11:17', 'Normal'),       # annular
    ('11:17', '11:27', 'Normal'),       # healthy
    ('11:27', '11:47', 'Slugging'),     # slugging
    ('11:47', '12:11', 'Normal'),       # healthy/bubbly
    ('12:11', '12:20', 'Normal'),       # healthy
    ('12:20', '12:29', 'Normal'),       # healthy
    ('12:29', '12:37', 'Slugging'),     # slugging
    ('12:37', '12:45', 'Slugging'),     # slugging
    ('12:45', '12:56', 'Slugging'),     # slugging
    ('12:56', '13:05', 'Slugging'),     # slugging
    ('13:05', '13:12', 'Normal'),       # healthy
    ('13:12', '13:26', 'Normal'),       # healthy
    ('13:26', '13:34', 'Normal'),       # healthy/churn
    ('13:34', '14:01', 'Normal'),       # healthy (until end of data)
]

# 0626Testday5: Normal only (Test11)
# Data starts 11:00; 11:00-11:18 discarded (startup), 11:33-11:37 (transition)
SEG_0626 = [
    ('11:18', '11:33', 'Normal'),       # Air 120, Water 0.1
    ('11:37', '11:52', 'Normal'),       # Air 150, Water 0.5
]

# 0907Testday2: Blockage (Test2+Test3)
# Data starts 11:50; 11:50-13:24 discarded (pre-test)
SEG_0907_BLOCKAGE = [
    ('13:24', '13:33', 'Normal'),       # Test2_90 (baseline)
    ('13:33', '13:41', 'Blockage_80'),
    ('13:41', '13:49', 'Blockage_70'),
    ('13:49', '13:57', 'Blockage_60'),
    ('13:57', '14:05', 'Blockage_50'),
    ('14:05', '14:14', 'Blockage_40'),
    ('14:14', '14:23', 'Blockage_30'),
    ('14:23', '14:32', 'Blockage_20'),
    ('14:32', '14:34', 'Blockage_10'),
    # 14:34-14:47 discarded (transition to Test3)
    ('14:47', '14:55', 'Normal'),       # Test3_90 (baseline)
    ('14:55', '15:03', 'Blockage_80'),
    ('15:03', '15:11', 'Blockage_70'),
    ('15:11', '15:19', 'Blockage_60'),
    ('15:19', '15:27', 'Blockage_50'),
    ('15:27', '15:35', 'Blockage_40'),
    ('15:35', '15:43', 'Blockage_30'),
    ('15:43', '15:51', 'Blockage_20'),
    ('15:51', '15:59', 'Blockage_10'),
]

# 0907Testday2: Leakage (Test4)
SEG_0907_LEAKAGE = [
    ('16:07', '16:12', 'Leakage_10'),   # Test4
    ('16:12', '16:17', 'Leakage_20'),
    ('16:17', '16:25', 'Leakage_30'),
    ('16:25', '16:30', 'Leakage_15'),
    ('16:30', '16:35', 'Leakage_25'),
    ('16:35', '16:42', 'Leakage_40'),
    ('16:42', '16:50', 'Leakage_90'),
]

# 0911Testday3: Leakage (Test5+Test6)
SEG_0911_LEAKAGE = [
    ('10:34', '10:47', 'Normal'),       # Test5_0 (baseline)
    ('10:47', '11:00', 'Leakage_5'),
    ('11:00', '11:25', 'Leakage_10'),
    ('11:25', '11:28', 'Leakage_15'),
    # 11:28-11:52 discarded (transition to Test6)
    ('11:52', '12:00', 'Normal'),       # Test6_0 (baseline)
    ('12:00', '12:10', 'Leakage_5'),
    ('12:10', '12:20', 'Leakage_10'),
    ('12:20', '12:29', 'Leakage_15'),
    ('12:29', '12:35', 'Leakage_20'),
    ('12:35', '12:45', 'Leakage_25'),
]

# 0911Testday3: Diverted (Test7+Test8)
SEG_0911_DIVERTED = [
    ('13:07', '13:15', 'Diverted_5'),   # Test7
    ('13:15', '13:23', 'Diverted_10'),
    ('13:23', '13:30', 'Diverted_15'),
    ('13:30', '13:40', 'Diverted_20'),
    ('13:40', '13:44', 'Diverted_30'),
    ('13:44', '15:00', 'Diverted_40'),  # gap (13:44-14:47) + Test7_40 (14:47-15:00)
    ('15:00', '15:02', 'Diverted_60'),  # Test7_60
    # 15:02-15:17 discarded (transition to Test8)
    ('15:17', '15:27', 'Diverted_10'),  # Test8
    ('15:27', '15:34', 'Diverted_20'),
    ('15:34', '15:43', 'Diverted_30'),
    ('15:43', '15:50', 'Diverted_40'),
    ('15:50', '15:57', 'Diverted_45'),
    ('15:57', '16:04', 'Diverted_50'),
    ('16:04', '16:12', 'Diverted_60'),
]


# ── Helper functions ───────────────────────────────────────────────────────

def parse_start_time(meta_row):
    """Parse start time from CSV row 1 metadata."""
    raw = meta_row[1].strip()
    for fmt in ('%m/%d/%Y %H:%M', '%m/%d/%Y  %H:%M'):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse start time: {raw!r}")


def find_col_indices(instrument_row):
    """Map target instruments to column indices using row 2."""
    lookup = {col.strip(): i for i, col in enumerate(instrument_row)}
    indices = []
    for inst in INSTRUMENTS:
        if inst not in lookup:
            raise KeyError(f"Instrument {inst} not found. "
                           f"Available: {[c for c in lookup if c]}")
        indices.append(lookup[inst])
    return indices


def hhmm_to_offset(start_time, hhmm):
    """Convert 'HH:MM' to seconds offset from start_time (same date)."""
    h, m = map(int, hhmm.split(':'))
    target = start_time.replace(hour=h, minute=m, second=0, microsecond=0)
    return int((target - start_time).total_seconds())


def read_raw(file_key):
    """Read a raw CSV, return (start_time, list_of_value_rows).

    Timestamps are generated as start_time + row_index * 1s.
    Each value row is a list of 17 string values for the target features.
    """
    path = RAW[file_key]
    print(f"  Reading {file_key}: {os.path.basename(path)} ...", end=' ',
          flush=True)

    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        meta_row = next(reader)
        inst_row = next(reader)
        _name_row = next(reader)  # row 3 (human-readable names) — skip

        start_time = parse_start_time(meta_row)
        col_idx = find_col_indices(inst_row)

        rows = []
        for row in reader:
            values = [row[i] for i in col_idx]
            rows.append(values)

    print(f"{len(rows)} samples from {start_time.strftime('%Y-%m-%d %H:%M')}")
    return start_time, rows


def extract_segments(start_time, all_rows, segments):
    """Extract labelled rows for given segment definitions.

    Returns list of (timestamp_str, values_list, label) tuples.
    """
    result = []
    n = len(all_rows)
    for (t_start, t_end, label) in segments:
        i0 = hhmm_to_offset(start_time, t_start)
        i1 = hhmm_to_offset(start_time, t_end)
        i1 = min(i1, n)  # clamp to file length
        if i0 >= n:
            print(f"    WARNING: segment {t_start}-{t_end} ({label}) "
                  f"starts beyond file end ({n} rows)")
            continue
        if i0 < 0:
            print(f"    WARNING: segment {t_start} ({label}) is before "
                  f"file start, clamping to 0")
            i0 = 0
        count = i1 - i0
        for i in range(i0, i1):
            ts = start_time + timedelta(seconds=i)
            result.append((ts.strftime('%Y-%m-%d %H:%M:%S'),
                           all_rows[i], label))
        print(f"    {t_start}-{t_end}  {label:20s}  {count:6d} rows")
    return result


def write_csv(filename, rows):
    """Write output CSV with header."""
    path = os.path.join(OUT, filename)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for ts, values, label in rows:
            writer.writerow([ts] + values + [label])
    print(f"  Wrote {filename}: {len(rows):,d} rows")


def verify(name, rows):
    """Run sanity checks on extracted data."""
    issues = []

    # Column count
    for i, (ts, vals, lbl) in enumerate(rows[:10]):
        ncols = 1 + len(vals) + 1
        if ncols != 19:
            issues.append(f"Row {i}: {ncols} cols (expected 19)")

    # Check numeric values
    for i, (ts, vals, lbl) in enumerate(rows[:5]):
        for j, v in enumerate(vals):
            try:
                float(v)
            except (ValueError, TypeError):
                issues.append(f"Row {i}, col {j} ({FEATURE_NAMES[j]}): "
                              f"non-numeric value {v!r}")

    # Monotonic timestamps within contiguous blocks
    # (leakage combines 0907+0911, so allow date resets)
    prev_ts = None
    prev_date = None
    for i, (ts, vals, lbl) in enumerate(rows):
        cur_date = ts[:10]
        if prev_ts and cur_date == prev_date and ts < prev_ts:
            issues.append(f"Row {i}: timestamp {ts} < previous {prev_ts}")
            if len(issues) > 5:
                break
        prev_ts = ts
        prev_date = cur_date

    labels = sorted(set(lbl for _, _, lbl in rows))
    if issues:
        print(f"  ISSUES in {name}:")
        for issue in issues[:10]:
            print(f"    - {issue}")
    else:
        print(f"  OK: {name} — {len(rows):,d} rows, labels: {labels}")

    # Print first data row as sample
    ts0, v0, l0 = rows[0]
    sample_vals = ', '.join(f'{float(x):.2f}' for x in v0[:4])
    print(f"       First row: {ts0}  [{sample_vals}, ...]  label={l0}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT, exist_ok=True)

    # Read all 4 raw CSVs
    print("=" * 60)
    print("Reading raw CSVs...")
    print("=" * 60)
    t0912, d0912 = read_raw('0912')
    t0626, d0626 = read_raw('0626')
    t0907, d0907 = read_raw('0907')
    t0911, d0911 = read_raw('0911')

    # Extract segments
    print()
    print("=" * 60)
    print("Extracting segments...")
    print("=" * 60)

    # ── normal.csv ──
    print("\n--- normal.csv (0912 Normal + 0626) ---")
    normal_0912 = extract_segments(
        t0912, d0912,
        [(s, e, l) for s, e, l in SEG_0912 if l == 'Normal'])
    normal_0626 = extract_segments(t0626, d0626, SEG_0626)
    normal_all = normal_0912 + normal_0626

    # ── slugging.csv ──
    print("\n--- slugging.csv (0912 Slugging) ---")
    slugging = extract_segments(
        t0912, d0912,
        [(s, e, l) for s, e, l in SEG_0912 if l == 'Slugging'])

    # ── blockage.csv ──
    print("\n--- blockage.csv (0907 Test2+Test3) ---")
    blockage = extract_segments(t0907, d0907, SEG_0907_BLOCKAGE)

    # ── leakage.csv ──
    print("\n--- leakage.csv (0907 Test4 + 0911 Test5+Test6) ---")
    leak_0907 = extract_segments(t0907, d0907, SEG_0907_LEAKAGE)
    leak_0911 = extract_segments(t0911, d0911, SEG_0911_LEAKAGE)
    leakage = leak_0907 + leak_0911

    # ── diverted.csv ──
    print("\n--- diverted.csv (0911 Test7+Test8) ---")
    diverted = extract_segments(t0911, d0911, SEG_0911_DIVERTED)

    # Write output CSVs
    print()
    print("=" * 60)
    print("Writing output CSVs...")
    print("=" * 60)
    write_csv('normal.csv', normal_all)
    write_csv('slugging.csv', slugging)
    write_csv('blockage.csv', blockage)
    write_csv('leakage.csv', leakage)
    write_csv('diverted.csv', diverted)

    # Verification
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)
    verify('normal.csv', normal_all)
    verify('slugging.csv', slugging)
    verify('blockage.csv', blockage)
    verify('leakage.csv', leakage)
    verify('diverted.csv', diverted)

    # Cross-file checks
    print()
    normal_ts_0912 = set(ts for ts, _, _ in normal_0912)
    slugging_ts = set(ts for ts, _, _ in slugging)
    overlap = normal_ts_0912 & slugging_ts
    if overlap:
        print(f"  WARNING: {len(overlap)} overlapping timestamps "
              f"between normal and slugging!")
    else:
        print("  OK: No overlap between normal and slugging (0912)")

    block_ts = set(ts for ts, _, _ in blockage)
    leak_0907_ts = set(ts for ts, _, _ in leak_0907)
    overlap2 = block_ts & leak_0907_ts
    if overlap2:
        print(f"  WARNING: {len(overlap2)} overlapping timestamps "
              f"between blockage and leakage (0907)!")
    else:
        print("  OK: No overlap between blockage and leakage (0907)")

    leak_0911_ts = set(ts for ts, _, _ in leak_0911)
    div_ts = set(ts for ts, _, _ in diverted)
    overlap3 = leak_0911_ts & div_ts
    if overlap3:
        print(f"  WARNING: {len(overlap3)} overlapping timestamps "
              f"between leakage and diverted (0911)!")
    else:
        print("  OK: No overlap between leakage and diverted (0911)")

    total = (len(normal_all) + len(slugging) + len(blockage)
             + len(leakage) + len(diverted))
    print(f"\nTotal: {total:,d} samples across 5 files")
    print("Done.")


if __name__ == '__main__':
    main()
