"""
sample_dataset.py
=================
Takes a chronological stride sample from outcome_labels.csv.
Creates a small working dataset that fits in RAM on any laptop.

Run this ONCE, then run pattern_analyzer.py against the sample.
"""

import csv
import os

SRC  = "scanner_data/outcomes/outcome_labels.csv"
OUT  = "scanner_data/outcomes/outcome_labels_sample.csv"
WANT = 30_000   # rows to keep — enough for reliable pattern analysis

if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found. Run export_labels.py first.")
    exit(1)

# Count total rows first (fast — just counts lines)
print(f"Counting rows in {SRC} ...")
with open(SRC, encoding="utf-8") as f:
    total = sum(1 for _ in f) - 1  # minus header
print(f"Total labeled rows: {total:,}")

if total <= WANT:
    print(f"Already <= {WANT} rows. No sampling needed.")
    print(f"Just use {SRC} directly.")
    exit(0)

stride = total // WANT
print(f"Stride: every {stride}th row → ~{total // stride:,} rows sampled")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

written = 0
with open(SRC, newline="", encoding="utf-8") as fin, \
     open(OUT, "w", newline="", encoding="utf-8") as fout:

    reader  = csv.DictReader(fin)
    writer  = csv.DictWriter(fout, fieldnames=reader.fieldnames,
                             extrasaction="ignore")
    writer.writeheader()

    for i, row in enumerate(reader):
        if i % stride == 0:
            writer.writerow(row)
            written += 1

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Done. Wrote {written:,} rows to {OUT} ({size_mb:.1f} MB)")
print()
print("Now run:")
print("  python pattern_analyzer.py --min-samples 5 --max-rows 30000")
