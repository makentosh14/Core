import csv
import os

src = "scanner_data/raw/research_events.csv"
out = "scanner_data/outcomes/outcome_labels.csv"

os.makedirs("scanner_data/outcomes", exist_ok=True)

if not os.path.exists(src):
    print(f"ERROR: File not found: {src}")
    exit(1)

size_mb = os.path.getsize(src) / 1024 / 1024
print(f"Source: {src} ({size_mb:.1f} MB)")
print("Streaming labeled rows to output ...")

exported = 0
writer = None
fout = None

with open(src, newline="", encoding="utf-8") as fin:
    reader = csv.DictReader(fin)
    for row in reader:
        v = str(row.get("outcome_labeled", "")).lower()
        if v not in ("true", "1"):
            continue
        if writer is None:
            fout = open(out, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        exported += 1
        if exported % 100000 == 0:
            print(f"  ... exported {exported} rows so far")

if fout:
    fout.close()

print(f"Done. Exported {exported} labeled rows to {out}")
