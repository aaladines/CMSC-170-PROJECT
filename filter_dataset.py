import csv
from pathlib import Path



def filter_dataset():
    folder = Path(__file__).resolve().parent
    source = folder / "dataset.csv"
    output = folder / "dataset_filtered.csv"
    kept = 0

    with source.open("r", encoding="utf-8-sig", newline="") as infile:
        with output.open("x", encoding="utf-8-sig", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            header = next(reader, None)
            if header is not None:
                writer.writerow(header)
            for row in reader:
                if len(row) < 2:
                    continue
                location = row[1].strip()
                if location.upper() == "PHILIPPINES" or (
                    location.startswith("..") and not location.startswith("....")
                ):
                    writer.writerow(row)
                    kept += 1

    print(f"Saved {kept:,} data rows to {output.name}")



filter_dataset()