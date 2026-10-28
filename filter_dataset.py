import csv
from pathlib import Path


# store the ten selected crop names in lowercase for matching
CROPS = set()
for crop in (
    "Ampalaya fruit [Bitter gourd]",
    "Camote [Sweet potato]",
    "Cassava, fresh tubers",
    "Eggplant",
    "Tomato",
    "Gabi [Taro]",
    "Ginger",
    "Upo [Bottle gourd]",
    "Kangkong [Swamp cabbage]",
    "Okra [Lady's finger]",
):
  CROPS.add(crop.lower())



def filter_dataset():
  folder = Path(__file__).resolve().parent
  source = folder / "dataset.csv"
  output = folder / "dataset_filtered.csv"
  kept = 0  # count saved data rows, excluding the header

  # read one row at a time so the entire dataset is never loaded into memory
  # mode "w" overwrites the previous filtered dataset
  with open(source, "r", encoding="utf-8", newline="") as infile:
    with open(output, "w", encoding="utf-8", newline="") as outfile:
      reader = csv.reader(infile)
      writer = csv.writer(outfile)

      # skip the dataset title, the second row contains the actual headers
      next(reader, None)
      header = next(reader, None)
      if (header is not None):
        # keep crop and geolocation (indices 0 and 1)
        # each year repeats: q1, q2, s1, q3, q4, s2, annual
        # after the first two columns, offsets 0, 1, 3, 4 select q1-q4
        columns = []
        for i in range(len(header)):
          if (i < 2 or (i - 2) % 7 in (0, 1, 3, 4)):
            columns.append(i)
        filtered_header = []
        for i in columns:
          filtered_header.append(header[i])
        writer.writerow(filtered_header)

        for row in reader:
          # ignore blank or incomplete rows without a geolocation field
          if (len(row) < 2):
            continue

          # compare crop names
          if (row[0].strip().lower() not in CROPS):
            continue

          location = row[1].strip().upper()
          # exclude nir
          if (location == "..NEGROS ISLAND REGION (NIR) 2/"):
            continue
          
          # regions start with two dots, provinces start with four
          # this also excludes philippines, which has no dot prefix
          if (location.startswith("..") and not location.startswith("....")):
            # write selected columns, filling any missing trailing values
            filtered_row = []
            for i in columns:
              if (i < len(row)):
                filtered_row.append(row[i])
              else:
                filtered_row.append("")
            writer.writerow(filtered_row)
            kept += 1

  print(f"Saved {kept} data rows to {output.name}")

filter_dataset()
