import csv
import math
from pathlib import Path
import re
import numpy as np

def parse_quarter_headers(header):
  # match quarterly headers to column positions
  if (header is None or len(header) < 3):
    raise ValueError("Expected crop, geolocation and quarterly columns.")

  periods = {}
  for column in range(2, len(header)):
    # match year and quarter, for example 2010 quarter1
    match = re.fullmatch(
      r"(\d{4}) Quarter([1-4])",
      header[column].strip()
    )
    if (match is None):
      raise ValueError(f"Unrecognized quarterly header: {header[column]!r}")

    year = int(match.group(1))
    quarter = int(match.group(2))
    # consecutive quarter numbers make gaps easy to detect across years
    period = year * 4 + quarter - 1
    if (period in periods):
      raise ValueError(f"Duplicate quarter: {header[column]!r}")
    periods[period] = column

  return periods


def parse_production_values(row, periods, row_number):
  # read production values, skip missing entries and preserve zero
  production = {}
  for period in periods:
    column = periods[period]
    value = ""
    if (column < len(row)):
      value = row[column].strip()

    # missing values must never become zero production
    if (value == ".."):
      continue
    try:
      amount = float(value.replace(",", ""))
    except ValueError as error:
      raise ValueError(
        f"Invalid production on row {row_number}: {value!r}"
      ) from error
    if (not math.isfinite(amount) or amount < 0):
      raise ValueError(f"Invalid production on row {row_number}: {value!r}")
    production[period] = amount

  return production


def read_filtered_dataset(path):
  # group quarterly production by crop and region
  series = {}

  # the filtered csv starts with its headers, without the dataset title
  with open(path, "r", encoding="utf-8", newline="") as file:
    reader = csv.reader(file)
    header = next(reader, None)
    periods = parse_quarter_headers(header)

    for row_number, row in enumerate(reader, start=2):
      if (not row):
        continue
      if (len(row) < 2 or not row[0].strip() or not row[1].strip()):
        raise ValueError(f"Missing crop or region on row {row_number}.")

      crop = row[0].strip()
      region = row[1].strip().lstrip(".").strip()
      key = (crop, region)
      if (key in series):
        raise ValueError(f"Duplicate crop and region: {key!r}")

      series[key] = parse_production_values(row, periods, row_number)

  return series


def build_comparison_records(series):
  # compare each quarter with the same quarter last year
  records = []

  for key in series:
    crop, region = key
    production = series[key]
    for target in sorted(production):
      # four quarters back is the same quarter of the previous year
      previous = target - 4
      if (previous not in production):
        continue

      result = 0
      if (production[target] > production[previous]):
        result = 1

      year = target // 4
      quarter = target % 4 + 1
      record = {
        "crop": crop,
        "region": region,
        "year": year,
        "quarter": quarter,
        "comparison": f"{year} Q{quarter} vs {year - 1} Q{quarter}",
        "same_quarter_previous_year": production[previous],
        "this_quarter": production[target],
        "production": result,
      }

      records.append(record)

  return records


def get_record_quarter(record):
  # return a quarter number for chronological sorting
  return record["year"] * 4 + record["quarter"] - 1


def assign_train_test_sets(records):
  # assign earlier quarters to train, later quarters to test, aim for 80 percent train
  if (not records):
    return records

  records.sort(key=get_record_quarter)
  counts = {}
  for record in records:
    period = get_record_quarter(record)
    if (period not in counts):
      counts[period] = 0
    counts[period] += 1

  periods = sorted(counts)
  cutoff = periods[-1] + 1

  # keep each quarter together, choose the boundary closest to 80 percent of rows
  if (len(periods) > 1):
    target_count = len(records) * 0.8
    train_count = 0
    best_difference = float("inf")
    for index in range(len(periods) - 1):
      train_count += counts[periods[index]]
      difference = abs(train_count - target_count)
      if (difference < best_difference):
        best_difference = difference
        cutoff = periods[index + 1]

  for record in records:
    if (get_record_quarter(record) < cutoff):
      record["set"] = "train"
    else:
      record["set"] = "test"

  return records


def prepare_dataset(path=None):
  # load comparison records with training and testing assignments
  if (path is None):
    path = Path(__file__).resolve().parent / "dataset_filtered.csv"

  series = read_filtered_dataset(path)
  records = build_comparison_records(series)
  assign_train_test_sets(records)

  return records


def print_records(records):
  for record in records:
    print(record)


def save_prepared_dataset(records, path=None):
  # save the arranged table, overwrite an earlier output
  if (path is None):
    path = Path(__file__).resolve().parent / "dataset_prepared.csv"

  columns = [
    "crop", "region", "year", "quarter", "comparison",
    "same_quarter_previous_year", "this_quarter", "production", "set",
  ]
  with open(path, "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)

# splits the training and testing data
def split_per_year(features, labels, years, year, seed=170):
  # same seed + year always gives the same shuffle, so every method gets the same split
  rand = np.random.default_rng(seed + year)

  idx = np.where(years == year)[0]  # rows that belong to a specific year
  rand.shuffle(idx)  # randomizes the ordering

  split_length = int(0.70 * len(idx))
  train_idx, test_idx = idx[:split_length], idx[split_length:]

  training = {"features": features[train_idx], "labels": labels[train_idx]}
  testing = {"features": features[test_idx]}
  actual_labels = labels[test_idx]

  return training, testing, actual_labels, test_idx

def main():
  records = prepare_dataset()
  save_prepared_dataset(records)
  print_records(records)


if (__name__ == "__main__"):
  main()
