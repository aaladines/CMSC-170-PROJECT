# CMSC-170-PROJECT

A project that has at least three AI or machine learning methods to one dataset, implement them from scratch, and compare how well they perform.

## Prepared dataset

Run `python data_loader.py` to read `dataset_filtered.csv`

Each output row compares one crop and region with the same quarter last year:

| Column                       | Meaning                                                           |
| ---------------------------- | ----------------------------------------------------------------- |
| `crop`                       | Crop name                                                         |
| `region`                     | Region name without the leading dots                              |
| `year`                       | Target year                                                       |
| `quarter`                    | Target quarter, 1 to 4                                            |
| `comparison`                 | For example, `2011 Q1 vs 2010 Q1`                                 |
| `same_quarter_previous_year` | Production in the same quarter last year                          |
| `this_quarter`               | Actual production in the target quarter                           |
| `production`                 | `1` if higher than last year, `0` if equal or lower               |
| `set`                        | `train` or `test`                                                 |

Both quarters must have actual values, unavailable values such as `..` are
skipped, actual zero production is preserved.

### Training and testing split

Records are sorted chronologically by target year and quarter. Earlier quarters
are assigned to training, later quarters to testing. The boundary is chosen
closest to 80% training rows and 20% testing rows, keeping every crop and region
from the same target quarter in the same set. The exact ratio can vary when a
quarter cannot be divided evenly. With only one target quarter, all rows are
training data because there is no later quarter to test on.

### Using the data

```python
from data_loader import prepare_dataset, print_records

records = prepare_dataset()
print_records(records)
```
