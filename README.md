# CMSC-170-PROJECT

A project that has at least three AI or machine learning methods to one dataset, implement them from scratch, and compare how well they perform.

## How to Run on Linux/WSL

### 1. Prerequisites

Make sure you have Python 3 installed.

### 2. Create a Virtual Environment

Create a virtual environment in the project's root directory:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
```bash
source venv/bin/activate
```

Once activated, your terminal should display `(venv)` before the command prompt.

### 4. Install Dependencies

With the virtual environment activated, install the required libraries:

```bash
pip install numpy plotly
```

### 5. Prepare the Dataset (if the individual methods are to be run one by one)

```bash
python filter_dataset.py
```

```bash
python data_loader.py
```

### 6. Run Main
```bash
python main.py
```

The main file performs the following operations:

1. **Data Preparation** – Processes the raw dataset and generates `dataset_prepared.csv`.
2. **Gaussian Naive Bayes (GNB)** – Runs the Gaussian Naive Bayes classifier.
3. **K-Nearest Neighbors (KNN)** – Runs the KNN classifier.
4. **Perceptron** – Runs the Perceptron classifier.
5. **Comparative Evaluation** – Displays Accuracy, Precision, Recall, and F1-Score for all three algorithms, evaluated per year.
6. **Visualization** – Generates performance comparison plots and confusion matrices.

### 7. Deactivate the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

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
