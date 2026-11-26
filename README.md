# CMSC-170-PROJECT

A project that has at least three AI or machine learning methods to one dataset, implement them from scratch, and compare how well they perform.

## GitHub IO Page Link
https://aaladines.github.io/CMSC-170-PROJECT/

## How to Run on Linux/WSL

### I. Prerequisites

Make sure you have Python 3 installed.

### II. Go to the source code folder

```bash
cd source_code
```

### III. Create a Virtual Environment

Create a virtual environment in the project's root directory:

```bash
python3 -m venv venv
```

### IV. Activate the Virtual Environment
```bash
source venv/bin/activate
```

Once activated, your terminal should display `(venv)` before the command prompt.

### V. Install Dependencies

With the virtual environment activated, install the required libraries:

```bash
pip install numpy plotly
```

### VI. Prepare the Dataset (if the individual methods are to be run one by one)

```bash
python filter_dataset.py
```

```bash
python data_loader.py
```

### VII. Run Main
```bash
python main.py
```

## Dataset Link
- [Vegetables and Root Crops: Volume of Production, by Region, Province, Quarter, and Semester, 2010-2026](https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2E__CS/0082E4EVCP3.px/?rxid=7b53d3cc-7504-4f45-a8af-781c15e70037)

## Task Log of Members

**Ken Kristoffer F. Enrile**

**Angelo Gabriel A. Ladines**  

**Lawrence Andrew Maminta**

**Arianne Marie R. Reyes**
- **From-Scratch Model & Metric Implementation (`k_nearest_neighbor.py`, `gaussian_naive_bayes.py`):** Implemented the Minkowski/Euclidean K-Nearest Neighbors classifier from scratch, tuned the neighborhood hyperparameter (`k = 19`), and added evaluation metric calculations (`evaluate_gnb`) for Gaussian Naive Bayes.
- **Unified Evaluation Pipeline (`main.py`) & Git Integration:** Built the `main.py` pipeline to run Gaussian Naive Bayes, KNN, and Perceptron across all evaluation years, resolved Perceptron output unpacking in `main.py`, and merged feature branches (`perceptron` into `milestone2`).
- **Dataset Scope & Splitting (`filter_dataset.py`, `data_loader.py`):** Updated the train/test split configuration to 80/20 and excluded partial 2026 records to standardize the 15-year (2011–2025; 9,600-row) evaluation dataset.
- **GitHub Pages UI & Dynamic Tables (`index.html`, `style.css`, `script.js`, `results.js`):** Styled the project page with updated typography and a green, yellow, and white theme (`style.css`), reorganized the layout of `index.html` for smoother section flow, and implemented dynamic yearly and 15-year summary table rendering via `results.js` and `script.js`.
- **Technical Write-Up & Documentation (`index.html`):** Documented dataset-specific implementation details under Methods, expanded the Discussion (external climatic/economic factors and Perceptron convergence analysis), Conclusion, and References.
