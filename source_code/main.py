# CMSC 170: Artificial Intelligence Project
# Group Name: Fourcast
# Master Pipeline: Preprocessing & Per-Year Comparative Evaluation of GNB, KNN, and Perceptron

import numpy as np
import plots
from data_loader import prepare_dataset, save_prepared_dataset, split_per_year

from gaussian_naive_bayes import (
    make_gnb_features,
    calculate_mean_and_var,
    class_priors,
    predict_increaase_decrease,
    evaluate_gnb,
)
from k_nearest_neighbor import (
    make_knn_features,
    standardize_features as knn_standardize,
    predict_minkowski_knn,
    evaluate_knn,
)
from perceptron import (
    load_dataset,
    make_perceptron_features,
    standardize_features as perceptron_standardize,
    train_perceptron,
    predict_perceptron,
    evaluate_perceptron,
)

GNB_K = 1e-9
KNN_K, KNN_P = 19, 2.0
PERCEPTRON_LR, PERCEPTRON_EPOCHS = 0.01, 100

# Per-year results
def run_per_year(features, labels, years, fit_predict, name):
    results = {}
    for year in sorted(set(years.tolist())):
        training, testing, actual_labels, _ = split_per_year(features, labels, years, year)

        if len(actual_labels) == 0 or len(set(training["labels"].tolist())) < 2:
            print(f"  [{name}] year {year}: not enough data, skipped")
            continue

        preds = fit_predict(training, testing)
        results[year] = (np.array(actual_labels), np.array(preds))
    return results


# Models
def gnb_fit_predict(training, testing):
    mean_var = calculate_mean_and_var(training["features"], training["labels"])
    priors = class_priors(mean_var)
    all_vars = np.concatenate([mean_var[0]["var"], mean_var[1]["var"]])
    epsilon = GNB_K * all_vars.max()
    return predict_increaase_decrease(testing["features"], mean_var, priors, epsilon)


def knn_fit_predict(training, testing):
    train_x, test_x = knn_standardize(training["features"], testing["features"])
    k = min(KNN_K, len(training["labels"]))
    return predict_minkowski_knn(train_x, training["labels"], test_x, k=k, p=KNN_P)


def perceptron_fit_predict(training, testing):
    x_tr, x_te = perceptron_standardize(training["features"], testing["features"])
    result = train_perceptron(
        x_tr, training["labels"],
        learning_rate=PERCEPTRON_LR, max_epochs=PERCEPTRON_EPOCHS,
    )
    weights, bias, *_ = result
    return predict_perceptron(x_te, weights, bias)


# Metrics for one year
def score(evaluator, actual, pred):
    out = evaluator(actual, pred)
    return out[-4:]

def to_binary(value):
    """Works for both 0/1 and -1/+1 labels: positive = Increase (1), else Decrease (0)."""
    return 1 if value > 0 else 0

def run_pipeline():
    print("=" * 95)
    print("               CMSC 170 PROJECT: FOURCAST AGRICULTURAL YIELD PREDICTION")
    print("=" * 95)

    # Step 1: Data preparation
    print("[1/4] Preparing dataset and generating dataset_prepared.csv...")
    records = prepare_dataset()
    save_prepared_dataset(records)
    print(f"Total dataset records: {len(records)}\n")

    # Load dataset using perceptron.py's load_dataset function
    data = load_dataset("dataset_prepared.csv")
    data_table = {}
    for r in data:
        key = (r["crop"], r["region"], int(r["year"]), int(r["quarter"]))
        data_table[key] = (
            float(r["same_quarter_previous_year"]),
            float(r["this_quarter"]),
            int(r["production"]),
        )

    # Step 2: Gaussian Naive Bayes
    print(f"[2/4] Running Gaussian Naive Bayes (k={GNB_K:g}), per year...")
    g_feats, g_labels, g_years = make_gnb_features(data_table)
    gnb_results = run_per_year(g_feats, g_labels, g_years, gnb_fit_predict, "GNB")

    # Step 3: Evaluate K-Nearest Neighbors
    print(f"[3/4] Running K-Nearest Neighbors (k={KNN_K}, p={KNN_P:g}), per year...")
    k_feats, k_labels, k_years = make_knn_features(data_table)
    knn_results = run_per_year(k_feats, k_labels, k_years, knn_fit_predict, "KNN")

    # Step 4: Evaluate Perceptron (learning_rate=0.01, max_epochs=100)
    print(f"[4/4] Running Perceptron (lr={PERCEPTRON_LR}, epochs={PERCEPTRON_EPOCHS}), per year...")
    p_feats, p_labels, p_records = make_perceptron_features(data)
    p_feats, p_labels = np.array(p_feats), np.array(p_labels)
    p_years = np.array([int(r["year"]) for r in p_records])
    perc_results = run_per_year(p_feats, p_labels, p_years, perceptron_fit_predict, "Perceptron")

    # Step 5: One comparison table per year (+ collect data for the charts)
    models = [
        ("Gaussian Naive Bayes", gnb_results, evaluate_gnb),
        ("K-Nearest Neighbors", knn_results, evaluate_knn),
        ("Perceptron", perc_results, evaluate_perceptron),
    ]
    all_years = sorted(set(gnb_results) | set(knn_results) | set(perc_results))
 
    scores = {}        # scores[model][year] = (acc, prec, rec, f1)
    confusion = {name: np.zeros((2, 2), dtype=int) for name, _, _ in models}  # rows=actual, cols=pred
    class_counts = {}  # class_counts[year] = (n_decrease, n_increase)
 
    for year in all_years:
        print("=" * 95)
        print(f"                     COMPARATIVE MODEL RESULTS SUMMARY: YEAR {year}")
        print("=" * 95)
 
        print(f"{'Method / Model':<26} {'no. of data':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 95)
        for name, results, evaluator in models:
            if year not in results:
                print(f"{name:<26} {'-':<12} {'-':<12} {'-':<12} {'-':<12} {'-':<12}")
                continue
            actual, pred = results[year]
            acc, prec, rec, f1 = score(evaluator, actual, pred)
            print(f"{name:<26} {len(actual):<12} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")
 
            # collect for charts
            scores.setdefault(name, {})[year] = (acc, prec, rec, f1)
            for a, p in zip(actual, pred):
                confusion[name][to_binary(a), to_binary(p)] += 1
            if year not in class_counts:
                bin_actual = [to_binary(a) for a in actual]
                class_counts[year] = (bin_actual.count(0), bin_actual.count(1))
 
        print()

    # THIS IS THE RESULT FOR THE TABLES AND CHART
    # Step 6: One table per metric (rows = year, columns = the three methods)
    model_names = [name for name, _, _ in models]
    for i, metric in enumerate(["Accuracy", "Precision", "Recall", "F1-Score"]):
        print("=" * 95)
        print(f"                                      {metric.upper()} PER YEAR")
        print("=" * 95)
        header = f"{'Year':<10}" + "".join(f"{n:<26}" for n in model_names)
        print(header)
        print("-" * len(header))
        for year in all_years:
            row = f"{year:<10}"
            for name in model_names:
                if year in scores.get(name, {}):
                    row += f"{scores[name][year][i]:<26.4f}"
                else:
                    row += f"{'-':<26}"
            print(row)
        print()
    
    plots.show_all(scores, confusion, class_counts)

if __name__ == "__main__":
    run_pipeline()