# CMSC 170: Artificial Intelligence Project
# Group Name: Fourcast
# Master Pipeline: Preprocessing & Comparative Evaluation of GNB, KNN, and Perceptron

import csv
import sys
import numpy as np

# Import preprocessing pipeline
from data_loader import prepare_dataset, save_prepared_dataset

# Import model-specific feature generators and prediction functions
from gaussian_naive_bayes import (
    make_gnb_features,
    calculate_mean_and_var,
    class_priors,
    predict_increaase_decrease
)
from k_nearest_neighbor import (
    make_knn_features,
    predict_minkowski_knn,
    evaluate_knn
)
from perceptron import (
    load_dataset,
    make_perceptron_features,
    split_dataset,
    standardize_features,
    train_perceptron,
    predict_perceptron,
    evaluate_perceptron
)


def run_pipeline():
    print("==================================================================")
    print("      CMSC 170 PROJECT: FOURCAST AGRICULTURAL YIELD PREDICTION    ")
    print("==================================================================\n")

    # Step 1: Data Preparation
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
            int(r["production"])
        )

    # Step 2: Evaluate Gaussian Naive Bayes
    print("[2/4] Running Gaussian Naive Bayes...")
    train_gnb, test_gnb, gnb_actuals = make_gnb_features(data_table)
    mean_var = calculate_mean_and_var(train_gnb["features"], train_gnb["labels"])
    priors = class_priors(mean_var)
    all_vars = np.concatenate([mean_var[0]["var"], mean_var[1]["var"]])
    epsilon = 1 * all_vars.max()
    gnb_preds = predict_increaase_decrease(test_gnb["features"], mean_var, priors, epsilon)
    gnb_corr, gnb_inc, gnb_acc, gnb_prec, gnb_rec, gnb_f1 = evaluate_knn(gnb_actuals, gnb_preds)

    # Step 3: Evaluate K-Nearest Neighbors (k=5, p=2)
    print("[3/4] Running K-Nearest Neighbors (k=5, p=2)...")
    train_knn, test_knn, knn_actuals, _ = make_knn_features(data_table)
    knn_preds = predict_minkowski_knn(
        train_knn["features"],
        train_knn["labels"],
        test_knn["features"],
        k=5,
        p=2.0
    )
    knn_corr, knn_inc, knn_acc, knn_prec, knn_rec, knn_f1 = evaluate_knn(knn_actuals, knn_preds)

    # Step 4: Evaluate Perceptron (learning_rate=0.01, max_epochs=100)
    print("[4/4] Running Perceptron (learning_rate=0.01, epochs=100)...")
    p_feats, p_labels, p_sets = make_perceptron_features(data)
    px_tr, px_te, py_tr, py_te = split_dataset(p_feats, p_labels, p_sets)
    px_tr_scaled, px_te_scaled = standardize_features(px_tr, px_te)
    p_weights, p_bias = train_perceptron(px_tr_scaled, py_tr, learning_rate=0.01, max_epochs=100)
    p_preds = predict_perceptron(px_te_scaled, p_weights, p_bias)
    p_acc, p_prec, p_rec, p_f1 = evaluate_perceptron(py_te, p_preds)

    # Step 5: Output Comparative Summary Table
    print("\n=========================================================================")
    print("                     COMPARATIVE MODEL RESULTS SUMMARY                   ")
    print("=========================================================================")
    print(f"{'Method / Model':<25} {'Accuracy (%)':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 75)
    print(f"{'Gaussian Naive Bayes':<25} {gnb_acc:<15.4f} {gnb_prec:<12.4f} {gnb_rec:<12.4f} {gnb_f1:<12.4f}")
    print(f"{'K-Nearest Neighbors (k=5)':<25} {knn_acc:<15.4f} {knn_prec:<12.4f} {knn_rec:<12.4f} {knn_f1:<12.4f}")
    print(f"{'Perceptron (lr=0.01)':<25} {p_acc*100:<15.4f} {p_prec:<12.4f} {p_rec:<12.4f} {p_f1:<12.4f}")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_pipeline()