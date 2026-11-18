# REFERENCES
# https://www.geeksforgeeks.org/k-nearest-neighbours/
# https://www.saedsayad.com/k_nearest_neighbors.htm
# https://www.krishnaik.in/ai-tutorials/statistics/standardization-and-normalization

import csv
import sys
import numpy as np
from data_loader import split_per_year

def make_knn_features(data_table):
    
    # Extracts features for the KNN model and standardizes them.
    # Three features are created:
    # 1. Log-transformed production volume for the previous year same quarter.
    # 2. Year-over-Year change in production compared to the previous year.
    # 3. Quarter-over-Quarter change in production from the preceding quarter.
    
    features, labels, years  = [], [], []
    for key, value in data_table.items():
        crop, region, year, quarter = key

        # previous quarter and current quarter volume
        prev_q, this_q, production = value 

        # Feature 1: Log-transformed previous year same quarter production
        prev_q_feat = np.log(1 + prev_q)

        # Feature 2: Year-over-Year change from last year's same quarter
        last_year_same_quarter = data_table.get((crop, region, year-1, quarter))
        if last_year_same_quarter is not None:
            last_year_change = np.log(1 + last_year_same_quarter[1]) - np.log(1 + last_year_same_quarter[0])
        else:
            last_year_change = 0.0

        # Feature 3: Quarter-over-Quarter change from previous quarter
        if quarter > 1:
            prev_quarter_year = year
            prev_quarter = quarter - 1
        else:
            prev_quarter_year = year - 1
            prev_quarter = 4

        prev_quarter_row = data_table.get((crop, region, prev_quarter_year, prev_quarter))
        if prev_quarter_row is not None:
            prev_quarter_change = np.log(1 + prev_quarter_row[1]) - np.log(1 + prev_quarter_row[0])
        else:
            prev_quarter_change = 0.0

        features.append([prev_q_feat, last_year_change, prev_quarter_change])
        labels.append(production)
        years.append(year)

    features, labels, years = np.array(features), np.array(labels), np.array(years)


    # Feature Standardization (Z-score normalization)
    # Distance-based algorithms like KNN require normalized features so large magnitudes do not dominate
    # mean = np.mean(features, axis=0)
    # std = np.std(features, axis=0)
    # std[std == 0] = 1.0  # Prevent division by zero
    # features_scaled = (features - mean) / std

    # split_length = int(0.8 * len(data_table)) # 80% train, 20% test split
    
    # training = {"features": features_scaled[:split_length], "labels": labels[:split_length]}
    # testing = {"features": features_scaled[split_length:]}
    # actual_labels = labels[split_length:]

    return features, labels, years

def standardize_features(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0] = 1.0  # Prevent division by zero

    return (x_train - mean) / std, (x_test - mean) / std
def predict_minkowski_knn(train_features, train_labels, test_features, k, p):
    
    # Classifies test instances using k-Nearest Neighbors with Minkowski distance.
    # Minkowski Distance formula: d(x, y) = ( sum(|x_i - y_i|^p) )^(1/p)
    # p = 1: Manhattan Distance
    # p = 2: Euclidean Distance
    
    predicted_labels = []

    for test_x in test_features:
        # Calculate Minkowski distance between test point and all training points
        distances = np.sum(np.abs(train_features - test_x) ** p, axis=1) ** (1.0 / p)
        
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_labels = train_labels[k_indices]

        # Majority voting
        counts = np.bincount(k_labels)
        predicted_class = np.argmax(counts)
        predicted_labels.append(predicted_class)

    return predicted_labels


def evaluate_knn(y_test, predictions):
    
    # Evaluates predictions against true test labels to compute
    # Accuracy, Precision, Recall, and F1-Score
    
    y_test = np.array(y_test)
    predictions = np.array(predictions)

    true_positive = np.sum((y_test == 1) & (predictions == 1))
    true_negative = np.sum((y_test == 0) & (predictions == 0))
    false_positive = np.sum((y_test == 0) & (predictions == 1))
    false_negative = np.sum((y_test == 1) & (predictions == 0))

    correct = true_positive + true_negative
    incorrect = false_positive + false_negative
    total = len(y_test)

    accuracy = (correct / total) if total > 0 else 0.0
    precision = (true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
    recall = (true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return correct, incorrect, accuracy, precision, recall, f1_score


def knn():
    # Opens the prepared dataset CSV file
    path = "dataset_prepared.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    try:
        with open(path, newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Cannot find {path}")
        return

    # Stores raw dataset in a lookup dictionary
    data_table = {}
    for row in data:
        key = (row["crop"], row["region"], int(row["year"]), int(row["quarter"]))
        data_table[key] = (float(row["same_quarter_previous_year"]), float(row["this_quarter"]), int(row["production"]))

    # Prepare features for every row (the split happens later, per year)
    features, labels, years = make_knn_features(data_table)

    print("Dataset loaded successfully")
    print(f"Total number of records: {len(data)}\n")

    # User inputs for k and p parameters
    while True:
        try:
            k_input = input("Enter the odd value of k: ")
            k = int(k_input)
            if k > 0 and k % 2 != 0:
                break
            else:
                print("Invalid input! k must be a positive odd integer.")
        except ValueError:
            print("Please enter a valid integer.")

    while True:
        try:
            p_input = input("Enter the value of p for Minkowski distance (1 for Manhattan, 2 for Euclidean): ")
            p = float(p_input)
            if p > 0:
                break
            else:
                print("Invalid input! p must be greater than 0.")
        except ValueError:
            print("Please enter a valid numeric value.")

    print(f"\nk = {k}")
    print(f"p = {p}\n")

    # One result per year
    for year in sorted(set(years.tolist())):
        # Separate the training data and testing data of this year
        training, testing, actual_labels, test_idx = split_per_year(features, labels, years, year)
        training_no = len(training["labels"])
        testing_no = len(actual_labels)

        # Makes sure that the data per year are enough
        if testing_no == 0 or len(set(training["labels"].tolist())) < 2:
            print(f"YEAR {year} has not enough data!\n")
            continue

        # Standardize using this year's training data only
        train_x, test_x = standardize_features(training["features"], testing["features"])

        # Run KNN classification (k cannot be larger than the training set)
        predicted_labels = predict_minkowski_knn(train_x, training["labels"], test_x, min(k, training_no), p)

        # Evaluate metrics matching perceptron.py format
        correct, incorrect, accuracy, precision, recall, f1_score = evaluate_knn(actual_labels, predicted_labels)

        # Collect incorrect classifications list (data_no = row number in the csv)
        incorrect_list = []
        for i in range(len(predicted_labels)):
            data_no = test_idx[i] + 1
            if predicted_labels[i] != actual_labels[i]:
                incorrect_list.append((data_no, actual_labels[i], predicted_labels[i]))

        # Print KNN Classification Results
        print("---------------------------------------------------------------")
        print(f"KNN CLASSIFICATION RESULTS - YEAR {year}\n")
        print(f"Records in year: {training_no + testing_no}")
        print(f"Training instances: {training_no}")
        print(f"Testing instances: {testing_no}\n")
        print(f"Correct predictions: {correct}")
        print(f"Incorrect predictions: {incorrect}")
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}\n")

        # Display first 20 incorrect predictions
        # print("Incorrect Classifications (first 20):")
        # print(f"{'Data Record#':<15} {'Actual':<11} {'Predicted':<8}")
        # for rec_num, actual_val, pred_val in incorrect_list[:20]:
        #     actual_str = "Increase" if actual_val == 1 else "Decrease"
        #     pred_str = "Increase" if pred_val == 1 else "Decrease"
        #     print(f"{rec_num:<15} {actual_str:<11} {pred_str:<8}")


if __name__ == "__main__":
    knn()