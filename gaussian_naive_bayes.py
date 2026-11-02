# REFERENCES
# https://stats.stackexchange.com/questions/163088/how-to-use-log-probabilities-for-gaussian-naive-bayes
# https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial4.pdf
# https://www.geeksforgeeks.org/machine-learning/gaussian-naive-bayes/
# https://www.saedsayad.com/naive_bayesian.htm

import csv
import sys
import numpy as np

# functions that make the features for the gaussian naive bayes method
# three features will be made.
# 1. production for the previous year
# 2. change from the previous quarter of the same year to the current quarter
# 3. change in production of the following year from last year 
def make_gnb_features(data_table):
    # empty lists for the 
    features, labels = [], []
    for key, value in data_table.items():
        crop, region, year, quarter = key

        # previous quarter and the current quarter
        prev_q, this_q, production = value 

        # feature 1: same_quarter_previous_year
        prev_q_feat = np.log(1 + prev_q)

        # feature 2: last year's change
        last_year_same_quarter = data_table.get((crop, region, year-1, quarter))

        if last_year_same_quarter is not None:
            last_year_change  = np.log(1 + last_year_same_quarter[1]) - np.log(1 + last_year_same_quarter[0])
        else:
            last_year_change = 0.0

        # feature 3: last quarter's change
        if quarter > 1: # if the quarter is 2-4, (the same year still)
            prev_quarter_year = year
            prev_quarter = quarter - 1
        else:
            prev_quarter_year = year - 1
            prev_quarter = 4

        prev_quarter_row = data_table.get((crop, region, prev_quarter_year, prev_quarter))

        # change in production in the earlier year
        if prev_quarter_row is not None:
            prev_quarter_change = np.log(1 + prev_quarter_row[1]) - np.log(1 + prev_quarter_row[0])
        else:
            prev_quarter_change = 0.0

        features.append([prev_q_feat, last_year_change,prev_quarter_change ])
        labels.append(production)
        # make it into a numpy array for easier manipulation
    features, labels = np.array(features), np.array(labels)

    split_length = int(0.8 * len(data_table)) #  the training data is 80% while the testing data is 20%
    
    training = {"features": features[:split_length], "labels": labels[:split_length]} # training includes the label
    testing = {"features": features[split_length:]} # testing does not include the label
    actual_labels = labels[split_length:] # this is the correct label for the testing data

    return training, testing, actual_labels

 # gets the mean and variance for classifications (0 and 1) -> increase/decrease
def calculate_mean_and_var(features, labels):
    mean_and_var = {}

    # 0 for decrease and 1 for increase
    for classification in (0, 1):
        # appends the rows that belongs in a particular classification
        rows = []
        for i in range(len(features)):
            if labels[i] == classification: #if label and classification are the same, append
                rows.append(features[i])

        mean_and_var[classification] = {
            "n": len(rows), #no of rows per classification
            "mean": np.mean(rows, axis=0), #mean
            "var": np.var(rows, axis=0) #variance
        }
    return mean_and_var

# priors 
def class_priors(mean_and_var):
    n_decrease = mean_and_var[0]["n"]
    n_increase = mean_and_var[1]["n"]
    total = n_decrease + n_increase # aka no. of training rows

    prior_proba_decrease = n_decrease / total
    prior_proba_increase = n_increase / total

    # if 0, decrease, else, increase
    return {0: prior_proba_decrease, 1: prior_proba_increase}

# the gaussian likelihood as well
def gaussian_pdf(x, mean, var):
    # Formula:
    # P(x | mean, sigma^2) = (1 / sqrt(2 * pi * sigma^2)) * exp(- (x - mean)^2 / (2 * sigma^2))
    return (1 / np.sqrt(2 * np.pi * var)) * (np.exp(-((x - mean)**2) / (2 * var)))    

# P(x|c) -> likelihood
# P(c) -> class prior probability
# P(x) -> predictor prior probability
# P(x) -> posterior probability
# P(c|x) = (P(x|c) * P(c)) / P(x)

# naive bayesian formula (https://www.saedsayad.com/naive_bayesian.htm)
# P(c | x) = P(x1|c) * P(x2|c) * ... P(xn|c) * P(c)
def predict_increaase_decrease(features, mean_and_var, priors, epsilon):
    predicted_label = []
    for x in features:
        scores = {}
        
        for classification in (0,1):
            # epsilon is a smoothing factor
            var = mean_and_var[classification]["var"] + epsilon
             # P(x1|c) * P(x2|c) * P(x3|c) * P(c) 
            #  x1, x2, x3 are the features
            scores[classification] = np.prod(gaussian_pdf(x, mean_and_var[classification]["mean"], var)) * priors[classification] 

        # returns 1 or 0 depends on which score is better
        if scores[classification] >= scores[0]:
            predicted_label.append(1) 
        else:
            predicted_label.append(0)

    return predicted_label

# Function to evaluate GNB predictions
def evaluate_gnb(y_test, predictions):
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

def gaussian_naive_bayes():
    # opens the prepared dataset
    path = "dataset_prepared.csv"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    try:
        with open(path, newline="", encoding="utf-8") as f:
            # stored as a dictionary based on the label on the first row. 
            # ex. {crop:... , region: ..., year, quarter, ... }
            data = list(csv.DictReader(f))

    except FileNotFoundError:
        print(f"Cannot find {path}")
        return
    # print(data)

    # stores the data into a dictionary
    data_table = {}

    # sample: ('Ampalaya fruit [Bitter gourd]', 'CORDILLERA ADMINISTRATIVE REGION (CAR)', 2011, 1) --> (162.58, 165.01)
    for row in data:
        key = (row["crop"], row["region"], int(row["year"]), int(row["quarter"]))
        data_table[key] = (float(row["same_quarter_previous_year"]), float(row["this_quarter"]), int(row["production"]))

    # print(data_table[:10])
    # for key, value in list(damta_table.items())[:10]:
    #     print(key, ":   ", value)

    # split the data (80/20) -> training and testing
    training, testing, actual_labels = make_gnb_features(data_table)
    training_no = int(len(data) * .80)
    testing_no = int(len(data) * .20)
    print("Dataset loaded successfully")
    print(f"Total number of records: {len(data)}")
    print(f"Training instances: {training_no}")
    print(f"Testing instances: {testing_no}")

    # this part is where learning in the training data occurs

    # gets the mean and variance for every row of every classification
    mean_and_var = calculate_mean_and_var(training["features"], training["labels"])
    priors = class_priors(mean_and_var)

    #  select a k value
    while True:
        k_input = input("Enter k value from 0 to 1 (ex. 1e-9):  ")
        try:
            k = float(k_input)
            if 0 < k <= 1:
                break
            else:
                print("Invalid input! Please enter a valid k value between 0 and 1!")
        except ValueError: #when the user enters a non numeric value
            print("Please enter a numeric value")
    print(f"\nk = {k_input}\n")

    # gets all the variances from the two classifications
    all_variances = np.concatenate([mean_and_var[0]["var"], mean_and_var[1]["var"]])

    # epsilon is just a very small number to prevent division by zero (smoothing factor only)
    epsilon = k * all_variances.max()

    predicted_labels = predict_increaase_decrease(testing["features"], mean_and_var, priors, epsilon)

    correct, incorrect, accuracy, precision, recall, f1_score = evaluate_gnb(actual_labels, predicted_labels)
    
    incorrect_list = []
    for i in range(len(predicted_labels)):
        data_no = training_no + i + 1
        if predicted_labels[i] != actual_labels[i]:
            incorrect_list.append((data_no, actual_labels[i], predicted_labels[i]))
    
    incorrect = len(predicted_labels) - correct
    accuracy = correct / len(predicted_labels) 

    # printing the result
    print("GAUSSIAN NAIVE BAYES CLASSIFICATION RESULTS\n")

    print(f"Correct predictions: {correct} ")
    print(f"Incorrect predictions: {incorrect} ")
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}\n")

    # wrong predictions
    print(f"\nIncorrect Classifications:(first 20)")
    print(f"Data Record#:   Actual      Predicted")

    # prints the first 20 only
    for i in incorrect_list[:20]:
        actual = "Increase" if i[1] == 1 else "Decrease"
        predicted = "Increase" if i[2] == 1 else "Decrease"
        print(f"{i[0]:<15} {actual:<11} {predicted:<8}")

# calls the 'main' function
if __name__ == "__main__":
    gaussian_naive_bayes()