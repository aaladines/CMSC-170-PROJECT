import csv
import numpy as np


# function to load the prepared dataset
def load_dataset(path="dataset_prepared.csv"):
	with open(path, "r", encoding="utf-8", newline="") as file:
		reader = csv.DictReader(file)
		data = list(reader)

	return data  


# function to create the 3 features that will be used by the perceptron
def make_perceptron_features(data):
	features = []
	labels = []
	sets = []

	# dictionary to store the records 
	data_table = {}

	for row in data:
		# uses crop, region, year, and quarter to identify each record
		key = (row["crop"], row["region"], int(row["year"]), int(row["quarter"]))
		
		# store the previous year's production, current production, and label
		data_table[key] = (float(row["same_quarter_previous_year"]), float(row["this_quarter"]), int(row["production"]))


	# this calculates the features for each record
	for row in data:
		crop = row["crop"]
		region = row["region"]
		year = int(row["year"])
		quarter = int(row["quarter"])

		# retrieves the production values and actual label of the current record
		prev_q, this_q, production = data_table[(crop, region, year, quarter)]

		# this is the previous year's production (feature 1)
		prev_q_feat = np.log(1 + prev_q)


		# this is the previous year's production change (feature 2)
		# finds the record for the same crop, region, and quarter from the previous year
		last_year_same_quarter = data_table.get((crop, region, year - 1, quarter))

		if last_year_same_quarter is not None:
			# gets the logarithmic production change of that earlier record
			last_year_change = np.log(1 + last_year_same_quarter[1]) - np.log(1 + last_year_same_quarter[0])
		else:
			# use 0 if the previous year's record is not available
			last_year_change = 0.0


		# previous quarter's production change (feature 3)
		if quarter > 1:
			prev_quarter_year = year 
			prev_quarter = quarter - 1
		else:
			# if the current quarter is q1, the previous quarter is q4 of the previous year
			prev_quarter_year = year - 1
			prev_quarter = 4

		# get the previous quarter's record
		prev_quarter_row = data_table.get((crop, region, prev_quarter_year, prev_quarter))

		if prev_quarter_row is not None:
			# gets the previous quarter's logarithmic production change
			prev_quarter_change = np.log(1 + prev_quarter_row[1]) - np.log(1 + prev_quarter_row[0])
		else:
			# use 0 if the previous quarter's record is not available
			prev_quarter_change = 0.0

		# save all the features for the current record
		features.append([prev_q_feat, last_year_change, prev_quarter_change])
		
		# save the actual classification of the current record
		labels.append(production)
		
		# save whether the record is part of the training set or testing set
		sets.append(row["set"])

	return np.array(features), np.array(labels), np.array(sets)


# function to separate the dataset into training and testing sets
def split_dataset(features, labels, sets):
	# get the positions of records assigned to training
	train_indices = np.where(sets == "train")[0]

	# get the positions of records assigned to testing
	test_indices = np.where(sets == "test")[0]

	# get the training features and their actual labels
	x_train = features[train_indices]
	y_train = labels[train_indices]

	# get the testing features and their actual labels
	x_test = features[test_indices]
	y_test = labels[test_indices]

	return x_train, x_test, y_train, y_test 


# function to standardize the features
def standardize_features(x_train, x_test):
	# calculate the mean and standard deviation using the training data
	mean = np.mean(x_train, axis=0)
	std = np.std(x_train, axis=0)

	# to avoid division by zero if a feature has no variation
	std[std == 0] = 1

	# standardize both training and testing features
	x_train = (x_train - mean) / std 
	x_test = (x_test - mean) / std 

	return x_train, x_test


# function to train the perceptron using the learning algorithm
def train_perceptron(x_train, y_train, learning_rate=0.01, max_epochs=100):
	# convert the labels from 0 and 1 to -1 and 1
	y_train = np.where(y_train == 0, -1, 1)

	# initialize the weights and bias to zero
	weights = np.zeros(x_train.shape[1])
	bias = 0.0

	# repeat the training process for the given number of epochs
	for epoch in range(max_epochs):
		updates = 0 

		# loops through each training record
		for i in range(len(x_train)):
			# calculate the weighted sum and bias
			net = np.dot(weights, x_train[i]) + bias

			# to apply the signum function
			if net >= 0:
				prediction = 1 
			else:
				prediction = -1

			# calculate the difference between the actual and desired output
			error = y_train[i] - prediction

			# update the weights and bias if the prediction is incorrect
			if error != 0:
				weights += learning_rate*error*x_train[i]
				bias += learning_rate*error
				updates += 1

		# stop training if there are no more weight updates
		if updates == 0:
			break

	return weights, bias


# function to make predictions using the trained perceptron
def predict_perceptron(x_test, weights, bias):
	predictions = []

	# go through each testing record
	for i in range(len(x_test)):
		# calculate the weighted sum and bias
		net = np.dot(weights, x_test[i]) + bias

		# to apply the signum function
		if net >= 0:
			prediction = 1
		else:
			prediction = -1

		# convert the prediction back to 0 or 1
		if prediction == -1:
			prediction = 0

		predictions.append(prediction)

	return np.array(predictions)


# function to evaluate the perceptron's predictions
def evaluate_perceptron(y_test, predictions):
	# count the correct and incorrect predictions
	true_positive = np.sum((y_test == 1) & (predictions == 1))
	true_negative = np.sum((y_test == 0) & (predictions == 0))
	false_positive = np.sum((y_test == 0) & (predictions == 1))
	false_negative = np.sum((y_test == 1) & (predictions == 0))

	# to calculate accuracy
	accuracy = (true_positive + true_negative) / len(y_test)

	# to calculate precision
	if true_positive + false_positive == 0:
		# to avoid division by zero error
		precision = 0 
	else:
		precision = true_positive / (true_positive + false_positive)

	# to calculate recall
	if true_positive + false_negative == 0:
		# to avoid division by zero error
		recall = 0
	else:
		recall = true_positive / (true_positive + false_negative)

	# to calculate f1 score
	if precision + recall == 0:
		f1_score = 0
	else:
		f1_score = (2*precision*recall) / (precision + recall)

	return accuracy, precision, recall, f1_score


# main function to run the perceptron
def main():
	# load the prepared dataset
	data = load_dataset()

	# create the features, labels, and dataset assignments
	features, labels, sets = make_perceptron_features(data)

	# separate the training data and testing data
	x_train, x_test, y_train, y_test = split_dataset(features, labels, sets)

	# standardize the features
	x_train, x_test = standardize_features(x_train, x_test)

	# train the perceptron
	weights, bias = train_perceptron(x_train, y_train)

	# make predictions using the testing data
	predictions = predict_perceptron(x_test, weights, bias)

	# evaluate the predictions
	accuracy, precision, recall, f1_score = evaluate_perceptron(y_test, predictions)

	# results
	print("Perceptron Results")
	print("------------------")
	print("Training records: ", len(x_train))
	print("Testing records: ", len(x_test))
	print("Accuracy: ", round(accuracy, 4))
	print("Precision: ", round(precision, 4))
	print("Recall: ", round(recall, 4))
	print("F1 score: ", round(f1_score, 4))


# to run the main function
if __name__ == "__main__":
    main()