from BasicDataProcess import BasicDataProcess
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import numpy as np
import os
from shutil import copyfile
import sys
import tensorflow as tf
from EEGModels import EEGNet
import datetime
from scipy.signal import resample

# tf.random.set_seed(73)
np.random.seed(73)
val_session_start_at = 1
if len(sys.argv) > 1 and sys.argv[1] is not None:
	timestr = sys.argv[1]
else:
	timestr = time.strftime("%Y%m%d-%H%M%S")
print(timestr)
save_path = "./results/" + timestr + "/"
model_name_prefix = save_path + "Model-SBJ"
model_name = model_name_prefix
session_amount = 7
all_labels = []
all_raw_results = []
def EventsToTargetsByLabelsAndRuns(events, labels, runs):
	result = []
	i = 0
	print(len(events))
	while i < len(events):
		label_no = i // (8 * runs)
		if labels[label_no] == events[i]:
			result.append(1)
		else:
			result.append(0)
		i += 1
	return np.array(result)
def DeepNormaliseTrain(data, targets, spectrum):
	result = []
	filter_feature_matrices = []
	for i in data:
		filter_feature = BasicDataProcess.GetNonTargetsAverage(i, targets)
		if spectrum:
			output = BasicDataProcess.ApplySpecialFilter(i, filter_feature)
		else:
			output = BasicDataProcess.ApplySpecialFilter(i, filter_feature, spectrum=False)
		filter_feature_matrices.append(filter_feature)
		result.append(output)
	return np.array(result), np.array(filter_feature_matrices)
def DeepNormaliseTest(data, filter_feature, spectrum):
	result = []
	# print(filter_feature.shape)
	for i, j in zip(data, filter_feature):
		if spectrum:
			output = BasicDataProcess.ApplySpecialFilter(i, j)
		else:
			output = BasicDataProcess.ApplySpecialFilter(i, j, spectrum=False)
		result.append(output)
	return np.array(result)
def Preprocessing(data, events, downsampling=0, targets=None, filter_feature_in=None, sort=False, reshape="None", norm=False, spectrum=True, low_pass=0):
	output = data
	filter_feature_out = None
	if low_pass != 0:
		for i in output:
			for j in i:
				j = BasicDataProcess.LowPassFilter(low_pass, j)

		# print(output.shape)
	if sort:
		output = BasicDataProcess.SortBasedOnEvents(output, events)
	if norm:
		if targets is not None:
			output, filter_feature_out = DeepNormaliseTrain(data, targets, spectrum)
		else:
			output = DeepNormaliseTest(data, filter_feature_in, spectrum)
	if downsampling != 0:
		output = resample(output, downsampling, axis=2)
	if reshape == "2D":
		output = BasicDataProcess.ReshapeTo2D(output)
	elif reshape == "1D":
		output = BasicDataProcess.ReshapeTo1D(output)
	if targets is not None:
		return output, filter_feature_out
	else:
		return output
true_labels_phaseI = BasicDataProcess.LoadCSV("./true_labels_phaseI.csv")[1:, 2:].astype(np.int)
true_labels_phaseII = BasicDataProcess.LoadCSV("./true_labels_phaseII.csv")[1:, 2:].astype(np.int)
# print(true_labels_phaseII)
true_labels = []
for i in range(15):
	for j in range(3):
		true_labels.append(true_labels_phaseI[i * 3 + j])
	for j in range(4):
		true_labels.append(true_labels_phaseII[i * 4 + j])

accuracy_matrics = {}

for sbj_no in range(1, 16):
	test_data = None
	train_data = None
	train_events = None
	print("===================SBJ%02d===================" % sbj_no)
	sbj_folder = "./SBJ%02d" % sbj_no
	model_name = model_name_prefix + "%02d.h5" % sbj_no

	print("Building model...")
	#====================================Buid model=======================================
	model = LinearDiscriminantAnalysis()
	if sbj_no == 1:
		# model.summary()
		os.mkdir(save_path)
		script_file_name = "Scripts-" + timestr + ".py"
		src = sys.argv[0]
		dst = save_path + script_file_name
		copyfile(src, dst)
	#======================================================================================

	print("Model built, now loading data....")
	# Read data cross-sessionly
	all_runs_per_block = []
	for session_no in range(1, session_amount + 1):
		# File IO
		data_dir = sbj_folder + "/S0" + str(session_no)
		runs_per_block = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/runs_per_block.txt")[0]
		if session_no >= val_session_start_at:
			all_runs_per_block.append(runs_per_block)
			pos = (sbj_no - 1) * session_amount + session_no - 1
		if session_no >= val_session_start_at:
			if test_data is None:
				test_data = BasicDataProcess.LoadEEGFromFile(data_dir, False)
				test_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
				test_targets = EventsToTargetsByLabelsAndRuns(test_events, true_labels[pos], runs_per_block)
			else:
				test_data = np.concatenate((test_data, BasicDataProcess.LoadEEGFromFile(data_dir, False)), axis=1)
				test_events_current = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
				test_events = np.concatenate((test_events, test_events_current))
				test_targets = np.concatenate((test_targets, EventsToTargetsByLabelsAndRuns(test_events_current, true_labels[pos], runs_per_block)))
		if train_data is None:
			train_data = BasicDataProcess.LoadEEGFromFile(data_dir, True)
			train_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainEvents.txt")
			train_labels = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainLabels.txt")
			train_targets = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")
		else:
			train_data = np.concatenate((train_data, BasicDataProcess.LoadEEGFromFile(data_dir, True)), axis=1)
			train_events = np.concatenate((train_events, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainEvents.txt")))
			train_labels = np.concatenate((train_labels, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainLabels.txt")))
			train_targets = np.concatenate((train_targets, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")))
	# sorted_train_data = BasicDataProcess.SortBasedOnEvents(train_data, train_events)
	# reshaped_train_data = BasicDataProcess.ReshapeTo2D(sorted_train_data)
	print("Data loaded, now preprocessing...")

	# train_data = train_data[:,:,50:200]
	# test_data = test_data[:,:,50:200]
	train_x, train_filter_feature = Preprocessing(train_data, train_events, 
		downsampling=0,
		targets=train_targets, 
		sort=False, 
		reshape="1D",
		norm=False,
		spectrum=False,
		low_pass=20
		)
	train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
	# train_x = train_x.reshape(train_x.shape[0], 1, 8, 350)
	test_x = Preprocessing(test_data, test_events, 
		downsampling=0,
		filter_feature_in=train_filter_feature,
		sort=False, 
		reshape="1D",
		norm=False,
		spectrum=False,
		low_pass=20
		)
	test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
	# test_x = test_x.reshape(test_x.shape[0], 1, 8, 350)
	# train_y = train_targets.reshape((-1, 8)).astype(np.float)
	train_y = train_targets
	# Split the data
	# x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.20, shuffle=True)
	x_train = train_x
	y_train = train_y
	x_valid = test_x
	y_valid = test_targets

	print("Data preprocessed, now fitting...")
	model.fit(
		# train_x, train_y,
		x_train, y_train
		)

	print("Data fitted, now classifying...")
	
	test_prediction = model.predict_proba(test_x)

	# print(val_dataset.GetAccuracyofCurrentSet(test_prediction))

	all_raw_results.append(test_prediction)
	start_pos = 0
	end_pos = 0
	counter = val_session_start_at
	for i in all_runs_per_block:
		end_pos = start_pos + i * 400
		raw_results = test_prediction[start_pos:end_pos]
		test_labels = BasicDataProcess.GetLabelsFromProbablities(raw_results, test_events[start_pos:end_pos], i)
		all_labels.append(test_labels)
		key = "SBJ{0:02d}SESSION{1:02d}".format(sbj_no, counter)
		value = BasicDataProcess.GetAccuracy(test_labels, true_labels[(sbj_no - 1) * session_amount + counter - 1])
		accuracy_matrics[key] = value
		print(key + ": " + str(value * 100) + "%")
		start_pos = end_pos
		counter += 1
	print("===========================================")

print("===================Summary===================")
print(timestr)
average_accuracy = 0.0
counter = 0.0
raw_results_file_name = save_path + "Result-RawMatrices-" + timestr
np.save(raw_results_file_name, all_raw_results)
result_accuracy_file_name = save_path + "Result-Accuracy-" + timestr
np.save(result_accuracy_file_name, accuracy_matrics)
result_labels_file_name = save_path + "Result-Labels-" + timestr
np.save(result_labels_file_name, all_labels)

with open("./results/Log-" + timestr + ".log", "w+") as log_file:
	log_file.write("===================Summary===================\n")
	for key, value in accuracy_matrics.items():
		counter += 1.0
		average_accuracy += value
		output = key + ":{0:6.2f}".format(value * 100) + "%"
		print(output)
		log_file.write(output + "\n")
	print("---------------------------------------")
	log_file.write("---------------------------------------\n")
	print("Average Accuracy:{0:6.2f}%".format(average_accuracy / counter * 100))
	log_file.write("Average Accuracy:{0:6.2f}%\n".format(average_accuracy / counter * 100))
	print("===================End of Classification===================")
	log_file.write("===================End of Classification===================")
