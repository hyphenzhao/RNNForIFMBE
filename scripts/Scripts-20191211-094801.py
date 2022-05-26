from BasicDataProcess import BasicDataProcess
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, DepthwiseConv2D, AveragePooling2D, SeparableConv2D, Reshape, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
from keras.constraints import max_norm
from keras import optimizers
import time
import numpy as np
import os
from shutil import copyfile
import sys
import tensorflow as tf
from EEGModels import EEGNet
import datetime
from scipy.signal import resample

tf.random.set_seed(73)
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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

	if downsampling != 0:
		output = resample(output, downsampling, axis=2)
		# print(output.shape)
	if sort:
		output = BasicDataProcess.SortBasedOnEvents(output, events)
	if norm:
		if targets is not None:
			output, filter_feature_out = DeepNormaliseTrain(data, targets, spectrum)
		else:
			output = DeepNormaliseTest(data, filter_feature_in, spectrum)
	if reshape == "2D":
		output = BasicDataProcess.ReshapeTo2D(output)
	elif reshape == "1D":
		output = BasicDataProcess.ReshapeTo1D(output)
	if targets is not None:
		return output, filter_feature_out
	else:
		return output

class ValidationDataset():
	def __init__(self, x_valid, y_valid, events):
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.test_events = events
		self.all_runs_per_block = None
	def GetAccuracyofCurrentSet(self, test_prediction):
		start_pos = 0
		end_pos = 0
		counter = val_session_start_at
		average_accuracy = 0.0
		for i in self.all_runs_per_block:
			end_pos = start_pos + i * 400
			raw_results = test_prediction[start_pos:end_pos]
			test_labels = BasicDataProcess.GetLabelsFromProbablities(raw_results, self.test_events[start_pos:end_pos], i)
			value = BasicDataProcess.GetAccuracy(test_labels, true_labels[(sbj_no - 1) * session_amount + counter - 1])
			average_accuracy += value
			start_pos = end_pos
			counter += 1
		return average_accuracy / len(self.all_runs_per_block)

class MyCustomCallback(tf.keras.callbacks.Callback):
	def __init__(self, patience=5, val_dataset=None):
		super(MyCustomCallback, self).__init__()
		self.patience = patience
		self.val_dataset = val_dataset
		self.best_weights = None

	def on_train_begin(self, logs=None):
		if os.path.exists(model_name):
			self.model = load_model(model_name)

	def on_train_begin(self, batch, logs=None):
		self.wait = 0
		self.stopped_epoch = 0
		self.best = 0.0

	def on_epoch_end(self, epoch, logs=None):
		test_prediction = self.model.predict(self.val_dataset.x_valid)
		current_accuracy = self.val_dataset.GetAccuracyofCurrentSet(test_prediction)
		if self.best < current_accuracy:
			self.best = current_accuracy
			self.wait = 0
			self.best_weights = self.model.get_weights()
			self.model.save(model_name)
		else:
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print('Restoring model weights from the end of the best epoch.')
				self.model.set_weights(self.best_weights)
		print(" - current_average_accuracy: " + str(current_accuracy) + " - current_best_accuracy: " + str(self.best) \
			+ " - epochs_to_earlystopping: " + str(self.patience - self.wait))
		if self.best == 1.0:
			self.stopped_epoch = epoch
			self.model.stop_training = True
			print('Model reached 100% accurate.')
			self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

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
	f1 = 8
	f2 = 16
	p = 0.25
	#create model
	#Instantiate an empty model
	model = Sequential()

	model.add(Reshape((1,8,140), input_shape=(8,140)))
	model.add(ZeroPadding2D(padding=(0, 32), data_format='channels_first'))
	# 1st Convolutional Layer
	model.add(Conv2D(filters=f1, kernel_size=(1,65), padding="valid", activation='linear', data_format='channels_first'))
	model.add(BatchNormalization(axis=1))
	# Depthwise Convlutional Layer
	model.add(DepthwiseConv2D(depth_multiplier=2, kernel_size=(8,1), padding="valid", activation='linear', data_format='channels_first', kernel_constraint=max_norm(1)))
	# Batch Normalisation before passing it to the next layer
	model.add(BatchNormalization(axis=1))
	model.add(Activation('elu'))	
	# Pooling 
	model.add(AveragePooling2D(pool_size=(1,4), data_format='channels_first'))
	model.add(Dropout(p))

	# 2nd Convolutional Layer
	model.add(ZeroPadding2D(padding=(0, 8), data_format='channels_first'))
	model.add(SeparableConv2D(filters=f2, kernel_size=(1,17), padding='valid', activation='linear', data_format='channels_first'))
	model.add(BatchNormalization(axis=1))
	model.add(Activation('elu'))		
	# Pooling 
	model.add(AveragePooling2D(pool_size=(1,8), data_format='channels_first'))
	# Batch Normalisation before passing it to the next layer
	model.add(Dropout(p))
	model.add(Flatten())

	# Output Layer
	model.add(Dense(2, activation='linear'))
	model.add(Activation('softmax'))

	if sbj_no == 1:
		model.summary()
		os.mkdir(save_path)
		script_file_name = "Scripts-" + timestr + ".py"
		src = sys.argv[0]
		dst = save_path + script_file_name
		copyfile(src, dst)
		# sys.exit()
	adam = optimizers.Adam(lr=0.0005)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
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
		downsampling=140,
		targets=train_targets, 
		sort=False, 
		reshape="1D",
		norm=False,
		spectrum=False,
		low_pass=20
		)
	# train_x = train_x.reshape(train_x.shape[0], 1, 8, 350)
	test_x = Preprocessing(test_data, test_events, 
		downsampling=140,
		filter_feature_in=train_filter_feature,
		sort=False, 
		reshape="1D",
		norm=False,
		spectrum=False,
		low_pass=20
		)
	# test_x = test_x.reshape(test_x.shape[0], 1, 8, 350)
	# train_y = train_targets.reshape((-1, 8)).astype(np.float)
	train_y = to_categorical(train_targets)
	# Split the data
	# x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.20, shuffle=True)
	x_train = train_x
	y_train = train_y
	x_valid = test_x
	y_valid = to_categorical(test_targets)
	val_dataset = ValidationDataset(x_valid, y_valid, test_events)
	val_dataset.all_runs_per_block = all_runs_per_block
	print("Data preprocessed, now fitting...")
	model.fit(
		# train_x, train_y,
		x_train, y_train, 
		batch_size=64, 
		epochs=1000, 
		shuffle=True, 
		callbacks=[MyCustomCallback(patience=100, val_dataset=val_dataset)], 
		validation_data=(x_valid, y_valid)
		)

	print("Data fitted, now classifying...")
	
	test_prediction = model.predict(test_x)

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
