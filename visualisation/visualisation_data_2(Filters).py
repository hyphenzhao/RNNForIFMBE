from keras.models import load_model
import numpy as np
from BasicDataProcess import BasicDataProcess
import matplotlib.pyplot as plt
import mne
from sklearn.manifold import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel   # noqa


def order_func(times, data):
    this_data = data[:, (times > 0.2) & (times < 0.40)]
    this_data /= np.sqrt(np.sum(this_data ** 2, axis=1))[:, np.newaxis]
    return np.argsort(spectral_embedding(rbf_kernel(this_data, gamma=1.),
                      n_components=1, random_state=0).ravel())

## Predefine visualisation information
# Define subject and session to be visualised
sbj_no = 14
session_no = 7
# Define cross-session amount
val_session_start_at = 1
session_amount = 7
# Define folders/paths to models and data
sbj_folder = "./SBJ%02d" % sbj_no
data_dir = sbj_folder + "/S0" + str(session_no)
path_prefix = "./results/"
rnn_folder = "20191219-194324"
cnn_folder = "20191218-101826"
model_affix = "/Model-SBJ{0:02d}.h5".format(sbj_no)
cnn_model_name = path_prefix + cnn_folder + model_affix
rnn_model_name = path_prefix + rnn_folder + model_affix

## Load true labels
true_labels_phaseI = BasicDataProcess.LoadCSV("./true_labels_phaseI.csv")[1:, 2:].astype(np.int)
true_labels_phaseII = BasicDataProcess.LoadCSV("./true_labels_phaseII.csv")[1:, 2:].astype(np.int)
true_labels = []
for i in range(15):
	for j in range(3):
		true_labels.append(true_labels_phaseI[i * 3 + j])
	for j in range(4):
		true_labels.append(true_labels_phaseII[i * 4 + j])

## Predefine functions
# Transfer test events to test targets
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
# NFBF Train
# input train data and train target
# return a filtered train data and a feature map
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
# NFBF Test
# input test data and feature map
# return a filtered test data
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

## Load data from file
# Initialise variables
train_data = BasicDataProcess.LoadEEGFromFile(data_dir, True)
train_data = np.moveaxis(train_data, 0, 1)
test_data = BasicDataProcess.LoadEEGFromFile(data_dir, False)
test_data = np.moveaxis(test_data, 0, 1)
train_targets = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")
test_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
runs_per_block = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/runs_per_block.txt")[0]
pos = (sbj_no - 1) * session_amount + session_no - 1
test_targets = EventsToTargetsByLabelsAndRuns(test_events, true_labels[pos], runs_per_block)
epochs = []
for i, val in enumerate(test_targets):
	if val == 1:
		epochs.append(test_data[i])
original_epochs = np.array(epochs)
filters = ['Original','Lowpass','NFBF', 'All(NFBF)']
# filters = ['NFBF']
for flt in filters:
	if flt == 'Lowpass':
		epochs = np.empty_like(original_epochs)
		for i, single_epoch in enumerate(original_epochs):
			for j, channel in enumerate(single_epoch):
				epochs[i][j] = BasicDataProcess.LowPassFilter(20, channel)
	elif flt == 'NFBF':
		epochs = original_epochs
		train_data_NFBF = None
		all_runs_per_block = []
		for session_no_NFBF in range(1, 8):
			data_dir_NFBF = sbj_folder + "/S0" + str(session_no_NFBF)
			runs_per_block_NFBF = BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Test/runs_per_block.txt")[0]
			all_runs_per_block.append(runs_per_block_NFBF)
			pos = (sbj_no - 1) * 7 + session_no_NFBF - 1
			if train_data_NFBF is None:
				train_data_NFBF = BasicDataProcess.LoadEEGFromFile(data_dir_NFBF, True)
				train_events_NFBF = BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainEvents.txt")
				train_labels_NFBF = BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainLabels.txt")
				train_targets_NFBF = BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainTargets.txt")
			else:
				train_data_NFBF = np.concatenate((train_data_NFBF, BasicDataProcess.LoadEEGFromFile(data_dir_NFBF, True)), axis=1)
				train_events_NFBF = np.concatenate((train_events_NFBF, BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainEvents.txt")))
				train_labels_NFBF = np.concatenate((train_labels_NFBF, BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainLabels.txt")))
				train_targets_NFBF = np.concatenate((train_targets_NFBF, BasicDataProcess.LoadDataFromFile(data_dir_NFBF + "/Train/trainTargets.txt")))
		train_data_NFBF, noise_features = DeepNormaliseTrain(train_data_NFBF, train_targets_NFBF, True)
		inputs = np.moveaxis(epochs,0,1)
		epochs = DeepNormaliseTest(inputs, noise_features, True)
		epochs = np.moveaxis(epochs,0,1)
	elif flt == 'All(NFBF)':
		new_epochs = np.empty_like(epochs)
		for i, single_epoch in enumerate(epochs):
			for j, channel in enumerate(single_epoch):
				new_epochs[i][j] = BasicDataProcess.LowPassFilter(20, channel)
		epochs = new_epochs

		# train_data_LP = np.empty_like(train_data_NFBF)
		# for i, single_epoch in enumerate(train_data_NFBF):
		# 	for j, channel in enumerate(single_epoch):
		# 		train_data_LP[i][j] = BasicDataProcess.LowPassFilter(20, channel)
		# train_data_LP_NFBF, noise_features = DeepNormaliseTrain(train_data_LP, train_targets_NFBF, True)
		# inputs = np.moveaxis(epochs,0,1)
		# epochs = DeepNormaliseTest(inputs, noise_features, True)
		# epochs = np.moveaxis(epochs,0,1)
	else:
		epochs = original_epochs
	## Load data into MNE
	channel_names = ['C3','Cz','C4','CPz','P3','Pz','P4','POz']
	info = mne.create_info(
		channel_names, 250.0
		, ch_types='eeg'
		)
	mne_epochs = mne.EpochsArray(
		epochs, info
		, tmin=-0.2
		)


	# title = "SBJ{0:02d}SESSION{1:02d} {2:s}(EEG)".format(sbj_no, session_no, flt)
	# mne_epochs.plot(
	# 	n_epochs=8
	# 	, n_channels=8
	# 	, title=title
	# 	, show=True
	# 	, block=True
	# 	, scalings='auto'
	# 	)
	# continue
	title = "SBJ{0:02d}SESSION{1:02d} {2:s}(Mean)".format(sbj_no, session_no, flt)
	plt_times = np.linspace(0.2, 0.4, len(epochs))
	fig = plt.figure(figsize=(7,10))
	heatmap_axis = fig.add_axes([0.17,0.2,0.6,0.73])
	fig.add_axes([0.17,0.07,0.6,0.1])
	fig.add_axes([0.8,0.2,0.05,0.73])
	mne_epochs.plot_image(
		sigma=0.5
		, picks='all'
		#, picks=['C3','Cz','C4','CPz','P3','Pz','P4','POz']
		#, picks=['P3']
		, colorbar=True
		, cmap=('RdBu_r', True)
		, vmin=-2e7
		, vmax=2e7
		, evoked=True
		, order=order_func
		, combine='mean'
		, show=False
		, fig=fig
		# , overlay_times=plt_timesv
		, title=title
		)
	# fig = mne_epochs.plot_psd(
	# 	picks='all'
	# 	, show=False
	# 	)
	fig_prefix = "./figures/{0:s}".format(title)
	# for line in heatmap_axis.get_lines():
	# 	line.set_linewidth(0.8)
	fig.savefig(
			"{0:s}.png".format(fig_prefix)
			, quality=100
			, dpi=100
			)

