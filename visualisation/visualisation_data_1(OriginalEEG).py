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
# test_data = None
# train_data = None
# train_events = None
# all_runs_per_block = []
# # Load data cross-sessionally
# for session_no in range(1, session_amount + 1):
# 	data_dir = sbj_folder + "/S0" + str(session_no)
# 	runs_per_block = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/runs_per_block.txt")[0]
# 	if session_no >= val_session_start_at:
# 		all_runs_per_block.append(runs_per_block)
# 		pos = (sbj_no - 1) * session_amount + session_no - 1
# 	if session_no >= val_session_start_at:
# 		if test_data is None:
# 			test_data = BasicDataProcess.LoadEEGFromFile(data_dir, False)
# 			test_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
# 			test_targets = EventsToTargetsByLabelsAndRuns(test_events, true_labels[pos], runs_per_block)
# 		else:
# 			test_data = np.concatenate((test_data, BasicDataProcess.LoadEEGFromFile(data_dir, False)), axis=1)
# 			test_events_current = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
# 			test_events = np.concatenate((test_events, test_events_current))
# 			test_targets = np.concatenate((test_targets, EventsToTargetsByLabelsAndRuns(test_events_current, true_labels[pos], runs_per_block)))
# 	if train_data is None:
# 		train_data = BasicDataProcess.LoadEEGFromFile(data_dir, True)
# 		train_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainEvents.txt")
# 		train_labels = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainLabels.txt")
# 		train_targets = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")
# 	else:
# 		train_data = np.concatenate((train_data, BasicDataProcess.LoadEEGFromFile(data_dir, True)), axis=1)
# 		train_events = np.concatenate((train_events, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainEvents.txt")))
# 		train_labels = np.concatenate((train_labels, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainLabels.txt")))
# 		train_targets = np.concatenate((train_targets, BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")))

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
epochs = np.array(epochs)#[:,:,0:250]

# train_events = mne.read_events(data_dir + "/Train/trainEvents.txt")
# print(train_events)
# exit()
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
plt_times = np.linspace(0.2, 0.4, len(epochs))
fig = plt.figure(figsize=(7,10))
heatmap_axis = fig.add_axes([0.17,0.2,0.6,0.73])
fig.add_axes([0.17,0.07,0.6,0.1])
fig.add_axes([0.8,0.2,0.05,0.73])
mne_epochs.plot_image(
	sigma=0.5
	, picks='all'
	#, picks=['C3','Cz','C4','CPz','P3','Pz','P4','POz']
	, colorbar=True
	, cmap=('RdBu_r', True)
	, vmin=-2e7
	, vmax=2e7
	, evoked=True
	, order=order_func
	, combine='mean'
	, show=False
	, fig=fig
	, overlay_times=plt_times
	, title="SBJ{0:02d}SESSION{1:02d} EEG(Mean)".format(sbj_no, session_no)
	)
fig_prefix = "./figures/SBJ{0:02d}SESS{1:02d}_Mean_Overlay".format(sbj_no, session_no)
for line in heatmap_axis.get_lines():
	line.set_linewidth(0.8)
fig.savefig(
		"{0:s}.png".format(fig_prefix)
		, quality=100
		, dpi=100
		)
# for i, fig in enumerate(figs):
# 	fig.savefig(
# 		"{0:s}_Channels{1:d}.png".format(fig_prefix, i)
# 		, quality=100
# 		#, dpi=1000
# 		)
# epochs_to_visualise = [2723,2721,2728]


