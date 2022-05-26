from keras.models import load_model
import numpy as np
from BasicDataProcess import BasicDataProcess
import matplotlib.pyplot as plt
import mne
from sklearn.manifold import spectral_embedding  # noqa
from sklearn.metrics.pairwise import rbf_kernel   # noqa
from keras import backend as K
from scipy.signal import resample

def order_func(times, data):
    this_data = data[:, (times >= 0.2) & (times <= 0.40)]
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
rnn_folder = "20200108-124539"
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
# train_data = BasicDataProcess.LoadEEGFromFile(data_dir, True)
# train_data = np.moveaxis(train_data, 0, 1)
# test_data = BasicDataProcess.LoadEEGFromFile(data_dir, False)
# test_data = np.moveaxis(test_data, 0, 1)
# train_targets = BasicDataProcess.LoadDataFromFile(data_dir + "/Train/trainTargets.txt")
# test_events = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/testEvents.txt")
# runs_per_block = BasicDataProcess.LoadDataFromFile(data_dir + "/Test/runs_per_block.txt")[0]
# pos = (sbj_no - 1) * session_amount + session_no - 1
# test_targets = EventsToTargetsByLabelsAndRuns(test_events, true_labels[pos], runs_per_block)
# epochs = []
# for i, val in enumerate(test_targets):
# 	if val == 1:
# 		epochs.append(test_data[i])
# epochs = np.load("filtered_data.npy", allow_pickle=True)
# epochs = np.load("filtered_data_notarget.npy", allow_pickle=True)
epochs = np.load("original_data.npy", allow_pickle=True)
# epochs = np.load("original_data_nontarget.npy", allow_pickle=True)
# Preprocessing
epochs = resample(epochs, 140, axis=2)
# original_epochs = np.array(epochs)
model = load_model(cnn_model_name)
inputs = model.input
outputs = [layer.output for layer in model.layers]
func = K.function([inputs, K.learning_phase()], outputs)
layer_outputs = func([epochs,1])
# print(model.summary())
# for i, layer_output in enumerate(layer_outputs):
# 	print("Shape of Layer {0}: {1}".format(i, layer_output.shape))
layer_no = 13
epochs_all = layer_outputs[layer_no]
# epochs_all_mean = np.mean(epochs_all, axis=0)
# plt.plot(epochs_all_mean)
# plt.show()
epochs_all = np.moveaxis(epochs_all, 0, 1)
#epochs_all = epochs_all.reshape(epochs_all.shape[0], epochs_all.shape[1], 1, epochs_all.shape[2])
# print(epochs_all.shape)
# print(epochs_all_mean.tolist())
# exit()


## Visualisation
for i, epochs in enumerate(epochs_all):
# if True:
	#epochs = epochs_all
	# print("epoch {0}: {1}".format(i, epochs[1].tolist()))
	# break
	## Load data into MNE
	# channel_names = ['C3','Cz','C4','CPz','P3','Pz','P4','POz']
	channel_names = ['Cz']
	freq = (4 * 25) / 35
	info = mne.create_info(
		channel_names, freq
		, ch_types='eeg'
		)
	mne_epochs = mne.EpochsArray(
		epochs, info
		, tmin=-0.2
		)

	flt = "Layer-Pooling(Block 2)_{0}".format(i)
	title = "SBJ{0:02d}SESSION{1:02d} {2:s}(Mean)".format(sbj_no, session_no, flt)
	plt_times = np.linspace(0.2, 0.4, len(epochs))
	fig = plt.figure(figsize=(7,10))
	heatmap_axis = fig.add_axes([0.17,0.2,0.6,0.73])
	fig.add_axes([0.17,0.07,0.6,0.1])
	fig.add_axes([0.8,0.2,0.05,0.73])
	try:
		mne_epochs.plot_image(
			sigma=0.5
			#, picks='all'
			#, picks=['C3','Cz','C4','CPz','P3','Pz','P4','POz']
			, picks=['Cz']
			, colorbar=True
			, cmap=('RdBu_r', True)
			# , vmin=-2e7
			# , vmax=2e7
			, evoked=True
			, order=order_func
			#, combine='mean'
			, show=False
			, fig=fig
			# , overlay_times=plt_timesv
			, title=title
			)
	except Exception as e:
		print(e)
		exit()
		# for epoch in epochs:
		# 	print(epoch.tolist())
		# break
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
	plt.close(fig)

