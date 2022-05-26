from sklearn.decomposition import PCA
import scipy.io
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from csv import reader
from scipy.fftpack import rfft, irfft

class BasicDataProcess:
	filter_threshold = 20
	@staticmethod
	def LoadCSV(filename):
		raw_data = open(filename)
		csv_data = reader(raw_data, delimiter = ',')
		x = list(csv_data)
		data = np.array(x)
		return data
	@staticmethod
	def ExtendTrainLabels(inputs, times):
		result = np.empty([inputs.shape[0] * times])
		for i in range(inputs.shape[0] * times):
			result[i] = inputs[i // times]
		return result
	@staticmethod
	def GetNonTargetsAverage(train_inputs, train_targets):
		# print("--Get non-targets average matrix--")
		non_targets = None
		count = 0
		for i in range(len(train_targets)):
			if train_targets[i] == 0:
				count += 1
				if non_targets is None:
					non_targets = train_inputs[i]
				else:
					non_targets = np.add(non_targets, train_inputs[i])
		non_targets = non_targets / float(count)
		# print("--Done--")
		return non_targets
	@staticmethod
	def ApplySpecialFilter(inputs, filter_feature, spectrum=True):
		# print("--Apply special filter to the inputs--")
		result = []
		for single_input in inputs:
			if spectrum:
				input_fft = rfft(single_input)
				filter_fft = rfft(filter_feature)
				output = irfft(input_fft - filter_fft)
			else:
				output = single_input - filter_feature
			result.append(output)
		# print("--Done--")
		return np.array(result)
	@staticmethod
	def ReshapeTo1D(data):
		# It is practically proven that channel first will have a better result.
		# return np.moveaxis(data, 0, -1)
		return np.moveaxis(data, 0, 1)
	@staticmethod
	def ReshapeTo2D(data):
		result = np.moveaxis(data, 0, -1)
		result = np.split(result, result.shape[0]//8)
		result = np.moveaxis(result, -1, 1)
		return np.array(result)
	@staticmethod
	def SortBasedOnEvents(data, events):
		result = np.empty_like(data)
		count = 0
		set_no = 0
		for i in events:
			for j in range(8):
				target_no = set_no * 8 + int(i) - 1
				# print("%02d, %02d" % (target_no, count))
				result[j][target_no] = data[j][count]
			count += 1
			if count % 8 == 0:
				set_no += 1
		return result
	@staticmethod
	def SlidingFFT(inputs, nseg, noverlap):
		results = []
		cutoff_pos = 0
		for x in inputs:
			epoch = None
			for y in x:
				f, t, Zxx = signal.stft(y, 
					fs=250, 
					nperseg=nseg, 
					noverlap=noverlap)
				if cutoff_pos == 0:
					for i in f:
						if i <= BasicDataProcess.filter_threshold:
							cutoff_pos += 1
						else:
							break
				Zxx = np.delete(Zxx, np.s_[cutoff_pos:len(Zxx)], 0)
				Zxx = np.abs(Zxx)
				Zxx = Zxx.transpose()
				if epoch is None:
					epoch = Zxx
				else:
					epoch = np.add(epoch, Zxx)
			epoch = epoch / float(len(x))
			results.append(epoch)
		return results
	@staticmethod
	def TransposeElements(inputs):
		outputs = []
		for i in inputs:
			outputs.append(i.transpose())
		return np.array(outputs)
	@staticmethod
	def LoadEEGFromFile(data_dir, is_train, trans_flag = True):
		if is_train:
			data = scipy.io.loadmat(data_dir + "/Train/trainData.mat")
			set_name = "trainData"
		else:
			data = scipy.io.loadmat(data_dir + "/Test/testData.mat")
			set_name = "testData"
		channels = []
		for i in data[set_name]:
			if trans_flag:
				channels.append(np.array(i).transpose())
			else:
				channels.append(np.array(i))
		return np.array(channels)
	@staticmethod
	def LoadDataFromFile(filename):
		file = open(filename)
		raw_data = file.readlines()
		file.close()
		data = []
		for i in raw_data:
			data.append(int(i))
		return np.array(data)
	@staticmethod
	def LowPassFilter(fc, data):
		fs = 250.0
		w = fc / (fs / 2.0) # Normalize the frequency
		b, a = signal.butter(9, w, 'low', analog=False)
		output = signal.filtfilt(b, a, data)
		return output
	@staticmethod
	def GetFeatureByPCA(channels, data_size, with_filter, pca_threshold, reshaped, width):
		train_input = []
		for i in range(data_size):
			p300_matrix = []
			for channel in channels:
				left = 125 - width // 3
				right = 125 + 2 * width // 3
				raw_channel_data = channel[i][int(left):int(right)]
				if with_filter:
					filtered_data = BasicDataProcess.LowPassFilter(BasicDataProcess.filter_threshold, raw_channel_data)
					p300_matrix.append(filtered_data.tolist())
				else:
					p300_matrix.append(raw_channel_data)
			pca = PCA()
			p300_pca = pca.fit_transform(p300_matrix)
			if reshaped:
				train_input.append(p300_pca[0:pca_threshold].reshape(-1).tolist())
			else:
				train_input.append(p300_pca[0:pca_threshold].tolist())
		return train_input
	@staticmethod
	def GetP300Inputs(channels, with_filter, data_size, reshaped, width):
		result = []
		for i in range(data_size):
			p300_matrix = []
			for channel in channels:
				if width == 0:
					raw_channel_data = channel[i]
				else:
					left = 125 - width // 3
					right = 125 + 2 * width // 3
					raw_channel_data = channel[i][int(left):int(right)]
				if(with_filter):
					filtered_data = BasicDataProcess.LowPassFilter(BasicDataProcess.filter_threshold, raw_channel_data)
					p300_matrix.append(filtered_data.tolist())
				else:
					p300_matrix.append(raw_channel_data)
			if reshaped:
				result.append(np.array(p300_matrix).reshape(-1).tolist())
			else:
				result.append(np.array(p300_matrix))
		return result
	@staticmethod
	def GetLabelsFromProbablities(proba_results, events, block_size):
		label_amt = len(proba_results) // (block_size * 8)
		labels_proba = np.zeros((label_amt, 8))
		for i in range(len(proba_results)):
			label_no = i // (block_size * 8)
			labels_proba[label_no][events[i] - 1] += proba_results[i][1]
		labels = np.argmax(labels_proba, axis=1) + 1
		# print(labels)
		return labels
	@staticmethod
	def GetTargetResults(proba_results):
		label_results = np.zeros(len(proba_results), dtype=np.int8)
		pos = 0
		max_proba = 0
		record = 0
		for i in proba_results:
			if(i[1] > max_proba):
				max_proba = i[1]
				record = pos
			pos += 1
			if pos % 8 == 0:
				label_results[record] = 1
				max_proba = 0
		return label_results.tolist()
	@staticmethod
	def GetLabelResults(targets, events):
		result = []
		pos = 0
		for i in targets:
			if i == 1:
				result.append(events[pos])
			pos += 1
		return result
	@staticmethod
	def GetDifferences(a, b, size):
		result = 0
		for i in range(size):
			if a[i] != b[i]:
				result += 1
		return result
	@staticmethod
	def GetMostFrequent(List): 
		counter = 0
		num = List[0] 

		for i in List: 
			curr_frequency = List.count(i) 
			if(curr_frequency > counter): 
				counter = curr_frequency 
				num = i 
		return num
	@staticmethod
	def GetLabels(votes, block_size):
		test_labels = []
		for i in range(len(votes) // block_size):
			block = np.array(votes[i * block_size : (i + 1) * block_size])
			test_labels.append(BasicDataProcess.GetMostFrequent(block.tolist()))
		return test_labels
	@staticmethod
	def GetAccuracy(a, b):
		fa = np.array(a).ravel()
		fb = np.array(b).ravel()
		counter = 0
		correct = 0
		for i, j in zip(fa, fb):
			counter += 1
			if i == j:
				correct += 1
		return float(correct) / float(counter)