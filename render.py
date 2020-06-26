import matplotlib.pyplot as plt
import matplotlib.cm as cm
import signal_processing

def show_mfccs(path, nr_mfccs, max_pad_len, seconds):
	data = signal_processing.extract_features_mfcc_seconds(path, nr_mfccs, max_pad_len, seconds)
	plt.imshow(data, cmap=cm.gray)