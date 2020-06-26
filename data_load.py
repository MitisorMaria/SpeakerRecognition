from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os
import signal_processing
import numpy as np
from sklearn.model_selection import train_test_split

def class_label_to_nr(class_label):
	if (class_label == 'A.J._Buckley'):
		classNo = 0
	elif (class_label == 'A.R._Rahman'):
		classNo = 1
	elif (class_label == 'Aamir_Khan'):
		classNo = 2
	elif (class_label == 'Aaron_Tveit'):
		classNo = 3
	elif (class_label == 'Aaron_Yoo'):
		classNo = 4
	elif (class_label == 'Abbie_Cornish'):
		classNo = 5
	elif (class_label == 'Abigail_Breslin'):
		classNo = 6
	elif (class_label == 'Abigail_Spencer'):
		classNo = 7
	elif (class_label == 'Adam_Beach'):
		classNo = 8
	elif (class_label == 'Adam_Brody'):
		classNo = 9
	elif (class_label == 'Adam_Copeland'):
		classNo = 10
	return classNo
	
	
def make_dataframe_melspectrogram(nr_speakers_total, base_path, nr_mels, n_fft, hop_length, max_pad_len, nr_seconds):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []
	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		
		speaker_dir = os.path.join(base_path, str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
					file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
					class_label = row["Class_name"]
					
					classNo = class_label_to_nr(class_label)
					data = signal_processing.extract_features_melspectrogram(file_name, n_fft, nr_mels, hop_length, max_pad_len, nr_seconds)
					features.append([data, class_label, classNo])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1          
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label','class'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf
	
def make_dataframe_2speakers(nr_speakers_total, base_path, nr_mfccs, max_pad_len, nr_seconds, name1, name2):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []
	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		
		speaker_dir = os.path.join(base_path, str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
					file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
					print(file_name)
					class_label = row["Class_name"]
					if ((class_label == name1) or (class_label == name2)):
						print("Yes")
						data = signal_processing.extract_features_mfcc_seconds(file_name, nr_mfccs, max_pad_len, nr_seconds)
						features.append([data, class_label])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1          
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf
	
	
def make_dataframe_sr(nr_speakers_total, base_path, nr_mfccs, max_pad_len, nr_seconds, sample_rate):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []
	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		
		speaker_dir = os.path.join(base_path, str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
					file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
					class_label = row["Class_name"]
					
					classNo = class_label_to_nr(class_label)
					data = signal_processing.extract_features_mfcc_sr(file_name, nr_mfccs, max_pad_len, nr_seconds, sample_rate)
					features.append([data, class_label, classNo])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1          
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label','class'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf

	
def make_dataframe_htk(nr_speakers_total, base_path, nr_mfccs, max_pad_len, nr_seconds):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []
	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		
		speaker_dir = os.path.join(base_path, str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
					file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
					class_label = row["Class_name"]
					
					classNo = class_label_to_nr(class_label)
					data = signal_processing.extract_features_mfcc_htk(file_name, nr_mfccs, max_pad_len, nr_seconds)
					features.append([data, class_label, classNo])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1          
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label','class'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf
	


def make_dataframe_class_no(nr_speakers_total, base_path, nr_mfccs, max_pad_len, nr_seconds):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []
	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		
		speaker_dir = os.path.join(base_path, str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
					file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
					class_label = row["Class_name"]
					
					classNo = class_label_to_nr(class_label)
					data = signal_processing.extract_features_mfcc_seconds(file_name, nr_mfccs, max_pad_len, nr_seconds)
					features.append([data, class_label, classNo])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1          
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label','class'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf
	
	

def make_dataframe_seconds(nr_speakers_total, base_path, nr_mfccs, max_pad_len, nr_seconds):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []

	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		speaker_dir = os.path.join(os.path.abspath(base_path), str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
				file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
				class_label = row["Class_name"]
				data = signal_processing.extract_features_mfcc_seconds(file_name, nr_mfccs, max_pad_len, nr_seconds)
				features.append([data, class_label])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1         
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf

def make_dataframe(nr_speakers_total, base_path, nr_mfccs, max_pad_len):
	metadata = pd.read_csv('vox1_meta.csv')
	features = []

	# Iterate through each sound file and extract the features 
	nr_of_speakers = 0
	for index, row in metadata.iterrows():
		speaker_dir = os.path.join(os.path.abspath(base_path), str(row["ID"]))
		entries = os.scandir(speaker_dir)
		
		for folder in entries:
			files = os.scandir(folder)
			for f in files:
				file_name = os.path.join(str(speaker_dir), str(folder.name), os.path.basename(f))
				class_label = row["Class_name"]
				data = signal_processing.extract_features_mfcc(file_name, nr_mfccs, max_pad_len)
				features.append([data, class_label])
			
		if nr_of_speakers == nr_speakers_total - 1:
			break
		else:
			nr_of_speakers += 1         
	# Convert into a Panda dataframe 
	featuresdf = pd.DataFrame(features, columns=['feature','class_label'])
	print('Finished feature extraction from ', len(featuresdf), ' files')
	return featuresdf
	
	
def make_train_test_sets(featuresdf, num_rows, num_columns, num_channels):
	# Convert features and corresponding classification labels into numpy arrays
	X = np.array(featuresdf.feature.tolist())
	y = np.array(featuresdf.class_label.tolist())

	# Encode the classification labels
	le = LabelEncoder()
	yy = to_categorical(le.fit_transform(y)) 

	# split the dataset 
	
	x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
	num_labels = yy.shape[1]
	
	if num_channels == 3:
		x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns)
		x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns)
	
		x_train = np.repeat(x_train[..., np.newaxis], 3, -1)
		x_test = np.repeat(x_test[..., np.newaxis], 3, -1)

		result_sets = [x_train, x_test, y_train, y_test, num_labels]
		return result_sets
	
	x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, 1)
	x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, 1)
	result_sets = [x_train, x_test, y_train, y_test, num_labels]
	return result_sets
	