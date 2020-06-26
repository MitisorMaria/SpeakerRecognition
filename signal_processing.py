import numpy as np
import librosa

def extract_features_melspectrogram(file_name, nr_fft, nr_mels, hop_length, max_pad_len, seconds):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=seconds) 
    #mai intai tai din clip
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=nr_fft, hop_length=hop_length, n_mels=nr_mels)
    pad_width = max_pad_len - mel.shape[1]
    mel = np.pad(mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel


def extract_features_mfcc_seconds(file_name, nr_mfccs, max_pad_len, seconds):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration = seconds) 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=nr_mfccs)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def extract_features_mfcc(file_name, nr_mfccs, max_pad_len):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=nr_mfccs)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def extract_features_mfcc_htk(file_name, nr_mfccs, max_pad_len, seconds):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=seconds)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=nr_mfccs, htk=True)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs

def extract_features_mfcc_sr(file_name, nr_mfccs, max_pad_len, seconds, sr):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=seconds, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=nr_mfccs)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfccs