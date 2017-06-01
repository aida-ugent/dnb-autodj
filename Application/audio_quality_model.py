''' 
	Implements the audio quality SVM model that evaluates the quality of the crossfades.
'''

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import librosa
from librosa.core import cqt
import numpy as np
from scipy.stats import kurtosis, skew

try:
	scaler = joblib.load('feature_scaler.pkl')
	model = joblib.load('svm_model.pkl') 
except Exception as e:
	raise Exception('Music quality model not installed!\nMake sure feature_scaler.pkl and svm_model.pkl are in the app its root directory\n ' + str(e))

def calculateFeaturesForDownbeat(audio):
	datarow = []		
	BINS_PER_OCTAVE = 12
	NUM_OCTAVES = 5
	spectogram = cqt(audio, sr=44100, bins_per_octave=BINS_PER_OCTAVE, n_bins=BINS_PER_OCTAVE*NUM_OCTAVES, hop_length=512)
	spec_db = np.absolute(spectogram)
	spec_db = librosa.amplitude_to_db(spectogram, ref=np.max)
	max_spec = np.max(spec_db) 
	min_spec = np.min(spec_db)
	if max_spec != min_spec:
		spec_db = (spec_db - min_spec) / (max_spec - min_spec)
	else:
		spec_db = (spec_db - np.min(spec_db))
	# Statistics for each frequency bin (each row => aggregate over axis 1 = columns)
	NUM_FRAMES = spectogram.shape[1]
	features_per_half_octave = []
	window_features = np.zeros((4*4*2-1,NUM_OCTAVES*2))
	for i_freq in range(NUM_OCTAVES*2):			# Frequency resolution: every half octave; hop size quarter octave
		features_cur_freq_window = []
		# Window along frequency axis
		f_bin_start = (i_freq * BINS_PER_OCTAVE) / 2
		f_bin_end = ((i_freq + 1) * BINS_PER_OCTAVE) / 2
		for i_time in range(4 * 4 * 2 - 1): 	# Aggregate along time axis in frames of one quarter beat, hop size an eight dbeat
			# Window along time axis
			frame_start = int(i_time * NUM_FRAMES / 32.0)
			frame_end = int((i_time + 2) * NUM_FRAMES / 32.0)
			# Current section of spectogram
			cur_w = spec_db[f_bin_start:f_bin_end,frame_start:frame_end]
			window_features[i_time,i_freq] = np.mean(cur_w)	# Determine how 'loud' this window is approx
	
	freq_axis_features = []
	freq_axis_features.extend(np.mean(window_features,axis=1))
	freq_axis_features.extend(np.var(window_features,axis=1))
	freq_axis_features.extend(skew(window_features,axis=1))
	freq_axis_features.extend(kurtosis(window_features,axis=1))
	
	time_means = np.mean(window_features,axis=0)
	time_vars = np.var(window_features, axis=0)
	freq_axis_features.extend([np.mean(time_means), np.var(time_vars), np.mean(time_vars), np.var(time_vars)])
	
	datarow = freq_axis_features
	return datarow

def evaluate_audio_quality(audio):
	'''
		Evaluates each frame of the output audio using the classifier and returns an aggregate score for the entire audio
		Assumes 175 BPM input, downbeat aligned
	'''
	LEN_DBEAT_175 = 60480
	num_dbeats = len(audio) / LEN_DBEAT_175
	features = [] # Feature matrix for entire audio array
	for dbeat in range(num_dbeats):
		start_idx = dbeat * LEN_DBEAT_175
		audio_cur = audio[start_idx : start_idx + LEN_DBEAT_175]
		features.append(calculateFeaturesForDownbeat(audio_cur))
	
	features = scaler.transform(features)
	y_pred = model.predict_log_proba(features)[:,1]
	
	#~ epsilon = 1e-10
	#~ y_pred[y_pred < epsilon] = epsilon	# Numeric stability of log
		
	return np.mean(y_pred)
