'''
	Script to train the quality model of downbeat segments.
	Can also be copied to the Application directory and used to visualize accuracy over one song. Uncomment the necessary lines for that
'''

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

#~ # Uncomment if script is in Application directory
#~ from song import Song
#~ import librosa
#~ from librosa.core import cqt
#~ from timestretching import time_stretch_sola

# Change feature CSV's here
# These feature CSV files were generated using the generate_downbeat_segments_for_class.py script
train_data_file = 'data/train_windowed_features_timeaggregate.csv'
test_data_file = 'data/test_windowed_features_timeaggregate.csv'

def load_training_data_csv(filename, shuffle=True):
	
	with open(filename) as csvfile:
		
		csvreader = csv.reader(csvfile)
		labels = []
		song_labels = []
		data = []
		
		previous_song = ''
		new_song_label = 0
		
		# Row format: song title, label (1 or 0), data
		song_labels_dict = {}
		for row in csvreader:
			title = row[0]
			song_labels.append(title)
			labels.append(float(row[1]))
			data.append([float(i) for i in row[2:]])
			
	return np.array(data), np.array(labels), np.array(song_labels)

def train():
	
	# Load the training data
	data, labels, song_labels = load_training_data_csv(train_data_file)
	
	# For plotting
	scores_val = []
	scores_train = []
	losses_val = []
	losses_train = []
	regularization = []
	
	# Train the model for different values of C; validate using CV
	# Use different values here as desired: now it contains the best options
	gamma_options = [1e-3]
	C_options = [1e2]
	tol = 1e-7 	# Stop tolerance
	
	# Best params for windowed features: train_moresegments_bothclasses C = 100, gamma = 0.1
	
	for C in C_options:
		for gamma in gamma_options:
			#~ gamma = 1.0
			regularization.append(C)
			# Create the model
			#~ model = LogisticRegression(penalty='l2', tol=tol, C=C, solver='liblinear', max_iter=100, multi_class='ovr', n_jobs=-1)
			model = SVC(tol=1e-4, C=C, gamma = gamma, kernel='rbf', probability=True)
			
			# Initialize the cross-validation
			num_cv = 5
			unique_song_labels = np.unique(song_labels)
			song_labels_perm = np.random.permutation(unique_song_labels)
			num_songs_per_val_set = float(len(unique_song_labels)) / num_cv	# Number of songs in a validation set
			
			score_train = 0
			score_val = 0
			loss_train = 0
			loss_val = 0
			
			for cv_iter in range(num_cv):
							
				# Select the train and validation set of songs for this iteration (no data leakage between songs)
				cv_song_labels_start = int(cv_iter * num_songs_per_val_set)
				cv_song_labels_end = int((cv_iter + 1) * num_songs_per_val_set)
				song_labels_val = song_labels_perm[cv_song_labels_start: cv_song_labels_end]
				song_labels_train = np.append(song_labels_perm[:cv_song_labels_start], song_labels_perm[cv_song_labels_end:])
				
				# Select the corresponding training samples
				train_mask = [(song_labels[i] in song_labels_train) for i in range(len(song_labels))]
				X_train = data[train_mask]
				print X_train.shape
				y_train = labels[train_mask]
				X_val = data[np.logical_not(train_mask)]
				y_val = labels[np.logical_not(train_mask)]
				
				# PCA analysis
				#~ pca = PCA(n_components=10)
				#~ pca.fit(X_train)
				#~ print pca.explained_variance_ratio_
				
				scaler = StandardScaler()
				
				X_train_new = scaler.fit_transform(X_train)
				X_val_new = scaler.transform(X_val)		
				
				# Train the model
				model.fit(X_train_new, y_train)
				
				# Evaluate the model
				score_train += model.score(X_train_new, y_train)
				score_val += model.score(X_val_new, y_val)
				# TODO logloss using log_proba and log_loss
				predicted = model.predict_proba(X_train_new)[:,1]
				loss_train += log_loss(y_train, predicted)
				predicted = model.predict_proba(X_val_new)[:,1]
				loss_val += log_loss(y_val, predicted)
				#~ print 'Support vectors: ' + str(model.n_support_)
				print '\tC = {:.2} gamma = {:.2} Train = {:.2}, {:.2} Val = {:.2}, {:.2}'.format(C, gamma, score_train/(cv_iter+1), loss_train/(cv_iter+1), score_val/(cv_iter+1), loss_val/(cv_iter+1)) 
				
			# Printing
			score_train = score_train / num_cv
			score_val = score_val / num_cv
			loss_train = loss_train / num_cv
			loss_val = loss_val / num_cv
			
			scores_val.append(score_val)
			scores_train.append(score_train)
			losses_val.append(loss_val)
			losses_train.append(loss_train)
			
			print 'C = {:.2} gamma = {:.2} Train = {:.2}, {:.2} Val = {:.2}, {:.2}'.format(C, gamma, score_train, loss_train, score_val, loss_val) 
			
	plt.figure()
	plt.semilogx(regularization, scores_val)
	plt.semilogx(regularization, scores_train)
	plt.semilogx(regularization, losses_val)
	plt.semilogx(regularization, losses_train)
	plt.show()

def test_evaluate():
	# Load the data
	X_train, y_train, _ = load_training_data_csv(train_data_file)
	X_test, y_test, titles = load_training_data_csv(test_data_file)
	# Scale the data
	scaler = StandardScaler()	
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)	
	# Build the model
	gamma, C = 0.001, 100	# Optimal parameters determined by CV
	model = SVC(tol=1e-4, C=C, gamma = gamma, probability=True)
	# Fit model
	model.fit(X_train, y_train)
	# Evaluate accuracy and loss
	score = model.score(X_test, y_test)
	loss  = log_loss(y_test, model.predict_proba(X_test)[:,1])
	
	for title, label, prob in zip(titles, y_test, model.predict_proba(X_test)[:,1]):
		print '{} {:.2} {} \t{}'.format(label, prob, np.round(prob), title)
	
	print 'Evaluation on test data: loss = {:.3}, acc = {:.3}'.format(loss, score)
	
	score = model.score(X_train, y_train)
	loss  = log_loss(y_train, model.predict_proba(X_train)[:,1])
	print 'Evaluation on test data: loss = {:.3}, acc = {:.3}'.format(loss, score)
	
	# Persist the model
	joblib.dump(scaler, 'feature_scaler.pkl')
	joblib.dump(model, 'svm_model.pkl')
	
	
def calculateFeaturesForDownbeat(audio):
	datarow = []		
	BINS_PER_OCTAVE = 12
	NUM_OCTAVES = 5
	spectogram = cqt(audio, sr=44100, bins_per_octave=BINS_PER_OCTAVE, n_bins=BINS_PER_OCTAVE*NUM_OCTAVES, hop_length=512)
	spec_db = np.absolute(spectogram)
	spec_db = librosa.amplitude_to_db(spectogram, ref=np.max)
	spec_db = (spec_db - np.min(spec_db)) / (np.max(spec_db) - np.min(spec_db))
	# Statistics for each frequency bin (each row => aggregate over axis 1 = columns)
	NUM_FRAMES = spectogram.shape[1]
	features_per_half_octave = []
	for i_freq in range(NUM_OCTAVES*2):			# Frequency resolution: every half octave
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
			features_cur_freq_window.append(np.mean(cur_w))	# Determine how 'loud' this window is approx
		# Aggregate over different time windows
		features_per_half_octave.append(np.mean(features_cur_freq_window))
		features_per_half_octave.append(np.var(features_cur_freq_window))
		features_per_half_octave.append(skew(features_cur_freq_window))
		features_per_half_octave.append(kurtosis(features_cur_freq_window))
	datarow = features_per_half_octave
	# Statistics for each time point (each column => aggregate over axis 0 = rows)
	#~ datarow.extend(np.std(spec_db, axis=0))
	#~ datarow.extend(skew(spec_db, axis=0))
	#~ datarow.extend(kurtosis(spec_db, axis=0))
	# Show
	return datarow
	
def evaluate_each_dbeat_of_song(path_to_file):
	
	song = Song(path_to_file)
	song.open()
	song.openAudio()
	
	TEMPO = 175.0
	audio = time_stretch_sola(song.audio, song.tempo / TEMPO, song.tempo, 0.0)
	
	features = []
	for dbeat in song.downbeats:
		dbeat_stretched = dbeat * song.tempo / TEMPO
		start_idx = int(dbeat_stretched * 44100)
		audio_cur = audio[start_idx : start_idx + 60480]
		
		features.append(calculateFeaturesForDownbeat(audio_cur))
	
	# Build model
	try:
		raise Exception('foo')
		model = joblib.load('svm_model.pkl') 
	except:
		X_train, y_train, _ = load_training_data_csv(train_data_file)
		gamma, C = 0.001, 100	# Optimal parameters determined by CV
		tol = 1e-7 
		model = SVC(tol=1e-4, C=C, gamma = gamma, probability=True)
		#~ model = LogisticRegression(penalty='l2', tol=tol, C=C, solver='liblinear', max_iter=100, multi_class='ovr', n_jobs=-1)
		model.fit(X_train, y_train)
		joblib.dump(model, 'svm_model.pkl')
	
	# Evaluate each segment of the song
	for score in model.predict_proba(features)[:,1]:
		print '{:.2}'.format(score)
	plt.figure()
	plt.plot(model.predict_proba(features)[:,1])
	plt.show()
	
if __name__ == '__main__':
	
	# Only one of the lines below should be uncommented!
	
	# Uncomment to train the model
	#~ train()
	# Uncomment to evaluate the accuracy on the test set
	#~ test_evaluate()
	# Uncomment when in Application directory, to evaluate each downbeat of a song and visualize the result
	evaluate_each_dbeat_of_song(sys.argv[1])
	
	
