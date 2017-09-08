'''
	This script is used to train and evaluate the downbeat classifier model. Example usage to train the best model so far:
	
	python downbeatModelTrainer.py 9999 featureLoudness.py featureMFCC.py featureOnsetIntegral.py featureOnsetIntegralCsd.py featureOnsetIntegralHfc.py --forceRecalculate
	
	
'''

import numpy as np
import sys, os, importlib, time
import csv

from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt

from sklearn import preprocessing, svm, linear_model
from sklearn.externals import joblib	# Model persistence
from sklearn.metrics import log_loss

ANNOT_SUBDIR = '_annot_beat_downbeat/'
ANNOT_DOWNB_PREFIX = 'downbeats_'
ANNOT_BEATS_PREFIX = 'beats_'
FEATURE_SUBDIR = '_features/'

options = {
	'--forceRecalculate' : False,
	'--reuseMatrix' : False,
	'--reuseTestMatrix' : False,
	'--reuseModel' : False,
	'--showFigures' : False,
	'--noFitModel' : False
}

MODEL_FILE = 'model.pkl'
SCALER_FILE = 'scaler.pkl'

class UnannotatedException(Exception):
    pass

def loadAnnotationFile(directory, song_title, prefix):
	'''
	Loads an input file with annotated times in seconds.
	
	-Returns: A numpy array with the annotated times parsed from the file.
	'''
	input_file = directory + ANNOT_SUBDIR + prefix + song_title + '.txt'
	result = []
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line[0] == '#':
					continue
				result.append(float(line))	
	else:
		raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
	return np.array(result)
	
def feature_file_name(directory, song_title, module_name):
	return directory + FEATURE_SUBDIR + song_title + '_' + module_name + '.csv'

def feature_file_exists(directory, song_title, module_name):
	return os.path.exists(feature_file_name(directory, song_title, module_name))
	
def read_features_and_labels_from_file(path, hasSongRowIndices = False):
	labels = []
	matrix = []
	song_row_indices = None
	with open(path, 'rb') as csvfile:
		r = csv.reader(csvfile, delimiter=',', quotechar='|')
		# Read the song row indices (number of feature vectors per song)
		if hasSongRowIndices:
			firstrow = next(r)
			song_row_indices = [int(i) for i in firstrow]
		# Read the matrix and the labels
		for row in r:
			labels.append(float(row[0]))
			matrix.append([ float(i) for i in row[1:] ])
	return np.array(matrix), np.transpose(labels), np.transpose(song_row_indices) if song_row_indices != None else None
	
def write_features_and_labels_to_file(matrix, labels, path, song_row_indices = None):
	with open(path, 'wb') as csvfile:
		w = csv.writer(csvfile, delimiter=',',	quotechar='|', quoting=csv.QUOTE_MINIMAL)
		# Write the song indices row
		if song_row_indices is not None:
			w.writerow(song_row_indices)
		# Write the matrix and its labels
		for mrow, l in zip(matrix, labels):
			row = [l]
			row.extend(mrow)
			w.writerow(row)

def read_features_from_file(directory, song_title, module_name):
	matrix, labels, song_row_indices = read_features_and_labels_from_file(feature_file_name(directory, song_title, module_name))
	return matrix
	
def write_features_to_file(matrix, labels, directory, song_title, module_name):
	write_features_and_labels_to_file(matrix, labels, feature_file_name( directory, song_title, module_name))
	
def readDownbeatIndexFromFile(directory, song_title):
	
	prefix = 'downbeats_'
	input_file = directory + ANNOT_SUBDIR + prefix + song_title + '.txt'
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line.startswith('#downbeat '):
					result = int(line[10])
					if result in range(4):
						return result
					else:
						Exception('Downbeat index should be in range(4)!')
	else:
		raise UnannotatedException('Attempting to load downbeat index annotations of unannotated audio!')
	raise UnannotatedException('Attempting to load downbeat index from file without downbeat index annotation!')

def trimAudio(audio, beats):
	
	beats = beats * 44100 # Beats in samples
	
	# Calculate beatwise RMS
	rms = []
	for i in range(len(beats) - 1):
		rms.append(np.sqrt(np.mean(np.square(audio[int(beats[i]) : int(beats[i+1])]))))
	def adaptive_mean(x, N):
		return np.convolve(x, [1.0]*int(N), mode='same')/N
	rms_adaptive = adaptive_mean(rms, 4)
	rms_adaptive_max = max(rms_adaptive)
	
	# Determine cut positions
	start, end, ratiox = 0,0,0
	ratios = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
	for ratio in ratios:
		# Extract beginning and end
		for i in range(len(rms)):
			if rms[i] > ratio*rms_adaptive_max:
				start = i
				break # Go to trail cutting
		for i in range(len(rms)):
			if rms[len(rms) - i - 1] > ratio*rms_adaptive_max:
				end = len(rms) - i - 1
				break # Go to check if file is not too small now
		# If beginning and end cut not more than 50% of the song, then it is ok
		if end - start >= len(rms) * 0.4:
			ratiox = ratio
			break
			
	return start, end

def getFeaturesForFile(directory, file_name, feature_modules, forceRecalculateFeatures = False):
	
	# Get the song title
	song_title, song_ext = os.path.splitext(os.path.basename(file_name))
	print 'Processing ' + song_title
	
	# Load the annotated downbeat file
	beats = loadAnnotationFile(directory, song_title, ANNOT_BEATS_PREFIX)
	
	# Load the audio			
	loader = essentia.standard.MonoLoader(filename = directory + file_name)
	audio = loader()
	
	# Cut the audio to the relevant (main) part
	FRAME_INDEXER_MIN = 4
	FRAME_INDEXER_MAX = len(beats) - 9
	trim_start_beat, trim_end_beat = trimAudio(audio, beats)
	indexer_start = max(FRAME_INDEXER_MIN, trim_start_beat)
	indexer_end = min(FRAME_INDEXER_MAX, trim_end_beat)
	frame_indexer = range(indexer_start, indexer_end) # -9 instead of -8 to prevent out-of-bound in featureLoudness
	# Uncomment the line below to test without audio cropping
	#~ frame_indexer = range(FRAME_INDEXER_MIN, FRAME_INDEXER_MAX) # -9 instead of -8 to prevent out-of-bound in featureLoudness
	
	# Read the downbeat index from the input file, and add it to the list
	# There is a difference in the definition of the label in the file, and the classification label:
	# The downbeat index stored in the file is the index of the first beat that is a downbeat. If the third beat of the song is a downbeat, then this label is '2' (start counting from 0)
	# The label that we use here is the classification label of the first full beat of the song. If the first beat of the song is the third beat in the measure, then the label is '2'
	# There will thus be difference in label if the downbeat index is 1, then classification label will be 3 and vice versa
	label = (4 - readDownbeatIndexFromFile(directory, song_title)) % 4
	labels_cur_file = np.tile(np.mod(range(label,label+4), 4), int(np.ceil(len(beats) / 4.0)))[frame_indexer]
	
	# Calculate the features on every frame in the audio
	features_cur_file = None
	for module in feature_modules:
		if not feature_file_exists(directory, song_title, module.__name__) or forceRecalculateFeatures:
			# Recalculate the features
			print '> Recalculating ', module.__name__
			absolute_feature_submatrix = module.feature_allframes(audio, beats, frame_indexer)
			write_features_to_file(absolute_feature_submatrix, labels_cur_file, directory, song_title, module.__name__)
		else:
			print '> Reading ', module.__name__
			# Load the features from the input files
			absolute_feature_submatrix = read_features_from_file(directory, song_title, module.__name__)
		
		# Append features (loaded from file or recalculated) to the feature matrix
		if features_cur_file is None:
			features_cur_file = absolute_feature_submatrix
		else:
			features_cur_file = np.append(features_cur_file, absolute_feature_submatrix, axis=1)
	return (features_cur_file, labels_cur_file)

def getFeaturesAndLabelsFromDirectory(directory, feature_modules, forceRecalculateFeatures = False):
	'''
	Loads the input data from the specified directory and extracts the features
	per audio input file.
	
		-Returns: 	A matrix X of dimensionality (num_fragments, num_features)
					An array y of dimensionality (num_fragments), the output labels
					An array u of dimensionality (num_songs + 1), with start and end points in matrix rows of the song features
	'''
	features, labels, num_frames_per_song = None, [], [0]
	
	test = 0
	
	for f in os.listdir(directory):
		
		test += 1
		TEST = NUM_SONGS
		if test > TEST:
			break
			
		if f.endswith('.mp3') or f.endswith('.wav'):
						
			# Get the features for the current file
			try:
				features_cur_file, labels_cur_file = getFeaturesForFile(directory, f, feature_modules, forceRecalculateFeatures)
			except UnannotatedException as e:
				print e
				continue
				
			# Add the features to the feature matrix
			if features is None:
				features = features_cur_file
			else:
				features = np.append(features, features_cur_file, axis = 0)
			
			labels.extend(labels_cur_file)
			
			num_frames_per_song.append(np.shape(features_cur_file)[0])
			
	print 'Features dimen: ', np.shape(features)
	labels = np.array(labels)
	num_frames_per_song = np.array(num_frames_per_song)
	return (features, labels, num_frames_per_song)
				
if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print 'Usage: ', sys.argv[0], ' featureModule1.py ... featureModuleN.py'
		exit()
	
	# Disable obnoxious INFO Log
	essentia.log.infoActive = False
	
	directory = '../music/'		# Input directory of the music and training data
	test_directory = '../music/test/'
	
	# Parse the input modules
	feature_modules = []
	
	global NUM_SONGS 
	NUM_SONGS = int(sys.argv[1])
		
	for arg in sys.argv[2:]:
		if arg[:2] == '--':
			# Parse option
			if arg in options.keys():
				options[arg] = True
			else:
				print arg + ' is not a valid option!'
				print 'Options: ' + str(options.keys())
				exit()
		else:
			# Parse feature module
			analysis_method_file = os.path.splitext(os.path.basename(arg))[0]
			analysis_method_module = importlib.import_module(analysis_method_file)
			if 'feature_allframes' in dir(analysis_method_module):
				feature_modules.append(analysis_method_module)
			else:
				raise Exception('File ' + analysis_method_file + ' does not contain any feature analysis functions!')
	
	model, scaler = None, None
	
	if options['--reuseModel']:
		print 'Loading model from file...'
		try:
			model = joblib.load(MODEL_FILE) 
			scaler = joblib.load(SCALER_FILE)
		except:
			options['--reuseModel'] = False
		print 'Loading unscaled feature matrix and labels from csv file...'
		matrix, labels, num_frames_per_song = read_features_and_labels_from_file('./features_train.csv', True)
		print matrix.shape
		matrix = scaler.transform(matrix)
	
	if not options['--reuseModel']:
		# Get training data
		if not options['--reuseMatrix']:
			print 'Building feature matrix from audio files...'
			matrix, labels, num_frames_per_song = getFeaturesAndLabelsFromDirectory(directory, feature_modules, options['--forceRecalculate'])
			print matrix.shape, labels.shape, num_frames_per_song.shape
			write_features_and_labels_to_file(matrix, labels, './features_train.csv', num_frames_per_song)
		else:
			print 'Loading unscaled feature matrix and labels from csv file...'
			matrix, labels, num_frames_per_song = read_features_and_labels_from_file('./features_train.csv', hasSongRowIndices = True)
			
		# Scale the features
		scaler = preprocessing.StandardScaler().fit(matrix)
		matrix = scaler.transform(matrix)
		
		# Learn model on training data		
		if not options['--noFitModel']:
			print 'Fitting model...'
			#~ model = svm.SVC(C=1.0, kernel='rbf', gamma='auto', tol=0.001, cache_size=1000)
			
			# Time how long it takes to fit
			model = linear_model.LogisticRegression(tol=0.000001, C=1000, fit_intercept=True, solver='sag', max_iter=100000, multi_class='ovr', n_jobs=-1)
			#~ model = svm.SVC(C=1.0, kernel='rbf', gamma='auto', tol=0.001, cache_size=1000, probability=True)
			time1 = time.time()
			model.fit(matrix, labels)
			time2 = time.time()
			print 'Fitting model took %0.3f ms' % ((time2 - time1) * 1000.0)
			# Save model data
			joblib.dump(model, MODEL_FILE) 
			joblib.dump(scaler, SCALER_FILE)
		else:
			print 'Skip fitting model!'
	
	# Get test data
	print 'Loading test data...'
	if not options['--reuseTestMatrix']:
		matrix_test, labels_test, num_frames_per_song_test = getFeaturesAndLabelsFromDirectory(test_directory, feature_modules, options['--forceRecalculate'])
		write_features_and_labels_to_file(matrix_test, labels_test, './features_test.csv', num_frames_per_song_test)
	else:
		matrix_test, labels_test, num_frames_per_song_test = read_features_and_labels_from_file('./features_test.csv', hasSongRowIndices = True)
	
	matrix_test = scaler.transform(matrix_test)
	
	# Some visualisation
	if options['--showFigures']:
		plt.figure(1)
		VMIN, VMAX = -5, 5
		plt.subplot(221)
		plt.imshow(matrix[labels == 0], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(222)
		plt.imshow(matrix[labels == 1], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(223)
		plt.imshow(matrix[labels == 2], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(224)
		plt.imshow(matrix[labels == 3], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		
		plt.figure(2)
		VMIN, VMAX = -5, 5
		plt.subplot(221)
		plt.imshow(matrix_test[labels_test == 0], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(222)
		plt.imshow(matrix_test[labels_test == 1], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(223)
		plt.imshow(matrix_test[labels_test == 2], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.subplot(224)
		plt.imshow(matrix_test[labels_test == 3], aspect='auto', interpolation='none', vmin=VMIN, vmax=VMAX, cmap='bwr')
		plt.show()
	
	
	# See how well we did
	print 'Mean accuracy train: ' + str(model.score(matrix, labels))
	print 'Mean accuracy test: ' + str(model.score(matrix_test, labels_test))
	
	# Print log loss
	loss_train = log_loss(labels, model.predict_proba(matrix))
	loss_test = log_loss(labels_test, model.predict_proba(matrix_test))
	print 'Logloss train: ' + str(loss_train)
	print 'Logloss test: ' + str(loss_test)
	
	# See how it does on a SONG basis
	indices = np.cumsum(num_frames_per_song)
	errors = 0
	
	for i in range(len(num_frames_per_song) - 1):
		
		probas = model.predict_log_proba(matrix[indices[i]:indices[i+1]])
		
		sum_log_probas = np.array([0,0,0,0], dtype='float64')
		permuted_row = [0] * 4
		
		for j, row in zip( np.array(range(len(probas))) % 4, probas):
			permuted_row[:4-j] = row[j:]
			permuted_row[4-j:] = row[:j]
			sum_log_probas += permuted_row
			if labels[indices[i] + j] != np.argmax(row):
				pass
				#~ print 'Incorrect detection ', labels[indices[i] + j], np.argmax(row), row
		
		if labels[indices[i]] != np.argmax(sum_log_probas):
			errors += 1
		
		print int(labels[indices[i]]), np.argmax(sum_log_probas), sum_log_probas
	print 'Wrongly detected downbeat of ', errors / float(len(num_frames_per_song) - 1), ' songs in training set (', errors, ' out of ', float(len(num_frames_per_song) - 1) , ')'

	
	indices = np.cumsum(num_frames_per_song_test)
	errors = 0
	for i in range(len(num_frames_per_song_test) - 1):
		probas = model.predict_log_proba(matrix_test[indices[i]:indices[i+1]])
		
		sum_log_probas = np.array([0,0,0,0], dtype='float64')
		permuted_row = [0] * 4
		for j, row in zip( np.array(range(len(probas))) % 4, probas):
			permuted_row[:4-j] = row[j:]
			permuted_row[4-j:] = row[:j]
			sum_log_probas += permuted_row
		if labels_test[indices[i]] != np.argmax(sum_log_probas):
			errors += 1
		print int(labels_test[indices[i]]), np.argmax(sum_log_probas), sum_log_probas
	print 'Wrongly detected downbeat of ', errors / float(len(num_frames_per_song_test) - 1) ,' songs in test set (', errors, ' out of ', float(len(num_frames_per_song_test) - 1), ')' 
