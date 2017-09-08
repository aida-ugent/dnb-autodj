'''
	Train script for the singing voice SVM model.
	Note that this is a working script, so not very clean, and you might need to adapt some boolean flags or something to make it work :)

	traindir = '../SingingVoiceDetection/'
	testdir = '../SingingVoiceDetection/test/'
	
	Output after last training:
	X_train: (36780, 125)
	X_test: (25572, 125)
	[acc, precision, recall, loss]
	[0.97805872756933121, 0.9017227877838685, 0.992244722102542, 0]
	[0.87290786798060382, 0.62270887883837178, 0.61107217939733705, 0]
	36780 6963 (55 train songs)
	25572 4281 (38 test songs)
'''

from song import Song
from songcollection import SongCollection
import sys, os, csv
from essentia import *
from essentia.standard import *

from timestretching import * # Data augmentation
import songtransitions # Data augmentation

import sklearn
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_features(audio):

	features = []
	
	FRAME_SIZE, HOP_SIZE = 2048, 1024
	low_f = 100
	high_f = 7000
	
	w = Windowing(type = 'hann')
	spec = Spectrum(size = FRAME_SIZE)
	mfcc = MFCC(lowFrequencyBound=low_f, highFrequencyBound=high_f)
	spectralContrast = SpectralContrast(lowFrequencyBound=low_f, highFrequencyBound=high_f)
	pool = Pool()
	
	for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
		frame_spectrum = spec(w(frame))
		spec_contrast, spec_valley = spectralContrast(frame_spectrum)
		mfcc_bands, mfcc_coeff = mfcc(frame_spectrum)
		pool.add('spec_contrast', spec_contrast)
		pool.add('spec_valley', spec_valley)
		pool.add('mfcc_coeff', mfcc_coeff)
		
	def add_moment_features(array):
		avg = np.average(array,axis=0)
		std = np.std(array,axis=0)
		skew = scipy.stats.skew(array,axis=0)
		#~ kurt = scipy.stats.kurtosis(array,axis=0)
		deltas = array[1:,:] - array[:-1,:]
		avg_d = np.average(deltas,axis=0)
		std_d = np.std(deltas,axis=0)
		#~ skew_d = scipy.stats.skew(deltas,axis=0)
	
		features.extend(avg)
		features.extend(std)
		features.extend(skew)
		#~ features.extend(kurt) # Does not help, even makes accuracy worse
		features.extend(avg_d)
		features.extend(std_d)
		#~ features.extend(skew_d) # Does not contribute a lot
	
	add_moment_features(pool['spec_contrast'])
	add_moment_features(pool['spec_valley'])
	add_moment_features(pool['mfcc_coeff'])
	
	return np.array(features,dtype='single')

if __name__ == '__main__':
	
	sc = SongCollection()
	for dir_ in sys.argv[1:]:
		sc.load_directory(dir_)
		
	# Read annotated files
	traindir = '../SingingVoiceDetection/'
	testdir = '../SingingVoiceDetection/test/'
	
	def get_songs_and_annots(csv_dir):
		songs = []
		annotations = []
		for filename in os.listdir(csv_dir):
			if filename.endswith('.csv'): 
				title, ext = os.path.splitext(filename)
				matching_songs = [s for s in sc.get_annotated() if s.title == title]
				if len(matching_songs) > 0:
					songs.extend(matching_songs)
					annot_cur = []
					with open(os.path.join(csv_dir, filename)) as csvfile:
						reader = csv.reader(csvfile)
						for line in reader:
							time = float(line[0])
							annot_cur.append(time)
					annotations.append(annot_cur)
		return songs, annotations	
		
	def extract_features_from_songs(songs, annotations_all_songs):
		pool = Pool()
		for song, annotations in zip(songs, annotations_all_songs):
			print 'Processing {}'.format(song.title)
			song.open()
			song.openAudio()
			
			audio_normal = song.audio
			# Data augmentation
			audio_semitone_up   = np.array(time_stretch_and_pitch_shift(audio_normal, 1.0, +1),dtype='single')
			audio_semitone_down = np.array(time_stretch_and_pitch_shift(audio_normal, 1.0, -1),dtype='single')
			
			audios = [audio_normal, audio_semitone_up, audio_semitone_down]
			
			annotations.insert(0,0.0)
			annotations.append(len(song.audio)/44100.0)
			
			annotation_idx = 1
			start_s = annotations[annotation_idx]
			cur_label = False # Start with non-vocal first
			
			for dbeat_idx in range(len(song.downbeats)-1):		
				
				start = int(song.downbeats[dbeat_idx]*44100)
				stop = int(song.downbeats[dbeat_idx+1]*44100)
				
				if abs(start_s - song.downbeats[dbeat_idx]) < abs(start_s - song.downbeats[dbeat_idx+1]):
					# Next downbeat is further away from annotation: this means that the current downbeat is closest
					# to the annotation instant: switch labeling	
					cur_label = not cur_label
					annotation_idx += 1
					start_s = annotations[annotation_idx]	
					#~ stream.write(audio_section, num_frames=len(audio_section), exception_on_underflow=False)
					
				for audio in audios:# Features on normal audio
					audio_section = audio[start:stop]
					features = calculate_features(audio_section)
					
					pool.add('features', features)
					pool.add('titles', song.title)
					pool.add('labels', 1 if cur_label else 0)
					
				#~ print dbeat_idx, cur_label
				while annotation_idx < len(annotations) and abs(start_s - song.downbeats[dbeat_idx]) < abs(start_s - song.downbeats[dbeat_idx+1]):
					annotation_idx += 1	
					cur_label = not cur_label
					start_s = annotations[annotation_idx]
			
			song.close()
		return pool['features'], pool['labels'], pool['titles']
					
	# Load features for each train and test song
	
	def annot_files_exist():
		return (
			os.path.isfile('singingvoice_X_train.bin.npy') and
			os.path.isfile('singingvoice_y_train.bin.npy') and
			os.path.isfile('singingvoice_t_train.bin.npy') and
			os.path.isfile('singingvoice_X_test.bin.npy') and
			os.path.isfile('singingvoice_y_test.bin.npy') and
			os.path.isfile('singingvoice_t_test.bin.npy')				
		)
	
	if not annot_files_exist():
		train_songs, train_annotations = get_songs_and_annots(traindir)
		test_songs, test_annotations = get_songs_and_annots(testdir)
		print '{} training files, {} test files'.format(len(train_songs), len(test_songs))
		X_train, y_train, t_train = extract_features_from_songs(train_songs, train_annotations)
		X_test, y_test, t_test = extract_features_from_songs(test_songs, test_annotations)
		
		np.save('singingvoice_X_train.bin', X_train)
		np.save('singingvoice_y_train.bin', y_train)
		np.save('singingvoice_t_train.bin', t_train)
		np.save('singingvoice_X_test.bin' , X_test )
		np.save('singingvoice_y_test.bin' , y_test )
		np.save('singingvoice_t_test.bin' , t_test )
	else:
		X_train = np.load('singingvoice_X_train.bin.npy')
		y_train = np.load('singingvoice_y_train.bin.npy')
		t_train = np.load('singingvoice_t_train.bin.npy')
		X_test  = np.load('singingvoice_X_test.bin.npy' )
		y_test  = np.load('singingvoice_y_test.bin.npy' )
		t_test  = np.load('singingvoice_t_test.bin.npy' )
		
		#~ samples = np.array([i%4!=0 for i in range(X_train.shape[0])])
		#~ X_train = X_train[samples,:]
		#~ y_train = y_train[samples]
		#~ t_train = t_train[samples]
		#~ samples = np.array([i%4!=0 for i in range(X_test.shape[0])])
		#~ X_test = X_test[samples,:]
		#~ y_test = y_test[samples]
		#~ t_test = t_test[samples]
	
	files = np.unique(t_train)
	N_files = files.size
	N_folds = 5
	
	def calculate_scores(svm, X, y):
		y_pred = svm.predict(X)
		#~ y_pred_prob = svm.predict_proba(X)[:,1]
		
		#~ y_pred = svm.predict(X) #np.array(y_pred_prob >= 0.7,dtype='single') # np.logical_and(y_pred[::3], y_pred[1::3], y_pred[2::3])
		y_agg = y
		
		#~ print sklearn.metrics.classification_report(y,y_pred)
		acc = sklearn.metrics.accuracy_score(y_agg,y_pred)
		precision = sklearn.metrics.precision_score(y_agg,y_pred)
		recall = sklearn.metrics.recall_score(y_agg,y_pred)
		#~ loss = sklearn.metrics.log_loss(y,y_pred_prob, labels=[0.0,1.0])
		loss = 0
		#~ print 'Score: acc {:.2f}\tloss {:.2f}\tf1 {:.2f}'.format(acc,loss,f1)
		return [acc, precision, recall, loss]
		
	cur_feature_mask = np.array(
		[
		True			
		for i in range(X_train.shape[1])
		])
	X_train = X_train[:,cur_feature_mask]
	X_test = X_test[:,cur_feature_mask]
	print 'X_train: {}'.format(X_train.shape)
	print 'X_test: {}'.format(X_test.shape)
	
	y_val_all = []
	t_val_all = []
	
	if True:
		for gamma in [0.01]: # Gamma > 0.01: high precision, low recall; < 0.01: other way around
			for C in [1.0]:
				
				print 'Results for C={:.3f}, gamma={}'.format(C,gamma)
				scores_train = []
				scores_val   = []
				scores_val_filter_only = []
				for cur_fold in range(N_folds):
					print '{} out of {} folds'.format(cur_fold, N_folds)
					cur_fold_test_files = files[cur_fold*N_files/N_folds:(cur_fold+1)*N_files/N_folds]
					cur_fold_train_files = [f for f in files if f not in cur_fold_test_files]
					cur_fold_mask = np.array([t in cur_fold_train_files for t in t_train])
					
					X_train_cv = X_train[cur_fold_mask,:]
					y_train_cv = y_train[cur_fold_mask]
					print X_train_cv.shape
					print y_train_cv.shape, y_train_cv[y_train_cv == 1].shape
					X_val_cv = X_train[np.logical_not(cur_fold_mask),:]
					y_val_cv = y_train[np.logical_not(cur_fold_mask)]
					t_val_cv = t_train[np.logical_not(cur_fold_mask)]
		
					scaler = preprocessing.StandardScaler()
					X_train_scaled = scaler.fit_transform(X_train_cv)
					
					model = sklearn.svm.SVC(C=C, gamma=gamma, probability=False, tol=0.01,class_weight='balanced')
					#~ model = sklearn.linear_model.LogisticRegression(C=C,class_weight='balanced')
					model.fit(X_train_scaled, y_train_cv)
					
					blub = calculate_scores(model,scaler.transform(X_train_cv), y_train_cv)
					print blub
					scores_train.append(blub)
					blub = calculate_scores(model,scaler.transform(X_val_cv), y_val_cv)
					print blub
					scores_val.append(blub)
					
					y_val_all.extend(model.predict(scaler.transform(X_val_cv)))
					t_val_all.extend(t_val_cv)
					
					#~ print scores_val[cur_fold]
					
					#~ y_pred = np.array(model.predict_proba(scaler.transform(X_val_cv))[:,1] >= 0.7,dtype='single')
					#~ for title in np.unique(t_val_cv):
						#~ cur_fold_mask = np.array([t == title for t in t_val_cv])
						#~ np.save('singingvoice_predicted_'+title, y_pred[cur_fold_mask])
					#~ for y_true, y_pred, title in zip(y_val_cv, model.predict(scaler.transform(X_val_cv)), t_val_cv):
						#~ print '{} {} {}'.format(y_true, y_pred, title)
				
				acc, prec, recall, f1 = tuple(np.average(scores_train,axis=0))
				#~ acc, prec, recall, f1 = tuple(np.average(scores_val_filter_only,axis=0))
				acc2, prec2, recall2, f12 = tuple(np.average(scores_val,axis=0))
				#~ print 'TRAIN acc: {:.3f}, prec: {:.3f}, recall: {:.3f}, f1: {:.3f}'.format(acc, prec, recall, f1)
				print 'TRAIN: {:.3f} {:.3f} {:.3f} {:.3f}; VAL:{:.3f} {:.3f} {:.3f} {:.3f}'.format(acc, prec, recall, f1,acc2, prec2, recall2, f12)
	
	np.save('singingvoice_y_val.bin', y_val_all)
	np.save('singingvoice_t_val.bin', t_val_all)
	
	# TODO pick best C and gamma and perform test
	scaler = preprocessing.StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	#~ model = sklearn.linear_model.LogisticRegression(C=1.0,class_weight='balanced')
	model = sklearn.svm.SVC(C=1.0, gamma=0.01, probability=False, tol=0.001,class_weight='balanced')
	model.fit(X_train_scaled, y_train)
	print calculate_scores(model,scaler.transform(X_train), y_train)
	print calculate_scores(model,scaler.transform(X_test), y_test)
	print len(y_train), np.sum(y_train == 1)
	print len(y_test), np.sum(y_test == 1)
	
	#~ joblib.dump(scaler, 'singingvoice_scaler.pkl')
	#~ joblib.dump(model, 'singingvoice_model.pkl')		

