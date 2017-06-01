import numpy as np
import sys, os
from sklearn.externals import joblib	# Model persistence
import featureLoudness, featureMFCC, featureOnsetIntegral, featureOnsetIntegralCsd, featureOnsetIntegralHfc

DOWNBEAT_DIR = './_annot_downbeat_temp'
FEATURE_MODULES = [featureLoudness, featureMFCC, featureOnsetIntegral, featureOnsetIntegralCsd, featureOnsetIntegralHfc] 

class DownbeatTracker:
	'''
		Detects the downbeat locations given the beat locations and audio
	'''
	def __init__(self, downbeatDirectory = DOWNBEAT_DIR):
		# Load the feature modules	
		self.model = joblib.load('model.pkl') 
		self.scaler = joblib.load('scaler.pkl')

		# Create annotation directory
		try:
			os.makedirs(DOWNBEAT_DIR)
		except OSError as exception:
			if exception.errno != errno.EEXIST:
				raise

	def getFeaturesForAudio(self, audio, beats):
	
		frame_indexer = range(4,len(beats) - 9) # -9 instead of -8 to prevent out-of-bound in featureLoudness
		
		# Calculate the features on every frame in the audio
		features_cur_file = None
		for module in feature_modules:
			absolute_feature_submatrix = module.feature_allframes(audio, beats, frame_indexer)
			if features_cur_file is None:
				features_cur_file = absolute_feature_submatrix
			else:
				features_cur_file = np.append(features_cur_file, absolute_feature_submatrix, axis=1)
		return features_cur_file

	def getDownbeats(self, audio, beats):
		'''
			Track the downbeats of the given audio file
		'''
		features = self.getFeaturesForAudio(audio, beats)
		probas = self.model.predict_log_proba(features)
			
		sum_log_probas = np.array([[0,0,0,0]], dtype='float64')		
		permuted_row = [0] * 4
		
		for i, j, row in zip(range(len(probas)), np.array(range(len(probas))) % 4, probas):
			permuted_row[:4-j] = row[j:]
			permuted_row[4-j:] = row[:j] # element i of permuted_row (i = 0,1,2,3) corresponds to the TRAJECTORY over the song starting with downbeat 0, 1, 2, 3
			perm_row_np = np.array([permuted_row])
			sum_log_probas = sum_log_probas + permuted_row
		
		downbeatIndex = (4 - np.argmax(sum_log_probas)) % 4
		return beats[downbeatIndex::4]
