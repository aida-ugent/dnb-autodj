import numpy as np
from essentia import *
from essentia.standard import Windowing, Loudness
from sklearn import preprocessing

def feature_allframes(audio, beats, frame_indexer = None):
	
	# Initialise the algorithms
	w = Windowing(type = 'hann')
	loudness = Loudness()
	
	if frame_indexer is None:
		frame_indexer = range(1,len(beats) - 1) # Exclude first frame, because it has no predecessor to calculate difference with
		
	# 1 loudness value by default
	loudness_values = np.zeros((len(beats), 1))
	# 1 difference value between loudness value cur and cur-1
	# 1 difference value between loudness value cur and cur-4
	# 1 difference value between differences above
	loudness_feature_vector = np.zeros((len(beats), 4))
	
	# Step 1: Calculate framewise for all output frames
	# Calculate this for all frames where this frame, or its successor, is in the frame_indexer
	for i in [i for i in range(len(beats)) if (i in frame_indexer) or (i-1 in frame_indexer) or (i-2 in frame_indexer) or (i-3 in frame_indexer)]:
		
		SAMPLE_RATE = 44100
		start_sample = int(beats[i] * SAMPLE_RATE)
		end_sample = int(beats[i+1] * SAMPLE_RATE) 
		#print start_sample, end_sample
		frame = audio[start_sample : end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1]
		loudness_values[i] = loudness(w(frame))
		
	loudness_values = preprocessing.scale(loudness_values)
	# Step 2: construct feature vector
	for i in frame_indexer:
		loudness_feature_vector[i] = np.reshape(loudness_values[i:i+4], (4,))
		
	# Include the raw values as absolute features
	result = loudness_feature_vector[frame_indexer]
	
	#~ print np.shape(result), np.shape(loudness_values), np.shape(loudness_differences)
	return result
