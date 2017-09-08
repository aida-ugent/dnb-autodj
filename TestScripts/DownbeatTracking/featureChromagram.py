import numpy as np
from essentia import *
from essentia.standard import Windowing, Spectrum, SpectralPeaks, HPCP
from sklearn import preprocessing

def feature_allframes(audio, beats, frame_indexer = None):
	
	# Initialise the algorithms
	w = Windowing(type = 'blackmanharris92')
	spectrum = Spectrum()
	specPeaks = SpectralPeaks()
	hpcp = HPCP()
	
	if frame_indexer is None:
		frame_indexer = range(1,len(beats) - 1) # Exclude first frame, because it has no predecessor to calculate difference with
		
	# 12 chromagram values by default
	chroma_values = np.zeros((len(beats), 12))
	# Difference between chroma vectors
	chroma_differences = np.zeros((len(beats), 3))
	
	# Step 1: Calculate framewise for all output frames
	# Calculate this for all frames where this frame, or its successor, is in the frame_indexer
	for i in [i for i in range(len(beats)) if (i in frame_indexer) or (i+1 in frame_indexer) or (i+1 in frame_indexer)]:
		
		SAMPLE_RATE = 44100
		start_sample = int(beats[i] * SAMPLE_RATE)
		end_sample = int(beats[i+1] * SAMPLE_RATE) 
		#print start_sample, end_sample
		frame = audio[start_sample : (end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1)]
		freq, mag = specPeaks(spectrum(w(frame)))
		chroma_values[i] = hpcp(freq, mag)
	
	# Step 2: Calculate the cosine distance between the MFCC values
	for i in frame_indexer:
		chroma_differences[i][0] = np.linalg.norm(chroma_values[i] - chroma_values[i-1])
		chroma_differences[i][1] = np.linalg.norm(chroma_values[i] - chroma_values[i+1])
		chroma_differences[i][2] = np.linalg.norm(chroma_values[i-1] - chroma_values[i+1])
		
	# Include the raw values as absolute features
	result = np.append(chroma_values[frame_indexer], chroma_differences[frame_indexer], axis=1)
	
	#~ print np.shape(result), np.shape(chroma_values), np.shape(chroma_differences)
	return preprocessing.scale(result)
