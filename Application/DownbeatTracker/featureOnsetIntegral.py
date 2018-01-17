# Copyright 2017 Len Vande Veire, IDLab, Department of Electronics and Information Systems, Ghent University
# This file is part of the source code for the Auto-DJ research project, published in Vande Veire, Len, and De Bie, Tijl, "From raw audio to a seamless mix: an artificial intelligence approach to creating an automated DJ system.", 2018 (submitted)
# Released under AGPLv3 license.

import numpy as np
from essentia import *
from essentia.standard import Spectrum, Windowing, OnsetDetection, FrameGenerator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

'''
	This feature vector contains correlations of the onset function between different frames.
	The reasoning behind it, is that on a downbeat, the correlation might suddenly change a lot 
	because in a downbeat more or less musical activity might be going on.
'''


def feature_allframes(song, frame_indexer = None):
	
	audio = song.audio
	beats = song.beats
	fft_result_mag = song.fft_mag_1024_512
	fft_result_ang = song.fft_phase_1024_512
	
	od_hfc = OnsetDetection(method = 'flux')
	pool = Pool()
	HOP_SIZE = 512

	for mag,phase in zip(fft_result_mag, fft_result_ang):
		pool.add('onsets.flux', od_hfc(mag, phase))
	
	# Normalize and half-rectify onset detection curve
	def adaptive_mean(x, N):
		return np.convolve(x, [1.0]*int(N), mode='same')/N
		
	novelty_mean = adaptive_mean(pool['onsets.flux'], 16.0)
	novelty_hwr = (pool['onsets.flux'] - novelty_mean).clip(min=0)
	novelty_hwr = novelty_hwr / np.average(novelty_hwr)
	
	# For every frame in frame_indexer, 
	if frame_indexer is None:
		frame_indexer = range(4,len(beats) - 1) # Exclude first frame, because it has no predecessor to calculate difference with
		
	# Feature: correlation between current frame onset detection f and of previous frame
	# Feature: correlation between current frame onset detection f and of next frame
	# Feature: diff between correlation between current frame onset detection f and corr cur and next
	onset_integrals = np.zeros((2 * len(beats), 1))
	frame_i = (np.array(beats) * 44100.0/ HOP_SIZE).astype('int')
	onset_correlations = np.zeros((len(beats), 21))
	
	for i in [i for i in range(len(beats)) if (i in frame_indexer) or (i+1 in frame_indexer)
		or (i-1 in frame_indexer) or (i-2 in frame_indexer) or (i-3 in frame_indexer)
		or (i-4 in frame_indexer) or (i-5 in frame_indexer) or (i-6 in frame_indexer) or (i-7 in frame_indexer)]:
		
		half_i = int((frame_i[i] + frame_i[i+1]) / 2)
		cur_frame_1st_half = novelty_hwr[frame_i[i] : half_i]
		cur_frame_2nd_half = novelty_hwr[half_i : frame_i[i+1]]
		onset_integrals[2*i] = np.sum(cur_frame_1st_half)
		onset_integrals[2*i + 1] = np.sum(cur_frame_2nd_half)
	
	# Step 2: Calculate the cosine distance between the MFCC values
	for i in frame_indexer:
		
		# Correlation gives symmetrical results, which is not necessarily what we want.
		# Better: integral of sum, multiplied by sign of difference
		# If integral large a, b large but difference positive (a-b): a contains more onsets than 
		
		onset_correlations[i][0] = max(np.correlate(novelty_hwr[frame_i[i-1] : frame_i[i]], novelty_hwr[frame_i[i] : frame_i[i+1]], mode='valid')) # Only 1 value
		onset_correlations[i][1] = max(np.correlate(novelty_hwr[frame_i[i] : frame_i[i+1]], novelty_hwr[frame_i[i+1] : frame_i[i+2]], mode='valid')) # Only 1 value
		onset_correlations[i][2] = max(np.correlate(novelty_hwr[frame_i[i] : frame_i[i+1]], novelty_hwr[frame_i[i+2] : frame_i[i+3]], mode='valid')) # Only 1 value
		onset_correlations[i][3] = max(np.correlate(novelty_hwr[frame_i[i] : frame_i[i+1]], novelty_hwr[frame_i[i+3] : frame_i[i+4]], mode='valid')) # Only 1 value
		
		# Difference in integrals of novelty curve between frames
		# Quantifies the difference in number and prominence of onsets in this frame
		onset_correlations[i][4] = onset_integrals[2*i] - onset_integrals[2*i-1]
		onset_correlations[i][5] = onset_integrals[2*i+2] + onset_integrals[2*i+3] - onset_integrals[2*i-1] - onset_integrals[2*i-2]
		for j in range(1,16):
			onset_correlations[i][5 + j] = onset_integrals[2*i + j] - onset_integrals[2*i]
		
			
	# Include the MFCC coefficients as features
	result = onset_correlations[frame_indexer]
	return preprocessing.scale(result)



