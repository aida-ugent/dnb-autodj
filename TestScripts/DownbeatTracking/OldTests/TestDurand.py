'''
	Implements the method proposed by Durand et al
	However, instead of mahalanobis distance, I use derivative along x-axis
	
'''

from essentia import *
from essentia.standard import MonoLoader, Spectrum, Windowing, MFCC, FrameGenerator, Spectrum
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from skimage import filters
import numpy as np

import sys
import matplotlib.pyplot as plt
from BeatTracker import *

if len(sys.argv) != 2:
	print 'Usage : ', sys.argv[0], ' <filename>'
	exit()

filename = sys.argv[1]	

# ================== Load the audio =========================
loader = MonoLoader(filename = filename)
audio = loader()
audio = audio[44100:]

# # ================== Bpm estimation and beat tracking ================== 
print 'Extracting beat information...'
beatTracker = BeatTracker()
beatTracker.run(audio)
beats = beatTracker.getBeats()
bpm = beatTracker.getBpm()
phase = beatTracker.getPhase()
beats = beats - phase
print 'Bpm: ', bpm
print 'Frame size in samples: ', 44100 * (60.0/bpm)


#  ================== Construct self-similarity matrix and perform analysis  ================== 
# Has to happen on beat-aligned frames (cf paper)
FRAME_SIZE = int(44100 * (60.0/bpm))
HOP_SIZE = FRAME_SIZE
frames_per_second = (44100.0 / FRAME_SIZE) * (FRAME_SIZE / HOP_SIZE)
beats = beats * frames_per_second
spec = Spectrum(size = FRAME_SIZE - FRAME_SIZE % 2)
w = Windowing(type = 'hann')
spectrum = Spectrum() # FFT would return complex FFT, we only want magnitude
mfcc = MFCC()
pool = Pool()

# Step 0: align audio with phase

beats = beats - 0.5
matrices = {}

for start_sample_offset_beats in range(4):

	start_sample = int((phase + start_sample_offset_beats) * (44100.0 * 60 / bpm))

	# Step 1: Calculate framewise MFCC
	for frame in FrameGenerator(audio[start_sample:], frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:FRAME_SIZE-(FRAME_SIZE % 2)])))
		pool.add('lowlevel.mfcc' + str(start_sample_offset_beats), mfcc_coeffs)
		pool.add('lowlevel.mfcc_bands' + str(start_sample_offset_beats), mfcc_bands)
	# Step 2: Self-correlation
	matrices[start_sample_offset_beats] = 1 - pairwise_distances(pool['lowlevel.mfcc' + str(start_sample_offset_beats)],\
		pool['lowlevel.mfcc' + str(start_sample_offset_beats)],metric='cosine')
		
	print np.shape(pool['lowlevel.mfcc'+ str(start_sample_offset_beats)] )
	
	# Step 3: stretch the results to (0,1)
	imgNorm = MinMaxScaler()
	imgNorm.fit(matrices[start_sample_offset_beats])
	matrices[start_sample_offset_beats] = imgNorm.transform(matrices[start_sample_offset_beats]) * 0.99

	# Plot the result
	plt.subplot(2,2,start_sample_offset_beats + 1)
	plt.imshow(matrices[start_sample_offset_beats], aspect='auto', interpolation='nearest')
	for beat in beats[::]:
		pass
		#~ plt.axvline(x = beat, color = 'black')
	for beat in beats[::4]:
		#~ pass
		plt.axvline(x = beat)
		
	# Calculate where the downbeat should be
	pool.add('result', np.sum(np.abs(np.diff(matrices[start_sample_offset_beats]))))
		
print pool['result']
plt.show()
