''' Selfsimilarity matrix test '''

from essentia import *
from essentia.standard import MonoLoader, Spectrum, Windowing, MFCC, FrameGenerator, Spectrum
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from skimage import filters
import numpy as np

import scipy.spatial.distance

import sys
import matplotlib.pyplot as plt
from BeatTracker import *

if len(sys.argv) != 2:
	print 'Usage : ', sys.argv[0], ' <filename>'
	exit()

filename = sys.argv[1]	
# Load the audio
loader = MonoLoader(filename = filename)
audio = loader()

# Beat tracking

print 'Extracting beat information...'
beatTracker = BeatTracker()
beatTracker.run(audio)
beats = beatTracker.getBeats()
bpm = beatTracker.getBpm()
phase = beatTracker.getPhase()
beats = beats - phase
print 'Bpm: ', bpm
print 'Frame size in samples: ', 44100 * (60.0/bpm)


# Followed approach from Foote

# Adjust the frame size to the length of a beat, to extract beat-aligned information (zelf-uitgevonden)
FRAME_SIZE = int(44100 * (60.0/bpm)) / 4
FRAME_SIZE = FRAME_SIZE - FRAME_SIZE % 2
HOP_SIZE = FRAME_SIZE
frames_per_second = (44100.0 / FRAME_SIZE) * (FRAME_SIZE / HOP_SIZE)
print frames_per_second
beats = beats * frames_per_second
spec = Spectrum(size = FRAME_SIZE)
w = Windowing(type = 'hann')
spectrum = Spectrum() # FFT would return complex FFT, we only want magnitude
mfcc = MFCC()
pool = Pool()

# Step 0: align audio with phase
start_sample = int(phase * (44100.0 * 60 / bpm))
beats = beats

# Step 1: Calculate framewise MFCC
for frame in FrameGenerator(audio[start_sample:], frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
    mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    pool.add('lowlevel.mfcc', mfcc_coeffs)
    pool.add('lowlevel.mfcc_bands', mfcc_bands)

# Step 2: correlate
print np.shape(np.array(pool['lowlevel.mfcc']))
matrix = cosine_similarity(np.array(pool['lowlevel.mfcc']))
#~ matrix = 1 - scipy.spatial.distance.pdist(np.array(pool['lowlevel.mfcc']), metric='cosine')
print np.shape(matrix)

np.clip(matrix, 0.0, 1.0, out=matrix)

a = [pool['lowlevel.mfcc'][127], pool['lowlevel.mfcc'][128]]
print matrix[127][127:129], matrix[128][127:129]
print cosine_similarity(np.array(a))


plt.figure()
plt.plot(pool['lowlevel.mfcc'])

plt.figure()
plt.imshow(matrix, aspect='auto', interpolation='none')
for beat in beats[::]:
	#~ pass
	plt.axvline(x = beat, color = 'black')
for beat in beats[::4]:
	#~ pass
	plt.axvline(x = beat)
plt.colorbar()

plt.figure()
plt.imshow(filters.sobel_v(matrix), aspect='auto', interpolation='none')
for beat in beats[::]:
	#~ pass
	plt.axvline(x = beat, color = 'black')
for beat in beats[::4]:
	#~ pass
	plt.axvline(x = beat)
plt.colorbar()
plt.show()
