import numpy as np
import sys
import os

# Load the libraries
print 'Loading Essentia...'
from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt

# Also see the default python example!
# Load a file using the command line
curFigure = 0
for f in os.listdir("."):
    if f.endswith(".mp3"):
		curFigure = curFigure + 1
		plt.figure(curFigure)
		filename = f
		
		# Load the audio
		print 'Loading audio file "', filename, '" ...'
		loader = essentia.standard.MonoLoader(filename = filename)
		audio = loader()

		#TODO test OnsetDetectionGlobal
		# ------------ Calculate the onset detection function
		print 'Initialising algorithms...'
		FRAME_SIZE = 1024
		HOP_SIZE = 512
		spec = Spectrum(size = FRAME_SIZE)
		w = Windowing(type = 'hann')
		# For calculating the phase and magnitude
		fft = FFT()
		c2p = CartesianToPolar()

		od_csd = OnsetDetection(method = 'complex')
		od_flux = OnsetDetection(method = 'complex')

		pool = Pool()

		print 'Calculating frame-wise onset detection curve...'
		for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
			mag, phase, = c2p(fft(w(frame)))
			pool.add('onsets.complex', od_csd(mag, phase))
			#pool.add('onsets.flux', od_flux(mag, phase))
			
		# Done! now show the result
		plt.subplot(411)
		plt.plot(pool['onsets.complex'])


		# ------------ Calculate the tempo function thingy (using method from paper)
		# Step 1: normalise the data using an adaptive mean threshold
		print 'Normalising result and half-wave rectifying it...'
		def adaptive_mean(x, N):
			#TODO efficient implementation instead of convolve
			return np.convolve(x, [1.0]*int(N), mode='same')/N
			
		novelty_mean = adaptive_mean(pool['onsets.complex'], 16.0)
		# Step 2: half-wave rectify the result
		novelty_hwr = (pool['onsets.complex'] - novelty_mean).clip(min=0)
		plt.subplot(412)
		plt.plot(novelty_hwr)   

		# Step 3: then calculate the autocorrelation of this signal
		print 'Autocorrelating resulting curve...'
		def autocorr(x):
			result = np.correlate(x, x, mode='full')
			return result[result.size/2:]

		novelty_autocorr = autocorr(novelty_hwr)
		plt.subplot(413)
		plt.plot(novelty_autocorr)   

		#( Step 4: Apply a shift-invariant comb filterbank) --> not yet
		# own implementation: sum over constant intervals
		print 'Iterating over valid BPM values...'
		valid_bpms = np.arange(170.0, 176.0, 0.01)
		for bpm in valid_bpms:
			num_frames_per_beat = (60.0 * 44100.0)/(512.0 * bpm)
			frames = (np.round(np.arange(0,np.size(novelty_autocorr),num_frames_per_beat)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
			pool.add('output.bpm', np.sum(novelty_autocorr[frames])/np.size(frames))

		plt.subplot(414)
		plt.plot(valid_bpms, pool['output.bpm'])   
		print 'Done!'

plt.show()
