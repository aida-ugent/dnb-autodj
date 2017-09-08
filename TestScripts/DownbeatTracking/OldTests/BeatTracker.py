import numpy as np
import sys
from essentia import *
from essentia.standard import Spectrum, Windowing, CartesianToPolar, OnsetDetection, FFT, FrameGenerator

class BeatTracker:
	'''Detects the BPM, phase and locations of the beats for the given input audio'''
	
	def __init__(self, minBpm = 100.0, maxBpm = 199.0, stepBpm = 0.01, FRAME_SIZE = 1024, HOP_SIZE = 512, SAMPLE_RATE = 44100.0, ):
		self.minBpm = minBpm
		self.maxBpm = maxBpm
		self.stepBpm = stepBpm
		self.FRAME_SIZE = FRAME_SIZE
		self.HOP_SIZE = HOP_SIZE
		self.SAMPLE_RATE = SAMPLE_RATE
		
		self.bpm = None
		self.phase = None
		self.beats = None
		
	def getBpm(self):
		'''Returns the BPM for the analysed audio.
		
		:returns Beats per minute
		'''
		if self.bpm is None:
			raise Exception('No BPM detected yet, you must run the BeatTracker first!')
		return self.bpm
		
	def getPhase(self):
		'''Returns the beat phase for the analysed audio.
		
		:returns Phase in seconds
		'''
		if self.phase is None:
			raise Exception('No phase detected yet, you must run the BeatTracker first!')
		return self.phase
		
	def getBeats(self):
		'''Returns the beat locations for the analysed audio. These beats are all equidistant (constant BPM is assumed).
		
		:returns Array of beat locations in seconds
		'''
		if self.beats is None:
			raise Exception('No beats detected yet, you must run the BeatTracker first!')
		return self.beats
		
	def run(self, audio):	
			
		# TODO put this in some util class
		def numFramesPerBeat(bpm):
			return (60.0 * self.SAMPLE_RATE)/(self.HOP_SIZE * bpm)			
		def autocorr(x):
			result = np.correlate(x, x, mode='full')
			return result[result.size/2:]
		def adaptive_mean(x, N):
			#TODO efficient implementation instead of convolve
			return np.convolve(x, [1.0]*int(N), mode='same')/N
					
		# Step 0: calculate the CSD (Complex Spectral Difference) features
		# and the associated onset detection function 
		spec = Spectrum(size = self.FRAME_SIZE)
		w = Windowing(type = 'hann')
		fft = FFT()
		c2p = CartesianToPolar()
		od_csd = OnsetDetection(method = 'complex')

		pool = Pool()
		
		# TODO test faster (numpy) way
		for frame in FrameGenerator(audio, frameSize = self.FRAME_SIZE, hopSize = self.HOP_SIZE):
			mag, phase, = c2p(fft(w(frame)))
			pool.add('onsets.complex', od_csd(mag, phase))
			
		# Step 1: normalise the data using an adaptive mean threshold					
		novelty_mean = adaptive_mean(pool['onsets.complex'], 16.0)
		
		# Step 2: half-wave rectify the result
		novelty_hwr = (pool['onsets.complex'] - novelty_mean).clip(min=0)

		# Step 3: then calculate the autocorrelation of this signal
		novelty_autocorr = autocorr(novelty_hwr)  

		# Step 4: Sum over constant intervals to detect most likely BPM
		valid_bpms = np.arange(self.minBpm, self.maxBpm, self.stepBpm)
		for bpm in valid_bpms:
			frames = (np.round(np.arange(0,np.size(novelty_autocorr), numFramesPerBeat(bpm))).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
			pool.add('output.bpm', np.sum(novelty_autocorr[frames])/np.size(frames))  
		bpm = valid_bpms[np.argmax(pool['output.bpm'])]

		# Step 5: Calculate phase information
		valid_phases = np.arange(0.0, 60.0/bpm, 0.001) # Valid phases in SECONDS
		for phase in valid_phases:
			# Convert phase from seconds to frames
			phase_frames = (phase * 44100.0) / (512.0)
			frames = (np.round(np.arange(phase_frames,np.size(novelty_hwr), numFramesPerBeat(bpm))).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
			pool.add('output.phase', np.sum(novelty_hwr[frames])/np.size(frames))
		phase = valid_phases[np.argmax(pool['output.phase'])]
		
		# Step 6: Determine the beat locations
		spb = 60./bpm #seconds per beat
		beats = (np.arange(phase, (np.size(audio)/44100) - spb + phase, spb).astype('single'))
		
		# Store all the results
		self.bpm = bpm
		self.phase = phase
		self.beats = beats
		
	

if __name__ == '__main__':
	
	import sys
	import essentia
	from essentia.standard import MonoLoader, AudioOnsetsMarker, MonoWriter
	
	if len(sys.argv) != 2:
		print 'Usage: ', sys.argv[0], ' <filename>'
	filename = sys.argv[1]
	
	# Load the audio
	print 'Loading audio file "', filename, '" ...'
	loader = essentia.standard.MonoLoader(filename = filename)
	audio = loader()

	# TESTING HERE
	tracker = BeatTracker()
	tracker.run(audio)
	print 'Detected BPM: ', tracker.getBpm()
	print 'Detected phase: ', tracker.getPhase()
	beats = tracker.getBeats()	
	# END TESTING
	
	# Overlay the audio file with onsets
	onsetMarker = AudioOnsetsMarker(onsets = beats)
	audioMarked = onsetMarker(audio/2.)

	# Stretch the result
	#from librosa.effects import time_stretch
	#audioMarked = time_stretch(audioMarked, 175./172.)

	# Output the marked file
	writer = MonoWriter(filename = 'test.wav')
	beginIndex = 0.2*np.size(audioMarked)
	endIndex = 0.5*np.size(audioMarked)
	writer(audioMarked[beginIndex:endIndex]) #Only write fragment

	# Play the result
	from subprocess import call
	call(["mplayer", 'test.wav'])

	plt.show()
