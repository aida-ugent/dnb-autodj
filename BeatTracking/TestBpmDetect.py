import numpy as np
import sys

# Also see the default python example!
# Load a file using the command line
try:
	filename = sys.argv[1]
	if len(sys.argv) > 2:
		MIN_VALID_BPM = int(sys.argv[2])
		MAX_VALID_BPM = int(sys.argv[3])
	else:
		MIN_VALID_BPM = 100.0
		MAX_VALID_BPM = 190.0
		
except:
	print "usage:", sys.argv[0], "<audiofile>"
	sys.exit()

# Load the libraries
print 'Loading Essentia...'
from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt

# Load the audio
print 'Loading audio file "', filename, '" ...'
loader = essentia.standard.MonoLoader(filename = filename)
audio = loader()

# ------------ Calculate the onset detection function
print 'Initialising algorithms...'
FRAME_SIZE = 1024
HOP_SIZE = 512
spec = Spectrum(size = FRAME_SIZE)
w = Windowing(type = 'hann')
# For calculating the phase and magnitude
fft = np.fft.fft#FFT()
c2p = CartesianToPolar()

od_csd = OnsetDetection(method = 'melflux')
od_flux = OnsetDetection(method = 'complex')

pool = Pool()

print 'Calculating frame-wise onset detection curve...'
for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
	pool.add('windowed_frames', w(frame))
	
# TODO Test if this is faster?
print 'windowed frames: ', (pool['windowed_frames']).shape
fft_result = fft(pool['windowed_frames']).astype('complex64')
print 'fftresult: ', fft_result.shape
fft_result_mag = np.absolute(fft_result)
fft_result_ang = np.angle(fft_result)

# Process every frame vector in the result
for mag,phase in zip(fft_result_mag, fft_result_ang):
	pool.add('onsets.complex', od_csd(mag, phase))
	#pool.add('onsets.flux', od_flux(mag, phase))
	
# Done! now show the result


# ------------ Calculate the tempo function thingy (using method from paper)
# Step 1: normalise the data using an adaptive mean threshold
print 'Normalising result and half-wave rectifying it...'
def adaptive_mean(x, N):
	#TODO efficient implementation instead of convolve
	return np.convolve(x, [1.0]*int(N), mode='same')/N
	
novelty_mean = adaptive_mean(pool['onsets.complex'], 16.0)
# Step 2: half-wave rectify the result
novelty_hwr = (pool['onsets.complex'] - novelty_mean).clip(min=0)  

# Step 3: then calculate the autocorrelation of this signal
print 'Autocorrelating resulting curve...'
def autocorr(x):
	result = np.correlate(x, x, mode='full')
	return result[result.size/2:]

novelty_autocorr = autocorr(novelty_hwr)

# Step 4: Apply a "shift-invariant comb filterbank"
# own implementation: sum over constant intervals
print 'Iterating over valid BPM values...'
#valid_bpms = np.arange(170.0, 176.0, 0.01)
valid_bpms = np.arange(MIN_VALID_BPM, MAX_VALID_BPM, 0.01)
for bpm in valid_bpms:
	num_frames_per_beat = (60.0 * 44100.0)/(512.0 * bpm) # TODO put this in a function
	frames = (np.round(np.arange(0,np.size(novelty_autocorr),num_frames_per_beat)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
	pool.add('output.bpm', np.sum(novelty_autocorr[frames])/np.size(frames))

bpm = valid_bpms[np.argmax(pool['output.bpm'])]
print 'Detected BPM: ', bpm

# Step 5: Calculate phase information
# Valid phases in SECONDS
valid_phases = np.arange(0.0, 60.0/bpm, 0.001)
num_frames_per_beat_final = (60.0 * 44100.0)/(512.0 * bpm) #TODO put this in a function

for phase in valid_phases:
	# Convert phase from seconds to frames
	phase_frames = (phase * 44100.0) / (512.0)
	frames = (np.round(np.arange(phase_frames,np.size(novelty_hwr),num_frames_per_beat_final)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
	pool.add('output.phase', np.sum(novelty_hwr[frames])/np.size(frames))  

phase = valid_phases[np.argmax(pool['output.phase'])]
print 'Detected phase: ', phase

spb = 60./bpm #seconds per beat
beats = (np.arange(phase, (np.size(audio)/44100) - spb + phase, spb).astype('single'))

plt.subplot(511)
plt.plot(audio[0*len(audio):0.01*len(audio)])
plt.xlim((0,len(audio)*0.01))
plt.title('Audio waveform')
plt.subplot(512)
plt.plot(novelty_hwr[0*len(novelty_hwr):0.01*len(novelty_hwr)]) 
plt.title('Half-wave rectified novelty detection curve')
plt.xlim((0,len(novelty_hwr)*0.01))
plt.subplot(513)
plt.plot(novelty_autocorr[0*len(novelty_autocorr):0.01*len(novelty_autocorr)])
plt.xlim((0,0.01*len(novelty_autocorr)))
plt.title('Correlation of half-wave rectified novelty detection curve')   
plt.subplot(514)
plt.title('BPM detection curve')
plt.plot(valid_bpms, pool['output.bpm'], linewidth=2.0)   
plt.subplot(515)
plt.title('Phase detection curve')
plt.plot(valid_phases, pool['output.phase'], linewidth=2.0) 
plt.show()

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

