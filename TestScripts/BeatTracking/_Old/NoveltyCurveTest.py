import sys

# Load a file using the command line
try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<audiofile>"
    sys.exit()

# Load the libraries
print 'Loading Essentia...'
import numpy as np
from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt # For plotting

# Load the audio
print 'Loading audio file "', filename, '" ...'
loader = MonoLoader(filename = filename)
audio = loader()

# Calculate frequency information on the audio for each frame

# Initialise all the algorithms

FRAME_SIZE = 2048
HOP_SIZE = 1024

win = Windowing(type = 'hann')
spec = Spectrum(size = FRAME_SIZE)
fqb = FrequencyBands(sampleRate = 44100.0)

pool = Pool()

for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
	freq_bands = fqb(spec(win(frame)))
	pool.add('lowlevel.freq_bands', freq_bands)

frequencyBands = array(pool['lowlevel.freq_bands'])
print 'frequencybands dim: ', frequencyBands.shape
plt.imshow(frequencyBands, aspect='auto')
plt.show()

# Calculate novelty curve
novcurve_calculator = NoveltyCurve(normalize=True, weightCurveType = 'inverse_quadratic', frameRate = 44100/float(HOP_SIZE))
novelty = novcurve_calculator(frequencyBands)

plt.plot(novelty)
plt.show()

# Calculate beat positions
print 'Calculating beat positions...'
beat_tracker = NoveltyCurveFixedBpmEstimator(hopSize=HOP_SIZE, tolerance=0.00001) #Tolerance: 172/172,5 = .9971
#beat_tracker = NoveltyCurveFixedBpmEstimator(hopSize=HOP_SIZE, tolerance=0.00001) #Tolerance: 172/172,5 = .9971
bpms, amplitudes = beat_tracker(novelty)
print '> BPM = ', bpms, ', with confidence: ', amplitudes
plt.plot(bpm_dist)
plt.show()
plt.plot(bpmIntervals)
plt.show()

# Calculate how stable the BPM is
delta_beats = beats[1:] - beats[:len(beats)-1]
BPM = 60. / np.mean(delta_beats)
lenAudioInMin = (len(audio)/(44100.0 * 60))
BPM2 = len(beats) / lenAudioInMin
BPM_std = np.std(delta_beats)
print '> BPM = ', BPM, '; using length of file: ', BPM2, '; std = ', BPM_std

delta_beats_run_mean = running_mean(delta_beats, 64)
print '> Mean of running mean (low pass filter) of instant BPM: ', np.average(60./delta_beats_run_mean)

# Plot how the beat distances are distributed
hist, bins = np.histogram(delta_beats, bins = 20)
valid_bins = np.where(hist >= len(delta_beats)/10)
print valid_bins
print bins
delta_beats_bin_index = np.digitize(delta_beats, bins) - 1
delta_beats_filtered = delta_beats[np.in1d(delta_beats_bin_index, valid_bins)]
delta_beats_filtered_plot = np.where(np.in1d(delta_beats_bin_index, valid_bins), delta_beats, 60./170)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.hist(delta_beats_filtered, bins=bins, color='red', width = 0.9*(bins[1]-bins[0]))
plt.show()

print '> BPM based on filtered histogram: ', np.mean(60./delta_beats_filtered)

binwidth = .5
hist_data = 60./delta_beats
plt.hist(hist_data, bins=np.arange(np.min(hist_data), np.max(hist_data) + binwidth, binwidth), cumulative=True)
plt.axvline(x=BPM, color='red')
plt.axvline(x=172, color='green')
plt.show()

# Plot the time difference between beats, plus a moving average
plt.plot(60./delta_beats)
plt.plot(60./delta_beats_run_mean, color='red')
plt.plot(60./delta_beats_filtered_plot, color='green')
plt.show()

# Round the BPM to a value above, and a value below
# BPM_floor = np.floor(BPM)
# valid_BPMs = [BPM_floor - 1, BPM_floor, BPM_floor+1, BPM_floor+2]
# form bpm in valid_BPMs:
# 	sPerBeat = (60./BPM)
# 	beatsInSong = (len(audio)/44100.) / sPerBeat
# 	perfectBeats = sPerBeat * range(int(beatsInSong) - 1) # One beat shorter than the original file for phase aligning
# 	np.correlate(beats, perfectBeats, mode = 'valid')

# Overlay the audio file with onsets
onsetMarker = AudioOnsetsMarker(onsets = beats)
audioMarked = onsetMarker(audio/2.)

# Stretch the result
#from librosa.effects import time_stretch
#audioMarked = time_stretch(audioMarked, 175./172.)

# Output the marked file
writer = MonoWriter(filename = 'test.wav')
writer(audioMarked[:]) #Only write fragment

# Play the result
from subprocess import call
call(["mplayer", 'test.wav'])

# Display the waveform
#print 'Displaying waveform'
#plt.plot(audio[::4410]) # 44.1 kHz, plotted every 4410th sample => 10 samples per second
#for b in beats:
#	plt.axvline(x=b*10)
#plt.show() # unnecessary if you started "ipython --pylab"
