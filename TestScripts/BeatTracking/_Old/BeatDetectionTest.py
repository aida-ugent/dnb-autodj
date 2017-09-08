import numpy as np
from scipy.signal import medfilt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

import sys

# Load a file using the command line
try:
    filename = sys.argv[1]
except:
    print "usage:", sys.argv[0], "<audiofile>"
    sys.exit()

# Load the libraries
print 'Loading Essentia...'
import essentia
from essentia.standard import RhythmExtractor2013, AudioOnsetsMarker, MonoWriter
import matplotlib.pyplot as plt # For plotting
import csv

# Load the audio
print 'Loading audio file "', filename, '" ...'
loader = essentia.standard.MonoLoader(filename = filename)
audio = loader()

# Calculate beat positions
print 'Calculating beat positions...'
beat_tracker = RhythmExtractor2013(minTempo=160, maxTempo=180, method='multifeature')
bpm, beats, conf, bpm_dist, bpmIntervals = beat_tracker(audio)
print '> BPM = ', bpm, ', with confidence: ', conf

# Calculate how stable the BPM is
delta_beats = beats[1:] - beats[:len(beats)-1]
BPM = 60. / np.mean(delta_beats)
lenAudioInMin = (len(audio)/(44100.0 * 60))
BPM2 = len(beats) / lenAudioInMin
BPM_std = np.std(delta_beats)
print '> BPM = ', BPM, '; using length of file: ', BPM2, '; std = ', BPM_std

# adaptation from 25/12: print beats to compare them with correct ones
with open('bla.csv', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for beat in beats:
		csvwriter.writerow([beat])

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

# Using IQR
q75, q25 = np.percentile(delta_beats_run_mean, [75,25])
iqr = q75 - q25
print q75,q75
q_epsilon = .0001
delta_beats_iqr = delta_beats[(delta_beats_run_mean >= q25 - q_epsilon) | (delta_beats_run_mean <= q75 + q_epsilon)]
print '> BPM based on IQR of RUNNING MEAN: ', np.mean(60./delta_beats_iqr)

binwidth = .5
hist_data = 60./delta_beats
plt.hist(hist_data, bins=np.arange(np.min(hist_data), np.max(hist_data) + binwidth, binwidth), cumulative=True)
plt.axvline(x=BPM, color='red')
plt.axvline(x=172, color='green')
plt.show()

# Plot the time difference between beats, plus a moving average
plt.plot(60./delta_beats)
plt.plot(60./delta_beats_run_mean, color='red')
plt.axhline(60./(q25-q_epsilon), color='green')
plt.axhline(60./(q75+q_epsilon), color='green')
plt.axhline(60./(q25), color='green')
plt.axhline(60./(q75), color='green')
#plt.plot(60./delta_beats_filtered_plot, color='green')
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
