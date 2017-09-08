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
from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt # For plotting

# Load the audio
print 'Loading audio file "', filename, '" ...'
loader = essentia.standard.MonoLoader(filename = filename)
audio = loader()

# Calculate beat positions
print 'Calculating beat positions...'
beat_tracker = PercivalBpmEstimator(minBPM=160, maxBPM=190)
bpm = beat_tracker(audio)
print '> BPM = ', bpm

#~ # Overlay the audio file with onsets
#~ onsetMarker = AudioOnsetsMarker(onsets = beats)
#~ audioMarked = onsetMarker(audio/2.)

#~ # Stretch the result
#~ #from librosa.effects import time_stretch
#~ #audioMarked = time_stretch(audioMarked, 175./172.)

#~ # Output the marked file
#~ writer = MonoWriter(filename = 'test.wav')
#~ writer(audioMarked[:]) #Only write fragment

#~ # Play the result
#~ from subprocess import call
#~ call(["mplayer", 'test.wav'])
