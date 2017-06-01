'''
	A simple demo, used somewhere in the beginning of the project (first semester).
	Annotates some audio with beats and downbeats and plays it back
'''

import essentia
from essentia import *
from essentia.standard import MonoLoader, Spectrum, Windowing, MFCC, FrameGenerator, Spectrum, AudioOnsetsMarker, MonoWriter
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from skimage import filters
import numpy as np

import sys
import matplotlib.pyplot as plt
from downbeatTracker import *
from BeatTracker import *

if len(sys.argv) != 2:
	print 'Usage : ', sys.argv[0], ' <filename>'
	exit()

filename = sys.argv[1]	
essentia.log.infoActive = False
# Load the audio
print 'Loading audio...'
loader = MonoLoader(filename = filename)
audio = loader()

# Beat tracking

print 'Extracting beat information...'
tracker = BeatTracker()
tracker.run(audio)
beats = tracker.getBeats()
bpm = tracker.getBpm()
print 'Bpm: ', bpm


print 'Extracting downbeat information...'
downbeatTracker = DownbeatTracker()
downbeats = downbeatTracker.getDownbeats(audio, beats)

print 'Annotating audio...'
onsetMarker = AudioOnsetsMarker(onsets = beats, type='noise')
audioMarked = onsetMarker(audio)
onsetMarker2 = AudioOnsetsMarker(onsets = downbeats*1.0, type='beep')
audioMarked2 = onsetMarker2(audioMarked)

# Stretch the result
#from librosa.effects import time_stretch
#audioMarked = time_stretch(audioMarked, 175./172.)

# Output the marked file
writer = MonoWriter(filename = 'test.wav')
beginIndex = 0.2*np.size(audioMarked2)
endIndex = 0.5*np.size(audioMarked2)
writer(audioMarked2[int(beginIndex):int(endIndex)]) #Only write fragment

# Play the result
from subprocess import call
call(["mplayer", 'test.wav'])
