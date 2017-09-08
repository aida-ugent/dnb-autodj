'''
Quick test script to test the song similarity method using the DTW algorithm
'''

# Test if saving the onset curve worked
from song import Song
import matplotlib.pyplot as plt
import numpy as np
import sys
from essentia import *
from essentia.standard import FrameGenerator	

s1 = Song('../music/FILL IN')
s1.open()

s2 = Song('../evenmoremusic/FILL IN')
s2.open()
s2.openAudio()

start1 = (64 + 0)*4
print s1.beats[start1], s1.tempo
start2 = (80 + 0)*4
print s2.beats[start2], s2.tempo

from scipy.interpolate import interp1d

a = s2.getOnsetCurveFragment(start1,start1 + 4*4)
b = s2.getOnsetCurveFragment(start2,start2 + 4*4)
baudio = s2.audio[int(s2.beats[start2]*44100):int(s2.beats[start2+4*4]*44100):441]
baudio = baudio *10 
c = s1.getOnsetCurveFragment(start1,start1 + 4*4)
c = np.zeros(a.shape)
print a.size, b.size
L = min(a.size,b.size,c.size)

from tracklister import calculateOnsetSimilarity
print 'aa', calculateOnsetSimilarity(a,a)
print 'bb', calculateOnsetSimilarity(b,b)
print 'ab', calculateOnsetSimilarity(a,b)
print 'ac', calculateOnsetSimilarity(a,c)
print 'bc', calculateOnsetSimilarity(b,c)


plt.plot(np.linspace(0,1,L-2), a[:L-2],c='grey')
plt.plot(np.linspace(0,s2.tempo/s1.tempo,L-2), -b[:L-2],c='grey')
#~ plt.plot(np.linspace(0,1,len(baudio)), baudio, c='blue')
plt.plot(np.linspace(0,s2.tempo/s1.tempo,L-2), -c[:L-2],c='grey')
#~ plt.plot(np.sqrt(a[:L-2]*b[:L-2]), color='b')
#~ plt.plot(np.sqrt(a[:L-2]*b[1:L-1]), color='g')
#~ plt.plot(np.sqrt(a[1:L-1]*b[:L-2]), color='r')
plt.show()
