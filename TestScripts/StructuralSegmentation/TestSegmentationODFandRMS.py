'''
Visualize ODF and RMS adaptive mean, to confirm that it correctly detects high segments
	usage: TestSegmentationODFandRMS.py path_to_song
'''

from song import Song
import matplotlib.pyplot as plt
import numpy as np
import sys
from essentia import *
from essentia.standard import FrameGenerator

s1 = Song(sys.argv[1])

s1.open()
s1.openAudio()
audio = s1.audio
FRAME_SIZE = int(44100 * (60.0 / s1.tempo) / 2)
HOP_SIZE = FRAME_SIZE / 2

def adaptive_mean(x, N):
	return np.convolve(x, [1.0]*int(N), mode='same')/N

pool = Pool()
for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
			pool.add('lowlevel.rms', np.average(frame**2))
			
adaptive_mean_rms = adaptive_mean(pool['lowlevel.rms'], 64) # Mean of rms in window of [-4 dbeats, + 4 dbeats]
mean_rms = np.mean(adaptive_mean_rms)
adaptive_mean_odf = adaptive_mean(s1.onset_curve, int((44100*60/s1.tempo)/512) * 4) # -4 dbeats, +4 dbeats
adaptive_mean_odf_2 = adaptive_mean(adaptive_mean_odf, 8)
mean_odf = np.mean(adaptive_mean_odf)

plt.plot(np.linspace(0.0,1.0,adaptive_mean_rms.size), adaptive_mean_rms / max(adaptive_mean_rms),c='r')
plt.plot(np.linspace(0.0,1.0,adaptive_mean_odf.size), adaptive_mean_odf / max(adaptive_mean_odf),c='b')
plt.plot(np.linspace(0.0,1.0,adaptive_mean_odf_2.size), adaptive_mean_odf_2 / max(adaptive_mean_odf_2),c='g')
plt.plot(np.linspace(0.0,1.0,s1.onset_curve.size), s1.onset_curve / (2*max(s1.onset_curve)),c='grey')
plt.axhline(mean_rms / max(adaptive_mean_rms),c='r')
plt.axhline(mean_odf / max(adaptive_mean_odf),c='b')
plt.show()

#~ start1 = (128 + 12)*4
#~ print s1.beats[start1], s1.tempo
#~ start2 = (128 + 0)*4
#~ print s2.beats[start2], s2.tempo

#~ a = s2.getOnsetCurveFragment(start1,start1 + 4*4)
#~ b = s2.getOnsetCurveFragment(start2,start2 + 4*4)
#~ L = min(a.size,b.size)

#~ plt.plot(a,c='grey')
#~ plt.plot(-b,c='grey')
#~ plt.plot(np.sqrt(a[:L]*b[:L]), color='b')
#~ plt.plot(np.sqrt(a[:L-1]*b[1:L]), color='r')
#~ plt.plot(np.sqrt(a[1:L]*b[:L-1]), color='g')
#~ plt.show()
