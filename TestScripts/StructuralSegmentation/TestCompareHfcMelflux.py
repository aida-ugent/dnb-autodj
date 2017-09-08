'''
	Quick script to visualize the difference between the HFC and Melflux ODFs
'''

from song import Song
import sys
import numpy as np
from essentia import *
from essentia.standard import *

song = Song(sys.argv[1])
song.open()
song.openAudio()

spec = Spectrum(size = 1024)
w = Windowing(type = 'hann')
fft = np.fft.fft
c2p = CartesianToPolar()
pool = Pool()
odf_hfc = OnsetDetection(method = 'hfc')
odf_mel = OnsetDetection(method = 'melflux')

for frame in FrameGenerator(song.audio, frameSize = 1024, hopSize = 512):
	pool.add('audio.windowed_frames', w(frame))	
fft_result = fft(pool['audio.windowed_frames']).astype('complex64')
print fft_result.shape
fft_result_mag = np.absolute(fft_result)
fft_result_ang = np.angle(fft_result)

HOP_SIZE = 512
for mag,phase in zip(fft_result_mag, fft_result_ang):
	pool.add('onsets.hfc', odf_hfc(mag, phase))
	pool.add('onsets.melflux', odf_mel(mag, phase))

a = pool['onsets.hfc'] / max(pool['onsets.hfc'])
b = pool['onsets.melflux'] / max(pool['onsets.melflux'])
plt.plot(np.linspace(0,1,len(a)), a, c='g')
plt.plot(np.linspace(0,1,len(b)), -b, c='r')
plt.show()
