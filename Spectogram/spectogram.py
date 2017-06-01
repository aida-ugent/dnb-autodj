import sys

import numpy as np

import matplotlib.pyplot as plt

import librosa
import librosa.display
from librosa.core import cqt

START = 44100*50
audio, sr = librosa.load(sys.argv[1], sr=44100)
audio = audio[START:START + int(sys.argv[2]) * 60480]

FRAME_SIZE = 1024
HOP_SIZE = FRAME_SIZE/2

#~ spectogram = np.array([np.log(spectrum(w(frame))) for frame in frames])
spectogram = cqt(audio, sr=44100, bins_per_octave=12, n_bins= 84)
spec_db = librosa.amplitude_to_db(spectogram, ref=np.max)
print (np.shape(spec_db), spec_db.dtype)

librosa.display.specshow(spec_db, sr=44100)
#~ plt.imshow(spec_db, aspect='auto')
plt.show()
