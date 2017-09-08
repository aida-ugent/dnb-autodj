# Calculate the Theme Descriptor features for each "odd" (uneven) 30 second segment in a DJ mix
# This is done to validate that the descriptor indeed describes if music fits together: in a DJ mix, songs surely fit together, so consecutive
# segments must lie close together in the feature space
from essentia import *
from essentia.standard import *
import numpy as np
from scipy.stats import skew
import sys

import matplotlib.pyplot as plt

from sklearn.externals import joblib
theme_pca = joblib.load('song_theme_pca_model_2.pkl') 
theme_scaler = joblib.load('song_theme_scaler_2.pkl') 

def calculateThemeDescriptor(audio):
	
	FRAME_SIZE = 2048
	HOP_SIZE = FRAME_SIZE/2

	spec = Spectrum(size = FRAME_SIZE)
	w = Windowing(type = 'hann')
	fft = np.fft.fft
	pool = Pool()
	
	specContrast = SpectralContrast(frameSize=FRAME_SIZE, sampleRate=44100,numberBands=12)
	#~ mfcc = MFCC()
	#~ specCentroid = Centroid()
	#~ keyExtractor = KeyExtractor()
	
	for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
		frame_spectrum = spec(w(frame))
		# Spectral contrast
		specCtrst, specValleys = specContrast(frame_spectrum)
		pool.add('audio.spectralContrast', specCtrst)
		pool.add('audio.spectralValleys', specValleys)
		#~ # MFCC
		#~ _, mfcc_coeff = mfcc(frame_spectrum)
		#~ pool.add('audio.mfcc', mfcc_coeff)
		#~ # Spectral centroid
		#~ centroid = specCentroid(frame_spectrum)
		#~ pool.add('audio.specCentroid', centroid)	
	def calculateDeltas(array):
			D = array[1:] - array[:-1]
			return D
	specCtrstAvgs = np.average(pool['audio.spectralContrast'], axis=0)
	specValleyAvgs = np.average(pool['audio.spectralValleys'], axis=0)
	specCtrstDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralContrast'])),axis=0)
	specValleyDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralValleys'])),axis=0)
	#~ mfccAvgs = np.average(pool['audio.mfcc'], axis=0)
	#~ mfccStds = np.std(pool['audio.mfcc'], axis=0)
	#~ specCentroidAvg = np.average(pool['audio.specCentroid'])
	#~ specCentroidStd = np.std(pool['audio.specCentroid'])
	#~ specCentroidSkew = skew(pool['audio.specCentroid'])
	#~ # Key
	#~ key, scale, strength = keyExtractor(audio)	
	
	features = np.concatenate((specCtrstAvgs, specValleyAvgs,specCtrstDeltas,specValleyDeltas))
	song_theme_descriptor = theme_pca.transform(theme_scaler.transform(
			features.reshape((1,-1)).astype('single')
			)).astype('single').reshape((1,-1))
	
	return song_theme_descriptor
	
# ---------------------Analyse this mix ---------------------------
print 'Loading audio'
title = sys.argv[1]
loader = MonoLoader(filename = title)
audio = loader().astype('single')

print 'Audio loaded!'

start_sample = 44100 * 30
stop_sample = audio.size
LEN_S = 90
length_samples = 44100 * LEN_S
HOP_S = 120
hop_size = 44100 * HOP_S

pool = Pool()

for idx in range(start_sample, stop_sample - length_samples, hop_size):
	print 'Analysed until {:.2f}'.format((idx/44100.0)/60)
	start_idx = idx
	end_idx = idx + length_samples
	fragment = audio[start_idx : end_idx]
	theme_descr = calculateThemeDescriptor(fragment)
	print theme_descr
	pool.add('theme_descriptors', theme_descr.tolist()[0])
	
# --------------------- Load all audio files ---------------------------
from songcollection import SongCollection
sc = SongCollection()
sc.load_directory('../music')
sc.load_directory('../moremusic')
sc.load_directory('../evenmoremusic')
sc.load_directory('../music/test')

songs = []

for song in sc.get_annotated():
	song.open()
	pool.add('song.themes', song.song_theme_descriptor.tolist()[0])
	songs.append(song.title)
	song.close()

# --------------------- Make nice plots ---------------------------
Y = pool['song.themes']	# All songs in "/music" and "/moremusic" libraries
X = pool['theme_descriptors']						# Evolution of mix theme descriptors

from scipy.spatial.distance import euclidean as euclidean_distance

for P1,P2 in zip(X[:-1,:],X[1:,:]):
	distance_path = euclidean_distance(P1,P2)
	num_songs_closer = 0
	for P in Y:
		dist = euclidean_distance(P1,P)
		if dist < distance_path:
			num_songs_closer += 1
	print distance_path, num_songs_closer

from mpl_toolkits.mplot3d import Axes3D
#~ fig = plt.figure()
#~ ax = fig.add_subplot(111, projection='3d')
#~ ax.scatter(X[:,0],X[:,1],X[:,2], marker='x')
#~ ax.scatter(Y[:,0],Y[:,1],Y[:,2], marker='o')
#~ plt.show()

# 01

for a,b in [(0,1),(1,2),(0,2)]:
	plt.figure()
	plt.scatter(X[:,a], X[:,b],marker='o',c='black',s=10)
	for x1,y1,x2,y2 in zip(X[:-1,a],X[:-1,b],X[1:,a],X[1:,b]):
		plt.arrow(x1, y1, x2-x1, y2-y1, fc="k", ec="k", head_width=0.05, head_length=0.1, lw=1)
	plt.scatter(Y[:,a],Y[:,b], c='grey', lw=0, s=12)
	#~ plt.savefig('{}_{}{}[{}_{}].png'.format(title,a,b,LEN_S,HOP_S))

plt.show()
