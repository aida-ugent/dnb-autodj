'''
	Visualize the predicted and the ground truth vocal labels for a song.
	This file must be in the Application folder to work as it uses the Song class.
	Can only work after running the train script, which saves the singingvoice_?_train.bin.npy files.
	Usage: python TestSingingVoicePredictionsSong.py path_to_song ../music ../moremusic <etc>
'''

from song import Song
from songcollection import SongCollection
import sys, os, csv
from essentia import *
from essentia.standard import *
import pyaudio

import sklearn
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy

if __name__ == '__main__':
	
	sc = SongCollection()
	for dir_ in sys.argv[2:]:
		sc.load_directory(dir_)
		
	# Read annotated files
	traindir = '../SingingVoiceDetection/'
	testdir  = '../SingingVoiceDetection/test/'
	
	#~ validatedir = TODO? wss makkelijker te implementeren
	def get_songs_and_annots(csv_dir):
		songs = []
		annotations = []
		for filename in os.listdir(csv_dir):
			if filename.endswith('.csv'): 
				title, ext = os.path.splitext(filename)
				matching_songs = [s for s in sc.get_annotated() if s.title == title]
				if len(matching_songs) > 0:
					songs.extend(matching_songs)
					annot_cur = []
					with open(os.path.join(csv_dir, filename)) as csvfile:
						reader = csv.reader(csvfile)
						for line in reader:
							time = float(line[0])
							annot_cur.append(time)
					annotations.append(annot_cur)
		return songs, annotations
	#~ train_songs, train_annotations = get_songs_and_annots(traindir)
	#~ test_songs, test_annotations   = get_songs_and_annots(testdir)
	
	pyAudio = pyaudio.PyAudio()
	stream = pyAudio.open(format = pyaudio.paFloat32,
						channels=1,
						rate=44100,
							output=True)
							
	try:	
		title = sys.argv[1][len('singingvoice_predicted_'):-4]
		y_train_all = np.load('singingvoice_y_train.bin.npy')
		t_train_all = np.load('singingvoice_t_train.bin.npy')
		print title
		cur_song_mask = np.array([t == title for t in t_train_all])
		y_train = y_train_all[cur_song_mask]
		y_train = y_train[np.array([i%4!=0 for i in range(len(y_train))])]
		y_pred = np.load('singingvoice_predicted_{}.npy'.format(title))
		song = [s for s in sc.get_annotated() if s.title == title][0]
		song.open()
		plt.figure()
		plt.fill_between(range(len(y_train[::3])),np.logical_and(y_pred[0::3],y_pred[1::3],y_pred[2::3]),alpha=0.5,color='blue')
		plt.fill_between(range(len(y_train[::3])),y_train[::3],alpha=0.5,color='red')
		plt.show()
		for y_true, y, t in zip(y_train[::4], y_pred[::4], song.downbeats):
			print '{} {} {}'.format(y_true, y, t)
		song.close()
		
		
	finally:	
		stream.stop_stream()
		stream.close()
		pyAudio.terminate()
		print 'Closing streams'

