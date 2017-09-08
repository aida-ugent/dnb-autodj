from song import Song
from songcollection import SongCollection
import sys, os, csv
from essentia import *
from essentia.standard import *
import pyaudio

import sklearn
'''
	This script evaluates all possible overlappings of songs that have been manually annotated with singing voice annotations.
	It creates a confusion matrix of two classes: either the overlap clashes in vocals, or the overlap does not clash in vocals.
	Several methods of smoothing the classifier output and estimating whether overlaps are conflicting are evaluated.
	
	On the ground truth annotations, an overlap is considered clashing if the number of downbeats that simultaneously have vocals is more than 4
	
	On the predicted results, several things have been tried:
	- median filtering of the output, kernel = 3
	[[23263  4027]
	 [ 1321  3166]]
	[[ 0.85243679  0.14756321]
	 [ 0.29440606  0.70559394]]

	- median filtering of the output, kernel = 5
	[[25033  2257]
	 [ 1856  2631]]
	[[ 0.91729571  0.08270429]
	 [ 0.4136394   0.5863606 ]]
	 
	- iterating over all 4-downbeat length segments of the overlap with hop size 2:
	a 4-downbeat segment is considered vocal if there are more than 2 downbeats in it with (estimated) vocal activity
	a clash occurs if two vocal 4-downbeat segments of both songs overlap
	[[22780  4510]
	 [ 1267  3220]]
	[[ 0.834738    0.165262  ]
	 [ 0.28237129  0.71762871]]
	
	- "low-pass" filtering the predictions as: a[i] = (a[i-1] + a[i+1] + 2*a[i]) >= 2
	an element in a_new is true if the corresponding element in a is true, or if both neighbors are true
	a clash occurs occurs if two downbeat segments of these smoothed results overlap
	
	Mean length of clash relative to crossfade length if incorrectly detected: 0.218728343728
	(so if there is an undetected vocal clash, about 1/5th of the crossfade contains clashing vocals)
	
	>= 4 ground truth

	[[22028  5262]
	 [  962  3525]]
	[[ 0.80718212  0.19281788]
	 [ 0.21439715  0.78560285]]
	 
	- without post processing, >=2, with >= 4 ground truth
	[[8265 1902]
	 [ 374 1150]]
	[[ 0.81292417  0.18707583]
	 [ 0.24540682  0.75459318]]
	 
	- without post processing with >= 2 ground truth
	[[ 0.82978283  0.17021717]
	 [ 0.3043048   0.6956952 ]]
	 
	- neighbor filter (see above) with >= 2 ground truth
	[[ 0.83464323  0.16535677]
	 [ 0.30677882  0.69322118]]
	 
	- without post processing, >=4, with >= 4 ground truth
	[[9363  804]
	 [ 650  874]]
	[[ 0.92092063  0.07907937]
	 [ 0.42650919  0.57349081]]
'''

from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy
import scipy.signal

import tracklister

def is_vocal_clash_true(master, slave):
	return sum(np.logical_and(master,slave)) >= 4
	
def is_vocal_clash_pred(master,slave):
	#~ HOP = 2
	#~ for i in range(0,len(master)-HOP+1,HOP):
		#~ m = np.sum(master[i:i+4]) >= 2
		#~ s = np.sum( slave[i:i+4]) >= 2
		#~ if m and s:
			#~ return True
	#~ return False
	
	# With median filtering
	#~ master = 2*master[1:-1] + master[:-2] + master[2:] >= 2
	#~ slave = 2*slave[1:-1] + slave[:-2] + slave[2:] >= 2
	return sum(np.logical_and(master,slave)) >= 2

if __name__ == '__main__':
	
	sc = SongCollection()
	for dir_ in sys.argv[1:]:
		sc.load_directory(dir_)
	
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
	
	#TODO make sure that these arrays are constructed and saved during training CV loop
	y_val_all = np.load('singingvoice_y_val.bin.npy')
	t_val_all = np.load('singingvoice_t_val.bin.npy')
	y_true_all = np.load('singingvoice_y_train.bin.npy')
	t_train_all = np.load('singingvoice_t_train.bin.npy')
	
	masks = {}
	
	for s in sc.get_annotated():
		if s.title in t_val_all:
							
			cur_song_mask = np.array([t == s.title for t in t_val_all])
			y_val = y_val_all[cur_song_mask]
			
			cur_song_mask_train = np.array([t == s.title for t in t_train_all])
			y_true = y_true_all[cur_song_mask_train]
			
			masks[s.title] = (y_val, y_true)
	
	for title_1, y_tuple_1 in masks.iteritems():
		song_1 = [s for s in sc.get_annotated() if s.title == title_1][0]
		song_1.open()
		
	# Combine all songs and report the accuracy on detecting overlaps in voice
	confusion_matrix = np.array([[0,0],[0,0]])
	average_clash_length = 0.0
	num_clashes = 0
	for title_1, y_tuple_1 in masks.iteritems():
		print title_1
		song_1 = [s for s in sc.get_annotated() if s.title == title_1][0]
		
		for title_2, y_tuple_2 in masks.iteritems():
			if title_1 == title_2:
				continue
			y_val_1, y_true_1 = y_tuple_1
			y_val_2, y_true_2 = y_tuple_2
			
			song_2 = [s for s in sc.get_annotated() if s.title == title_2][0]
			
			for t_type in [tracklister.TYPE_ROLLING]:
				
				# TODO Now transitions are not exactly the same as in tracklister
				master_cues = tracklister.getAllMasterCues(song_1, t_type)
				slave_cues = tracklister.getAllSlaveCues(song_2, t_type)
				
				for cue_m, L_in_m, L_out_m in master_cues:
					for cue_s, L_in_s in slave_cues:
						L_in = min(L_in_m, L_in_s)
						L_out = L_out_m
						L = L_in + L_out
						
						# Times 3 because in y_true, the labels are present three times (3 data augmented versions)
						master_singing = y_true_1[3*(cue_m - L_in): 3*(cue_m + L_out) : 3]
						slave_singing = y_true_2[3*(cue_s - L_in): 3*(cue_s + L_out) : 3]
						is_singing_clash = 1 if is_vocal_clash_true(master_singing, slave_singing) else 0
						
						master_singing_pred = y_val_1[3*(cue_m - L_in): 3*(cue_m + L_out) : 3] > 0
						slave_singing_pred = y_val_2[3*(cue_s - L_in): 3*(cue_s + L_out) : 3] > 0
						is_singing_clash_pred = 1 if is_vocal_clash_pred(master_singing_pred, slave_singing_pred) else 0
						
						confusion_matrix[is_singing_clash][is_singing_clash_pred] += 1
						
						if is_singing_clash == 1 and is_singing_clash_pred == 0:
							# False negative, which is the worst thing that can happen: an undetected vocal clash.
							# How long are these on average?
							average_clash_length += np.sum(np.logical_and(master_singing, slave_singing),dtype='single') / len(master_singing)
							num_clashes += 1
						
						# TODO consider overlapping if number of overlapping downbeats is >= 2?
						
						#~ print t_type, title_1[:15], cue_m, title_2[:15], cue_s, is_singing_clash, is_singing_clash_pred
						#~ print master_singing.astype('int')
						#~ print master_singing_pred.astype('int')
						#~ print slave_singing.astype('int')
						#~ print slave_singing_pred.astype('int')
				
						# TODO estimate for the overlapping portions if there is a vocal clash
						
						# TODO save result and report
	
		print average_clash_length / num_clashes
		print confusion_matrix
		print np.array(confusion_matrix) / np.sum(confusion_matrix,axis=1,dtype='float').reshape((-1,1))
	
	for title_1, y_tuple_1 in masks.iteritems():
		song_1 = [s for s in sc.get_annotated() if s.title == title_1][0]
		song_1.close()

