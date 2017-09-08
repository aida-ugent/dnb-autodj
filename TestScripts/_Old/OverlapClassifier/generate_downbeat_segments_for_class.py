'''
	Opens the songs in the provided directory and extracts training samples from them.
	Song is cut up into downbeat length segments, and for each song it is overlapped in a specific way
	with an extract of another song to emulate a bad mix.
	Afterwards, the spectogram is calculated and exported to .bin file
	as flattened numpy array data 
	
	Put this script in the Application/ directory or it won't work!
	Example usage:
	
	Generate training data:
	python generate_downbeats_for_class.py ../music/
	Generate test data:
	python generate_downbeats_for_class.py ../music/test/
'''

import sys, os, glob, csv

from essentia import *
from essentia.standard import *
import librosa
import librosa.display
import matplotlib.pyplot as plt
from librosa.core import cqt

from song import Song
from timestretching import time_stretch_sola
import songtransitions

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.stats.mstats import gmean
from random import random, randint
from scipy.spatial import distance

# CONSTANTS
TEMPO = 175
LENGTH_SEGMENT = 60480 # One downbeat at 175 bpm and 44100 kHz sample rate
SQRT1over2 = np.sqrt(0.5)

# PARAMETERS
writeAudio = True	
isPlot = False
num_h = 15
num_l = 5
num_examples_to_generate = num_h + num_l

def calcAndWriteFeatureFile(audio, filename):
	'''
		Generate the CQT matrix and write it as a binary array to a file, so that it can be later read by tensorflow with a fixedlengthrecordreader
	'''
	spectogram = cqt(audio, sr=44100, bins_per_octave=12, n_bins= 84)
	spec_db = librosa.amplitude_to_db(spectogram, ref=np.max)
	
	#~ librosa.display.specshow(spec_db, sr=44100)
	#~ plt.show()
	
	spec_db.tofile(filename)

def calculateFeaturesForDownbeat(audio):
	datarow = []		
	BINS_PER_OCTAVE = 12
	NUM_OCTAVES = 5
	spectogram = cqt(audio, sr=44100, bins_per_octave=BINS_PER_OCTAVE, n_bins=BINS_PER_OCTAVE*NUM_OCTAVES, hop_length=512)
	spec_db = np.absolute(spectogram)
	spec_db = librosa.amplitude_to_db(spectogram, ref=np.max)
	max_spec = np.max(spec_db) 
	min_spec = np.min(spec_db)
	if max_spec != min_spec:
		spec_db = (spec_db - min_spec) / (max_spec - min_spec)
	else:
		print 'Max equals min!'
		spec_db = (spec_db - np.min(spec_db))
	# Statistics for each frequency bin (each row => aggregate over axis 1 = columns)
	NUM_FRAMES = spectogram.shape[1]
	features_per_half_octave = []
	window_features = np.zeros((4*4*2-1,NUM_OCTAVES*2))
	for i_freq in range(NUM_OCTAVES*2):			# Frequency resolution: every half octave; hop size quarter octave
		features_cur_freq_window = []
		# Window along frequency axis
		f_bin_start = (i_freq * BINS_PER_OCTAVE) / 2
		f_bin_end = ((i_freq + 1) * BINS_PER_OCTAVE) / 2
		for i_time in range(4 * 4 * 2 - 1): 	# Aggregate along time axis in frames of one quarter beat, hop size an eight dbeat
			# Window along time axis
			frame_start = int(i_time * NUM_FRAMES / 32.0)
			frame_end = int((i_time + 2) * NUM_FRAMES / 32.0)
			# Current section of spectogram
			cur_w = spec_db[f_bin_start:f_bin_end,frame_start:frame_end]
			window_features[i_time,i_freq] = np.mean(cur_w)	# Determine how 'loud' this window is approx
	
	freq_axis_features = []
	freq_axis_features.extend(np.mean(window_features,axis=1))
	freq_axis_features.extend(np.var(window_features,axis=1))
	freq_axis_features.extend(skew(window_features,axis=1))
	freq_axis_features.extend(kurtosis(window_features,axis=1))
	
	time_means = np.mean(window_features,axis=0)
	time_vars = np.var(window_features, axis=0)
	freq_axis_features.extend([np.mean(time_means), np.var(time_vars), np.mean(time_vars), np.var(time_vars)])
	
	datarow = freq_axis_features
	# Statistics for each time point (each column => aggregate over axis 0 = rows)
	#~ datarow.extend(np.std(spec_db, axis=0))
	#~ datarow.extend(skew(spec_db, axis=0))
	#~ datarow.extend(kurtosis(spec_db, axis=0))
	# Show
	if isPlot:
		plt.figure()
		plt.subplot(2,1,1)
		librosa.display.specshow(spec_db, x_axis='time', y_axis='cqt_note')
		plt.subplot(2,1,2)
		#~ cosine_distances_t = [distance.cosine(spec_db[i+1,:], spec_db[i,:]) for i in range(spec_db.shape[0]-1)]
		plt.plot(datarow[::4])
		plt.plot(datarow[1::4])
		plt.plot(datarow[2::4])
		plt.plot(datarow[3::4])
	return datarow
	
def getAudioSegmentAtDownbeat(song, start_db):
	dbeats = song.downbeats
	start_db_s = dbeats[start_db]
	end_db_s = dbeats[start_db + 2] # +2 to have some slack for time stretching
	start_idx = int(start_db_s * 44100)
	end_idx = int(end_db_s * 44100)
	audio = song.audio[start_idx:end_idx]
	audio = time_stretch_sola(audio, song.tempo / TEMPO, song.tempo, 0.0)
	audio = audio[:LENGTH_SEGMENT]
	return audio
		
if __name__ == '__main__':
	
	directory = sys.argv[1]

	num_segments_generated = 0
	generated_seg_types = {
		'L' : 0,
		'H' : 0
	}
	songs = {}

	print 'Opening all songs...'
	i = 0

	for f in [f for f in os.listdir(directory) if os.path.splitext(f)[1] == '.mp3' or os.path.splitext(f)[1] == '.wav']:
		if isPlot:
			if i > 3:
				break
			i += 1
		# Open the current song, but make sure you don't need to open it twice
		if f not in songs.keys():
			song = Song(os.path.join(directory, f))
			song.open()
			song.openAudio()
			songs[f] = song
		else:
			song = songs[f]
			
	print 'Opened all songs...'
		
	with open('output/features.csv', 'w+') as csvfile:
		csvwriter = csv.writer(csvfile)
			
		for _, song in songs.iteritems():
			
			# Store the generated audio afterwards to evaluate
			audio_out_good = np.array((1,0)).astype('single')
			audio_out_bad = np.array((1,0)).astype('single')
			
			# Extract necessary info from the current song
			dbeats = song.downbeats
			segtypes = song.segment_types
			segidxs = song.segment_indices
			segs_H = [segidxs[j] for j in range(len(segidxs)-2) if segtypes[j] == 'H']	# Exclude the last two segments (end of the song)
			segs_L = [segidxs[j] for j in range(len(segidxs)-2) if segtypes[j] == 'L']
			
			print song.title, num_examples_to_generate
		
			for i in range(num_examples_to_generate):
				
				# Open a song for bad mixing
				mixed_song = song
				while mixed_song == song:
					rand_song_key = (songs.keys())[int(random() * len(songs.keys()))]
					mixed_song = songs[rand_song_key]
				
				# Extract necessary info from the mixed song
				dbeats_mixed = mixed_song.downbeats
				segtypes_mixed = mixed_song.segment_types
				segidxs_mixed = mixed_song.segment_indices
				segs_H_mixed = [segidxs_mixed[j] for j in range(len(segidxs_mixed)-2) if segtypes_mixed[j] == 'H']	# Exclude the last two segments (end of the song)
				segs_L_mixed = [segidxs_mixed[j] for j in range(len(segidxs_mixed)-2) if segtypes_mixed[j] == 'L']
				
				# Generate either a high or a low segment
				if i < num_h:
					# --- Type 1 of bad extracts: song in H segment, mixed in H segment, no filtering (bass clash) ---
					# Select a starting point for the current song
					start_db = segs_H[i % len(segs_H)] + (i*3 % 16)			# +i to have some rotation in downbeat position wrt segment boundary
					# Select a starting point for the bad song
					mixed_start_db = segs_H_mixed[i % len(segs_H_mixed)] + (i*3 % 16)	
				else:
					# Type 2 of bad extracts: song in L segment, mixed in H segment, mixed is filtered
					# Select a starting point for the current song
					start_db = segs_L[i % len(segs_L)] + (i*3 % 16)			# +i to have some rotation in downbeat position wrt segment boundary
					# Select a starting point for the bad song
					mixed_start_db = segs_H_mixed[i % len(segs_H_mixed)] + (i*3 % 16)	
				
				# Check if the start_dbs are okay
				if start_db + 2 >= len(dbeats) or start_db < 0:
					continue
				if mixed_start_db + 2 >= len(dbeats_mixed) or mixed_start_db < 0:
					continue
				
				# Time stretching master audio segment
				audio = getAudioSegmentAtDownbeat(song, start_db)
				audio_mixed = getAudioSegmentAtDownbeat(mixed_song, mixed_start_db)
				
				# Apply filtering if necessary
				if i >= num_h:
					audio_mixed = songtransitions.linear_fade_filter(audio_mixed, 'low_shelf', start_volume=0.0, end_volume=0.0)
					audio_mixed = songtransitions.linear_fade_filter(audio_mixed, 'high_shelf', start_volume=0.0, end_volume=0.0)
				
				# Overlap segments
				audio_mixed = ((audio + audio_mixed) * SQRT1over2).astype('single')
				
				# Calculate features for good segment
				datarow_good = calculateFeaturesForDownbeat(audio)
				datarow_bad = calculateFeaturesForDownbeat(audio_mixed)
				if isPlot:
					print len(datarow_good)
					print 'Good ', datarow_good
					print 'Bad ', datarow_bad
					plt.show()
				
				# Write features to feature file
				csvwriter.writerow([song.title, 1] + datarow_good)
				csvwriter.writerow([song.title, 0] + datarow_bad)
				
				if writeAudio:
					audio_out_good = np.append(audio_out_good, audio).astype('single')
					audio_out_bad = np.append(audio_out_bad, audio_mixed).astype('single')
				
				# Logging of stats
				num_segments_generated = num_segments_generated + 1
			
			print num_segments_generated
			
			if writeAudio:
				writer = MonoWriter(filename='output/good_'+song.title+'.wav')
				writer(audio_out_good)
				writer2 = MonoWriter(filename='output/bad_'+song.title+'.wav')
				writer2(audio_out_bad)
