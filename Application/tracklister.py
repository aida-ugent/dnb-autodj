from song import *
from songcollection import SongCollection
from util import *

import numpy as np

from essentia import *
from essentia.standard import MonoLoader
from BeatTracker import *
from DownbeatTracker.downbeatTracker import *
from timestretching import time_stretch_sola

from songtransitions import TransitionProfile, CrossFade
from audio_quality_model import evaluate_audio_quality

from random import random

import logging
logger = logging.getLogger('colorlogger')

from essentia.standard import MonoWriter
import csv

TYPE_DOUBLE_DROP = 'double drop'
TYPE_ROLLING = 'rolling'
TYPE_CHILL = 'relaxed'

TRANSITION_PROBAS = {
	TYPE_CHILL : [0.0, 0.6, 0.4],	# chill -> chill, rolling, ddrop
	TYPE_ROLLING : [0.5, 0.5, 0.0], # rolling> chill, rolling, ddrop
	TYPE_DOUBLE_DROP : [0.5, 0.5, 0.0] # ddrop -> chill, rolling, ddrop
	}

LENGTH_ROLLING_IN = 16
LENGTH_ROLLING_OUT = 32
LENGTH_DOUBLE_DROP_IN = LENGTH_ROLLING_IN
LENGTH_DOUBLE_DROP_OUT = 32
LENGTH_CHILL_IN = 16
LENGTH_CHILL_OUT = 16

ROLLING_START_OFFSET = 32 	# Start N downbeats after previous queue point.

def getMasterQueue(song, start_dbeat, cur_fade_type):
	
	'''
		Get the (potential) next queue point
		Returns the queue point, fade type, maximum fade in length and fade out length
		
		start_dbeat: first downbeat from where fade can start (causality constraint)
		
		returns:
		- queue: the point of SWITCHING (because fade in length not known yet)
		- fade_type: double drop, rolling or chill?
	'''
	
	# Get the segment types and indices that occur from start_dbeat and later
	segment_types, segment_dbeats = zip(*[tb for tb in zip(song.segment_types, song.segment_indices) if tb[1] >= start_dbeat])
	segment_dbeats = list(segment_dbeats)
	segment_types = list(segment_types)
	current_type = song.getSegmentType(start_dbeat) # TODO implement (get which segment the given downbeat is in)
	
	# Get the upcoming 'H' segments
	upcoming_H_dbeats = [segment_dbeats[j] for j in range(len(segment_dbeats)) if segment_types[j] == 'H' and segment_dbeats[j] != start_dbeat]
	upcoming_L_dbeats = [segment_dbeats[j] for j in range(len(segment_dbeats)) if segment_types[j] == 'L' and segment_dbeats[j] != start_dbeat]
		
	def getDbeatBefore(dbeat, options, n=1):
		''' Get the first segment downbeat before dbeat and after start_dbeat that is in options (dbeat not included) '''
		if dbeat is None:
			return None
		candidates = [b for b in options[::-1] if b < dbeat]
		if len(candidates) < n:
			return None
		else:
			return candidates[n-1]			
	def getHBefore(dbeat, n=1):
		''' Get the first segment downbeat before dbeat and after start_dbeat that is H (dbeat not included) '''
		return getDbeatBefore(dbeat, upcoming_H_dbeats, n=n)
	def getLBefore(dbeat, n=1):
		''' Get the first segment downbeat before dbeat and after start_dbeat that is L (dbeat not included) '''
		return getDbeatBefore(dbeat, upcoming_L_dbeats, n=n)
		
	def getDbeatAfter(dbeat, options, n=1):
		''' Get the nth segment downbeat after dbeat and after start_dbeat that is in options (dbeat not included)'''
		if dbeat is None:
			return None
		candidates = [b for b in options if b > dbeat]
		if len(candidates) < n:
			return None
		else:
			return candidates[n-1]
	def getHAfter(dbeat, n=1):
		''' Get the first segment downbeat before dbeat and after start_dbeat that is H (dbeat not included) '''
		return getDbeatAfter(dbeat, upcoming_H_dbeats, n=n)
	def getLAfter(dbeat, n=1):
		''' Get the first segment downbeat before dbeat and after start_dbeat that is L (dbeat not included) '''
		return getDbeatAfter(dbeat, upcoming_L_dbeats, n=n)
	
	P_chill, P_roll, P_ddrop = TRANSITION_PROBAS[cur_fade_type]
	
	# Determine if this should (and can) be a double drop
	# If there are no 'H' segments anymore, then double drop is impossible
	# If the next 'H' segment is too late in the song (less than 32 downbeats before the last L segment), then this doesn't make sense either
	if P_ddrop > 0:
		isDoubleDrop = (random() <= P_ddrop)
		# Get the first downbeat following a 'L' segment
		doubleDropDbeat = getHAfter(start_dbeat) if current_type == 'L' else getHAfter(getLAfter(start_dbeat))
		# Check if this doesn't take too long
		isDoubleDrop = isDoubleDrop and not doubleDropDbeat is None 
		
		if isDoubleDrop and doubleDropDbeat <= song.segment_indices[-1] - LENGTH_DOUBLE_DROP_OUT:
			fade_type = TYPE_DOUBLE_DROP
			queue = doubleDropDbeat
			# Length of double drop = 32 downbeats OR number of dbeats till next 'L' (+ 1 so that it switches 1 downbeat before the L->H transition)
			fade_out_len = min(LENGTH_DOUBLE_DROP_OUT, getLAfter(doubleDropDbeat) - doubleDropDbeat) + 1 # Switch 1 downbeat before L->H
			# Length of the fade in = 16 downbeats OR remaining time from start_dbeat till double drop
			max_fade_in_len = min(LENGTH_DOUBLE_DROP_IN, doubleDropDbeat - start_dbeat) - 1
			return queue-1, fade_type, max_fade_in_len, fade_out_len
	
	P_roll = P_roll / (P_roll + P_chill)
	P_chill = P_chill / (P_roll + P_chill)
	
	if P_roll > 0:	
		# Determine if it should (and can) be a rolling transition
		isRolling = (random() <= P_roll) and len(upcoming_H_dbeats) != 0
		rollingDbeat = getHAfter(start_dbeat)
		rollingDbeat = rollingDbeat + ROLLING_START_OFFSET if not rollingDbeat is None else start_dbeat + ROLLING_START_OFFSET # if current_type == 'H' else getHAfter(start_dbeat) + ROLLING_START_OFFSET
		if isRolling and rollingDbeat <= song.segment_indices[-1] - LENGTH_DOUBLE_DROP_OUT:		# If there are no 'H' segments anymore, then double drop is impossible
			fade_type = TYPE_ROLLING
			queue = rollingDbeat
			# Length of transition = 16 downbeats or time till next low segment (+1 to switch just before queue point)
			fade_out_len = min(LENGTH_ROLLING_OUT, getLAfter(queue) - queue) + 1
			# Length of fade in = 16 downbeats or remainder of time till queue (-1 to switch just before queue point)
			max_fade_in_len = min(LENGTH_ROLLING_IN, queue - start_dbeat) - 1
			return queue-1, fade_type, max_fade_in_len, fade_out_len
			
	# No rolling transition or double drop: must be a chill transition
	if True:		# Only reason this is here is to have the same indentation as above :)
		if True:	
			# Transition point: first low segment after the first high segment (or this dbeat if it is H)
			# The song must play a bit before doing a chill transition!
			queue = getLAfter(start_dbeat) if current_type == 'H' else getLAfter(getHAfter(start_dbeat))
			queue = queue if not queue is None else start_dbeat # If queue was none, there are no high segments coming anymore: start now!		
			fade_type = TYPE_CHILL
			# Transition length = 16 downbeats, or time till next H, or time till end
			must_end_before_dbeat = getHAfter(queue)
			must_end_before_dbeat = must_end_before_dbeat if not must_end_before_dbeat is None else song.segment_indices[-1]
			fade_out_len = min(LENGTH_CHILL_OUT, must_end_before_dbeat - queue)
			# Fade-in length = 16 downbeats, or from the end of the H segment preceding or from start_dbeat if 16 is too long
			max_fade_in_len = min(LENGTH_CHILL_IN, queue-start_dbeat)
			return queue, fade_type, max_fade_in_len, fade_out_len

def getSlaveQueue(song, fade_type, min_playable_length = 64):
	''' Search the slave song for a good transition point with type fade_type (chill, rolling, double drop) '''
	segment_dbeats = song.segment_indices
	segment_types = song.segment_types
	# Upcoming beginning of segments
	last_dbeat = segment_dbeats[-1]
	H_dbeats = [segment_dbeats[j] for j in range(len(segment_dbeats)-1) if segment_types[j] == 'H' and j > 0 and (segment_types[j-1] != 'H') and (last_dbeat - segment_dbeats[j] >= min_playable_length)]
	L_dbeats = [segment_dbeats[j] for j in range(len(segment_dbeats)-1) if segment_types[j] == 'L' and (j == 0 or segment_types[j-1] != 'L') and (last_dbeat - segment_dbeats[j] >= min_playable_length)]

	if fade_type == TYPE_DOUBLE_DROP or fade_type == TYPE_ROLLING:
		# Choose the first drop
		queue = H_dbeats[np.random.randint(len(H_dbeats))]-1	# -1 to switch just before drop
		fade_in_len = min(LENGTH_DOUBLE_DROP_IN + 1 if fade_type == TYPE_DOUBLE_DROP else LENGTH_ROLLING_IN + 1, queue - 1)
	else:
		# Choose the first L segment (beginning of song)
		queue = L_dbeats[0] + LENGTH_CHILL_IN
		fade_in_len = LENGTH_CHILL_IN
	
	return queue, fade_in_len

class TrackLister:
	
	def __init__(self, song_collection):
		self.songs = None 		# Ordered list of the songs in this tracklist
		self.crossfades = None	# Ordered list of crossfade objects: one less than number of songs
		self.song_collection = song_collection
		self.songsUnplayed = song_collection.get_annotated()	# Subset of song collection containing all unplayed songs
		self.songsPlayed = []									# List of songs already played
		self.song_file_idx = 0
		
	def getNextCrossfade(self, master_song, master_audio, drop_at_dbeat, transition_length, fade_type, tempo):
		'''
			Choose a song that overlaps best with the given audio, when dropping it at downbeat drop_at_dbeat.
			The type of transition is also given (rolling, double drop, chill).
		'''
		# Choose some random songs from the pool of unplayed songs
		NUM_SONGS = 3
		song_options = np.random.choice([s for s in self.songsUnplayed if s != master_song], size=NUM_SONGS, replace=False)
		
		samples_per_dbeat = int(44100 * 4 * 60.0 / tempo)
		
		# Iterate these songs and choose the best slave song
		cf = None
		best_score = -np.inf
		best_master_audio = None
		best_song = None
		best_master_queue = None
		best_anchor_sample = None	
		
		for s in song_options:
			# Open the song
			next_song = s
			next_song.open()
			next_song.openAudio()
			logger.debug('Selected song {}'.format(s.title))
			
			# Determine the queue points for the current song
			queue_slave, fade_in_len = getSlaveQueue(next_song, fade_type, min_playable_length = transition_length + 16)
			fade_in_len = min(fade_in_len, drop_at_dbeat)
			queue_master = drop_at_dbeat - fade_in_len
			queue_slave = queue_slave - fade_in_len
			
			# Crop the master audio so that it aligns with the queue point (in case fade_in_len is not drop_at_dbeat)
			master_audio_unmixed = master_audio[ : samples_per_dbeat * queue_master]
			master_audio_mixed = master_audio[samples_per_dbeat * queue_master: ]
			
			# Construct the cross-fade for this transition
			if queue_slave >= 16:
				cf = CrossFade(0, [queue_slave, queue_slave-16, queue_slave+16], transition_length - queue_master, fade_in_len, fade_type)
			else:
				cf = CrossFade(0, [queue_slave, queue_slave+16, queue_slave+32], transition_length - queue_master, fade_in_len, fade_type)
				
			# Iterate over the different options for queue_slave
			for queue_slave_cur in cf.queue_2_options:
									
				# Select the current slave audio and prepare it for mixing
				anchor_sample = int(next_song.downbeats[queue_slave_cur] * 44100)
				f = next_song.tempo / tempo
				next_audio_stretched = time_stretch_sola(next_song.audio[anchor_sample:], f, next_song.tempo, 0.0)
			
				# Now mix the audio
				master_audio_mixed_deepcpy = np.array(master_audio_mixed,dtype='single',copy=True)
				crossfaded_audio = cf.apply(master_audio_mixed_deepcpy, next_audio_stretched, tempo)
				master_audio_temp = np.append(master_audio_unmixed, crossfaded_audio)
				score = evaluate_audio_quality(master_audio_temp[:samples_per_dbeat * transition_length])
				if score > best_score:
					best_song = next_song
					best_score = score
					best_master_audio = master_audio_temp
					best_anchor_sample = anchor_sample
					best_master_queue = queue_slave_cur
					
				# Save the audio and its score
				#~ dbgfilename = 'output/'+str(self.song_file_idx)+'_'+master_song.title+'_'+s.title+'_'+str(queue_slave_cur)+'.wav'
				#~ writer = MonoWriter(filename=dbgfilename)
				#~ writer(master_audio_temp[:samples_per_dbeat*transition_length])	
				#~ self.song_file_idx = self.song_file_idx + 1
				#~ with open('output/bluabd.csv','a+') as csvfile:
					#~ csvwriter = csv.writer(csvfile)
					#~ csvwriter.writerow([dbgfilename,score])
					
				# Logging
				type_fade_dbg_str = '{} [{}:{}]: {}'.format(next_song.title[:10], fade_type, queue_slave_cur, score)
				logger.debug(type_fade_dbg_str)
				
		for s in song_options:
			if s != best_song:
				s.closeAudio()
				
		self.songsPlayed.append(best_song)
		self.songsUnplayed.remove(best_song)
		if len(self.songsUnplayed) <= NUM_SONGS: # If there are too few songs remaining, then restart
			logger.debug('Replenishing song pool')
			self.songsPlayed = []
			self.songsUnplayed = self.song_collection.get_annotated()	
			
		logger.debug('Songs unplayed: ' + str(len(self.songsUnplayed)))
		
		return best_song, best_master_queue, best_master_audio, best_anchor_sample
	
	def generate(self, N, overwrite = False):
		
		# First compose the tracklist
		if overwrite or self.songs is None:
			if len(self.song_collection.get_annotated()) > 0:
				self.songs = np.random.choice(self.song_collection.get_annotated(), size = N)
				self.songsUnplayed = self.song_collection.get_annotated()	# Subset of song collection containing all unplayed songs
		
		if self.songs is None:
			raise Exception('There must be songs in the library before you can create a tracklist!')
