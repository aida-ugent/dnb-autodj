import numpy as np
import bisect

# For threading and queueing of songs
import threading
import multiprocessing
from multiprocessing import Process, Queue, Event
from time import sleep
# For live playback of songs
import pyaudio

import tracklister
from songtransitions import CrossFade
from timestretching import time_stretch_sola

import logging
logger = logging.getLogger('colorlogger')

import time, os, sys

class DjController:
	
	def __init__(self, tracklister):
		self.tracklister = tracklister
		
		self.audio_thread = None
		self.dj_thread = None
		self.playEvent = multiprocessing.Event()
		self.isPlaying = multiprocessing.Value('b', True)
		self.skipFlag = multiprocessing.Value('b', False)
		self.queue = Queue(6)				# A blocking queue of to pass at most N audio fragments between audio thread and generation thread
		
		self.pyaudio = None
		self.stream = None
		
		self.djloop_calculates_crossfade = False
			
	def play(self):
								
		self.tracklister.generate(1,overwrite=True)
		self.playEvent.set()
			
		if self.dj_thread is None and self.audio_thread is None:
			self.dj_thread = Process(target = self._dj_loop, args=(self.playEvent, self.isPlaying))
			self.audio_thread = Process(target = self._audio_play_loop, args=(self.playEvent, self.isPlaying))
			self.isPlaying.value = True
			self.dj_thread.start()
			
			while self.queue.empty():
				# wait until the queue is full
				sleep(0.1)
				
			self.audio_thread.start()
			
		elif self.dj_thread is None or self.audio_thread is None:
			raise Exception('dj_thread and audio_thread are not both Null!')
	
	def skipToNextSegment(self):
		if not self.queue.empty():
			self.skipFlag.value = True
		else:
			self.skipFlag.value = False
			logger.warning('Cannot skip to next segment, no audio in queue!')
			
	
	def pause(self):
		if self.audio_thread is None:
			return
		self.playEvent.clear()
		
	def stop(self):
		# If paused, then continue playing (deadlock prevention)
		try:
			self.play()
		except Exception as e:
			logger.debug(e)
		# Notify the threads to stop working
		self.isPlaying.value = False
		# Empty the queue so the dj thread can terminate
		while not self.queue.empty():
			self.queue.get_nowait()
		if not self.dj_thread is None:
			self.dj_thread.terminate()
		# Reset the threads
		self.queue = Queue(6)
		self.audio_thread = None
		self.dj_thread = None
		# Reset pyaudio resources
		if not self.stream is None:
			self.stream.stop_stream()
			self.stream.close()
		if not self.pyaudio is None:
			self.pyaudio.terminate()
		self.pyaudio = None
			
	def _audio_play_loop(self, playEvent, isPlaying):
		
		if self.pyaudio is None:
			# Disable output for a while, because pyaudio prints annoying error messages that are irrelevant but that cannot be surpressed :(
			# http://stackoverflow.com/questions/977840/redirecting-fortran-called-via-f2py-output-in-python/978264#978264
			null_fds = [os.open(os.devnull, os.O_RDWR) for x in xrange(2)]
			save = os.dup(1), os.dup(2)
			os.dup2(null_fds[0], 1)
			os.dup2(null_fds[1], 2)
			
			# Open the audio
			self.pyaudio = pyaudio.PyAudio()
			
			# Reset stderr, stdout
			os.dup2(save[0], 1)
			os.dup2(save[1], 2)
			os.close(null_fds[0])
			os.close(null_fds[1])
			
		if self.stream is None:
			self.stream = self.pyaudio.open(format = pyaudio.paFloat32,
						channels=1,
						rate=44100,
						output=True)
						
		while isPlaying.value:
			time0 = time.time()
			toPlay, toPlayStr = self.queue.get()
			time1 = time.time()
			logger.info(toPlayStr)
			#~ logger.debug('Time waiting for queue: ' + str(time1-time0))
			if toPlay is None:
				logger.debug('Stopping music')
				self.stop()
				return
				
			FRAME_LEN = 1024
			last_frame_start_idx = int(len(toPlay)/FRAME_LEN) * FRAME_LEN
			for cur_idx in range(0,last_frame_start_idx+1,FRAME_LEN):
				playEvent.wait()
				if not self.isPlaying.value:
					return
				if self.skipFlag.value:
					self.skipFlag.value = False
					break
				if cur_idx == last_frame_start_idx:
					end_idx = len(toPlay)
				else:
					end_idx = cur_idx + FRAME_LEN
				toPlayNow = toPlay[cur_idx:end_idx]
				#~ logger.debug(str(len(toPlayNow)) + '='*int(rms))
				if toPlayNow.dtype != 'float32':
					logger.debug('Incorrect type of toPlayNow: {}'.format(toPlayNow.dtype))
				self.stream.write(toPlayNow, num_frames=len(toPlayNow), exception_on_underflow=False)
			
	def _dj_loop(self, playEvent, isPlaying):
		
		try:
			# Set parameters for the first song
			master_song = self.tracklister.songs[0]
			master_song.open()
			master_song.openAudio()
			master_audio = master_song.audio
			logger.debug('FIRST SONG: {}'.format(master_song.title))
			
			queue_master = master_song.segment_indices[0] + 32 # Start at least 32 downbeat into the first song, enough time to fill the buffer
			fade_type = tracklister.TYPE_CHILL
			
			TEMPO = 175 # Keep tempo fixed for classification of audio in segment evaluation
			f = master_song.tempo / TEMPO
			
			anchor_sample = 0
			master_audio = time_stretch_sola(master_audio[anchor_sample:], f, master_song.tempo, 0.0)
			
			# Array with all songs somewhere in queue at the moment (playing or to be played)
			song_titles_in_buffer = [master_song.title]
			# Sorted list of fade in points in samples relative to start of buffer
			tracklist_changes = []
			# The total number of songs hearable right now
			num_songs_playing = 1
			# The idx of the master in the subset of songs that is playing right now
			songs_playing_master = 0
			
			def curPlayingString(fade_type_str):
				outstr = 'Now playing:\n'
				for i in range(len(song_titles_in_buffer)):
					if i != songs_playing_master:
						outstr += song_titles_in_buffer[i] + '\n'
					else:
						outstr += song_titles_in_buffer[i].upper() + '\n'
				if fade_type_str != '':
					outstr += '['+fade_type_str+']'
				return outstr
				
			samples_per_dbeat = int(44100 * 4 * 60.0 / TEMPO)
			
			#~ NUM_SONGS = 20
			#~ for i in range(NUM_SONGS):
			
			# Drum and Bass ad infinitum :)
			i = 0
			NUM_SONGS = 999 # "i" will be 1 max
			while True:
				
				# Determine the queue point in the master audio
				queue_master, next_fade_type, max_fade_in_len, fade_out_len = tracklister.getMasterQueue(master_song, queue_master, fade_type)
				
				# Queue the part of the master audio buffer up till the next queue point
				# The remainder of the audio is queued in three parts: the part till the fade-in is over, till the fade-out is over
				# and the part till the new fade-in begins. If the new fade-in begins earlier then only one
				# audio segment is queued of course. This allows the user to skip between segments	
				next_in_start = int(f * (master_song.downbeats[queue_master - max_fade_in_len] * 44100 - anchor_sample))
				bisect.insort(tracklist_changes, (next_in_start, 'in', fade_type))
				# Add it to the list of change points
				if i > 0:
					cur_in_end = int(f * (master_song.downbeats[prev_queue_slave + prev_fade_in_len] * 44100 - anchor_sample))
					cur_out_end = int(f * (master_song.downbeats[prev_queue_slave + prev_fade_in_len + prev_fade_out_len] * 44100 - anchor_sample))
					bisect.insort(tracklist_changes, (cur_in_end, 'switch', fade_type))
					bisect.insort(tracklist_changes, (cur_out_end, 'out', fade_type))
				i = 1
				
				prev_end_sample = 0
				for end_sample, in_or_out, cur_fade_type in tracklist_changes:						
												
					# If its a double drop, then end_sample and prev_end_sample might be the same! Don't queue empty segments..
					if end_sample > prev_end_sample:
						toPlay = master_audio[prev_end_sample : end_sample]
						cur_fade_type_str = cur_fade_type if num_songs_playing > 1 else ''
						toPlayTuple = (toPlay,curPlayingString(cur_fade_type_str))
						self.queue.put(toPlayTuple, True)	# Block until slot available
						prev_end_sample = end_sample
					
					if in_or_out == 'in':
						num_songs_playing += 1
					elif in_or_out == 'out':
						num_songs_playing -= 1
						songs_playing_master -= 1
						song_titles_in_buffer = song_titles_in_buffer[1:]
					else: # switch
						songs_playing_master += 1
							
					if end_sample == next_in_start:
						break
						
				# Strip the queued audio from the master_audio buffer
				master_audio = master_audio[ next_in_start : ]
				# Modify the tracklist changes array to reflect the stripping of the master_audio buffer
				tracklist_changes = [(tc[0] - next_in_start, tc[1],tc[2]) for tc in tracklist_changes if tc[0] >= next_in_start]
				fade_type = next_fade_type
					
				if not isPlaying.value:
					logger.debug('Stopping DJ!')
					self.queue.put(None)
					return
				
				# Select a new song that overlaps best with the given audio and given fade type
				if i < NUM_SONGS - 1:
					# Try different combinations and return the mixed audio that fits best
					# The new song then becomes the master song and its queue becomes the master queue point
					self.djloop_calculates_crossfade = True
					master_song, queue_master, master_audio, anchor_sample = self.tracklister.getNextCrossfade(master_song, master_audio, max_fade_in_len, max_fade_in_len + fade_out_len, fade_type, TEMPO) 
					self.djloop_calculates_crossfade = False
					f = master_song.tempo / TEMPO
					prev_queue_slave = queue_master
					prev_fade_in_len = max_fade_in_len
					prev_fade_out_len = fade_out_len
					song_titles_in_buffer = song_titles_in_buffer + [master_song.title]
					logger.debug('NEXT SONG: [{}] {}'.format(fade_type, master_song.title))

			# Play the remainder of the current song and stop the mix
			toPlay = master_audio
			cur_fade_type_str = 'End of the mix, thanks for listening!'
			toPlayTuple = (toPlay,curPlayingString(cur_fade_type_str))
			self.queue.put(toPlayTuple, True)
			self.queue.put((None, 'The end'))	# Notify that the DJ is done giving segments
		except Exception as e:
			logging.exception('Exception occurred: ' + str(e))
			self.stop()
			
		
