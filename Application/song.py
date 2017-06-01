from util import *
import time
from essentia import *
from essentia.standard import *

from BeatTracker import *
from DownbeatTracker.downbeatTracker import *
from structuralsegmentation import *

import logging
logger = logging.getLogger('colorlogger')

def normalizeAudioGain(audio, rgain, target = -16):
	# Normalize to a level of -16, which is often value of signal here
	factor = 10**((-(target - rgain)/10.0) / 2.0) # Divide by 2 because we want sqrt (amplitude^2 is energy)
	audio *= factor	
	return audio

class Song:
	
	def __init__(self, path_to_file):
		
		self.dir_, self.title = os.path.split(os.path.abspath(path_to_file))
		self.title, self.extension = os.path.splitext(self.title)
		self.dir_annot = self.dir_ + '/' + ANNOT_SUBDIR
		
		if not os.path.isdir(self.dir_annot):
			logger.debug('Creating annotation directory : ' + self.dir_annot)
			os.mkdir(self.dir_annot)
		
		self.audio = None
		self.beats = None
		self.tempo = None
		self.downbeats = None
		self.segment_indices = None
		self.segment_types = None
		self.songBeginPadding = 0	# Number of samples to pad the song with, if first segment index < 0
		self.replaygain = None
	
	def getSegmentType(self, dbeat):
		''' Get the segment type ('H' or 'L') of the segment the dbeat falls in '''	
		for i in range(len(self.segment_types)-1):
			if self.segment_indices[i] <= dbeat and self.segment_indices[i+1] > dbeat:
				return self.segment_types[i]
		raise Exception('Invalid downbeat ' + str(dbeat) + ', should be between ' + str(self.segment_indices[0]) + ', ' + str(self.segment_indices[-1]))
	
	def hasBeatAnnot(self):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX))
		
	def hasDownbeatAnnot(self):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX))
		
	def hasSegmentationAnnot(self):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX))
		
	def hasReplayGainAnnot(self):
		return self.title in loadCsvAnnotationFile(self.dir_, ANNOT_GAIN_PREFIX)
		
	def hasAllAnnot(self):
		'''
		Check if this file has annotation files
		'''
		return self.hasBeatAnnot() and self.hasDownbeatAnnot() and self.hasSegmentationAnnot() and self.hasReplayGainAnnot()
		
	def annotate(self):
		# This doesn't store the annotations and audio in memory yet, this would cost too much memory: writes the annotations to disk and evicts the data from main memory until the audio is loaded for playback
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		audio = loader()
		
		# Beat annotations
		if not self.hasBeatAnnot():
			logger.debug('Annotating beats of ' + self.title)
			btracker = BeatTracker()
			btracker.run(audio)
			beats = btracker.getBeats()	# Do not keep the annotations in memory, so NOT self.beats = ...
			tempo = btracker.getBpm()
			phase = btracker.getPhase()
			writeAnnotFile(self.dir_, self.title, ANNOT_BEATS_PREFIX, beats, {'tempo' : tempo , 'phase' : phase})
		else:
			# Load the beats for annotating the downbeats
			beats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX)
			beats = [float(b) for b in beats_str]
			tempo = res_dict['tempo']
			
		# Downbeat annotations
		if not self.hasDownbeatAnnot():
			logger.debug('Annotating downbeats of ' + self.title)
			dbtracker = DownbeatTracker()
			downbeats = dbtracker.track(audio, beats)
			writeAnnotFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX, downbeats)
		else:
			downbeats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX)
			downbeats = [float(b) for b in downbeats_str]
			
		# Segmentation annotations
		if not self.hasSegmentationAnnot():
			logger.debug('Annotating structural segment boundaries of ' + self.title)
			structuralSegmentator = StructuralSegmentator()
			segment_db_indices, segment_types = structuralSegmentator.analyse(audio, downbeats, tempo)
			writeAnnotFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX, zip(segment_db_indices, segment_types))			
		else:
			segments_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX)
			
		if not self.hasReplayGainAnnot():
			logger.debug('Annotating replay gain of ' + self.title)
			# Dividing by 2 decreases the replay gain (related to energy in db) by 6,... dB
			# Indeed, amplitude / 2 => energy / 4 => 10log(4) = 6,...
			replayGain = ReplayGain()
			rgain = replayGain(audio)
			writeCsvAnnotation(self.dir_, ANNOT_GAIN_PREFIX, self.title, rgain)
		
	# Open the audio file and read the annotations
	def open(self):
				
		beats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX)
		self.beats = [float(b) for b in beats_str]
		self.tempo = res_dict['tempo']
		self.phase = res_dict['phase']
		downbeats_str, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX)
		self.downbeats = [float(b) for b in downbeats_str]
		
		segment_annot_strs, segment_annot_dict  = loadAnnotationFile(self.dir_, self.title, ANNOT_SEGMENT_PREFIX)
		self.segment_indices = []
		self.segment_types = []
		
		for s in segment_annot_strs:
			s1,s2 = s.split(' ')[:2]
			self.segment_indices.append(int(s1))
			self.segment_types.append(s2)
		
		# Some songs have a negative first segment index because the first downbeat got cropped a bit
		if self.segment_indices[0] < 0:
			# Calculate the amount of padding
			beat_length_s = 60.0 / self.tempo
			songBeginPaddingSeconds = (-self.segment_indices[0] * 4 * beat_length_s - self.downbeats[0])
			self.songBeginPadding = int(songBeginPaddingSeconds * 44100 )
			self.downbeats = [dbeat + songBeginPaddingSeconds for dbeat in self.downbeats]
			self.downbeats = [i*4*beat_length_s for i in range(-self.segment_indices[0])] + self.downbeats
			self.beats = [beat + songBeginPaddingSeconds for beat in self.beats]
			self.beats = [i*beat_length_s for i in range(int(self.beats[0] / beat_length_s))] + self.beats
			offset = self.segment_indices[0] 
			self.segment_indices = [idx - offset for idx in self.segment_indices]
		
		self.replaygain = loadCsvAnnotationFile(self.dir_, ANNOT_GAIN_PREFIX)[self.title]
			
	def openAudio(self):
		#~ logger.debug('Opening audio ' + str(self.title))
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		time0 = time.time()
		audio = loader().astype('single')
		time1 = time.time()
		#~ logger.debug('Time waiting for audio loading: ' + str(time1-time0))
		self.audio = normalizeAudioGain(audio, self.replaygain)
		if self.songBeginPadding > 0:
			self.audio = np.append(np.zeros((1,self.songBeginPadding),dtype='single'), self.audio)
	
	def closeAudio(self):
		# Garbage collector will take care of this later on
		self.audio = None
		
	# Close the audio file and reset all buffers to None
	def close(self):
		self.audio = None
		self.beats = None
		self.downbeats = None
		self.queuepts = None 
