from song import *
from util import *
import os

import logging
logger = logging.getLogger('colorlogger')

class SongCollection:
	
	def __init__(self):
		self.songs = []			# A list holding all (annotated) songs
		self.directories = []	# A list containing all loaded directories
	
	def load_directory(self, directory):
		directory_ = os.path.abspath(directory)
		if directory_ in self.directories:
			return				# Don't add the same directory twice
		logger.info('Loading directory ' + directory + '...')
		self.directories.append(directory_)
		self.songs.extend([Song(os.path.join(directory_, f)) for f in os.listdir(directory_) if os.path.isfile(os.path.join(directory_, f)) and (f.endswith('.wav') or f.endswith('.mp3'))])
	
	def annotate(self):
		for s in self.get_unannotated():
			s.annotate()
	
	def get_unannotated(self):
		return [s for s in self.songs if not s.hasAllAnnot()]
		
	def get_annotated(self):
		return [s for s in self.songs if s.hasAllAnnot()]
		
if __name__ == '__main__':
	
	from djcontroller import DjController
	from tracklister import TrackLister
	
	# Open the long library
	sc = SongCollection()
	sc.load_directory('../music/')
	sc.songs[0].open()
	logger.debug(sc.songs[0].tempo)
	# Generate a tracklist
	tl = TrackLister(sc)
	tl.generate(10)
	# Play!
	sm = DjController(tl)
	sm.play()
		
