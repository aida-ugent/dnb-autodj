'''
	This old demo creates mashups of different songs by switching each downbeat to another song.
	This demo is currently very outdated, was only used in the first semester.
	Might need additional changes to be compatible with the latest version of the beat and downbeat tracker.
	Kept here as a reference for later.
'''


from BeatTracker import * 
from downbeatTracker import *
import os, sys, random
from essentia import *
from essentia.standard import *

import pyaudio

ANNOT_SUBDIR = '_annot_beat_downbeat/'
ANNOT_DOWNB_PREFIX = 'downbeats_'
ANNOT_BEATS_PREFIX = 'beats_'

class UnannotatedException(Exception):
    pass

def loadAnnotationFile(directory, song_title, prefix):
	'''
	Loads an input file with annotated times in seconds.
	
	-Returns: A numpy array with the annotated times parsed from the file.
	'''
	input_file = directory + ANNOT_SUBDIR + prefix + song_title + '.txt'
	result = []
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line[0] == '#':
					continue
				result.append(float(line))	
	else:
		raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
	return result

if __name__ == '__main__':
	
	if len(sys.argv) != 2:
		print 'Usage: ', sys.argv[0], ' <directory>'
		exit()
	
	directory = sys.argv[1]
	files = [f for f in os.listdir(directory) if os.path.isfile(directory + f) and (f.endswith('.wav') or f.endswith('.mp3'))]
	random.shuffle(files)
	files = files[:2]
	
	essentia.log.infoActive = False
	
	output = []
	
	#~ pygame.mixer.pre_init(frequency=44100, size=-16, channels=1, buffer=4096*4096)
	#~ pygame.init()
	#~ channel = pygame.mixer.Channel(0)
	song = np.array([], dtype='single')
	
	index = 64
	f = None
	
	audio = {}	
	
	# instantiate PyAudio (1)
	p = pyaudio.PyAudio()

	# open stream (2)
	stream = p.open(format = pyaudio.paFloat32,
					channels=1,
					rate=44100,
					output=True)
	
	for i in range(2):
		f = files[i%2]
		# Detect beat and downbeat
		print directory + f
		loader = essentia.standard.MonoLoader(filename = directory + f)
		audio[i] = loader()
	
	for i in range(16):
		
		#~ if i%2 == 0:
			#~ f_old = f
			#~ while f == f_old:
				#~ #f = random.choice(files[1:])
				#~ f = files[1]
		#~ else:
			#~ f = files[0]
		f = files[i%2]
		
		f_stripped = os.path.splitext(f)[0]
		try:
			beats = loadAnnotationFile(directory, f_stripped, ANNOT_BEATS_PREFIX)
			downbeats = loadAnnotationFile(directory, f_stripped, ANNOT_DOWNB_PREFIX)
		except Exception as e:
			print e
			continue
		
		downbeatIndex = 0
		for j in range(4):
			if beats[j] == downbeats[0]:
				downbeatIndex = j
				break
				
		offset = {
			0:0, 1:1, 2:2, 3:-1
		}
		
		start_s = downbeats[index + offset[downbeatIndex]]
		end_s = downbeats[index + 1 + offset[downbeatIndex]]
		print start_s - end_s
		index = index + 1
		start = start_s * 44100
		end = end_s * 44100
		
		#song = np.append(song, audio[i%2][start:end])
		toPlay = audio[i%2][start:end]
		print toPlay.dtype, len(toPlay), len(toPlay)/44100.0
		stream.write(toPlay, num_frames=len(toPlay), exception_on_underflow=True)
		print(len(audio[i%2][start:end]) / 44100.0)
		
	stream.stop_stream()
	stream.close()

	# close PyAudio (5)
	p.terminate()

		
