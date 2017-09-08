from BeatTracker import *
import numpy as np
import sys, os
import essentia
from essentia.standard import AudioLoader

def process(indir, outdir, filename):

	# Load the audio
	print 'Processing "', filename, '" ...'
	loader = essentia.standard.MonoLoader(filename = indir + filename)
	#~ audio, sampleRate, numberChannels = loader()[0:3]
	audio = loader()
	#~ if(sampleRate != 44100):
		#~ raise Exception('Sample rate must be 44.1 kHz!')

	# TESTING HERE
	tracker = BeatTracker()
	tracker.run(audio)
	print 'Detected BPM: ', tracker.getBpm()
	print 'Detected phase: ', tracker.getPhase()
	beats = (tracker.getBeats())
	
	# Write beats to file
	with open(outdir + filename + '.txt', 'w') as f:
		for beat in beats:
			f.write(str(beat) + '\n')
	
if __name__ == '__main__':
	
	if len(sys.argv) != 3:
		print 'Usage: ', sys.argv[0], ' <directory> <out_directory>'
		exit()
	
	directory = sys.argv[1]
	out_directory = sys.argv[2]
	
	for filename in os.listdir(directory):
		if filename.endswith(".mp3") or filename.endswith(".wav"): 
			process(directory, out_directory, filename)
		else:
			continue


