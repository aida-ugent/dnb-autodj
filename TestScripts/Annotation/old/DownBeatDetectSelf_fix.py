#from BeatTracker import *
import numpy as np
import sys, os
import essentia
from essentia.standard import AudioLoader, AudioOnsetsMarker, MonoWriter
	
'''
Create correct (ground-truth) annotations for downbeats based on downbeat annotations made so far.
'''

def writeAnnotFile(beats, bpm, phase, filename, outDir):
	
	with open(outDir + 'beats_' + filename + '.txt', 'w+') as f:
		f.write('#bpm '+str(bpm)+'\n')
		f.write('#phase '+str(phase)+'\n')
		for beat in beats:
			f.write("{:.9f}".format(beat) + '\n')

if __name__ == '__main__':
	
	if len(sys.argv) != 3:
		print 'Usage: ', sys.argv[0], ' <directory> <out_directory>'
		exit()
	
	directory = sys.argv[1]
	out_directory = sys.argv[2]
	
	for filename in os.listdir(directory):
		if filename.endswith(".mp3") or filename.endswith(".wav"): 
			
			filename, ext = os.path.splitext(filename)
			if os.path.isfile(out_directory + 'downbeats_' + filename + '.txt'):
				# Step 0: Load the audio
				print 'Processing "', filename, '" ...'
				loader = essentia.standard.MonoLoader(filename = directory + filename + ext)
				audio = loader()

				# Step 1: Get the BPM
				tracker = BeatTracker()
				tracker.run(audio)
				bpm = tracker.getBpm()
				phase = tracker.getPhase()
				beats = tracker.getBeats()
				
				# Step 2: overlay with beeps
				writeAnnotFile(beats, bpm, phase, filename, out_directory)
			else:
				print 'Could not parse downbeat annotations for ', filename
			
			
				
			
			

