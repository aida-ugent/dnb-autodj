from BeatTracker import *
import numpy as np
import sys, os
import essentia
from essentia.standard import AudioLoader, AudioOnsetsMarker, MonoWriter
	
	
def overlayAudio(offset, interval):
	onsetMarker = AudioOnsetsMarker(onsets = 1.0*beats[offset::interval])
	audioMarked = onsetMarker(audio)
	writer = MonoWriter(filename = 'test.wav')
	beginIndex = 0.25*np.size(audioMarked)
	endIndex = 0.35*np.size(audioMarked)
	writer(audioMarked[beginIndex:endIndex]) #Only write fragment

def writeAnnotFile(beats, bpm, phase, value, filename, outDir):
	
	with open(outDir + 'downbeats_' + filename + '.txt', 'w+') as f:
		f.write('#phase '+str(phase)+'\n')
		f.write('#downbeat '+str(value)+'\n')
		for beat in beats[value::4]:
			f.write("{:.9f}".format(beat) + '\n')
			
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
				continue 
			
			# Step 0: Load the audio
			print 'Processing "', filename, '" ...'
			loader = essentia.standard.MonoLoader(filename = directory + filename + ext)
			audio = loader()

			# Step 1: Get the BPM
			tracker = BeatTracker(minBpm = 150.0)
			tracker.run(audio)
			print 'Detected BPM: ', tracker.getBpm()
			print 'Detected phase: ', tracker.getPhase()
			beats = (tracker.getBeats())
			
			# Step 2: overlay with beeps
			overlayAudio(0, 4)
			from subprocess import call
			call(["mplayer", 'test.wav'])
			
			# Step 3: ask for feedback
			userInput = ''
			validOptions = ['a','z','e','r','x']
			
			print 'Was the input correct? (a: 0, z:1, e:2, r:3, x:bpm wrong, t:play again)'
			while len(userInput) != 1 or (userInput not in validOptions):
				userInput = raw_input(':').lower()
				if userInput == 't':
					# Replay output
					from subprocess import call
					call(["mplayer", 'test.wav'])
			
			if userInput != 'x':
				onsetMappings = {
					'a' : 0, # onset on correct location
					'z' : 3, # onset one beat too late => downbeat is on 3rd
					'e' : 2, # onset two beats too late => next downbeat on second
					'r' : 1  # onset three beats too late
				}
				writeAnnotFile(beats, tracker.getBpm(), tracker.getPhase(), onsetMappings[userInput], filename, out_directory)
			else:
				print 'ERROR: INCORRECT BPM DETECTION'
				with open(out_directory + 'ERROR_beats_' + filename + '.txt', 'w+') as f:
					for beat in beats[::]:
						f.write(str(beat) + '\n')
			
			
				
			
			

