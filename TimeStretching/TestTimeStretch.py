import sys
import numpy as np
import BeatTracker
import pyrubberband as prb
import essentia
from essentia.standard import MonoLoader, MonoWriter

def linear(u):
	return (1-u, u)
	
def quadratic_out(u):
    u = u * u
    return (1-u, u)

def quadratic_in(u):
    u = 1-u
    u = u * u
    return (u, 1-u)

def linear_bounce(u):
    u = 2 * ( 0.5-u if u > 0.5 else u)
    return (1-u, u)


def crossfade(audio1, audio2, start1, start2, length, method=linear):
	'''Crossfade two audio clips, using linear fading by default.
	Assumes MONO input'''
	totalLength = start1 + audio2.size - start2 #total length in samples
	output = np.zeros((totalLength))
	output[:start1] = audio1[:start1]			# Beginning
	output[start1+length:] = audio2[start2+length:]	# Ending
	for i, sample1, sample2 in zip(range(start1, start1+length), audio1[start1:start1+length], audio2[start2:start2+length]):
		a1, a2 = method((i-start1)/float(length))
		new1, new2 =  (a1*sample1, a2*sample2)
		output[i] = new1 + new2
	return output
	

if __name__ == '__main__':
	
	if len(sys.argv) != 3:
		print 'Usage: ', sys.argv[0], ' <file1> <file2>'
		exit()
		
	file1 = sys.argv[1]
	file2 = sys.argv[2]
	
	# Load first audio file
	b = BeatTracker.BeatTracker(minBpm = 160.0, maxBpm = 195.0)
	print 'Loading audio file "', file1, '" ...'
	loader = essentia.standard.MonoLoader(filename = file1)
	audio1 = np.array(loader())
	print 'Processing...'
	b.run(audio1)
	bpm1 = b.getBpm()
	phase1 = b.getPhase()
	print 'Bpm ', bpm1, ' and phase ', phase1
	
	# Load second audio file
	print 'Loading audio file "', file2, '" ...'
	loader = essentia.standard.MonoLoader(filename = file2)
	audio2 = np.array(loader())
	print 'Processing...'
	b.run(audio2)
	bpm2 = b.getBpm()
	phase2 = b.getPhase()
	print 'Bpm ', bpm2, ' and phase ', phase2
	
	# Timestretching
	#TODO pyrubberband uses temp files, which is actually not that nice :(
	if(bpm2 != bpm1):
		print 'Time stretching with rate ', bpm1/bpm2
		audio2_stretched = prb.time_stretch(audio2, 44100, bpm1/bpm2)
	
	# Cross-fading
	start1 = int(44100.0 * (phase1 + 537*(60./bpm1)))
	start2 = int(44100.0 * (phase2 * bpm2/bpm1 + 1*(60./bpm1))) #bpm1 because audio2 has been stretched!
	length = int(44100.0 * (8*(60./bpm1)))
	print 'Crossfade from ', start1, ' till ', start1+length, '; start2 = ', start2
	result = crossfade(audio1, audio2_stretched, start1, start2, length).astype('single')
	
	# Write the result to a file
	# Output the marked file
	writer = MonoWriter(filename = 'test.wav')
	beginIndex = int(44100.0 * (phase1 + 514*(60./bpm1)))
	endIndex = int(44100.0 * (phase1 + 592*(60./bpm1)))
	writer(result[beginIndex:endIndex]) #Only write fragment

	# Play the result
	from subprocess import call
	call(["mplayer", 'test.wav'])
	
	
		
	
	
