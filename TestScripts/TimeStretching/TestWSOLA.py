import sys
import numpy as np
import BeatTracker
import pyrubberband as prb
import essentia
from essentia.standard import MonoLoader, MonoWriter
from scipy import signal
import time

'''
WSOLA: find a frame around the theoretical offset so that this frame matches the 'natural' frame that comes after the input frame in the ORIGINAL stream
e.g. stream: f1 f2 f3 ... 
f2 naturally follows f1, so a frame f that overlaps-adds best with f1 must closely resemble f2
'''

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
    
def time_stretch_sola(audio, f, bpm, phase, sample_rate = 44100, fragment_s = 0.1, overlap_s = .020, seek_window_s = .015):
	
	# Initialise time offsets and window lengths
	#~ frame_len_2 = int(fragment_s * sample_rate)		# Length of a fragment, including overlap at both sides
	#~ overlap_len = int(overlap_s * sample_rate)
	#~ frame_len_1 = frame_len_2 - overlap_len			# Length of a fragment, including only one overlap
	#~ frame_len_0 = frame_len_1 - overlap_len			# Length of a fragment, excluding overlaps (unmixed part)
	#~ next_frame_offset_f = f * frame_len_1 	# keep as a float to prevent rounding errors
	#~ next_frame_offset = int(next_frame_offset_f) 	# keep as a float to prevent rounding errors
	#~ seek_win_len = int(seek_window_s * sample_rate)
	
	frame_len_1 = 4410		# Length of a fragment, including overlap at one side; about 100 ms, see above
	overlap_len = 252		# About 8 ms
	frame_len_2 = frame_len_1 + overlap_len			# Length of a fragment, including overlap at both sides
	frame_len_0 = frame_len_1 - overlap_len			# Length of a fragment, excluding overlaps (unmixed part)
	next_frame_offset_f =  frame_len_1 / f			# keep as a float to prevent rounding errors
	next_frame_offset = int(next_frame_offset_f) 	# keep as a float to prevent rounding errors
	seek_win_len_half = int(955) / 2	# window total ~ 21,666 ms
	
	#~ frame_len_1 = int((60./ bpm) * (sample_rate / 8.0))		# Length of a fragment, including overlap at one side; about 40 ms
	#~ overlap_len = int(frame_len_1 / 5)
	#~ frame_len_2 = frame_len_1 + overlap_len			# Length of a fragment, including overlap at both sides
	#~ frame_len_0 = frame_len_1 - overlap_len			# Length of a fragment, excluding overlaps (unmixed part)
	#~ next_frame_offset_f =  frame_len_1 / f			# keep as a float to prevent rounding errors
	#~ next_frame_offset = int(next_frame_offset_f) 	# keep as a float to prevent rounding errors
	#~ seek_win_len_half = int(frame_len_2 / 2) / 2	# window total ~ 15 ms
	
	def find_matching_frame(frame, theor_center):
		'''
		Find a frame in the neighbourhood of theor_center that maximizes the autocorrelation with the given frame as much as possible.
			
			:returns The start index in the given audio array of the most matching frame. Points to beginning of STABLE part. 
		'''
		# minus overlap_len because theor_start_frame is at the beginning of the constant part, but you have to convolve with possible intro overlap parts
		# minus len(frame) and not overlap_len to avoid errors when frame is at the very end of the input audio, and is not a full overlap part anymore
		cur_win_min = theor_center - seek_win_len_half
		cur_win_max = theor_center + seek_win_len_half
		# TODO You can correlate entire signal with itself beforehand, and just read interesting info
		correlation = signal.fftconvolve(audio[cur_win_min:cur_win_max], frame[::-1], mode='same') # Faster than np.correlate! cf http://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
		#correlation = np.correlate(audio[cur_win_min:cur_win_max], frame, mode = 'same')
		return theor_center  + (np.argmax(correlation) - seek_win_len_half)
	
	# Algorithm
	
	# TODO test with aligning with BPM: first align with phase. Insert zeros so that starts with beat
	phase_inv_len = int((60./bpm - phase) * sample_rate)
	audio = np.insert(audio, 0, [0] * phase_inv_len)
	
	# Initialise output buffer
	# TODO currently assumes mono 44100 kHz audio (audio.size)
	num_samples_out = int(f * audio.size)
	output = np.zeros(num_samples_out)	# Note the definition of f: stretch factor of audio!; prealloc
	
	num_frames_out = num_samples_out / frame_len_1
	in_ptr_th_f = 0.0		# Theoretical starting point (equidistant intervals)
	in_ptr = 0
	isLastFrame = False
	
	# frame_index is aligned at the beginning of the constant part of the next frame that will be written in output
	for out_ptr in range(0, num_frames_out * frame_len_1, frame_len_1):
					
		# Write the constant part of the frame
		frame_to_copy = audio[in_ptr : in_ptr + frame_len_0]
		output[out_ptr : out_ptr + len(frame_to_copy)]  = frame_to_copy # intermediate step to prevent mismatch between out : out+len and in:in+len' 		
		
		# Check if it is still useful to look for a next frame
		# This is not the case when the part that is overlapped with next frame is not complete (or even completely missing) 
		if (in_ptr + frame_len_1 > audio.size):
			frame_to_copy = audio[in_ptr + frame_len_0 : in_ptr + frame_len_1]
			output[out_ptr + frame_len_0 : out_ptr + frame_len_0 + len(frame_to_copy)] = frame_to_copy
			return output
		
		# Look for the next frame that matches best when overlapped
		
		# Method 1: only consider overlaps
		#~ frame1_overlap = audio[in_ptr + frame_len_0 : in_ptr + frame_len_1]
		#~ match_ptr = find_matching_frame(frame1_overlap, int(in_ptr_th_f + next_frame_offset_f) - overlap_len)
		#~ frame2_overlap = audio[match_ptr - overlap_len : match_ptr]
		
		# Method 2: pure WSOLA: match the next frame in the INPUT audio as closely as possible, since this frame is the natural follower of the original
		frame_to_match = audio[in_ptr + frame_len_0 : in_ptr + frame_len_0 + frame_len_1]
		match_ptr = find_matching_frame(frame_to_match, int(in_ptr_th_f + next_frame_offset_f) - overlap_len)
		
		frame1_overlap = audio[in_ptr + frame_len_0 : in_ptr + frame_len_1]
		frame2_overlap = audio[match_ptr : match_ptr + overlap_len]
		
		# Mix the overlap parts of the frames
		output[out_ptr + frame_len_0 : out_ptr + frame_len_1] = crossfade(frame1_overlap, frame2_overlap, length = overlap_len, method=linear)
		print out_ptr + frame_len_0
		
		# Increase the input pointers
		in_ptr = match_ptr + overlap_len
		in_ptr_th_f += next_frame_offset_f
		
		
	return np.array(output).astype('single')
		
# Simple crossfade of two perfectly overlapping pieces of audio
# Assumes audio1 and audio2 have the same length
def crossfade(audio1, audio2, start1 = 0, start2 = 0, length = None, method=quadratic_out):
	'''Crossfade two audio clips, using linear fading by default.
	Assumes MONO input'''
	if(length is None):
		length = audio1.size
	profile = np.arange(0.0, len(audio1)) / len(audio1)
	output = (audio1 * profile[::-1]) + (audio2 * profile)
	return output 
	

if __name__ == '__main__':
	
	if len(sys.argv) != 3:
		print 'Usage: ', sys.argv[0], ' <file1> <bpm>'
		exit()
		
	file1 = sys.argv[1]
	
	# Parse bpm input
	bpm2 = int(sys.argv[2])
	
	# Load audio file
	b = BeatTracker.BeatTracker(minBpm = 160.0, maxBpm = 195.0)
	print 'Loading audio file "', file1, '" ...'
	loader = essentia.standard.MonoLoader(filename = file1)
	audio1 = np.array(loader())
	print 'Processing...'
	b.run(audio1)
	bpm1 = b.getBpm()
	phase1 = b.getPhase()
	print 'Bpm ', bpm1, ' and phase ', phase1
	
	# Timestretching
	beginIndex = int(44100.0 * (phase1 + 300*(60./bpm1)))
	endIndex = int(44100.0 * (phase1 + 600*(60./bpm1)))
	
	if(bpm1 != bpm2):
		print 'Time stretching with factor ', bpm1/bpm2
		audio_stretched = time_stretch_sola(audio1[beginIndex : endIndex], bpm1/bpm2, bpm1, phase1)
	else:
		audio_stretched = audio1
	
	# Cross-fading
	result = audio_stretched
	
	# Write the result to a file
	# Output the marked file
	writer = MonoWriter(filename = 'test.wav')
	writer(result.astype('single')) #Only write fragment

	# Play the result
	from subprocess import call
	call(["mplayer", 'test.wav'])
	
	
		
	
	
