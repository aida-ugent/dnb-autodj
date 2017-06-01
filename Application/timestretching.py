import numpy as np
from scipy import signal

def crossfade(audio1, audio2, length = None):
	'''Crossfade two audio clips, using linear fading by default.
	Assumes MONO input'''
	# TODO add checks that start1/start2 don't go out of bounds
	# TODO read about fading types
	if length is None:
		length = min(audio1.size, audio2.size)
	profile = ((np.arange(0.0, length)) / length)
	output = (audio1[:length] * profile[::-1]) + (audio2[:length] * profile)
	return output[:length]

def time_stretch_sola(audio, f, bpm, phase, sample_rate = 44100, fragment_s = 0.1, overlap_s = .020, seek_window_s = .015):
	# Assumes mono 44100 kHz audio (audio.size)
	
	if f == 1.0:
		return audio
	
	# Initialise time offsets and window lengths
	frame_len_1 = 4410								# Length of a fragment, including overlap at one side; about 100 ms, see above
	overlap_len = 252								# About 8 ms
	frame_len_2 = frame_len_1 + overlap_len			# Length of a fragment, including overlap at both sides
	frame_len_0 = frame_len_1 - overlap_len			# Length of a fragment, excluding overlaps (unmixed part)
	next_frame_offset_f =  frame_len_1 / f			# keep as a float to prevent rounding errors
	next_frame_offset = int(next_frame_offset_f) 	# keep as a float to prevent rounding errors
	seek_win_len_half = int(400)/2 					# window total ~ 21,666 ms

	def find_matching_frame(frame, theor_center):
		'''
		Find a frame in the neighbourhood of theor_center that maximizes the autocorrelation with the given frame as much as possible.
			
			:returns The start index in the given audio array of the most matching frame. Points to beginning of STABLE part. 
		'''
		# minus overlap_len because theor_start_frame is at the beginning of the constant part, but you have to convolve with possible intro overlap parts
		# minus len(frame) and not overlap_len to avoid errors when frame is at the very end of the input audio, and is not a full overlap part anymore
		cur_win_min = theor_center - seek_win_len_half
		cur_win_max = theor_center + seek_win_len_half
		correlation = signal.fftconvolve(audio[cur_win_min:cur_win_max], frame[::-1], mode='same') # Faster than np.correlate! cf http://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html
		return theor_center  + (np.argmax(correlation) - seek_win_len_half)

	# --------------Algorithm------------------
	# Initialise output buffer
	num_samples_out = int(f * audio.size)
	output = np.zeros(num_samples_out)		# f: stretch factor of audio!; prealloc

	num_frames_out = num_samples_out / frame_len_1
	in_ptr_th_f = 0.0
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
		# Method 2: pure WSOLA: match the next frame in the INPUT audio as closely as possible, since this frame is the natural follower of the original
		frame_to_match = audio[in_ptr + frame_len_0 : in_ptr + frame_len_0 + frame_len_1]
		match_ptr = find_matching_frame(frame_to_match, int(in_ptr_th_f + next_frame_offset_f) - overlap_len)
		
		frame1_overlap = audio[in_ptr + frame_len_0 : in_ptr + frame_len_1]
		frame2_overlap = audio[match_ptr : match_ptr + overlap_len]
		
		# Mix the overlap parts of the frames
		temp = crossfade(frame1_overlap, frame2_overlap)
		output[out_ptr + frame_len_0 : out_ptr + frame_len_0 + len(temp)] = temp
		
		# Increase the input pointers
		in_ptr = match_ptr + overlap_len
		in_ptr_th_f += next_frame_offset_f
		
	return np.array(output).astype('single')
