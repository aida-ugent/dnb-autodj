''' Simple test script to mess with shelf filtering '''

import os, sys, random
from essentia import *
from essentia.standard import *
import yodel.filter
import pyaudio
import numpy as np

import scipy.signal


if __name__ == '__main__':
	
	filepath = sys.argv[1]
	essentia.log.infoActive = False
	loader = essentia.standard.MonoLoader(filename = filepath)
	audio = loader()
	
	SAMPLE_RATE = 44100
	LOW_CUTOFF = 70
	MID_CENTER = 1000
	HIGH_CUTOFF = 13000
	Q = 1.0 / np.sqrt(2)
	
	start1 = int(sys.argv[2])
	start2 = int(sys.argv[3])
	start3 = int(sys.argv[4])
	start4 = int(sys.argv[5])
	
	lowpass = yodel.filter.Biquad()
	lowpass.low_shelf(SAMPLE_RATE, LOW_CUTOFF, 1, -26)
	highpass = yodel.filter.Biquad()
	highpass.high_shelf(SAMPLE_RATE, HIGH_CUTOFF, 1, -26)
	
	output_audio = np.zeros(44100*(start3-start2))
	output_audio_2 = np.zeros(44100*(start4-start3))
	print 
	
	print 'Filtering...'
	
	NUM_STEPS = 20
	
	
	profile = np.append((np.arange(0, NUM_STEPS/2) / float(NUM_STEPS/2)), (NUM_STEPS/2 * [1]))
	print profile
	
	
	first_idx = int(44100 * start2)
	for i in range(NUM_STEPS):
		start_idx = int(44100 * (start2 + (i / float(NUM_STEPS))*(start3 - start2)))
		end_idx = int(44100 * (start2 + ((i + 1) / float(NUM_STEPS))*(start3 - start2)))
		lowpass.low_shelf(SAMPLE_RATE, LOW_CUTOFF, Q, -int(26 * profile[i]))
		#~ lowpass.process(audio[start_idx:end_idx], output_audio[start_idx - first_idx:end_idx - first_idx]) 
		b = lowpass._b_coeffs
		a = lowpass._a_coeffs
		a[0] = 1.0 # This is already done in the yodel object, but a[0] is never reset to 1.0 after division!
		output_audio[start_idx - first_idx:end_idx - first_idx] = scipy.signal.lfilter(b, a, audio[start_idx : end_idx]).astype('float32')
		
	first_idx_2 = int(44100 * start3)
	for i in range(NUM_STEPS):
		start_idx = int(44100 * (start3 + (i / float(NUM_STEPS))*(start4 - start3)))
		end_idx = int(44100 * (start3 + ((i + 1) / float(NUM_STEPS))*(start4 - start3)))
		highpass.high_shelf(SAMPLE_RATE, HIGH_CUTOFF, Q, -int(26 * profile[i]))
		#~ highpass.process(audio[start_idx:end_idx], output_audio_2[start_idx - first_idx_2:end_idx - first_idx_2]) 
		b = highpass._b_coeffs
		a = highpass._a_coeffs
		a[0] = 1.0 # This is already done in the yodel object, but a[0] is never reset to 1.0 after division!
		output_audio_2[start_idx - first_idx_2:end_idx - first_idx_2] = scipy.signal.lfilter(b, a, audio[start_idx : end_idx]).astype('float32')
	
	#~ highpass.process(audio[44100*start2:44100*start3], output_audio)
	
	output_audio = output_audio.astype('float32')
	output_audio_2 = output_audio_2.astype('float32')
	print 'Filtered'
	
	print output_audio.dtype
	
	p = pyaudio.PyAudio()
	stream = p.open(format = pyaudio.paFloat32,
					channels=1,
					rate=44100,
					output=True)
	
	print 'Normal playing'
	stream.write(audio[44100*start1:44100*start2], num_frames=44100*(start2-start1), exception_on_underflow=True)
	print 'Low cutoff playing'
	stream.write(output_audio, num_frames=len(output_audio), exception_on_underflow=True)
	print 'High cutoff playing'
	stream.write(output_audio_2, num_frames=len(output_audio_2), exception_on_underflow=True)
	
	stream.stop_stream()
	stream.close()

	# close PyAudio (5)
	p.terminate()

		
