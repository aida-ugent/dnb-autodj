import numpy as np
import sys, os
import csv, time

'''
	This script evaluates if using the RMS onset detection function changes the prediction any file
	compared to when it used the 'complex' onset detection function.
'''

# Also see the default python example!
# Load a file using the command line
try:
	directory = sys.argv[1]
	if len(sys.argv) > 2:
		MIN_VALID_BPM = int(sys.argv[2])
		MAX_VALID_BPM = int(sys.argv[3])
	else:
		MIN_VALID_BPM = 150.0
		MAX_VALID_BPM = 190.0
		
except:
	print "usage:", sys.argv[0], "<directory>"
	sys.exit()

# Load the libraries
print 'Loading Essentia...'
from essentia import *
from essentia.standard import *
import matplotlib.pyplot as plt

# Initialize the algorithms
FRAME_SIZE = 1024
HOP_SIZE = 512
spec = Spectrum(size = FRAME_SIZE)
w = Windowing(type = 'hann')
fft = np.fft.fft
c2p = CartesianToPolar()

od = {
	'hfc' : OnsetDetection(method = 'hfc'),
	'rms' : OnsetDetection(method = 'rms'),
	'complex' : OnsetDetection(method = 'complex'),
	'melflux' : OnsetDetection(method = 'melflux')
}

phase_localmaxima_i = {
	'hfc' : [],
	'rms': [],
	'complex' : [],
	'melflux' : []
}

phase_localmaxima_val = {
	'hfc' : [],
	'rms': [],
	'complex' : [],
	'melflux' : []
}

plotFigures = False

def local_maxima(a):
	'''
		Returns the indices of the local maxima of the phase curve. Only maxima far enough apart are considered
	'''
	
	local_max_indexer = np.r_[a[0] > a[len(a)-1], a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], a[0] < a[len(a)-1]]
	indices_tmp = np.array(range(len(a)))
	indices_tmp = indices_tmp[local_max_indexer]
	indices = indices_tmp
	#~ indices = indices_tmp
	#~ for i in indices_tmp:
		#~ isFound = False
		#~ for j in indices:
			#~ dist1 = abs(j - i) 
			#~ dist2 = abs(j + len(a) - i)
			#~ dist3 = abs(i + len(a) - j)
			#~ dist = min(dist1, dist2, dist3)
			#~ if dist < len(a)/5:
				#~ isFound = True
				#~ break
		
		#~ if not isFound:
			#~ indices.append(i)
	
	values = a[indices]
	zipped = zip(indices, values)
	zipped.sort(key= lambda t: t[1])
	return zipped
	
timestr = '-' + time.strftime("%Y%m%d-%H%M%S")	
	
with open('./figures/evaluation_localmaxima'+timestr+'.csv', 'wb') as csvfile:
	csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	for f in os.listdir(directory):
		if f.endswith(".mp3") or f.endswith('.wav'):
			
			print '============== ', f , ' =================='
			loader = essentia.standard.MonoLoader(filename = directory + '/' + f)
			audio_orig = loader()
			lowpass = LowPass(cutoffFrequency=1500)
			audio = lowpass(audio_orig)
			
			csv_bpms, csv_phases, csv_times = [], [], []
			csv_peaks = {}
			
			i = 0
			pool = Pool()
			
			# Calculate framewise FFT
			for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
				pool.add('windowed_frames', w(frame))
			fft_result = fft(pool['windowed_frames']).astype('complex64')
			fft_result_mag = np.absolute(fft_result)
			fft_result_ang = np.angle(fft_result)
						
			for od_str, od_f in od.iteritems():
				#~ if os.path.exists('./figures/'+f+'.png'):
					#~ continue
					
				print od_str				

				error = False
				
				# ----------------------------- TIME FROM HERE
				start_time = time.time()
				
				# Calculate onset detection function on frames
				for mag,phase in zip(fft_result_mag, fft_result_ang):
					pool.add('onsets.'+od_str, od_f(mag, phase))

				# Subtract adaptive mean and HWR it
				def adaptive_mean(x, N):
					return np.convolve(x, [1.0]*int(N), mode='same')/N
					
				novelty_mean = adaptive_mean(pool['onsets.'+od_str], 16.0)
				novelty_hwr = (pool['onsets.'+od_str] - novelty_mean).clip(min=0)  

				# Calculate autocorrelation
				def autocorr(x):
					result = np.correlate(x, x, mode='full')
					return result[result.size/2:]

				novelty_autocorr = autocorr(novelty_hwr)

				# Apply shift-invariant comb filterbank
				valid_bpms = np.arange(MIN_VALID_BPM, MAX_VALID_BPM, 0.01)
				for bpm in valid_bpms:
					num_frames_per_beat = (60.0 * 44100.0)/(512.0 * bpm) 
					frames = (np.round(np.arange(0,np.size(novelty_autocorr),num_frames_per_beat)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
					pool.add('output.'+ od_str +'.bpm', np.sum(novelty_autocorr[frames])/np.size(frames))

				bpm = valid_bpms[np.argmax(pool['output.'+od_str+'.bpm'])]

				#~ if not (True): #or bpm_rms == bpm_csd and bpm_rms == bpm_melflux and bpm_rms == bpm_hfc):
					#~ print 'BPMs did not match!'
					#~ error = True
				#~ else:
					#~ print 'BPMs matched'

				# Calculate phase
				valid_phases = np.arange(0.0, 60.0/bpm, 0.001)
				num_frames_per_beat = (60.0 * 44100.0)/(512.0 * bpm)

				for phase in valid_phases:
					# Convert phase from seconds to frames
					phase_frames = (phase * 44100.0) / (512.0)
					frames = (np.round(np.arange(phase_frames,np.size(novelty_hwr),num_frames_per_beat)).astype('int'))[:-1] # Discard last value to prevent reading beyond array (last value rounded up for example)
					pool.add('output.'+od_str+'.phase', np.sum(novelty_hwr[frames])/np.size(frames))  

				phase = valid_phases[np.argmax(pool['output.' + od_str + '.phase'])]
				
				
				# --------------------- TIMER TILL HERE
				end_time = time.time()
				delta_time = end_time - start_time
				
				csv_bpms.append(bpm)
				csv_phases.append(phase)
				csv_times.append(delta_time)
				
				peaks = local_maxima(pool['output.' + od_str + '.phase'])
				
				if plotFigures:
					plt.figure(1)
					plt.subplot(2*len(od), 1, i + 1)
					plt.title('BPM ' + od_str)
					plt.plot(valid_bpms, pool['output.' + od_str + '.bpm'])
					plt.subplot(2*len(od), 1, len(od) + i + 1)
					plt.title('Phase ' + od_str)
					plt.plot(valid_phases, pool['output.' + od_str + '.phase'])
				
				for peak_i, peak_val in peaks:
					plt.plot(60./bpm * peak_i / float(len(valid_phases)), peak_val, marker='o', color='r')
					
				with open('./figures/evaluation_localmaxima_'+od_str+timestr+'.csv', 'a') as curfile:
					max_peak_i = np.argmax(pool['output.' + od_str + '.phase']) 
					max_peak_val = pool['output.' + od_str + '.phase'][max_peak_i]
					
					curcsvwriter = csv.writer(curfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					curcsvwriter.writerow(['#' + f])
					for peak_i, peak_val in peaks:
						curcsvwriter.writerow([len(valid_phases), max_peak_i, max_peak_val, peak_i, peak_val])
						
						dist = abs(peak_i - max_peak_i)
						dist = min(dist, len(valid_phases) - dist)						
						
						phase_localmaxima_i[od_str].append(float(dist) / len(valid_phases))
						phase_localmaxima_val[od_str].append(peak_val / max_peak_val)
						
				i = i+1
				
			csvwriter.writerow([f] + csv_bpms + csv_phases + csv_times)
			#~ csvwriter.writerow([f, bpm_hfc, phase_hfc])
			
			if plotFigures:
				#~ plt.show()
				plt.savefig('./figures/'+f+timestr+'.png')
				plt.close()
		
		# Plot this for all songs together		
	if True:
		for od_str, od_f in od.iteritems():
			plt.figure()
			plt.title(od_str)
			plt.scatter(phase_localmaxima_i[od_str], phase_localmaxima_val[od_str], marker = 'o')
			plt.xlim([-.01,.51])
			plt.ylim([-0.01,1.01])
			plt.savefig('./figures/phases-'+od_str+timestr+'.png')
			plt.close()

			#~ spb = 60./bpm #seconds per beat
			#~ beats_csd = (np.arange(phase_csd, (np.size(audio)/44100) - spb + phase_csd, spb).astype('single'))
			#~ beats_rms = (np.arange(phase_rms, (np.size(audio)/44100) - spb + phase_rms, spb).astype('single'))
			
			#~ # Overlay the audio file with onsets
			#~ onsetMarker = AudioOnsetsMarker(onsets = beats_csd)
			#~ onsetMarker2 = AudioOnsetsMarker(onsets = beats_rms)
			#~ audioMarked = onsetMarker2(onsetMarker(audio/2.))

			#~ # Stretch the result
			#~ #from librosa.effects import time_stretch
			#~ #audioMarked = time_stretch(audioMarked, 175./172.)

			#~ # Output the marked file
			#~ writer = MonoWriter(filename = 'test.wav')
			#~ beginIndex = 0.2*np.size(audioMarked)
			#~ endIndex = 0.5*np.size(audioMarked)
			#~ writer(audioMarked[beginIndex:endIndex]) #Only write fragment

			#~ # Play the result
			#~ from subprocess import call
			#~ call(["mplayer", 'test.wav'])
