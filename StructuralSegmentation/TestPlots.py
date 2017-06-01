'''
	Initial test script for structural segmentation
'''


from essentia import *
from essentia.standard import *
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from skimage import filters
import numpy as np
import scipy
import pyaudio

from sklearn.metrics.pairwise import cosine_similarity

import sys, os
import matplotlib.pyplot as plt

ANNOT_SUBDIR = '_annot_beat_downbeat/'
ANNOT_DOWNB_PREFIX = 'downbeats_'
ANNOT_BEATS_PREFIX = 'beats_'

def pathAnnotationFile(directory, song_title, prefix):
	return os.path.join(directory, ANNOT_SUBDIR, prefix + song_title + '.txt')

def loadAnnotationFile(directory, song_title, prefix):
	'''
	Loads an input file with annotated times in seconds.
	
	-Returns: A numpy array with the annotated times parsed from the file.
	'''
	input_file = pathAnnotationFile(directory, song_title, prefix)
	result = []
	result_dict = {}
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line[0] == '#':
					try:
						key, value = str.split(line[1:], ' ')
						result_dict[key] = float(value)
					except ValueError:
						# "too many values to unpack" because it's a normal comment line
						pass
				else:
					result.append(float(line))	
	else:
		raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
	return result, result_dict
	
def writeAnnotFile(directory, song_title, prefix, array, values_dict = {}):
	
	output_file = pathAnnotationFile(directory, song_title, prefix)	
	with open(output_file, 'w+') as f:
		# Write the dict
		for key, value in values_dict.iteritems():
			f.write('#' + str(key) + ' ' + '{:.9f}'.format(value) + '\n')
		# Write the annotations
		for value in array:
			f.write("{:.9f}".format(value) + '\n')

class Song:
	
	def __init__(self, path_to_file):
		
		self.dir_, self.title = os.path.split(os.path.abspath(path_to_file))
		self.title, self.extension = os.path.splitext(self.title)
		self.dir_annot = self.dir_ + '/' + ANNOT_SUBDIR
		
		if not os.path.isdir(self.dir_annot):
			print 'Creating annotation directory : ' + self.dir_annot
			os.mkdir(self.dir_annot)
		
		self.audio = None
		self.beats = None
		self.bpm = None
		self.downbeats = None
		self.queuepts = None
		
	def hasBeatAnnot(self):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX))
		
	def hasDownbeatAnnot(self):
		return os.path.isfile(pathAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX))
		
	def hasAllAnnot(self):
		'''
		Check if this file has annotation files
		'''
		hasSegmentationAnnot = True # TODO implement
		return self.hasBeatAnnot() and self.hasDownbeatAnnot() and hasSegmentationAnnot
		
	def annotate(self):
		#TODO for now, this doesn't store the annotations and audio in memory yet: happens only on load time
		
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		audio = loader()
		
		# Beat annotations
		if not self.hasBeatAnnot():
			btracker = BeatTracker()
			btracker.run(audio)
			beats = btracker.getBeats()
			tempo = btracker.getBpm()
			phase = btracker.getPhase()
			writeAnnotFile(self.dir_, self.title, ANNOT_BEATS_PREFIX, beats, {'tempo' : tempo , 'phase' : phase})
		else:
			beats, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX)
			
		# Downbeat annotations
		if not self.hasDownbeatAnnot():
			dbtracker = DownbeatTracker()
			downbeats = dbtracker.track(audio, beats)
			writeAnnotFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX, downbeats)
		else:
			downbeats, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX)
		
	# Open the audio file and read the annotations
	def open(self):
		loader = MonoLoader(filename = os.path.join(self.dir_, self.title + self.extension))
		self.audio = loader()		
		self.beats, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_BEATS_PREFIX)
		self.tempo = res_dict['bpm']
		self.phase = res_dict['phase']
		self.downbeats, res_dict = loadAnnotationFile(self.dir_, self.title, ANNOT_DOWNB_PREFIX)	
		print 'Opened ' + str(self.title) + '; tempo: ' + str(self.tempo)
		
	# Close the audio file and reset all buffers to None
	def close(self):
		self.audio = None
		self.beats = None
		self.downbeats = None
		self.queuepts = None 

def calculateCheckerboardCorrelation(matrix, N):
	
	M = min(matrix.shape[0], matrix.shape[1])
	result = np.zeros(M)
	
	u1 = scipy.signal.gaussian(2*N, std=N/2.0).reshape((2*N,1))
	u2 = scipy.signal.gaussian(2*N, std=N/2.0).reshape((2*N,1))
	U = np.dot(u1,np.transpose(u2))
	U[:N,N:] *= -1
	U[N:,:N] *= -1
	
	matrix_padded = np.pad(matrix, N, mode='edge')
	
	for index in range(N, N+M):
		submatrix = matrix_padded[index-N:index+N, index-N:index+N]
		result[index-N] = np.sum(submatrix * U)
	return result

def adaptive_mean(x, N):
	return np.convolve(x, [1.0]*int(N), mode='same')/N
	
'''
Construct self-similarity matrices with bar-length frames.
Do this four times, i.e. once for every candidate downbeat candidate
'''

if len(sys.argv) != 2 and len(sys.argv) != 3:
	print 'Usage : ', sys.argv[0], ' <filename> [<start_segment>]'
	exit()

filename = sys.argv[1]	
# Load the audio
song = Song(filename)

# Beat tracking
print 'Loading beat and downbeat information...'
song.open()
print song.title, song.tempo, song.phase
audio = song.audio

# Initializing algorithms
pool = Pool()
w = Windowing(type = 'hann')
mfcc = MFCC()
hpcp = HPCP()
speaks = SpectralPeaks()
spectrum = Spectrum()

# Cut the audio so it starts at a (guessed) downbeat
first_downbeat_sample = int(44100 * song.downbeats[0])
audio = audio[ first_downbeat_sample : ]

# MFCC self-similarity matrix and novelty curve
# TODO: These features are also calculated for beat tracking -> can be optimized
FRAME_SIZE = int(44100 * (60.0 / song.tempo) / 2)
HOP_SIZE = FRAME_SIZE / 2
for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
	mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:FRAME_SIZE-(FRAME_SIZE % 2)])))
	pool.add('lowlevel.mfcc', mfcc_coeffs)
	pool.add('lowlevel.mfcc_bands', mfcc_bands)

selfsim_mfcc = cosine_similarity(np.array(pool['lowlevel.mfcc']), np.array(pool['lowlevel.mfcc']))
selfsim_mfcc -= np.average(selfsim_mfcc)
selfsim_mfcc *= (1.0 / np.max(selfsim_mfcc))
novelty_mfcc = calculateCheckerboardCorrelation(selfsim_mfcc, N = 32)
novelty_mfcc *= 1.0/np.max(novelty_mfcc)

# RMS self-similarity matrix and novelty curve
for frame in FrameGenerator(audio, frameSize = FRAME_SIZE, hopSize = HOP_SIZE):
	pool.add('lowlevel.rms', np.average(frame**2))

selfsim_rms = pairwise_distances(pool['lowlevel.rms'].reshape(-1, 1))
selfsim_rms -= np.average(selfsim_rms)
selfsim_rms *= (1.0 / np.max(selfsim_rms))
novelty_rms = np.abs(calculateCheckerboardCorrelation(selfsim_rms, N = 32))
novelty_rms *= 1.0/np.max(np.abs(novelty_rms))

novelty_product = novelty_rms * novelty_mfcc
novelty_product = [i if i > 0 else 0 for i in novelty_product]
novelty_product = np.sqrt(novelty_product)

# Peak picking
# Find the most dominant peaks in the product signal
# Product-signal accentuates peaks that are present in both signals at the same time
peaks_absmax_i = np.argmax(novelty_product)	# The absolute maximum
peaks_absmax = novelty_product[peaks_absmax_i]
threshold = peaks_absmax * 0.05	# Detect peaks at least half as high
# Detect other peaks
peakDetection = PeakDetection(interpolate=False, maxPeaks=100, orderBy='amplitude', range=len(novelty_product), maxPosition=len(novelty_product), threshold=threshold)
peaks_pos, peaks_ampl = peakDetection(novelty_product.astype('single'))
peaks_ampl = peaks_ampl[np.argsort(peaks_pos)]
peaks_pos = peaks_pos[np.argsort(peaks_pos)]

# Filter the peaks
# Shift the peaks that are in window (-delta * downbeatLength, delta * downbeatLength) to the peak in the center of that interval
# Peaks that are not within that interval are removed as false positives
peaks_pos_modified, peaks_ampl_modified = [], []
peaks_pos_dbindex = []

peak_idx = 0
peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
num_filtered_out = 0

downbeat_len_s = 4 * 60.0 / song.tempo
delta = 0.4
print len(peaks_pos), peaks_pos[-1] * HOP_SIZE / float(len(audio))
for dbindex, downbeat in zip(range(len(song.downbeats)), np.array(song.downbeats) - song.downbeats[0]):
	# Skip the peaks prior to the acceptance interval
	while peak_cur_s < downbeat - delta * downbeat_len_s and peak_idx < len(peaks_pos):
		num_filtered_out += 1
		peak_idx += 1
		if peak_idx != len(peaks_pos):
			peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
	if peak_idx == len(peaks_pos):
		break
	# Adjust the peaks within the acceptance interval
	while peak_cur_s < downbeat + delta * downbeat_len_s and peak_idx < len(peaks_pos):
		peak_newpos = int(downbeat * 44100.0 / HOP_SIZE)	# seconds to frames
		peaks_pos_modified.append(peak_newpos)
		peaks_ampl_modified.append(peaks_ampl[peak_idx])
		peaks_pos_dbindex.append(dbindex)
		
		peak_idx += 1
		if peak_idx != len(peaks_pos):
			peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
	if peak_idx == len(peaks_pos):
		break
		
peaks_pos_modified, peaks_ampl_modified = np.array(peaks_pos_modified), np.array(peaks_ampl_modified)
peaks_pos_dbindex = np.array(peaks_pos_dbindex)

print 'Number of peaks filtered out: ', num_filtered_out

# Determine the most dominant peaks and see if they lie at a multiple of 8 downbeats from each 
# Assumption 1: high peaks are important; assumption 2: they should lie at multiples of 8 downbeats (phrase) from each other
NUM_HIGHEST_PEAKS = 10
highest_peaks_db_indices = (peaks_pos_dbindex[np.argsort(peaks_ampl_modified)])[-NUM_HIGHEST_PEAKS:]
distances = []		# Number of total peaks at multiple of 4
distances_high = [] # Number of high peaks at multiple of 4 for each of the highest 
distances8 = []		# Number of total peaks at multiple of 8
distances8_high = [] # Number of high peaks at multiple of 8 for each of the highest 

#~ # First look for the offset in 4 downbeats where the segments lie, as they are always at a multiple of 4 from each other
#~ for i in range(4): #highest_peaks_db_indices:
	#~ distances.append(len( [p for p in peaks_pos_dbindex if (p - i) % 4 == 0] ))
	#~ distances_high.append(len( [p for p in highest_peaks_db_indices if (p - i) % 4 == 0] ))
#~ for i in range(8): #highest_peaks_db_indices:
	#~ distances8.append(len( [p for p in peaks_pos_dbindex if (p - i) % 8 == 0] ))
	#~ distances8_high.append(len( [p for p in highest_peaks_db_indices if (p - i) % 8 == 0] ))
#~ most_likely_4db_index = np.argmax(distances)
#~ if most_likely_4db_index != np.argmax(distances_high):
	#~ raise Exception('Cannot determine most likely 4-downbeat segment index')
	#~ print ('Cannot determine most likely 4-downbeat segment index')
#~ print distances, distances_high
#~ print distances8, distances8_high
#~ most_likely_8db_index = most_likely_4db_index if np.argmax(distances8_high) == most_likely_4db_index else most_likely_4db_index + 4


# First look for the offset in 4 downbeats where the segments lie, as they are always at a multiple of 4 from each other
for i in range(8): #highest_peaks_db_indices:
	distances8.append(len( [p for p in peaks_pos_dbindex if (p - i) % 8 == 0] ))
	distances8_high.append(len( [p for p in highest_peaks_db_indices if (p - i) % 8 == 0] ))

# For the positions where the highest downbeats are detected, determine which one has the most alignments
# Discard the peaks where no highest peak has been detected
print distances8 * np.array([p / sum(distances8_high) if p > 0 else 0 for p in distances8_high])
most_likely_8db_index = np.argmax(distances8 * np.array([float(p) / sum(distances8_high) if p > 0 else 0 for p in distances8_high]))
print distances, distances_high
print distances8, distances8_high
print most_likely_8db_index

last_downbeat = song.downbeats[-1]
print 'Total length in downbeats: ' + str(last_downbeat)
segment_indices = [most_likely_8db_index if most_likely_8db_index <= 4 else most_likely_8db_index - 8] 	# Always have the start of the song as a likely db index
segment_indices.extend([db for db in highest_peaks_db_indices if (db - most_likely_8db_index) % 8 == 0]) 	# Also have all important segments in there 
segment_indices.extend([db+1 for db in highest_peaks_db_indices if (db + 1 - most_likely_8db_index) % 8 == 0])
segment_indices.extend([db+2 for db in highest_peaks_db_indices if (db + 2 - most_likely_8db_index) % 8 == 0])
segment_indices.extend([db-1 for db in highest_peaks_db_indices if (db - 1 - most_likely_8db_index) % 8 == 0])
segment_indices.extend([db-2 for db in highest_peaks_db_indices if (db - 2 - most_likely_8db_index) % 8 == 0])
segment_indices = np.unique(sorted(segment_indices))

# TODO extend with segment +- 16, so that segments are approximately equally spaced? Maybe less segments needed in the middle of the high part though
# e.g. only add these segments if they represent a H->L, L->H or L->L

# Determine the type of segment boundary: HL, LH, same
adaptive_mean_rms = adaptive_mean(pool['lowlevel.rms'], 64) # Mean of rms in window of [-4 dbeats, + 4 dbeats]
mean_rms = np.mean(adaptive_mean_rms)
segment_types = []

def getSegmentType(dbindex):
	before_index = int((dbindex - 4) * 4 * 60.0/song.tempo * 44100.0/HOP_SIZE)	# 2 downbeats before the fade
	after_index = int((dbindex + 4) * 4 * 60.0/song.tempo * 44100.0/HOP_SIZE)		# 2 downbeats after the fade
	rms_before = adaptive_mean_rms[before_index] / mean_rms
	rms_after = adaptive_mean_rms[after_index] / mean_rms
	return 'L' if rms_after < 1.0 else 'H'			

for segment in segment_indices:
	segment_types.append(getSegmentType(segment))
	
# Add more segments in between existing ones if the distance is too small
additional_segment_indices = []
additional_segment_types = []
for i in range(len(segment_indices) - 1):
	if segment_indices[i+1] - segment_indices[i] >= 32:				# Segments are too far apart
		previous_type = segment_types[i]
		for offset in range(16,segment_indices[i+1] - segment_indices[i],16): 	# 16, 32, 48, ..., distance - 16
			if getSegmentType(segment_indices[i] + offset) != previous_type:	# This offset introduces a new type of segment
				additional_segment_indices.append(segment_indices[i] + offset)
				previous_type = 'H' if previous_type == 'L' else 'H'
				additional_segment_types.append(previous_type)
				
segment_indices = np.append(segment_indices, additional_segment_indices)
segment_types = np.append(segment_types, additional_segment_types)
permutation = np.argsort(segment_indices)
segment_indices = segment_indices[permutation].astype('int')
segment_types = segment_types[permutation]

print (segment_indices)
print (segment_types)

# TODO use rhythm self-similarity to infer where parts are repeated, and attempt to repeat the locations of the segments in repeating parts

# Determine the most likely downbeat position that corresponds to the 

# -------- recurrence plot
recurrencePlot = False
if recurrencePlot:
	
	# --- HPCP features	
	FRAME_SIZE_2 = int(44100 * (60.0 / song.tempo) / 2)
	print 44100 * (60.0 / song.tempo) / 2
	HOP_SIZE_2 = FRAME_SIZE_2 / 2
	for frame in FrameGenerator(audio, frameSize = FRAME_SIZE_2, hopSize = HOP_SIZE_2):
		freqs, mags = speaks(spectrum(w(frame[:FRAME_SIZE_2-(FRAME_SIZE_2 % 2)])))
		mfcc_coeffs = hpcp(freqs, mags)
		#~ mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:FRAME_SIZE-(FRAME_SIZE % 2)])))
		pool.add('lowlevel.mfcc_quarternote', mfcc_coeffs)

	num_concat_notes = 32 # One downbeat of repetition info at quarter note level
	num_samples, num_features = pool['lowlevel.mfcc_quarternote'].shape

	qnote_features = pool['lowlevel.mfcc_quarternote']
	extended_features = qnote_features[:num_samples-num_concat_notes]
	e_length_fraction = float(num_samples - num_concat_notes) / float(num_samples)
	for i in range(1, num_concat_notes):
		extended_features = np.append(extended_features, qnote_features[i:num_samples - num_concat_notes + i], axis=1)

	R_temp = pairwise_distances(extended_features, metric = 'euclidean')
	
	e = calculateCheckerboardCorrelation(R_temp, 32)
	e *= 1.0/np.max(np.abs(e))
	
	# Find nearest neighbours
	#TODO Enforce that 'similar' segments are downbeat-aligned
	neighbrs = np.zeros(R_temp.shape)	# neighbrs[i,j] == 1 if i is a nearest neighbor of j
	num_nearest_neighbors = 25
	for i in range(R_temp.shape[0]):
		neighbrs_indices = np.argsort(R_temp[i,:])[:num_nearest_neighbors]
		neighbrs[i,neighbrs_indices] = 1
	R = neighbrs * np.transpose(neighbrs)	# R[i,j] == 1 if i is a nearest neighbor of j and vice versa
	# Create a gaussian kernel
	g1 = scipy.signal.gaussian(87, std=87.0/4).reshape((87,1))
	g2 = scipy.signal.gaussian(3, std=3.0/4).reshape((3,1))
	G = np.dot(g1,np.transpose(g2))
	# Rotate rows of R to construct structure features
	L = R
	for i in range(R.shape[0]):
		L[i,:] = np.roll(R[i,:], -i)
	L = scipy.signal.convolve2d(L, G, mode='same')

	d = np.linalg.norm(L[1:,:] - L[:-1,:], axis=1)
	d -= np.min(d)
	d *= (1.0/np.max(d))


plt.figure()
plt.imshow(selfsim_mfcc, aspect='auto', interpolation='none', vmin=-3.5, vmax=np.max(selfsim_mfcc))
range_in_sec = np.linspace(0, len(audio) / 44100.0, selfsim_mfcc.shape[0])
len_song_sec = len(audio) / 44100
xticks = np.linspace(0,selfsim_mfcc.shape[0], len_song_sec / 20)
plt.xticks(xticks, range(0,len_song_sec, 20), fontsize='large')
plt.yticks(xticks, range(0,len_song_sec, 20), fontsize='large')
plt.xlabel('Time (s)',fontsize=16)
plt.ylabel('Time (s)',fontsize=16)
ax = plt.gca()
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.show()


plt.figure()
plt.subplot(221)
plt.imshow(selfsim_mfcc, aspect='auto', interpolation='none', vmin=-1, vmax=np.max(selfsim_mfcc))
plt.subplot(222)
plt.imshow(selfsim_rms, aspect='auto', interpolation='none')
plt.subplot(223)
if recurrencePlot:
	plt.imshow(L, aspect='auto', cmap='gray')
plt.subplot(224)
if recurrencePlot:
	plt.imshow(R_temp, aspect='auto', interpolation='none')

plt.figure()
x_a = np.linspace(0,1,len(novelty_mfcc))
x_b = np.linspace(0,1,len(novelty_rms))
x_audio = np.linspace(0,1,len(audio[::441]))
x_rms = np.linspace(0,1,len(pool['lowlevel.rms']))
x_beats = np.linspace(0,1,len(audio)/100)
if recurrencePlot:
	x_e = np.linspace(0,e_length_fraction, len(e))

# Plot the audio
plt.plot(x_rms, adaptive_mean_rms, color='blue', alpha=0.5)
# Plot the downbeats

dbeats = (np.array(song.downbeats) - song.downbeats[0])
print dbeats[0]
dbeats = dbeats * 44100
for i, beat in zip(range(len(dbeats)), dbeats):
	plt.axvline(x_beats[int(beat)/100], linewidth = 1, c='black')
for i, beat in zip(range(len(dbeats)/4), dbeats[::4]):
	plt.axvline(x_beats[int(beat)/100], linewidth = 2, c='black')

plt.plot(x_a, novelty_mfcc, color='red', linewidth=2)
plt.plot(x_b, novelty_rms, color='black', linewidth=2)
plt.plot(x_b, novelty_product, color='orange', linewidth=2)
#~ for peak in peaks_pos_modified:
	#~ plt.axvline(x_b[peak], linewidth = 1, c='orange')
for peak in dbeats[peaks_pos_dbindex]:
	plt.axvline(x_beats[int(peak/100)], linewidth = 1, c='orange')

for peak in dbeats[highest_peaks_db_indices]:
	plt.axvline(x_beats[int(peak/100)], linewidth=1, c='red')
for peak in dbeats[segment_indices]:
	plt.axvline(x_beats[int(peak/100)], linewidth=3, c='red')
	

if recurrencePlot:
	print e_length_fraction
	plt.plot(np.linspace(0,e_length_fraction,len(d)), -d, color='black', linewidth=2)
	plt.plot(x_e, e, color='red', linewidth=2)
	

# Play the song
# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format = pyaudio.paFloat32,
				channels=1,
				rate=44100,
				output=True)
				
try:
	seg_to_play = int(sys.argv[2])
except:
	seg_to_play = 2
toPlay = audio[int(dbeats[segment_indices[seg_to_play]]):int(dbeats[min(len(dbeats)-1, segment_indices[seg_to_play] + 8)])]
stream.write(toPlay, num_frames=len(toPlay), exception_on_underflow=True)
	
plt.show()
