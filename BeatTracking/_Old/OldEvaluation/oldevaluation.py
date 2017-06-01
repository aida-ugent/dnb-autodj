''' Generates the figure that shows the inaccuracies created by e.g. BeatRoot (see dissertation chapter BeatTracking)

Example usage: 
python oldevaluation.py Bensley\ -\ Tiptoe.wav bla.csv beats_Bensley\ -\ Tiptoe.txt

with the two txt or csv files the two beat files to compare.

You might need to zoom in on the X-axis to see the results instead of loads of red and black stripes
'''

print 'Loading Essentia...'
import essentia
from essentia.standard import *
import matplotlib.pyplot as plt # For plotting
import sys, csv, os

audiofile = sys.argv[1]
file1 = sys.argv[2]
file2 = sys.argv[3]

def getBeatsFromFile(input_file):
	result = []
	if os.path.exists(input_file):
		with open(input_file) as f:
			for line in f:
				if line[0] == '#':
					continue
				result.append(float(line))
	return result

# Load the audio
print 'Loading audio file "', audiofile, '" ...'
loader = essentia.standard.MonoLoader(filename = audiofile)
audio = loader()

plt.plot(audio[::441], color='blue', ls='-')

# Load the correct annotations
beats1 = getBeatsFromFile(file1)
beats2 = getBeatsFromFile(file2)

for b in beats1:
	plt.axvline(b * 100, color='black', ls='-', lw=2.0)
for b in beats2:
	plt.axvline(b * 100, color='red', ls='--', lw=2.0)

plt.show()
