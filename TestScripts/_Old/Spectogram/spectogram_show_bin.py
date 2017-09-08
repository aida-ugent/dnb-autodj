''' Open .bin CQT feature file(s) and show it '''

import sys, os
import matplotlib.pyplot as plt
import numpy as np

filenames = sys.argv[1:]
if os.path.isdir(filenames[0]):
	directory = filenames[0]
	filenames = sorted([f for f in os.listdir(directory) if f[-4:] == '.bin'])
	print filenames
else:
	directory = ''
	
plt.figure()

numrows = 1+len(filenames)/10

for f,i in zip(filenames, range(len(filenames))):
	data = np.fromfile(os.path.join(directory,f)).reshape((119,84))
	plt.subplot((len(filenames)+1)/numrows, numrows, i+1)
	row_wise_mean = np.mean((data-np.min(data)) / (np.max(data)-np.min(data)), axis=1)
	row_wise_min = np.min((data-np.min(data)) / (np.max(data)-np.min(data)), axis=1)
	row_wise_max = np.max((data-np.min(data)) / (np.max(data)-np.min(data)), axis=1)
	#~ plt.imshow(data)
	plt.plot(row_wise_mean)
	plt.plot(row_wise_min)
	plt.plot(row_wise_max)
	plt.title(f[:4])
	

plt.show()
