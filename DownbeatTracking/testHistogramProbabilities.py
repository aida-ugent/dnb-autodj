'''
	This testing script generates nice plots of the input file, with the trajectories plotted.
	Example usage:
	
	python testHistogramProbabilities ../music/somesong.mp3
'''

# Detect downbeats on every frame of the song and visualise where it made most mistakes

def entropy(log_loss_vector):
	return np.sum([-i*np.exp(i) for i in log_loss_vector])

if __name__ == '__main__':
	
	from sklearn.externals import joblib	# Model persistence
	from essentia import *
	from essentia.standard import MonoLoader
	import sys, os
	import numpy as np
	from downbeatModelTrainer import getFeaturesForFile, readDownbeatIndexFromFile
	
	import matplotlib.pyplot as plt
	
	# feature modules used in this model
	import featureLoudness, featureOnsetIntegral, featureOnsetIntegralCsd, featureOnsetIntegralHfc, featureChromagram, featureMFCC
	# Change feature modules here if desired
	feature_modules = [featureLoudness, featureMFCC, featureOnsetIntegral, featureOnsetIntegralCsd, featureOnsetIntegralHfc] 
	
	if len(sys.argv) <= 2:
		print 'Usage: ', sys.argv[0], ' <filename>'
	
	# Load the trained models
	model = joblib.load('model.pkl') 
	scaler = joblib.load('scaler.pkl')

	# Load the features corresponding to the audio
	directory, filename = os.path.split(sys.argv[1])
	directory = directory if directory[len(directory) - 1] == '/' else directory + '/'
	features, labels = getFeaturesForFile(directory, filename, feature_modules)
	features = scaler.transform(features)
	
	# Load the audio
	loader = MonoLoader(filename = sys.argv[1])
	audio = loader()
	
	# Do framewise predictions
	probas = model.predict_log_proba(features)
		
	sum_log_probas = np.array([[0,0,0,0]], dtype='float64')
	sums_log_probas = np.array([[0],[0],[0],[0]], dtype='float64')
	
	log_proba_hist_16 = np.zeros((16,4),dtype='float64')
	sum_log_proba_hist_16_plot = np.zeros((4,1), dtype='float64')
	
	permuted_row = [0] * 4
	framewise_predictions = np.array([[],[],[],[]], dtype='float64')
	correct_plot = []  		# Holds logloss value of correct downbeat
	predicted_plot = [] 	# Holds logloss value of predicted downbeat
	entropy_plot = []
	
	downbeatIndex = readDownbeatIndexFromFile(directory, os.path.splitext(filename)[0])
		
	for i, j, row in zip(range(len(probas)), np.array(range(len(probas))) % 4, probas):
		permuted_row[:4-j] = row[j:]
		permuted_row[4-j:] = row[:j] # element i of permuted_row (i = 0,1,2,3) corresponds to the TRAJECTORY over the song starting with downbeat 0, 1, 2, 3
		perm_row_np = np.array([permuted_row])
		sum_log_probas = sum_log_probas + permuted_row
		sums_log_probas = np.append(sums_log_probas, np.transpose(sum_log_probas)/(sums_log_probas.shape[1]), axis=1)
		framewise_predictions = np.append(framewise_predictions, np.transpose(perm_row_np), axis=1)	
		
		log_proba_hist_16[i%16] = permuted_row
		bla = np.reshape(np.sum(log_proba_hist_16, axis=0), (4,1))
		sum_log_proba_hist_16_plot = np.append(sum_log_proba_hist_16_plot, bla, axis=1)
		
		correct_plot.append(row[(j+(4-downbeatIndex)) % 4])
		predicted_plot.append(np.max(row))
		
		entropy_plot.append(entropy(row))
			
	# Framewise_predictions holds the predictions for each trajectory
	# => Plot each trajectory!
	# => Compute histogram
	
	print 4-downbeatIndex, np.argmax(sum_log_probas), sum_log_probas
	
	# Plot things
	#~ plt.figure(1)
	#~ plt.subplot(411)
	#~ plt.plot(framewise_predictions[0])
	#~ plt.subplot(412)
	#~ plt.plot(framewise_predictions[1])
	#~ plt.subplot(413)
	#~ plt.plot(framewise_predictions[2])
	#~ plt.subplot(414)
	#~ plt.plot(framewise_predictions[3])
	plt.figure(2)
	plt.subplot(211)
	plt.plot(audio[::441])
	plt.title('Audio waveform',fontsize=20)
	plt.xticks([])
	plt.yticks([])
	plt.subplot(212)
	plt.plot(framewise_predictions[0], linewidth=2)
	plt.plot(framewise_predictions[1], linewidth=2)
	plt.plot(framewise_predictions[2], linewidth=2)
	plt.plot(framewise_predictions[3], linewidth=2)
	plt.title('Log probabilities predicted classes vs. time',fontsize=20)
	plt.xticks([])
	plt.yticks([])
	plt.legend()
	plt.show()
	#~ plt.subplot(614)
	#~ plt.plot(sums_log_probas[0])
	#~ plt.plot(sums_log_probas[1])
	#~ plt.plot(sums_log_probas[2])
	#~ plt.plot(sums_log_probas[3])
	#~ plt.xticks([])
	#~ plt.title('Average log probabilities vs. time')
	#~ plt.subplot(615)
	#~ plt.plot(correct_plot, label='Correct path')
	#~ plt.plot(predicted_plot, label='Predicted labels (highest instant class probabilities)')
	#~ plt.legend(loc='lower right')
	#~ plt.title('Correct path vs. predicted labels')
	#~ plt.xticks([])
	#~ plt.subplot(613)
	#~ plt.plot(sum_log_proba_hist_16_plot[0], label='Trajectory 0')
	#~ plt.plot(sum_log_proba_hist_16_plot[1], label='Trajectory 1')
	#~ plt.plot(sum_log_proba_hist_16_plot[2], label='Trajectory 2')
	#~ plt.plot(sum_log_proba_hist_16_plot[3], label='Trajectory 3')
	#~ plt.legend()
	#~ plt.title('Running sum of log probabilities over 16 frames')
	#~ plt.xticks([])
	#~ plt.subplot(616)
	#~ plt.plot(entropy_plot)
	#~ plt.xticks([])
	#~ plt.title('Entropy of predicted probability vector')
	#~ plt.show()
