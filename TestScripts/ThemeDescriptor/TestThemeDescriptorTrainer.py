from song import Song
from songcollection import SongCollection
import sys
from essentia import *
from essentia.standard import *

from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np



if __name__ == '__main__':
	scaler = preprocessing.StandardScaler()
	pca = PCA(n_components=3)

	sc = SongCollection()
	for dir_ in sys.argv[1:]:
		sc.load_directory(dir_)

	pool = Pool()
	songs = []

	for song in sc.get_annotated():
		song.open()
		pool.add('themefeatures', song.spectral_contrast)
		songs.append(song.title)
		song.close()
		
	#~ # Test 1: spectral centroid on itself
	#~ X = pool['spectral_centroid']
	#~ plt.figure()  
	#~ plt.scatter(X[:,0],X[:,1])
	#~ for label, x, y in zip(songs, X[:, 0], X[:, 1]):
		#~ plt.annotate(
			#~ label,
			#~ xy=(x, y), xytext=(-20, 20),
			#~ textcoords='offset points', ha='right', va='bottom',
			#~ bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			#~ arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			#~ fontsize=7)
	   
	#~ plt.figure()     
	#~ plt.scatter(X[:,0],X[:,2])
	#~ for label, x, y in zip(songs, X[:, 0], X[:, 2]):
		#~ plt.annotate(
			#~ label,
			#~ xy=(x, y), xytext=(-20, 20),
			#~ textcoords='offset points', ha='right', va='bottom',
			#~ bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			#~ arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			#~ fontsize=7)
			
	#~ plt.figure()     
	#~ plt.scatter(X[:,1],X[:,2])
	#~ for label, x, y in zip(songs, X[:, 1], X[:, 2]):
		#~ plt.annotate(
			#~ label,
			#~ xy=(x, y), xytext=(-20, 20),
			#~ textcoords='offset points', ha='right', va='bottom',
			#~ bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			#~ arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			#~ fontsize=7)

	#~ plt.show()
	#~ #---------------------
		
	# Test 2: all features
	A = pool['themefeatures']
	plt.figure()
	A = scaler.fit_transform(A)
	plt.imshow(A,aspect='auto',interpolation='none')
	plt.show()
	X = pca.fit_transform(A)
	print pca.explained_variance_ 

	plt.figure()  
	plt.scatter(X[:,0],X[:,1])

	for label, x, y in zip(songs, X[:, 0], X[:, 1]):
		plt.annotate(
			label,
			xy=(x, y), xytext=(-20, 20),
			textcoords='offset points', ha='right', va='bottom',
			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			fontsize=7)
	   
	plt.figure()     
	plt.scatter(X[:,0],X[:,2])
	for label, x, y in zip(songs, X[:, 0], X[:, 2]):
		plt.annotate(
			label,
			xy=(x, y), xytext=(-20, 20),
			textcoords='offset points', ha='right', va='bottom',
			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			fontsize=7)
			
	plt.figure()     
	plt.scatter(X[:,2],X[:,1])
	for label, x, y in zip(songs, X[:, 2], X[:, 1]):
		plt.annotate(
			label,
			xy=(x, y), xytext=(-20, 20),
			textcoords='offset points', ha='right', va='bottom',
			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'),
			fontsize=7)
			
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(X[:,0],X[:,1],X[:,2])

	plt.show()


	# ----------------
	# Save the PCA model
	from sklearn.externals import joblib
	joblib.dump(pca, 'song_theme_pca_model_2.pkl') 
	joblib.dump(scaler, 'song_theme_scaler_2.pkl') 
