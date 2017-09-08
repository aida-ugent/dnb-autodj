'''
	Visualizes the theme descriptor progression of a generated DJ mix.
	The theme descriptors of the successive songs are read from the xtestfile.csv file, which has been saved during generation.
	
	Usage:
		python TestThemeDescriptorMixAutodj.py ../music ../moremusic ../evenmoremusic ../music/test
		
	The files xtestfile.csv and xtestfile_goals.csv must be in the working directory.
	This file needs to be in the autodj/Applications folder to work, as it uses e.g. the Song and SongCollection class
'''

from song import Song
from songcollection import SongCollection
import sys

from essentia import *
from essentia.standard import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import csv

if __name__ == '__main__':
	
	name = 'xtestfile'
	sc = SongCollection()
	for dir_ in sys.argv[1:]:
		sc.load_directory(dir_)
	pool = Pool()
	
	for song in sc.get_annotated():
		song.open()
		pool.add('themes_allsongs', song.song_theme_descriptor.tolist()[0])
		song.close()
	
	with open(name+'.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			pool.add('themes_mix', [float(i) for i in line])
	with open(name+'_goals.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			pool.add('goals', [float(i) for i in line])
	
	X = pool['themes_allsongs']
	Y = pool['themes_mix']
	goals = pool['goals']
	
	for a,b in [(0,1),(0,2),(1,2)]:
		
		plt.figure()
		
		plt.scatter(goals[:,a],goals[:,b],marker='o',color='red',s=40,zorder=2)
		plt.scatter(X[:,a],X[:,b],marker='o',s=12,c='gray',lw=0)
		plt.scatter(Y[:,a],Y[:,b],marker='o',s=10,c='black')
		
		for x1, y1, x2, y2 in zip(Y[:-1, a], Y[:-1, b], Y[1:, a], Y[1:, b]):
			plt.arrow(x1, y1, x2-x1, y2-y1, fc="k", ec="k", head_width=0.01, head_length=0.01,lw=1)
			
		
	plt.show()		
