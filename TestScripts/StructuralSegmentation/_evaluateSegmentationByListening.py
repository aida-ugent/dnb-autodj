from songcollection import SongCollection
import pyaudio, csv
import numpy as np
from essentia.standard import AudioOnsetsMarker

if __name__ == '__main__':
	
	# Open the long library
	sc = SongCollection()
	sc.load_directory('../moremusic/')
	
	analysedSongs = []
	with open('evaluateMoresongs.csv', 'a+') as csvfile:
		for row in csv.reader(csvfile):
			analysedSongs.append(row[0])
		
	print analysedSongs
	
	with open('evaluateMoresongs.csv', 'a') as csvfile:
		writer = csv.writer(csvfile)
		p = pyaudio.PyAudio()
		stream = p.open(format = pyaudio.paFloat32,channels=1,rate=44100,output=True)
		
		if len(analysedSongs) == 0:
			initrow = ['Song title', 'Segment index (downbeats)', 'Is aligned', 'Is low-to-high']
			writer.writerow(initrow)
		
		for song in sc.songs:
			
			if song.title in analysedSongs:
				continue

			print song.title
			song.open()
			song.openAudio()
			indices = song.segment_indices
			types = song.segment_types
			lowhighsegments = [j + 1 for j in range(len(indices) - 1) if types[j] == 'L' and types[j+1] == 'H']
			lowhighsegments = [lowhighsegments[0]] # Only evaluate one transition for now, comment if more are wanted
			
			#~ commands = ['r','y','n']
			commands = ['r','y','b','d','s','z']
			cmd = 'r'
			
			for idx in lowhighsegments:
				
				row = [song.title]
				
				cmd = 'r'
				while(cmd == 'r'):
					
					# Play the audio for evaluation					
					START_BEFORE = 2
					start_db = song.downbeats[max(0, indices[idx]-START_BEFORE)]
					end_db = song.downbeats[min(len(song.downbeats)-1, indices[idx] + 2)]	
					print start_db, end_db, song.tempo			
					dbeats = np.arange(0,end_db - start_db,4*60.0/song.tempo).astype('float32')
					onsetsMarker = AudioOnsetsMarker(onsets = 1.0 * dbeats[START_BEFORE::8])	
					toPlay = onsetsMarker(song.audio[int(start_db * 44100):int(end_db * 44100)])
					beats = np.arange(0,end_db - start_db,60.0/song.tempo).astype('float32')
					onsetsMarker2 = AudioOnsetsMarker(onsets = 1.0 * beats, type='noise')	
					toPlay = onsetsMarker2(onsetsMarker(song.audio[int(start_db * 44100):int(end_db * 44100)]))
					stream.write(toPlay, num_frames=len(toPlay[::]), exception_on_underflow=False)
					
					cmd = 'blub'
					while cmd not in commands:
						cmd = raw_input('Correct annotation? y:yes b:beat wrong d:dbeat wrong s:segment wrong z:seg correct, wrong offset')
					if cmd != 'r':
						row.append(indices[idx])
						row.append(cmd)
					#~ if cmd == 'y':
						#~ # correct
						#~ row.append('1')
					#~ elif cmd == 'n':
						#~ row.append('0')
					
					
					#~ # Is the segment aligned?
					#~ cmd = 'blub'
					#~ while cmd not in commands:
						#~ cmd = raw_input('Aligned with boundary? y:1 n:0 r:redo')
					#~ row.append(indices[idx])
					#~ if cmd == 'y':
						#~ # correct
						#~ row.append('1')
					#~ elif cmd == 'n':
						#~ row.append('0')
						
					#~ # Is the segment a L->H segment?
					#~ cmd = 'blub'
					#~ while cmd not in commands:
						#~ cmd = raw_input('L->H? y:1 n:0 r:redo')
					#~ if cmd == 'y':
						#~ # correct
						#~ row.append('1')
					#~ elif cmd == 'n':
						#~ row.append('0')
						
				writer.writerow(row)
				
				
				
					
