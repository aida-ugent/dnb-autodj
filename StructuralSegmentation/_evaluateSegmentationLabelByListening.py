from songcollection import SongCollection
import pyaudio, csv
import numpy as np
from essentia.standard import AudioOnsetsMarker

''' Evaluate if L->H segments are indeed drops or not'''

if __name__ == '__main__':
	
	# Open the long library
	sc = SongCollection()
	sc.load_directory('../music/')
	sc.load_directory('../music/test/')
	
	analysedSongs = []
	with open('evaluateSegmentLH.csv', 'a+') as csvfile:
		for row in csv.reader(csvfile):
			analysedSongs.append(row[0])
		
	print analysedSongs
	
	with open('evaluateSegmentLH.csv', 'a') as csvfile:
		writer = csv.writer(csvfile)
		p = pyaudio.PyAudio()
		stream = p.open(format = pyaudio.paFloat32,channels=1,rate=44100,output=True)
		
		if len(analysedSongs) == 0:
			initrow = ['Song title', 'Segment index (downbeats)', 'Is low-to-high']
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
			
			commands = ['r','y','n']
			cmd = 'r'
			
			for idx in lowhighsegments:
				
				row = [song.title, indices[idx]]
				
				cmd = 'r'
				while(cmd == 'r'):
					
					# Play the audio for evaluation					
					start_db = song.downbeats[max(0, indices[idx]-1)]
					end_db = song.downbeats[min(len(song.downbeats)-1, indices[idx] + 2)]			
					dbeats = np.arange(0,end_db - start_db,4*60.0/song.tempo).astype('float32')
					onsetsMarker = AudioOnsetsMarker(onsets = 1.0 * dbeats[1::8])		
					print start_db, end_db		
					toPlay = onsetsMarker(song.audio[int(start_db * 44100):int(end_db * 44100)])
					stream.write(toPlay, num_frames=len(toPlay[::]), exception_on_underflow=False)
					
					# Is the segment aligned?
					cmd = 'blub'
					while cmd not in commands:
						cmd = raw_input('Is this L->H segment a drop? y:1 n:0 r:redo')
					if cmd == 'y':
						# correct
						row.append('1')
					elif cmd == 'n':
						row.append('0')
						
				writer.writerow(row)
				
				
				
					
