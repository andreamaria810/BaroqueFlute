from music21 import *
from music21 import converter, roman, stream 


f = chord.Chord('G Bb E')
kf = key.Key('g')
sf = stream.Measure([kf, f])
#sf.show()

rf = roman.romanNumeralFromChord(f, kf)

print(rf.figure)



