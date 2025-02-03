from music21 import *
from music21 import converter, chord

try:
    score = music21.converter.parse(r"c:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\samples\Sonata No. 3_4.xml")
except Exception as e:
    print(f"Error parsing file: {e}")
'''
def extract_chords_from_mxml(file_path):
    # Load the MusicXML file
    score = converter.parse(file_path)
    
    # List to store extracted chord labels
    chord_labels = []
    
    # Iterate through all elements in the score and extract Chords
    for c in score.flat.getElementsByClass('Chord'):
        # Get the root note of the chord (the first pitch of the chord)
        root = c.root()
        # Get the chord quality (e.g., major, minor, diminished)
        quality = c.quality
        # You can also get the full chord (e.g., C-E-G for a C major chord)
        chord_notes = [note.nameWithOctave for note in c.pitches]
        
        # Format the chord information (root + quality or just the notes)
        chord_label = f"{root.name}{quality} ({', '.join(chord_notes)})"
        chord_labels.append(chord_label)
    
    return chord_labels

# Path to your labeled MuseScore MusicXML file
file_path = r"c:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\samples\Sonata No. 3_4.xml"

# Extract the chords and their labels
chords = extract_chords_from_mxml(file_path)

# Output the extracted chords
for chord in chords:
    print(chord)
'''