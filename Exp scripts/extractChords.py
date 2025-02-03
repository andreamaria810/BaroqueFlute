from music21 import converter, harmony

def extract_labeled_chords(file_path):
    # Load the MusicXML file
    score = converter.parse(file_path)
    
    # List to store extracted labeled chords
    labeled_chords = []
    
    # Extract all Harmony objects (labeled chords)
    for h in score.flat.getElementsByClass(harmony.Harmony):
        # Get the full chord symbol (e.g., "Cmaj7", "G7")
        chord_symbol = h.figure
        labeled_chords.append(chord_symbol)
    
    return labeled_chords

# Path to your labeled MuseScore MusicXML file
file_path = r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.musicxml"
# Extract the labeled chords
labeled_chords = extract_labeled_chords(file_path)

# Output the labeled chords
for chord in labeled_chords:
    print(chord)