import partitura as pt
from partitura.musicanalysis.pitch_spelling import compute_morphetic_pitch, compute_morph_array, compute_chroma_array, compute_chroma_vector_array, chromatic_pitch_from_midi
import numpy as np


# Path to the MusicXML file
score_fn = r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.xml"

# Load the score into a `Part` object
score = pt.load_musicxml(score_fn)

part = score.parts[0]
# Get note array.
score_note_array = score.note_array()

#print(score_note_array.dtype.names)
#print(score_note_array)
#print(score_note_array[:10])

#print(part.time_sigs)
'''
all_notes = []

for part in pt.score.iter_parts(score.parts):
    for note in part.notes:
        midi_pitch = note.midi_pitch
        start_time = note.start.t
        duration = note.duration
        all_notes.append((midi_pitch, start_time, duration))


for note in all_notes:
    print(f"MIDI Pitch: {note[0]}, Start Time: {note[1]}, Duration: {note[2]}")


for part in pt.score.iter_parts(score.parts):
    # Print the part name
    print(f"Part name: {part.id}")

    # Example: Print all notes in this part
    for note in part.notes:
        print(f"Note: Pitch {note.midi_pitch}, Start {note.start.t}, Duration {note.duration}")

arr = np.array([2, 3, 1, 4, 5])
print(enumerate(arr))
'''

note_array = np.array([
    (64, 0.0, 1.0),  # MIDI pitch 64, onset 0.0, duration 1.0
    (62, 1.0, 0.5),  
    (60, 1.5, 0.75), 
    (60, 0.5, 0.5),
     (62, 0.25, 1.0) 
], dtype=[('midi_pitch', 'i4'), ('onset', 'f4'), ('duration', 'f4')])


def morphetic_pitch_array(note_array, K_pre=10, K_post=40):
    # Adapted from partitura.musicanalysis.pitch_spelling.ps13s1
    # Sorts note array by MIDI pitch and then onset time
    sort_idx = np.lexsort((note_array['onset'], note_array['midi_pitch']))
    
    sorted_ocp = np.column_stack(
        (
            note_array[sort_idx]['onset'],
            chromatic_pitch_from_midi(note_array[sort_idx]['midi_pitch']),
        )
    )

    chroma_array = compute_chroma_array(sorted_ocp=sorted_ocp)
    chroma_vector_array = compute_chroma_vector_array(chroma_array, K_pre, K_post)
    morph_array = compute_morph_array(chroma_array, chroma_vector_array)

    return compute_morphetic_pitch(sorted_ocp, morph_array)

print(morphetic_pitch_array(note_array, K_pre=10, K_post=40))







sorted_pitches = np.argsort(note_array['midi_pitch']) # Sort the note array by MIDI pitch
#print(sorted_pitches) # Pitches are now in ascending order

# Note array reordered according to sorted_pitches (ascending MIDI pitch) and then by onset time
sorted_onsets = np.argsort(note_array[sorted_pitches]['onset'], kind='mergesort') 
#print(sorted_onsets)

 



