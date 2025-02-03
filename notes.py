import partitura as pt
import numpy as np
import pandas as pd  
from partitura.musicanalysis.pitch_spelling import compute_morphetic_pitch, compute_morph_array, compute_chroma_array, compute_chroma_vector_array, chromatic_pitch_from_midi

score = pt.load_musicxml(r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.musicxml")

# Get the first part (flute) from the score
part = score.parts[0]

# Initialize variables
staff_number = 0
notes = []

# Assume `part` is a musical part object
for note_or_staff in part.iter_all():
    # Check if the element is a Staff
    if isinstance(note_or_staff, pt.score.Staff):
        staff_number = note_or_staff.number # Update the current staff number
    # Check if the element is a Note
    elif isinstance(note_or_staff, pt.score.Note):
        start_time = note_or_staff.start.t # Get the start time of the note
        end_time = note_or_staff.end_tied.t # Get the end time of the note
        # Append the note information to the list
        notes.append((part.quarter_map(start_time), note_or_staff.midi_pitch, part.quarter_map(note_or_staff.duration_tied), staff_number, part.measure_number_map(start_time)))


note_array = np.array(notes, dtype=[('onset', 'f4'), ('midi_pitch', 'i4'), ('duration', 'f4'), ('staff', 'i4'), ('measure', 'i4')])

# Compute morphetic pitch per note in the note array
def morphetic_pitch_array(note_array, K_pre=10, K_post=40):
    # Adapted from partitura.musicanalysis.pitch_spelling.ps13s1
    sort_idx = np.lexsort((note_array['onset'], note_array['midi_pitch'])) # Sorts note array by MIDI pitch and then onset time
    # Create 2D array of onset times and chromatic pitches
    sorted_ocp = np.column_stack(
        (
            note_array[sort_idx]['onset'],
            chromatic_pitch_from_midi(note_array[sort_idx]['midi_pitch']),
        )
    )
     
    chroma_array = compute_chroma_array(sorted_ocp=sorted_ocp) # Computes chroma feature representation (pitch classes over time)
    chroma_vector_array = compute_chroma_vector_array(chroma_array, K_pre, K_post)  # Computes context-aware chroma features
    morph_array = compute_morph_array(chroma_array, chroma_vector_array) # Computes morphetic pitch representation 

    return compute_morphetic_pitch(sorted_ocp, morph_array) # Returns the morphetic pitch array

# Compute morphetic pitch per note in the note array
morphetic_pitches = np.array(morphetic_pitch_array(note_array), dtype=[('morphetic_pitch', 'i4')])
# Merge all the columns into a single structured array
all_note_info = np.lib.recfunctions.merge_arrays((note_array, morphetic_pitches), flatten=True)

column_order = ['onset', 'midi_pitch', 'morphetic_pitch', 'duration', 'staff', 'measure']
df = pd.DataFrame(all_note_info)[column_order]
df.to_csv("notes.csv", index=False, header=False) # Save the note information to a CSV file


#print(morphetic_pitches.shape)
#print(all_note_info[:10])
#print(morphetic_pitch_array(note_array))
#print(all_note_info.dtype.names)

print(note_array[:10])