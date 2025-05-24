import partitura as pt
import numpy as np
import pandas as pd  
from pathlib import Path
import numpy.lib.recfunctions as rfn 
from partitura.musicanalysis.pitch_spelling import (
    compute_morphetic_pitch, compute_morph_array, compute_chroma_array, 
    compute_chroma_vector_array, chromatic_pitch_from_midi)
import warnings



def morphetic_pitch_array(note_array, K_pre=10, K_post=40):
    # Adapted from partitura.musicanalysis.pitch_spelling.ps13s1
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


def notes_to_csv(input_file, folder_path):
    # Load the MusicXML file
    score = pt.load_musicxml(input_file)
    all_notes = []  # List to store notes from all parts
    
    #for part in score.parts:  # Iterate over all parts
    if len(score.parts) > 0:
        part = score.parts[0]  

        for note in part.notes:
            onset = note.start.t
            offset = note.end.t
            #duration = onset - offset
            time_signature = part.time_signature_map(onset)
                
            raw_onset = part.quarter_map(onset)
            raw_offset = part.quarter_map(offset)
            raw_duration = abs(raw_onset - raw_offset)
            
            # Apply the same time signature scaling as in extract_harmonies
            if np.array_equal(time_signature, [2., 2., 2.]):
                mapped_onset = float(raw_onset) / 2
                mapped_duration = float(raw_duration) / 2
            elif np.array_equal(time_signature, [3., 2., 3.]):
                mapped_onset = float(raw_onset) / 2
                mapped_duration = float(raw_duration) / 2
            elif np.array_equal(time_signature, [3., 8., 3.]):
                mapped_onset = float(raw_onset) / 1.5
                mapped_duration = float(raw_duration) / 1.5
            elif np.array_equal(time_signature, [6., 8., 2.]):  
                mapped_onset = float(raw_onset) / 1.5
                mapped_duration = float(raw_duration) / 1.5
            elif np.array_equal(time_signature, [9., 8., 3.]):  
                mapped_onset = float(raw_onset) / 1.5
                mapped_duration = float(raw_duration) / 1.5
            elif np.array_equal(time_signature, [12., 8., 4.]):  
                mapped_onset = float(raw_onset) / 1.5
                mapped_duration = float(raw_duration) / 1.5
            else:
                mapped_onset = float(raw_onset)
                mapped_duration = float(raw_duration)
                
            all_notes.append((
                mapped_onset,
                note.midi_pitch,
                mapped_duration,
                note.staff,
                part.measure_number_map(note.start.t),
            ))
        
    print(f"Extracted {len(all_notes)} notes from {len(score.parts)} parts")
    
    # Convert list to NumPy structured array
    note_array = np.array(all_notes, dtype=[
        ('onset', 'f4'), ('midi_pitch', 'i4'), ('duration', 'f4'),
        ('staff', 'i4'), ('measure', 'i4')
    ])
    
    # Sort notes by measure, and within each measure, by onset time
    sorted_notes = np.sort(note_array, order=['measure', 'onset'])

    # Get morphetic pitch information.
    morphetic_pitches = np.array(morphetic_pitch_array(sorted_notes), dtype=[('morphetic_pitch', 'i4')])

    # Merge note data with morphetic pitch information
    all_note_info = rfn.merge_arrays((sorted_notes, morphetic_pitches), flatten=True)

    # Convert to pandas DataFrame
    column_order = ['onset', 'midi_pitch', 'morphetic_pitch', 'duration', 'staff', 'measure']
    df = pd.DataFrame(all_note_info)[column_order]

    # Save to CSV
    output_file = folder_path / "notes.csv"
    df.to_csv(output_file, index=False, header=False)

    print(f"Saved {output_file}")




#print(morphetic_pitches.shape)
#print(all_note_info[:10])
#print(morphetic_pitch_array(note_array))
#print(all_note_info.dtype.names)