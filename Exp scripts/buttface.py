
import partitura as pt
import numpy as np
import pandas as pd 


# Load the music score
score = pt.load_musicxml(r"C:\Users\amari\OneDrive\Documents\Sonata in G Major_sample.musicxml")

# Access the second part
part = score.parts[1]
for harmony in harmonies:
        roman_numeral = harmony.text
        #print(f"Harmony: {roman_numeral}") 
        roman_numerals.append(roman_numeral)
        onset_time = harmony.start.t
        offset_time = harmony.end.t if harmony.end else None

        if offset_time is None and len(part.notes) > 0:
            for i in range(len(harmonies)):
                current_harmony = harmonies[i]
                if i + 1 < len(harmonies):
                    next_harmony = harmonies[i + 1]
                    offset_time = next_harmony.start.t
                elif part.notes:  # If no next harmony, use the last note's end time
                    offset_time = part.notes[-1].end.t #

##############################################################

roman_numerals = []  # List to store the Roman numerals along with their onsets and offsets

# Assuming part is already defined and has the harmonies
if hasattr(part, 'harmony'):
    harmonies = part.harmony  # List or iterable of harmonies

    # For each note in the part, we will check which Roman numeral applies
    for i, harmony in enumerate(harmonies):
        roman_numeral = harmony.text
        onset_time = harmony.start.t  # Start time of the harmony
        offset_time = None

        # Check if harmony has a valid end time, otherwise calculate the offset_time
        if harmony.end:
            offset_time = harmony.end.t
        else:
            # If no end time, use the next harmony's start time or the last note's end time
            if i + 1 < len(harmonies):
                next_harmony = harmonies[i + 1]
                if harmony.start.t != next_harmony.start.t:
                    offset_time = next_harmony.start.t
            elif part.notes:
                offset_time = part.notes[-1].end.t  # Last note's end time as the default

        # If no valid offset_time, use the onset_time
        if offset_time is None:
            offset_time = onset_time

        # Now, iterate through the notes and assign Roman numerals
        for note in part.notes:
            # Check if the note's time lies between the onset and offset of the current harmony
            if onset_time <= note.start.t < offset_time:
                roman_numerals.append({
                    'note': note, 
                    'roman_numeral': roman_numeral, 
                    'onset': note.start.t,
                    'offset': note.end.t if note.end else offset_time
                })

# Output the collected Roman numerals and notes
for entry in roman_numerals:
    print(f"Note: {entry['note']}, Roman Numeral: {entry['roman_numeral']}, Onset: {entry['onset']}, Offset: {entry['offset']}")
