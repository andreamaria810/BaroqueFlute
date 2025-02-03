import ms3
import pandas as pd

# Open a MuseScore file (e.g., .mscz)
score = ms3.Score(r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.mscz")

# Extract note data (e.g., onset, pitch, duration, etc.)
note_data = []
for part in score.parts:
    for measure in part.measures:
        for note in measure.notes:
            note_info = {
                'part': part.name,
                'measure': measure.number,
                'note': note.name,
                'onset': note.offset,
                'duration': note.quarterLength,
                'pitch': note.pitch.nameWithOctave,
                'chord_symbol': note.getChordSymbol() if note.hasChordSymbol() else ''
            }
            note_data.append(note_info)

# Create a DataFrame from the note data
df_notes = pd.DataFrame(note_data)

# Print the first few rows of the dataframe
print(df_notes.head())