import os
import partitura
import warnings

warnings.filterwarnings("ignore", message=".*Found repeat without start.*")
warnings.filterwarnings("ignore", message=".*Found repeat without end.*")
warnings.filterwarnings("ignore", message=".*ignoring direction type: metronome.*")
warnings.filterwarnings("ignore", message=".*error parsing.*")

def extract_roman_numerals_from_musicxml(directory):
    """Extracts distinct Roman numeral labels from all MusicXML files in a directory."""

    unique_roman_numerals = set() 

    for filename in os.listdir(directory):
        if filename.endswith(".musicxml") or filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            #print(f"Processing: {filename}")

            score = partitura.load_musicxml(file_path)
            part = score.parts[1]

            for part in score.parts:
                if hasattr(part, 'harmony'):
                    for harmony in part.harmony:
                        if harmony.text and harmony.text.strip():
                            roman_numeral = harmony.text.strip()
                            if "{" not in roman_numeral and "}" not in roman_numeral:
                                unique_roman_numerals.add(roman_numeral)

    return sorted(unique_roman_numerals)

    
musicxml_directory = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\.musicxml"

roman_numerals = extract_roman_numerals_from_musicxml(musicxml_directory)

print("\nUnique Roman Numerals Found:")
print(sorted(roman_numerals))