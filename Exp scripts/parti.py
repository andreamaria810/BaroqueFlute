import partitura

# Path to your MusicXML file
musicxml_file = r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.musicxml"

# Load the MusicXML file
score = partitura.load_musicxml(musicxml_file)
part = score.parts[1]



# Iterate over the elements in the score to extract Roman numerals
roman_numerals = []
for element in part.iter_all(partitura.score.Harmony):
    # Get the Roman numeral text
    roman_text = element.function 
    # Append it to the list of Roman numerals
    roman_numerals.append({
        'measure': element.measure,
        'position': element.start_t,
        'roman_numeral': roman_text,
    })

# Print the extracted Roman numerals
for rn in roman_numerals:
    print(f"Measure: {rn['measure']}, Position: {rn['position']}, Roman Numeral: {rn['roman_numeral']}")
