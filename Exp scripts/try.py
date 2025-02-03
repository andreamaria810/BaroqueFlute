import music21 

try:
    score = music21.converter.parse(r"c:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\samples\Sonata No. 3_4.xml")
except Exception as e:
    print(f"Error parsing file: {e}")