import partitura as pt
import numpy as np
import pandas as pd 
import re 
import warnings 

######################### ONSET/OFFSET #########################

warnings.filterwarnings("ignore", message=".*error parsing.*")  # Catch unexpected characters in directions
warnings.filterwarnings("ignore", message=".*Found repeat without start.*")  # Repeat without start
warnings.filterwarnings("ignore", message=".*Found repeat without end.*")  # Repeat without end
warnings.filterwarnings("ignore", message=".*ignoring direction type: metronome.*")  # Metronome markings

score = pt.load_musicxml(r"C:\Users\amari\OneDrive\Documents\experiment.musicxml")

part = score.parts[1]

roman_numerals = []

# Check if part has harmonies
if hasattr(part, 'harmony'):
    harmonies = part.harmony

    for i, harmony in enumerate(harmonies):
        roman_numeral = harmony.text
        onset = harmony.start.t
        offset = None
  
        if harmony.end:
            offset = harmony.end.t
        else:
            if i + 1 < len(harmonies):
                next_harmony = harmonies[i + 1]
                if harmony.start.t != next_harmony.start.t:
                    offset = next_harmony.start.t
                elif part.notes:
                    offset = part.notes[-1].end.t     
        if offset is None:
            offset = onset
    
        for note in part.notes:
            if onset <= note.start.t <= offset:
                roman_numerals.append((onset, offset, roman_numeral))
                break

def convert_roman_numerals(roman_numerals):
    return [
        (onset, offset, roman_numeral.replace("o", "-").replace("0", "=").replace("0", "=").replace("#", "+").replace("b", "-").replace("{", "").replace("}", ""))
        if isinstance(roman_numeral, str) else (onset, offset, roman_numeral)
        for onset, offset, roman_numeral in roman_numerals
    ]

converted = convert_roman_numerals(roman_numerals)

#converted_sublist = [tup[2] for tup in converted]   


chord_array = np.array(converted, dtype=[('onset', 'f4'), ('offset', 'f4'), ('roman_numeral', 'U10')])  

################################ TONAL ATTRIBUTES ################################

pattern = r"""
# Adapted from DCML....
^                                          
(\{)?\.?                                   
((?P<globalkey>[a-gA-G]([-+]*))\.)?       
((?P<localkey>([-+]*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\.)?  
((?P<pedal>([-+]*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i))\[)?   
(?P<chord>                                 
    (?P<numeral>([+=-]*)(b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i|Ger|It|Fr|@none)[-+=]*)? 
    (?P<form>(%|o|\+|M|\+M))?             
    (?P<figbass>(7|65|43|42|2|64|6))?     
    (\((?P<changes>((\+|-|\^|v)?(b*|\#*)\d)+)\))?  
    (/(?P<relativeroot>((b*|\#*)(VII|VI|V|IV|III|II|I|vii|vi|v|iv|iii|ii|i)/?)*))?  
)
(?P<pedalend>\])?                          
(\|(?P<cadence>((HC|PAC|IAC|DC|EC|PC)(\..+?)?)))?  
(?P<phraseend>(\\\\|\{|\}|\}\{))?         
$                                           
"""

regex = re.compile(pattern, re.VERBOSE)

def scale_degree_array(roman_numerals):

    scale_degrees = []

    roman_numerals_to_scale_degree = {
    'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
    'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7
    }   

    for _, _, roman_numeral in roman_numerals:
        #print(f"Processing: {roman_numeral}")
        if isinstance(roman_numeral, str):
            match = regex.search(roman_numeral)
            #print(f"Match found: {match is not None}")
            
            if match:
                numeral = match.group("numeral")
                relative_root = match.group("relativeroot")
                #print(f"Numeral: {numeral}")
                #print(f"Relative root: {relative_root}")

            numeral = numeral.rstrip('+-=')

            if numeral:
                accidental = None
                if numeral.startswith('+') or numeral.startswith('-'):   
                    accidental = numeral[0]
                    numeral = numeral[1:] 

                if relative_root and numeral:
                    #print("\nProcessing secondary function...")
                    second_accidental = None
                    if relative_root.startswith('+') or relative_root.startswith('-'):
                        second_accidental = relative_root[0]
                        relative_root = relative_root[1:]
                        #print(f"Found prefix in second numeral: {second_accidental}")
                        #print(f"Second numeral after prefix removal: {relative_root}")

                    second_numeral = re.sub(r'^[b#]+', '', relative_root)
                    #print(f"Second numeral after removing accidentals: {second_numeral}")

                    if numeral in roman_numerals_to_scale_degree:
                        first_scale_degree = roman_numerals_to_scale_degree[numeral]
                        #print(f"Converted first numeral {numeral} to {first_scale_degree}")
                        if second_numeral in roman_numerals_to_scale_degree:
                            second_scale_degree = roman_numerals_to_scale_degree[second_numeral]
                            #print(f"Converted second numeral {second_numeral} to {second_scale_degree}")
                            first_degree_str = f"{accidental}{first_scale_degree}" if accidental else str(first_scale_degree)
                            second_degree_str = f"{second_accidental}{second_scale_degree}" if second_accidental else str(second_scale_degree)
                            first_and_second_degree = f"{first_degree_str}/{second_degree_str}"
                            #print(f"Final secondary function result: {first_and_second_degree}")
                            scale_degrees.append(f"{first_and_second_degree}")
                           
                
                else:
                    #print("\nProcessing single numeral...")
                    if numeral in roman_numerals_to_scale_degree:
                        scale_degree = roman_numerals_to_scale_degree[numeral]
                        #print(f"Converted numeral {numeral} to {scale_degree}")
                        
                        scale_degree_str = f"{accidental}{scale_degree}" if accidental else str(scale_degree)
                        #print(f"Final single numeral result: {scale_degree_str}")
                        scale_degrees.append(str(scale_degree_str))
      
    
    #print(f"\nFinal scale_degrees array: {scale_degrees}")
    return scale_degrees


def chord_inversion_array(roman_numerals):

    chord_inversions = []

    figured_bass_to_inversion = {
    '7': '0', '6': '1', '65' : '1', 
    '64': '2', '43' : '2', '42' : '3'
    }  

    for _, _, roman_numeral in roman_numerals:
        if isinstance(roman_numeral, str):
            match = re.search(regex, roman_numeral)

            if match:
                inversion = match.group("figbass")
                inversion_symbol = figured_bass_to_inversion.get(inversion, "0")
              
            else:
                inversion_symbol = '0'

            chord_inversions.append(inversion_symbol)

    return chord_inversions


def chord_quality_array(roman_numerals):

    chord_quality = []

    for _, _, roman_numeral in roman_numerals:
        #print(f"Processing roman_numeral: {roman_numeral}")

        if isinstance(roman_numeral, str):
            match = re.search(regex, roman_numeral)
            
            if match:
                numeral = match.group("numeral") 
                figured_bass = match.group("figbass") 
                change = match.group("changes")
                #print(f"  Numeral: {numeral}, Figured Bass: {figured_bass}")

                accidental = None

                if '+' in numeral:
                    accidental = '+'
                elif '-' in numeral:
                    accidental = '-'
                elif '=' in numeral:
                    accidental = '='

                if accidental:
                    if accidental == '+':
                        if numeral.islower():
                            #print(" Adding 'a'")
                            chord_quality.append('a')
                        if numeral.upper():
                            pass
                    elif accidental == '=':
                        if numeral.islower():
                            if figured_bass in ['7', '65', '43', '42']:
                                #print("  Adding 'h7'")
                                chord_quality.append('h7')
                        elif numeral.isupper():
                            pass

                    elif accidental == '-':
                        if numeral.islower():
                            #print(" Adding 'd'")
                            chord_quality.append('d')
                        elif numeral.isupper():
                            #print(" Adding 'M'")
                            chord_quality.append('M')
                
                else:
                    if numeral.islower():
                        if figured_bass in ['7', '65', '43', '42']:
                            #print(" Adding 'm7'")
                            chord_quality.append('m7')
                        else:
                            #print(" Adding 'm'")
                            chord_quality.append('m')
                    elif numeral.isupper():
                        if figured_bass in ['7', '65', '43', '42']:
                            if numeral == 'V':
                                #print(" Adding 'D7'")
                                chord_quality.append('D7')
                            else:
                                #print(" Adding 'M7'")
                                chord_quality.append('M7')
                        else:
                            #print(" Adding 'M'")
                            chord_quality.append('M')
    return chord_quality


def key_array(roman_numerals):

    keys = []

    for _, _, roman_numeral in roman_numerals:
        if isinstance(roman_numeral, str):
            match = re.search(regex, roman_numeral)

            if match:
                numeral = match.group('numeral')
                key = match.group('globalkey')
                if key:
                    current_key = key

            if numeral:
                keys.append(current_key)

    return keys


def clean_roman_numerals(roman_numerals):

    cleaned_numerals = [re.sub(r'[a-gA-G]\.', '', roman_numeral) for _, _, roman_numeral in roman_numerals]

    return cleaned_numerals



other_chord_info = np.array(
    list(zip(
        key_array(converted), 
        scale_degree_array(converted), 
        chord_quality_array(converted), 
        chord_inversion_array(converted),   
        clean_roman_numerals(converted),
    )),
    dtype=[('key', 'U10'), ('degree', 'U10'), ('quality', 'U10'), ('inversion', 'i4'), ('roman_numeral', 'U10')]
)


all_chord_info = np.lib.recfunctions.merge_arrays((chord_array[['onset', 'offset']], other_chord_info), flatten=True)

df = pd.DataFrame(all_chord_info)

df.to_excel("output.xlsx", index=False, header=False, engine='openpyxl')


