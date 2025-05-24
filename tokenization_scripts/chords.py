import partitura as pt
from partitura.score import RomanNumeral
from pathlib import Path
import numpy as np
import pandas as pd 
import re
from regex import regex
from key_maps import key_maps
import warnings 

warnings.filterwarnings("ignore", message=".*error parsing.*")  # Catch unexpected characters in directions
warnings.filterwarnings("ignore", message=".*Found repeat without start.*")  # Repeat without start
warnings.filterwarnings("ignore", message=".*Found repeat without end.*")  # Repeat without end
warnings.filterwarnings("ignore", message=".*ignoring direction type: metronome.*")  # Metronome markings
warnings.filterwarnings("ignore", message="Quality for [}{] was not found") # Phrase markings found
warnings.filterwarnings("ignore", message="grace note without recoverable same voice main note")
warnings.filterwarnings("ignore", message="Did not find a wedge start element for wedge stop!")
warnings.filterwarnings("ignore", message="Cannot parse fingering info for")
warnings.filterwarnings("ignore", message="ignoring empty <harmony> tag")
warnings.filterwarnings("ignore", message="might be cadenza notation")


def extract_harmonies(part):
    # Extract harmonmies and their onset/offset times.

    if not hasattr(part, 'harmony'):
        return []

    harmonies = part.harmony
    roman_numerals = []
 
    for i, harmony in enumerate(harmonies):
        onset = harmony.start.t
        time_signature = part.time_signature_map(onset)
        beats_per_measure = int(time_signature[0])

        roman_numeral = harmony.text

        if i < len(harmonies) - 1:
            offset = harmonies[i + 1].start.t  # Normal case
        else:
            offset = max(note.end.t for note in part.notes)

        raw_onset = part.quarter_map(onset)
        raw_offset = part.quarter_map(offset)
         
        if np.array_equal(time_signature, [2., 2., 2.]):
            mapped_onset = float(raw_onset) / 2
            mapped_offset = float(raw_offset) / 2
        elif np.array_equal(time_signature, [3., 2., 3.]):
            mapped_onset = float(raw_onset) / 2
            mapped_offset = float(raw_offset) / 2
        elif np.array_equal(time_signature, [3., 8., 3.]):
            mapped_onset = float(raw_onset) / 1.5
            mapped_offset = float(raw_offset) / 1.5
        elif np.array_equal(time_signature, [6., 8., 2.]):  
            mapped_onset = float(raw_onset) / 1.5
            mapped_offset = float(raw_offset) / 1.5            
        elif np.array_equal(time_signature, [9., 8., 3.]):  
            mapped_onset = float(raw_onset) / 1.5
            mapped_offset = float(raw_offset) / 1.5            
        elif np.array_equal(time_signature, [12., 8., 4.]):  
            mapped_onset = float(raw_onset) / 1.5
            mapped_offset = float(raw_offset) / 1.5
        else:
            mapped_onset = float(raw_onset)
            mapped_offset = float(raw_offset)


        roman_numerals.append((mapped_onset, mapped_offset, roman_numeral))

    return roman_numerals


def convert_roman_numerals(roman_numerals):
    # Convert Roman numerals to Tsung-Ping's format.
    
    return [
        (
            onset,
            offset,
            re.sub(r'(?<![a-zA-Z])(b)(?=\.)', 'b',  # Keep standalone 'b' before periods
                re.sub(r'(b)', '-',  # Replace all other 'b'
                    re.sub(r'#', '+', 
                        roman_numeral.replace("o", "-")
                                    .replace("0", "=")
                                    .replace("#", "+")
                                    .replace("{", "")
                                    .replace("}", ""))))
        ) if isinstance(roman_numeral, str) else (onset, offset, roman_numeral)
        for onset, offset, roman_numeral in roman_numerals
    ]



def remove_global_key(roman_numerals):
    # Remove the global key indicator from the first roman numeral. 
    cleaned_roman_numerals = [
        re.sub(r'[A-Ga-g][b#]?[+-]?\.', '', roman_numeral)  
        for _, _, roman_numeral in roman_numerals
    ]

    return cleaned_roman_numerals


def fix_zero_duration_chords(roman_numerals):
    # Fix chords with zero duration from file conversion errors (.mscz --> .musicxml)
    if not roman_numerals:
        return []
    
        
    fixed_numerals = []
    processed_indices = set()

    for i, (onset, offset, roman_numeral) in enumerate(roman_numerals):
        if i in processed_indices:
            continue

        if i + 1 < len(roman_numerals):
            next_onset = roman_numerals[i + 1][0]
            next_offset = roman_numerals[i + 1][1]
        
        # For the last chord, ensure there's a previous one to compare to
        if i - 1 >= 0:
            prev_offset = roman_numerals[i - 1][1]

        # Check if the current chord has zero duration
        if onset == offset:
            # Handle zero duration chord: split it with the next one
            if i + 1 < len(roman_numerals):  # Ensure there is a next chord
                if next_offset - onset == 1:
                    new_offset = offset + ((next_offset - onset) / 2)
                    # Update both current and next chords
                    fixed_numerals.append((onset, new_offset, roman_numeral))
                    fixed_numerals.append((new_offset, next_offset, roman_numerals[i + 1][2]))
                    processed_indices.add(i + 1)
                else:
                    fixed_numerals.append((onset, offset, roman_numeral))  # No change needed
            else:
                fixed_numerals.append((onset, offset, roman_numeral))  # No change needed for last chord
        else:
            # Otherwise, just append the normal chord
            fixed_numerals.append((onset, offset, roman_numeral))
   
    return fixed_numerals   


def process_chords(input_file):
    score = pt.load_musicxml(input_file)
    part = score.parts[1]  

    # Extract harmonies from the part
    roman_numerals = extract_harmonies(part)
    
    # Convert roman numerals to the desired format
    converted_roman_numerals = convert_roman_numerals(roman_numerals)

    # Fix any zero-duration chords
    fixed_roman_numerals = fix_zero_duration_chords(converted_roman_numerals)

    # Create two separate array with only roman numerals:
    # - one without global key (for scale degree, quality, and inversion arrays)
    rns_without_global_key = remove_global_key(fixed_roman_numerals)   

    key_changes = [rn for rn in rns_without_global_key if '.' in rn]

    # - one with global key (for key array)
    rns_with_global_key = [roman_numeral for _, _, roman_numeral in fixed_roman_numerals]

    # Create chord array from fixed roman numerals
    chord_array = np.array(
        [(onset, offset, roman_numeral) for (onset, offset, _), roman_numeral in zip(fixed_roman_numerals, rns_without_global_key)],
        dtype=[('onset', 'f4'), ('offset', 'f4'), ('roman_numeral', 'U20')]
    )

    return chord_array, fixed_roman_numerals, rns_without_global_key, rns_with_global_key


# TONAL ATTRIBUTES 

def scale_degree_array(roman_numerals):
    scale_degrees = []

    roman_numerals_to_scale_degree = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
        'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7
    }

    for roman_numeral in roman_numerals:
   
        # Process each Roman numeral
        if isinstance(roman_numeral, str):
            match = regex.search(roman_numeral)

            numeral = None
            relative_root = None
                
            if match:
                numeral = match.group("numeral")
                relative_root = match.group("relativeroot")
                
            # If no match, append empty string and continue
            if numeral is None:
                scale_degrees.append("")
                continue

            # Clean up numeral by removing accidentals
            numeral = numeral.rstrip('+-=') 

            accidental = None
            if numeral.startswith('+') or numeral.startswith('-'):
                accidental = numeral[0]
                numeral = numeral[1:]

            # Process secondary function if there is a relative root
            if relative_root and numeral:
                second_accidental = None
                if relative_root.startswith('+') or relative_root.startswith('-'):
                    second_accidental = relative_root[0]
                    relative_root = relative_root[1:]

                second_numeral = re.sub(r'^[b#]+', '', relative_root)

                if numeral in roman_numerals_to_scale_degree:
                    first_scale_degree = roman_numerals_to_scale_degree[numeral]
                        
                    if second_numeral in roman_numerals_to_scale_degree:
                        second_scale_degree = roman_numerals_to_scale_degree[second_numeral]

                        first_degree_str = f"{accidental}{first_scale_degree}" if accidental else str(first_scale_degree)
                        second_degree_str = f"{second_accidental}{second_scale_degree}" if second_accidental else str(second_scale_degree)
                        first_and_second_degree = f"{first_degree_str}/{second_degree_str}"

                        scale_degrees.append(f"{first_and_second_degree}")
                    else:
                        # Log warning for unrecognized second numeral and append empty string
                        print(f"Warning: Unrecognized relative root '{relative_root}'.")
                        scale_degrees.append("")
                else:
                    # Log warning for unrecognized numeral and append empty string
                    print(f"Warning: Unrecognized numeral '{numeral}'.")
                    scale_degrees.append("")
            else:
                if numeral in roman_numerals_to_scale_degree:
                    scale_degree = roman_numerals_to_scale_degree[numeral]
                    scale_degree_str = f"{accidental}{scale_degree}" if accidental else str(scale_degree)
                    scale_degrees.append(str(scale_degree_str))
                else:
                    # Log warning for unrecognized numeral and append empty string
                    print(f"Warning: Unrecognized numeral '{numeral}'.")
                    scale_degrees.append("")
                    
    return scale_degrees


def chord_inversion_array(roman_numerals):

    chord_inversions = []

    '''
    for roman_numeral in roman_numerals:          # Using partitura...
        roman_obj = RomanNumeral(roman_numeral)
        chord_inversions.append(roman_obj.inversion)
    '''

    figured_bass_to_inversion = {
    '7': '0', '6': '1', '65' : '1', 
    '64': '2', '43' : '2', '42' : '3'
    }  

    for roman_numeral in roman_numerals:
        if isinstance(roman_numeral, str):
            match = re.search(regex, roman_numeral)

            if match:
                inversion = match.group("figbass")

                if not inversion:
                    inversion_symbol = "0"  # Default root position
                else:
                    inversion_symbol = figured_bass_to_inversion.get(inversion, None)             

                    if inversion_symbol is None:
                        print(f"Error: Unrecognized inversion '{inversion}' for roman_numeral '{roman_numeral}'")
                        inversion_symbol = None  # Append None to indicate missing inversion
            else:
                print(f"Error: Regex match failed for roman_numeral '{roman_numeral}'")
                inversion_symbol = None  # Append None to indicate no match

            chord_inversions.append(inversion_symbol)
    
    return chord_inversions


def chord_quality_array(roman_numerals):
    chord_quality = []

    def get_accidental(numeral):
        if '+' in numeral:
            return '+'
        elif '-' in numeral:
            return '-'
        elif '=' in numeral:
            return '='
        return None

    def add_quality(numeral, accidental, figured_bass):
        if accidental == '+':
            if numeral.islower():
                return 'a'
        elif accidental == '=':
            if numeral.islower() and figured_bass in ['7', '65', '43', '42']:
                return 'h7'
        elif accidental == '-':
            if numeral.islower():
                return 'd7' if figured_bass in ['7', '65', '43', '42'] else 'd'
            elif numeral.isupper():
                return 'M'
        else:
            if numeral.islower():
                return 'm7' if figured_bass in ['7', '65', '43', '42'] else 'm'
            elif numeral.isupper():
                if figured_bass in ['7', '65', '43', '42']:
                    return 'D7' if numeral == 'V' else 'M7'
                else:
                    return 'M'
        return None

    for roman_numeral in roman_numerals:
        original_numeral = roman_numeral
        roman_numeral = re.sub(r"\(.*?\)", "", roman_numeral).split('/')[0].strip()

        # Skip empty numerals
        if not roman_numeral:
            print(f"Skipping empty numeral: {original_numeral}")
            chord_quality.append(None)  # Append None for missing numeral
            continue

        if isinstance(roman_numeral, str):
            # Perform regex search
            match = re.search(regex, roman_numeral)
            if match:
                numeral = match.group("numeral")
                figured_bass = match.group("figbass")
                accidental = get_accidental(numeral)

                # Generate the chord quality based on the numeral, accidental, and figured bass
                quality = add_quality(numeral, accidental, figured_bass)
                
                # If a valid quality is generated, append it
                if quality:
                    chord_quality.append(quality)
                else:
                    # If no valid quality found, append None
                    print(f"Skipping numeral: {original_numeral} (no valid quality found)")
                    chord_quality.append(None)

            else:
                # If regex does not match, append None and print the error
                print(f"Skipping numeral: {original_numeral} (no regex match)")
                chord_quality.append(None)

        else:
            # If the numeral isn't a string, append None
            print(f"Skipping non-string value: {original_numeral}")
            chord_quality.append(None)

    return chord_quality


def key_array(roman_numerals):
    keys = []
    current_key = None

    for roman_numeral in roman_numerals:
        if isinstance(roman_numeral, str):
            match = re.search(regex, roman_numeral)

            if match:
                numeral = match.group('numeral')
                global_key = match.group('globalkey')
                local_key = match.group('localkey')

                # Update current key if global key is present
                if global_key:
                    current_key = global_key
                
                # Update current key based on local key mapping
                if current_key and local_key in key_maps.get(current_key, {}):
                    mapped_key = key_maps[current_key][local_key]
                    current_key = mapped_key

            # Only append the current key if a numeral exists
            if numeral:
                keys.append(current_key)

        else:
            # If roman_numeral is not a string, append None
            print(f"Skipping non-string value: {roman_numeral}")
            keys.append(None)

    return keys

 
def clean_roman_numerals(roman_numerals):
    '''Remove the global key indicator from the first roman numeral.'''
    cleaned_numerals = [re.sub(r'[a-gA-G]\.', '', roman_numeral) for _, _, roman_numeral in roman_numerals]

    return cleaned_numerals

'''
def chords_to_xlsx(input_file, folder_path):   
    # Process chords from the input file
    folder_path.mkdir(parents=True, exist_ok=True)

    # Get chord_array from process_chords function
    fixed_roman_numerals = process_chords(input_file)  

    other_chord_info = np.array(
        list(zip(
            key_array(fixed_roman_numerals), 
            scale_degree_array(fixed_roman_numerals), 
            chord_quality_array(fixed_roman_numerals), 
            chord_inversion_array(fixed_roman_numerals),   
            clean_roman_numerals(fixed_roman_numerals),
        )),
        dtype=[('key', 'U10'), ('degree', 'U10'), ('quality', 'U10'), ('inversion', 'i4'), ('roman_numeral', 'U20')]
    )

    # Merge chord_array with other_chord_info
    all_chord_info = np.lib.recfunctions.merge_arrays((
        fixed_roman_numerals[['onset', 'offset']], other_chord_info), flatten=True)

    # Convert to dataframe 
    df = pd.DataFrame(all_chord_info)

    # Save to Excel
    folder_path.mkdir(parents=True, exist_ok=True) 
    output_file = folder_path / "chords.xlsx"
    df.to_excel(output_file, index=False, header=True, engine='openpyxl')
    
    print(f"Saved {output_file}")
'''

def chords_to_xlsx(input_file, folder_path):  
    folder_path.mkdir(parents=True, exist_ok=True)

    # Process chords from the input file and unpack the returned arrays
    chord_array, fixed_roman_numerals, rns_without_global_key, rns_with_global_key = process_chords(input_file)  

    key = key_array(rns_with_global_key) 
    degree = scale_degree_array(rns_without_global_key) 
    quality = chord_quality_array(rns_without_global_key) 
    inversion = chord_inversion_array(rns_without_global_key)   

    # Check if any arrays have different lengths
    arrays = {
        "chord_array": len(chord_array),
        "key": len(key),
        "degree": len(degree),
        "quality": len(quality),
        "inversion": len(inversion)
    }

    expected_length = len(chord_array)
    mismatched = [name for name, length in arrays.items() if length != expected_length]

    if mismatched:
        print(f"⚠️ Error: Arrays with mismatched lengths: {mismatched}")
        return f"Error: Mismatched array lengths: {mismatched}"

    # Check for NaN/None values
    has_problems = False
    problem_messages = []

 
    nan_onset_count = np.isnan(chord_array['onset']).sum()
    nan_offset_count = np.isnan(chord_array['offset']).sum()
    na_roman_numeral_count = pd.isna(chord_array['roman_numeral']).sum()
    none_key_count = sum(x is None for x in key)
    none_degree_count = sum(x is None for x in degree)
    none_quality_count = sum(x is None for x in quality)
    none_inversion_count = sum(x is None for x in inversion)
    
    if nan_onset_count > 0:
        problem_messages.append(f"⚠️ Onset NaN values found: {nan_onset_count}")
        has_problems = True

    if nan_offset_count > 0:
        problem_messages.append(f"⚠️ Offset NaN values found: {nan_offset_count}")
        has_problems = True

    if na_roman_numeral_count > 0:
        problem_messages.append(f"⚠️ Roman Numeral NaN values found: {na_roman_numeral_count}")
        has_problems = True

    if none_key_count > 0:
        problem_messages.append(f"⚠️ Key None values found: {none_key_count}")
        has_problems = True

    if none_degree_count > 0:
        problem_messages.append(f"⚠️ Degree None values found: {none_degree_count}")
        has_problems = True

    if none_quality_count > 0:
        problem_messages.append(f"⚠️ Quality None values found: {none_quality_count}")
        has_problems = True

    if none_inversion_count > 0:
        problem_messages.append(f"⚠️ Inversion None values found: {none_inversion_count}")
        has_problems = True

    # If there are issues, print problem messages
    if has_problems:
        print("\n".join(problem_messages))

    # Prepare zipped data for export
    zipped_data = list(zip(
        chord_array['onset'],
        chord_array['offset'],
        key,
        degree,
        quality,
        inversion,
        chord_array['roman_numeral']
    ))

    # If problems exist, print a preview of affected data 
    if has_problems:
        print(f"🔍 Preview of affected data (first 10 rows): {zipped_data[:10]}")

    # Convert to NumPy structured array
    all_chord_info = np.array(
        zipped_data,
        dtype=[('onset', 'f4'), ('offset', 'f4'), ('key', 'U10'), 
               ('degree', 'U10'), ('quality', 'U10'), ('inversion', 'i4'), 
               ('roman_numeral', 'U20')]
    )

    # Convert to pandas dataframe 
    df = pd.DataFrame(all_chord_info)


    # Save to Excel
    folder_path.mkdir(parents=True, exist_ok=True) 
    output_file = folder_path / "chords.xlsx"
    df.to_excel(output_file, index=False, header=False, engine='openpyxl')

    # If there were problems, mention them in the return message
    if has_problems:
        return f"⚠️ Chords saved to {output_file} with data quality issues (see above)"
    else:
        return f"✅ Chords successfully saved to {output_file}"


    