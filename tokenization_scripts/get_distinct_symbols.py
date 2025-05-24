import os
import partitura
import warnings
import partitura as pt
import numpy as np
from pathlib import Path
import pandas as pd
import csv


warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", message=".*Found repeat without start.*")
warnings.filterwarnings("ignore", message=".*Found repeat without end.*")
warnings.filterwarnings("ignore", message=".*ignoring direction type: metronome.*")
warnings.filterwarnings("ignore", message=".*error parsing.*")



def extract_roman_numerals_from_musicxml(directory):

    unique_roman_numerals = set() 

    for filename in os.listdir(directory):
        if filename.endswith(".musicxml") or filename.endswith(".xml"):
            file_path = os.path.join(directory, filename)
            #print(f"Processing: {filename}")

            score = pt.load_musicxml(file_path)
            part = score.parts[1]

            for part in score.parts:
                if hasattr(part, 'harmony'):
                    for harmony in part.harmony:
                        if harmony.text and harmony.text.strip():
                            roman_numeral = harmony.text.strip()
                           
                            if '.' in roman_numeral:
                                if "{" not in roman_numeral and "}" not in roman_numeral:
                                    try:
                                        unique_roman_numerals.add(roman_numeral)
                                    except Exception as e:
                                        print(f"Error processing harmony: {harmony.text}. Error: {e}")

    return sorted(unique_roman_numerals)

#musicxml_directory = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\.musicxml"
#roman_numerals = extract_roman_numerals_from_musicxml(musicxml_directory)


#print("\nUnique Roman Numerals Found:")
#print(roman_numerals)



def get_time_signature_matches(directory, target_time_signature=np.array([12., 8., 4.])):
   
    matching_files = set()
    
    # Get all musicxml files
    all_files = list(Path(directory).glob("*.musicxml"))
    print(f"Found {len(all_files)} MusicXML files to check")
    
    for i, filename in enumerate(all_files):
        try:
            # Print progress every 5 files
            if i % 5 == 0:
                print(f"Checking file {i+1}/{len(all_files)}: {filename.name}")
                
            score = pt.load_musicxml(filename)
            
            if len(score.parts) < 2:
                continue
            
            part = score.parts[1]
            
            # Only check the first few measures to save time
            measure_count = 0
            for measure in part.iter_all():
                if isinstance(measure, pt.score.Measure):
                    start_time = measure.start.t
                    time_signature = part.time_signature_map(start_time)
                    
                    if np.array_equal(time_signature, target_time_signature):
                        matching_files.add(filename.name)
                        print(f"Match found: {filename.name}")
                        break
                    
                    measure_count += 1
                    if measure_count >= 10:  # Only check first 10 measures
                        break
        except Exception as e:
            print(f"Error with {filename.name}: {str(e)}")
    
    result = list(matching_files)
    print(f"Found {len(result)} matching files")
    return result

directory_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\bad_xml_files"  # Replace with your actual path
matching_files = get_time_signature_matches(directory_path)

#print("Files with the target time signature:")
#for file in matching_files:
#    print(file)



def find_zero_duration_chords_in_files(musicxml_directory):
    files_with_zero_duration_chords = {}

    # Process each file in the directory
    for musicxml_file in os.listdir(musicxml_directory):
        # Ensure we are only processing .musicxml files
        if musicxml_file.endswith(".musicxml"):
            try:
                print(f"Processing file: {musicxml_file}")
                # Full path to the MusicXML file
                file_path = os.path.join(musicxml_directory, musicxml_file)

                # Load MusicXML file
                score = pt.load_musicxml(file_path)

                # Extract Roman numerals w/ onsets and offsets
                part = score.parts[1]  
                roman_numerals = extract_harmonies(part)

                # Check for zero-duration chords
                fixed_roman_numerals = fix_zero_duration_chords(roman_numerals)

                zero_duration_indices = []
                 
                for idx, (onset, offset, _) in enumerate(fixed_roman_numerals):
                    if onset == offset:
                        zero_duration_indices.append(idx)
                
                if zero_duration_indices:
                    # Store the fixed_roman_numerals along with indices for this file
                    files_with_zero_duration_chords[musicxml_file] = {
                        "Count": len(zero_duration_indices),
                        "Indices": zero_duration_indices,
                        "Data": fixed_roman_numerals  # Store the data for later reference
                    }

            except Exception as e:
                warnings.warn(f"An error occurred while processing {musicxml_file}: {str(e)}")

    return files_with_zero_duration_chords


# Example usage
#files_with_issues = find_zero_duration_chords_in_files(musicxml_directory)

####################

def check_excel_file(file_path):

    try:
        # Load Excel file into pandas DataFrame
        df = pd.read_excel(file_path)
        
        # Ensure the DataFrame has at least 2 columns
        if len(df.columns) < 2:
            print(f"File has less than 2 columns.")
            return []
        
        # Get the first two columns (onset and offset)
        onset_col = df.iloc[:, 0]
        offset_col = df.iloc[:, 1]
        
        # Find rows where onset equals offset
        equal_indices = []
        
        for idx, (onset, offset) in enumerate(zip(onset_col, offset_col)):
            if onset == offset:
                equal_indices.append(idx)
        
        return equal_indices
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def process_directory(directory_path):

    # Get all Excel files in the directory
    excel_files = [f for f in os.listdir(directory_path) 
                  if f.endswith(('.xlsx', '.xls'))]
    
    # Process each file
    for excel_file in excel_files:
        file_path = os.path.join(directory_path, excel_file)
        
        # Get indices where onset equals offset
        equal_indices = check_excel_file(file_path)
        
        # Print results for this file
        if equal_indices:
            print(f"{excel_file} - Equal values at indices: {equal_indices}")
        else:
            print(f"{excel_file} - No equal values found")

# Example usage
# Replace with your directory path
# directory_path = "/path/to/your/excel/files"
# process_directory(directory_path)

# To process a single file
#file_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\Baroque_Flute_Dataset\HWV359b_04_Allegro\chords.xlsx"
#indices = check_excel_file(file_path)
#print(f"Equal values at indices: {indices}")



