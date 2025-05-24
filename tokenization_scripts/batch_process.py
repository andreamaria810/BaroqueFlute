import os
import argparse
from pathlib import Path
import warnings


from chords import chords_to_xlsx 
from notes import notes_to_csv 

import partitura as pt
import numpy as np
import pandas as pd

# Set up warnings to ignore
warnings.filterwarnings("ignore", message=".*error parsing.*")
warnings.filterwarnings("ignore", message=".*Found repeat without start.*")
warnings.filterwarnings("ignore", message=".*Found repeat without end.*")
warnings.filterwarnings("ignore", message=".*ignoring direction type: metronome.*")
warnings.filterwarnings("ignore", message="Quality for [}{] was not found")

def process_file(input_file, output_dir, process_notes=True, process_chords=True):
    """
    Process a single musicXML file to extract notes and/or chord information.
    
    Args:
        input_file (str or Path): Path to the input musicXML file
        output_dir (str or Path): Directory to save output files
        process_notes (bool): Whether to process and save note data
        process_chords (bool): Whether to process and save chord data
    """
    input_path = Path(input_file)
    output_path = Path(output_dir) / input_path.stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {input_path.name}...")
    
    # Process notes if requested
    if process_notes:
        try:
            notes_to_csv(input_file, output_path)
            print(f"  ✓ Notes extracted to {output_path}/notes.csv")
        except Exception as e:
            print(f"  ✗ Error extracting notes: {str(e)}")
    
    # Process chords if requested
    if process_chords:
        try:
            chords_to_xlsx(input_file, output_path)          
            print(f"  ✓ Chords extracted to {output_path}/chords.xlsx")
        except Exception as e:
            print(f"  ✗ Error extracting chords: {str(e)}")


def batch_process(input_dir, output_dir, file_pattern="*.musicxml", process_notes=True, process_chords=True):
    """
    Process all musicXML files in a directory.
    
    Args:
        input_dir (str or Path): Directory containing input files
        output_dir (str or Path): Directory to save output files
        file_pattern (str): Pattern to match input files
        process_notes (bool): Whether to process and save note data
        process_chords (bool): Whether to process and save chord data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    files = list(input_path.glob(file_pattern))

    if not files:
        print(f"No files matching '{file_pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] ", end="")
        process_file(file, output_path, process_notes, process_chords)
    
    print(f"\nProcessing complete. Results saved to {output_path}")


def main():
    """
    Main function to handle command line arguments and run the batch processor.
    """
    parser = argparse.ArgumentParser(description="Process musicXML files to extract note and chord data")
    
    parser.add_argument("input", help="Input file or directory containing musicXML files")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: './output')")
    parser.add_argument("--pattern", "-p", default="*.musicxml", help="File pattern for batch processing (default: '*.musicxml')")
    parser.add_argument("--no-notes", action="store_true", help="Skip note extraction")
    parser.add_argument("--no-chords", action="store_true", help="Skip chord extraction")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine if input is a file or directory
    if input_path.is_file():
        # Process single file
        output_dir = Path(args.output)
        process_file(input_path, output_dir, not args.no_notes, not args.no_chords)
    elif input_path.is_dir():
        # Process directory
        print("Starting batch process...")
        batch_process(input_path, args.output, args.pattern, not args.no_notes, not args.no_chords)
        print("Batch process completed.")
    else:
        print(f"Error: Input path '{args.input}' does not exist")

if __name__ == "__main__":
    main()

# python3 batch_process.py --help