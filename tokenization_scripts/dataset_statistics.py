import partitura as pt
import warnings
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import openpyxl
import matplotlib.pyplot as plt
import re

warnings.filterwarnings("ignore", category=UserWarning, module="partitura")
warnings.filterwarnings("ignore", message="ignoring direction type: metronome")


def extract_metadata(input_file):
    # Load the MusicXML file
    score = pt.load_musicxml(input_file)
    part = score.parts[0]
    
    # Extract basic metadata
    composer = score.composer if score.composer else "Unknown"
    work_number = score.work_number if score.work_number else "Unknown"
    movement = score.title if score.title else "Unknown"
    
    # Extract first time signature (if available)
    time_sig = "Unknown"
    if part.time_sigs and len(part.time_sigs) > 0:
        ts = part.time_sigs[0]
        time_sig = f"{ts.beats}/{ts.beat_type}"
    
    # Extract first key signature (if available)
    key_sig = "Unknown"
    if part.key_sigs and len(part.key_sigs) > 0:
        key = part.key_sigs[0].fifths
        if key < 0:
            flats = abs(key)
            key_sig = f"{flats} flats"
        elif key > 0:
            sharps = key
            key_sig = f"{sharps} sharps"
        else:
            key_sig = "0 flats or sharps"
    
    # Return a fixed structure with just one value for each column
    return {
        'Composer': composer,
        'No.': work_number,
        'Movement': movement,
        'Time': time_sig,
        'Key': key_sig
    }


"""
def create_visualizations(df, directory):
    
    # Create a copy of the DataFrame to clean up without modifying the original
    viz_df = df.copy()
    
    # Function to properly format composer names
    def format_composer_name(name):
        if isinstance(name, str):
            # Only add spaces after periods if they don't already have one
            formatted = re.sub(r'\.([A-Za-z])', lambda m: '. ' + m.group(1), name)
            # Ensure proper capitalization
            formatted = formatted.title()
            # Make sure there aren't double spaces
            formatted = re.sub(r'\s+', ' ', formatted).strip()
            return formatted
        return name

    if 'Composer' in viz_df.columns:
        viz_df['Composer'] = viz_df['Composer'].apply(format_composer_name)
    
        print("Composers after formatting:")
        for composer in sorted(viz_df['Composer'].unique()):
            print(f"  {composer}")

        # Count pieces per composer
        composer_counts = viz_df['Composer'].value_counts().reset_index()
        composer_counts.columns = ['Composer', 'Number of Pieces']
        
        # Print the count of pieces by each composer
        print("\nNumber of pieces by composer:")
        for _, row in composer_counts.iterrows():
            print(f"  {row['Composer']}: {row['Number of Pieces']} pieces")
        
        # Save the count as a separate CSV file
        count_file = os.path.join(directory, "composer_counts.csv")
        composer_counts.to_csv(count_file, index=False)
        print(f"\nComposer counts saved to {count_file}")

        # Create a directory for visualizations
        viz_dir = os.path.join(directory, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. Composer distribution
        plt.figure(figsize=(10, 6))
        composer_counts = viz_df['Composer'].value_counts()
        composer_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Pieces by Composer')
        plt.xlabel('Composer')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "composer_distribution.png"))
        plt.close()
        
        # 2. Time signature distribution
        plt.figure(figsize=(8, 6))
        time_counts = viz_df['Time'].value_counts()
        time_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribution of Time Signatures')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "time_signature_distribution.png"))
        plt.close()
        
        # 3. Key signature distribution
        plt.figure(figsize=(10, 6))
        key_counts = viz_df['Key'].value_counts()
        key_counts.plot(kind='bar', color='lightgreen')
        plt.title('Distribution of Key Signatures')
        plt.xlabel('Key Signature')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "key_distribution.png"))
        plt.close()
        
        # 4. Simple crosstab visualization
        plt.figure(figsize=(12, 8))
        composer_time = pd.crosstab(viz_df['Composer'], viz_df['Time'])
        composer_time.plot(kind='bar', stacked=True)
        plt.title('Composers vs Time Signatures')
        plt.xlabel('Composer')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Time Signature')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "composer_time_chart.png"))
        plt.close()
    
        print(f"Visualizations saved to {viz_dir}")
        
    else:
        print("Warning: 'Composer' column not found in the dataset.")
"""


def batch_process(directory):
    # List all files in the specified directory
    files = os.listdir(directory)
    musicxml_files = [f for f in files if f.endswith('.musicxml')]
    
    # Initialize an empty list to store metadata from all files
    all_metadata = []
    
    # Process each file
    for file in musicxml_files:
        input_file = os.path.join(directory, file)
        try:
            file_metadata = extract_metadata(input_file)
            all_metadata.append(file_metadata)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metadata)
    
    # Define output path with XLSX extension
    output_file = os.path.join(directory, "metadata_good.xlsx")  # Changed extension here
    
    # Save to Excel
    df.to_excel(output_file, header=True, index=False, engine='openpyxl')
    
    #create_visualizations(df, directory)

    print(f"Metadata saved to {output_file}")
    
    return df

# Example usage: specify the directory with MusicXML files
directory_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Sonatas\correct_xml_files"
batch_process(directory_path)




