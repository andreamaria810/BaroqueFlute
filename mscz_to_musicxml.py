import os
import subprocess

# Directory containing the .mscz files
input_directory = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\.mscz"
output_directory = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\.musicxml"

musescore_exe = r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"

os.makedirs(output_directory, exist_ok=True)

# Iterate over all .mscz files
for filename in os.listdir(input_directory):
    if filename.endswith(".mscz"):
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.musicxml")
        
        # Run the MuseScore command-line tool to convert the file
        subprocess.run([musescore_exe, "-o", output_path, input_path])

        print(f"Converted: {filename} → {output_path}")

print("Batch conversion complete!")