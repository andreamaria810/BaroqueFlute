import xml.etree.ElementTree as ET

# Global variables for XML tree and root
tree = None
root = None

# Function to parse XML and initialize tree and root
def parse_xml(xml_file_path):
    global tree, root
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

def ListOfPitches(xml_file_path):
    # Parse the XML file
    parse_xml(xml_file_path)
    # List to store extracted pitches
    pitches = []
    # Find all <note> elements in the XML file
    notes = root.findall('.//note')
    # Loop through each <note> element
    for note in notes:
        # Find the <pitch> element within the <note> element
        pitch = note.find('pitch')
        if pitch is not None:
            step = pitch.find('step').text if pitch.find('step') is not None else ""
            octave = pitch.find('octave').text if pitch.find('octave') is not None else ""
            pitches.append(f"{step}{octave}")
        else:
            print("Pitch: None")
    return pitches

def HarmonyLabel(xml_file_path):
    # Parse the XML file
    parse_xml(xml_file_path)
    # List to store extracted harmony labels
    harmony_labels = [] 
    # Find all <harmony> elements in the XML file
    harmonies = root.findall('.//harmony') 
    #Loop through each <harmony> elements 
    for harmony in harmonies:
        # Extract the <function> tag's text
        function = harmony.find('function')
        label = function.text if function is not None else "None"
        harmony_root = harmony.find('root') 
        # Add formatted harmony label to the list
        harmony_labels.append(f"Harmony: {label}")
    return harmony_labels

def HarmonyLabelCount(xml_file_path):
    # Parse the XML file
    parse_xml(xml_file_path)
    # Find all <harmony> elements in the XML file
    harmonies = root.findall('.//harmony') 
    return len(harmonies)

def ListOfPitchesCount(xml_file_path):
    # Parse the XML file
    parse_xml(xml_file_path)
    # Find all <note> elements in the XML file
    notes = root.findall('.//note')
    return len(notes)

xml_file_path = r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.xml"
# Print extracted onsets

print(ListOfPitchesCount(xml_file_path))


labels = HarmonyLabel(xml_file_path)
for label in labels:
    print(label)

'''''
pitches = ListOfPitches(file_path)
for pitch in pitches:
    print(pitch)
'''''



