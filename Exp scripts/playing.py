from lxml import etree
import csv 

def extract_harmony_annotations(musicxml_file):
    """
    Extract harmony (Roman numeral) annotations from a MusicXML file using lxml.etree.

    :param file_path: Path to the MusicXML file
    :return: List of harmony annotations with measure and beat positions
    """
    # Parse the MusicXML file
    tree = etree.parse(musicxml_file)
    root = tree.getroot()

    # Namespace for MusicXML
    ns = {'mxl': 'http://www.musicxml.org/ns/musicxml'}

    # Extract harmony elements
    harmonies = []
    for harmony in root.xpath('//mxl:harmony', namespaces=ns):
        # Get Roman numeral function
        roman = harmony.xpath('./mxl:root/mxl:root-step', namespaces=ns)
        roman = roman[0].text if roman else "Unknown"

        # Get bass note if present
        bass = harmony.xpath('./mxl:bass/mxl:bass-step', namespaces=ns)
        bass = bass[0].text if bass else None

        # Extract measure and beat information
        measure = harmony.getparent().get('number')  # Parent is a measure
        beat = harmony.xpath('./mxl:offset', namespaces=ns)
        beat = beat[0].text if beat else "0"

        harmonies.append({
            'roman': roman,
            'bass': bass,
            'measure': measure,
            'beat': beat,
        })

    return harmonies

# Example usage
# Path to your MusicXML file
file_path = r"C:\Users\amari\OneDrive\Documents\Sonata in G Major - selection.musicxml"
annotations = extract_harmony_annotations(file_path)

output_file = 'harmony_annotations.csv'
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=['roman', 'bass', 'measure', 'beat'])
    writer.writeheader()
    writer.writerows(annotations)

print(f"Harmony annotations extracted to {output_file}")