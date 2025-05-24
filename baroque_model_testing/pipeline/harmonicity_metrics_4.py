import numpy as np
import math
from collections import Counter
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d


''' 1. Chord Histogram Entropy (CHE)

    Create a histogram of chord occurrences
    Normalize counts to sum to 1
    Calculate entropy: H = -Σ(pi * log(pi))    '''


def calculate_che(chord_labels):
    # Check the format of input data
    if chord_labels and isinstance(chord_labels[0], tuple):
        # If input is a list of tuples like ('A+', 'M')
        # Assuming first element is root and second is quality
        chord_sequence = [label[0] + label[1] for label in chord_labels]
    elif chord_labels and isinstance(chord_labels[0], dict):
        # Original format expecting dictionaries
        chord_sequence = [label['root'] + label['tquality'] for label in chord_labels]
    else:
        raise TypeError("chord_labels must be a list of tuples or dictionaries")
    
    # Calculate histogram
    chord_counts = Counter(chord_sequence)
    total_chords = len(chord_sequence)
    
    # Normalize and calculate entropy
    entropy = 0
    for chord, count in chord_counts.items():
        p_i = count / total_chords
        entropy -= p_i * np.log(p_i)
    
    return entropy


''' 2.  Chord Coverage (CC)

    Count unique chord labels in the sequence  '''

def calculate_cc(chord_labels):
    chord_sequence = [label['root'] + label['tquality'] for label in chord_labels]
    return len(set(chord_sequence))



''' 3. Chord Tonal Distance (CTD)

    Calculate tonal distance between adjacent chords
    Average these distances across the sequence     '''

def chord_to_pcp(root, quality):
    """
    Convert a chord to a pitch class profile (PCP).
    
    Args:
        root: Root note of the chord (e.g., 'C', 'F+', 'B-')
        quality: Quality of the chord ('M', 'm', 'O', etc.)
    
    Returns:
        A 12-element array representing the pitch class profile
    """
    # Define pitch class mapping
    pitch_classes = {
        'C': 0, 'C+': 1, 'D-': 1, 'D': 2, 'D+': 3, 'E-': 3, 'E': 4,
        'F': 5, 'F+': 6, 'G-': 6, 'G': 7, 'G+': 8, 'A-': 8, 'A': 9,
        'A+': 10, 'B-': 10, 'B': 11
    }
    
    # Handle variant notations
    if root not in pitch_classes:
        # Convert sharps/flats notation if needed
        root_normalized = root.replace('♯', '+').replace('♭', '-')
        if root_normalized in pitch_classes:
            root = root_normalized
        else:
            # Default to C if unrecognized
            root = 'C'
    
    # Get root pitch class
    root_pc = pitch_classes.get(root, 0)
    
    # Initialize PCP
    pcp = np.zeros(12)
    
    # Define intervals for different chord qualities
    if quality == 'M':  # Major
        intervals = [0, 4, 7]  # Root, major third, perfect fifth
    elif quality == 'm':  # Minor
        intervals = [0, 3, 7]  # Root, minor third, perfect fifth
    elif quality == 'O':  # Other (augmented or diminished)
        intervals = [0, 3, 6]  # Approximation for dim/aug chords
    else:
        intervals = [0, 4, 7]  # Default to major if unknown
    
    # Add chord tones to PCP
    for interval in intervals:
        pcp[(root_pc + interval) % 12] = 1.0
    
    return pcp


def pcp_to_tonal_centroid(pcp):
    """
    Convert a single pitch class profile to a tonal centroid.
    
    Based on the compute_Tonal_centroids function but for a single vector.
    
    :param pcp: A 12-element array representing the pitch class profile
    :return: A 6-element array representing the tonal centroid
    """
    # Define transformation matrix - phi (same as in compute_Tonal_centroids)
    Pi = math.pi
    r1, r2, r3 = 1, 1, 0.5
    phi_0 = r1 * np.sin(np.array(range(12)) * 7 * Pi / 6)
    phi_1 = r1 * np.cos(np.array(range(12)) * 7 * Pi / 6)
    phi_2 = r2 * np.sin(np.array(range(12)) * 3 * Pi / 2)
    phi_3 = r2 * np.cos(np.array(range(12)) * 3 * Pi / 2)
    phi_4 = r3 * np.sin(np.array(range(12)) * 2 * Pi / 3)
    phi_5 = r3 * np.cos(np.array(range(12)) * 2 * Pi / 3)
    phi_ = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]
    phi = np.concatenate(phi_).reshape(6, 12)  # [6, 12]
    
    # Calculate tonal centroid (no time dimension here)
    tonal_centroid = phi.dot(pcp)
    
    return tonal_centroid


def euclidean(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
    point1 (array-like): First point coordinates
    point2 (array-like): Second point coordinates
    
    Returns:
    float: Euclidean distance between the points
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))


def calculate_ctd(chord_labels):
    if len(chord_labels) < 2:
        return 0
    
    distances = []
    for i in range(1, len(chord_labels)):
        # Convert chord labels to PCP features
        prev_chord = chord_to_pcp(chord_labels[i-1]['root'], chord_labels[i-1]['tquality'])
        curr_chord = chord_to_pcp(chord_labels[i]['root'], chord_labels[i]['tquality'])
        
        # Project to 6D tonal space
        prev_tc = pcp_to_tonal_centroid(prev_chord)
        curr_tc = pcp_to_tonal_centroid(curr_chord)
        
        # Calculate Euclidean distance
        distance = euclidean(prev_tc, curr_tc)
        distances.append(distance)
    
    return np.mean(distances)



''' 4. Chord Tone to Non-Chord Tone Ratio (CTnCTR)

    Count chord tones and non-chord tones in melody
    Identify proper non-chord tones (within 2 semitones of next note)
    Calculate ratio: (nc + np) / (nc + nn)              '''

def get_chord_notes(root, quality, with_octaves=False, base_octave=60):
    """
    Get the notes that make up a chord.
    
    Parameters:
    root (int or str): Root note of the chord as pitch class (0-11) or string representation
    quality (str): Chord quality/type (e.g., 'maj', 'min', '7', 'dim', etc.)
    with_octaves (bool): If True, returns actual pitches with octave information
                         If False, returns only pitch classes (0-11)
    base_octave (int): Base MIDI note number for the root when with_octaves=True
    
    Returns:
    list: List of pitch classes (0-11) or actual pitches depending on with_octaves parameter
    """
    # Convert root to integer if it's a string
    if isinstance(root, str):
        # If root is a note name like "C", "C#", etc.
        pitch_class_map = {'C': 0, 'C+': 1, 'D-': 1, 'D': 2, 'D+': 3, 'E-': 3, 
                          'E': 4, 'F': 5, 'F+': 6, 'G-': 6, 'G': 7, 'G+': 8, 
                          'A-': 8, 'A': 9, 'A+': 10, 'B-': 10, 'B': 11}
        root = pitch_class_map.get(root, 0)  # Default to C if not found
    else:
        # Ensure root is an integer
        root = int(root)
    
    # Define intervals for common chord types
    intervals = {
        'M': [0, 4, 7],         # Major: root, major 3rd, perfect 5th
        'm': [0, 3, 7],         # Minor: root, minor 3rd, perfect 5th
        'd': [0, 3, 6],         # Diminished: root, minor 3rd, diminished 5th
        'a': [0, 4, 8],         # Augmented: root, major 3rd, augmented 5th
        'D7': [0, 4, 7, 10],       # Dominant 7th: root, major 3rd, perfect 5th, minor 7th
        'M7': [0, 4, 7, 11],    # Major 7th: root, major 3rd, perfect 5th, major 7th
        'm7': [0, 3, 7, 10],    # Minor 7th: root, minor 3rd, perfect 5th, minor 7th
        'd7': [0, 3, 6, 9],     # Diminished 7th: root, minor 3rd, diminished 5th, diminished 7th
        'h7': [0, 3, 6, 10],   # Half-diminished 7th: root, minor 3rd, diminished 5th, minor 7th
    }

    
    # Get the intervals for the chord quality
    chord_intervals = intervals.get(quality, intervals['M'])
    
    if with_octaves:
        # Calculate the base octave where all notes will be
        base_pitch = base_octave - (base_octave % 12) + root
        
        # Generate chord pitches in the specified register
        return [base_pitch + interval for interval in chord_intervals]
    else:
        # Return only pitch classes (0-11)
        return [(root + interval) % 12 for interval in chord_intervals]


def calculate_ctnctr(melody_notes, chord_labels):
    n_chord_tones = 0
    n_nonchord_tones = 0
    n_proper_nonchord_tones = 0
    
    for i, (note, chord) in enumerate(zip(melody_notes, chord_labels)):
        # Skip rests
        if note['pitch'] == 0:
            continue
        
        # Get pitch class of melody note
        pitch_class = note['pitch'] % 12
        
        # Get chord tones based on root and quality
        chord_tones = get_chord_notes(chord['root'], chord['tquality'])
        
        if pitch_class in chord_tones:
            n_chord_tones += 1
        else:
            n_nonchord_tones += 1
            
            # Check if it's a proper non-chord tone (resolves within 2 semitones)
            if i < len(melody_notes) - 1 and melody_notes[i+1]['pitch'] > 0:
                next_pitch = melody_notes[i+1]['pitch']
                if abs(note['pitch'] - next_pitch) <= 2:
                    n_proper_nonchord_tones += 1
    
    # Calculate ratio
    if n_chord_tones + n_nonchord_tones == 0:
        return 0
    
    return (n_chord_tones + n_proper_nonchord_tones) / (n_chord_tones + n_nonchord_tones)



''' 5. Pitch Consonance Score (PCS)

    Calculate consonance scores between melody notes and chord notes
    Average these scores across 16th-note windows   '''

def calculate_pcs(melody_notes, chord_labels):
    consonance_scores = []
    
    for note, chord in zip(melody_notes, chord_labels):
        # Skip rests
        if note['pitch'] == 0:
            continue
        
        # Get melody pitch
        melody_pitch = note['pitch']
        
        # Get chord pitches based on root and quality
        chord_pitches = get_chord_notes(chord['root'], chord['tquality'])
        
        # Calculate consonance for each chord note
        note_scores = []
        for chord_pitch in chord_pitches:
            # Ensure melody note is higher (as per the paper)
            while chord_pitch > melody_pitch:
                chord_pitch -= 12
                
            # Calculate interval
            interval = (melody_pitch - chord_pitch) % 12
            
            # Assign consonance score
            if interval in [0, 3, 4, 7, 8, 9]:  # unison, M/m 3rd, P5th, M/m 6th
                score = 1
            elif interval == 5:  # perfect 4th
                score = 0
            else:  # dissonant intervals
                score = -1
                
            note_scores.append(score)
        
        # Average scores for this melody note
        consonance_scores.append(sum(note_scores)/len(note_scores) * note['duration'])
    
    # Weight by duration and calculate average
    total_duration = sum(note['duration'] for note in melody_notes if note['pitch'] > 0)
    if total_duration == 0:
        return 0
        
    return sum(consonance_scores) / total_duration



''' 6. Melody-Chord Tonal Distance (MCTD)

    Calculate tonal distance between melody notes and corresponding chords
    Weight by note duration     '''

def calculate_mctd(melody_notes, chord_labels):
    total_distance = 0
    total_duration = 0
    
    for note, chord in zip(melody_notes, chord_labels):
        # Skip rests
        if note['pitch'] == 0:
            continue
            
        # Get melody PCP (one-hot vector for pitch class)
        melody_pcp = np.zeros(12)
        melody_pcp[note['pitch'] % 12] = 1
        
        # Get chord PCP
        chord_pcp = chord_to_pcp(chord['root'], chord['tquality'])
        
        # Project to 6D tonal space
        melody_tc = pcp_to_tonal_centroid(melody_pcp)
        chord_tc = pcp_to_tonal_centroid(chord_pcp)
        
        # Calculate distance and weight by duration
        distance = euclidean(melody_tc, chord_tc) * note['duration']
        
        total_distance += distance
        total_duration += note['duration']
    
    if total_duration == 0:
        return 0
        
    return total_distance / total_duration



# --- Calculate chord evaluation metrics --- 


def evaluate_chord_predictions(predicted_chords, ground_truth_chords, melody_notes):
    """
    Evaluate predicted chord sequences against ground truth using multiple metrics
    
    :param predicted_chords: List of predicted chord labels with root and quality
    :param ground_truth_chords: List of ground truth chord labels with root and quality
    :param melody_notes: List of melody notes with pitch and duration
    :return: Dictionary of evaluation metrics
    """

    metrics = {}
    
    # Chord progression metrics
    metrics['predicted_CHE'] = calculate_che(predicted_chords)
    metrics['ground_truth_CHE'] = calculate_che(ground_truth_chords)
    metrics['CHE_difference'] = abs(metrics['predicted_CHE'] - metrics['ground_truth_CHE'])
    
    metrics['predicted_CC'] = calculate_cc(predicted_chords)
    metrics['ground_truth_CC'] = calculate_cc(ground_truth_chords)
    metrics['CC_difference'] = abs(metrics['predicted_CC'] - metrics['ground_truth_CC'])
    
    metrics['predicted_CTD'] = calculate_ctd(predicted_chords)
    metrics['ground_truth_CTD'] = calculate_ctd(ground_truth_chords)
    metrics['CTD_difference'] = abs(metrics['predicted_CTD'] - metrics['ground_truth_CTD'])
    
    # Chord/melody harmonicity metrics
    metrics['predicted_CTnCTR'] = calculate_ctnctr(melody_notes, predicted_chords)
    metrics['ground_truth_CTnCTR'] = calculate_ctnctr(melody_notes, ground_truth_chords)
    metrics['CTnCTR_difference'] = abs(metrics['predicted_CTnCTR'] - metrics['ground_truth_CTnCTR'])
    
    metrics['predicted_PCS'] = calculate_pcs(melody_notes, predicted_chords)
    metrics['ground_truth_PCS'] = calculate_pcs(melody_notes, ground_truth_chords)
    metrics['PCS_difference'] = abs(metrics['predicted_PCS'] - metrics['ground_truth_PCS'])
    
    metrics['predicted_MCTD'] = calculate_mctd(melody_notes, predicted_chords)
    metrics['ground_truth_MCTD'] = calculate_mctd(melody_notes, ground_truth_chords)
    metrics['MCTD_difference'] = abs(metrics['predicted_MCTD'] - metrics['ground_truth_MCTD'])
    
    # Overall accuracy (exact match percentage)
    correct_predictions = sum(1 for p, g in zip(predicted_chords, ground_truth_chords) 
                            if p['root'] == g['root'] and p['tquality'] == g['tquality'])
    metrics['chord_accuracy'] = correct_predictions / len(ground_truth_chords) if len(ground_truth_chords) > 0 else 0
    
    return metrics


"""
def calculate_metrics(predictions, ground_truth, melody_notes):
    # Filter out padding entries
    valid_indices = np.where(ground_truth[root] != 'pad')[0]
    valid_predictions = predictions[valid_indices]
    valid_ground_truth = ground_truth[valid_indices]
    valid_melody = [melody_notes[i] for i in valid_indices if i < len(melody_notes)]
    
    results = {}
    
    # Calculate metrics...
    results['CHE_pred'] = calculate_che(valid_predictions)
    results['CHE_truth'] = calculate_che(valid_ground_truth)
    results['CHE_diff'] = abs(results['CHE_pred'] - results['CHE_truth'])
    
    # Add other metrics similarly
    
    # Add chord accuracy
    matching = (valid_predictions['root'] == valid_ground_truth['root']) & (valid_predictions['tquality'] == valid_ground_truth['tquality'])
    results['accuracy'] = np.mean(matching)
    
    return results
"""