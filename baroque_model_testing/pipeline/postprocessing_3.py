import numpy as np
import pickle
from collections import Counter
from scipy.spatial.distance import euclidean
import math
from harmonicity_metrics_4 import evaluate_chord_predictions


# Reverse mapping
reverse_key_dict = {0: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G', 5: 'A', 6: 'B', 7: 'c', 8: 'd', 9: 'e', 10: 'f', 11: 'g', 12: 'a', 13: 'b', 14: 'C+', 15: 'D+', 16: 'E+', 17: 'F+', 18: 'G+', 19: 'A+', 20: 'B+', 21: 'c+', 22: 'd+', 23: 'e+', 24: 'f+', 25: 'g+', 26: 'a+', 27: 'b+', 28: 'C-', 29: 'D-', 30: 'E-', 31: 'F-', 32: 'G-', 33: 'A-', 34: 'B-', 35: 'c-', 36: 'd-', 37: 'e-', 38: 'f-', 39: 'g-', 40: 'a-', 41: 'b-', 42: 'pad'}
reverse_quality_dict = {0: 'M', 1: 'm', 2: 'a7', 3: 'd', 4: 'M7', 5: 'm7', 6: 'D7', 7: 'd7', 8: 'h7', 9: 'a6', 10: 'pad'}
reverse_degree1_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '-2', 8: '-7', 9: '+6', 10: 'pad'}
reverse_degree2_dict = {0: 'none', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '+1', 9: '+3', 10: '+4', 11: '-2', 12: '-3', 13: '-6', 14: '-7', 15: 'pad'}
reverse_inversion_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: 'pad'}
reverse_extra_info_dict = {0: 'none', 1: '2', 2: '4', 3: '6', 4: '7', 5: '9', 6: '-2', 7: '-4', 8: '-6', 9: '-9', 10: '+2', 11: '+4', 12: '+5', 13: '+6', 14: '+7', 15: '+9', 16: '+72', 17: '72', 18: '62', 19: '42', 20: '64', 21: '94', 22: 'pad'}

# --- Gather all chord metrics  ---

def prepare_comparison_data(test_results):

    """
    Compiles ground truth, predictions, and melody 
    from the test results pickle file.    

    Calls upon transform_indices_to_chord_symbols() 
    and extract_melody_notes().

    Exports 'all_metrics' and 'avg_metrics' to chord 
    evaluation metrics functions.
    """

    # Load ground truth and predictions from the evaluation pickle file
    with open(test_results, 'rb') as f:
        data = pickle.load(f)   

    # Extract ground truth and predictions
    ground_truth = data['baroque_test_data']
    predictions = data['baroque_on_baroque_preds']

    # Initialize metrics
    all_metrics = []

    n_sequences = ground_truth['key'].shape[0]

    for seq_idx in range(n_sequences):
        # Get sequence length (use 'len' array or just use full length)
        seq_len = ground_truth['len'][seq_idx] if seq_idx < len(ground_truth['len']) else 128       

        # Extract ground truth chord data for this sequence
        gt_keys = ground_truth['key'][seq_idx, :seq_len]
        gt_degree1 = ground_truth['degree1'][seq_idx, :seq_len]
        gt_degree2 = ground_truth['degree2'][seq_idx, :seq_len]
        gt_quality = ground_truth['quality'][seq_idx, :seq_len]
        gt_inversion = ground_truth['inversion'][seq_idx, :seq_len]
        
        # Extract prediction chord data for this sequence
        pred_keys = predictions['key'][seq_idx, :seq_len]
        pred_degree1 = predictions['degree1'][seq_idx, :seq_len]
        pred_degree2 = predictions['degree2'][seq_idx, :seq_len]
        pred_quality = predictions['quality'][seq_idx, :seq_len]
        pred_inversion = predictions['inversion'][seq_idx, :seq_len]

        # Transform to chord symbols
        gt_chords = [transform_indices_to_chord_symbols(
            gt_keys[i], gt_degree1[i], gt_degree2[i], gt_quality[i], gt_inversion[i]
        ) for i in range(seq_len)]
        
        pred_chords = [transform_indices_to_chord_symbols(
            pred_keys[i], pred_degree1[i], pred_degree2[i], pred_quality[i], pred_inversion[i]
        ) for i in range(seq_len)]
 
        # Extract melody notes from pianoroll
        pianoroll = ground_truth['pianoroll'][seq_idx, :seq_len]
        melody_notes = extract_melody_notes(pianoroll)
        
        # Calculate metrics
        metrics = evaluate_chord_predictions(gt_chords, pred_chords, melody_notes)
        metrics['sequence_idx'] = seq_idx
        all_metrics.append(metrics)     # Length = 170 for the number of sequences
    
        
    # Calculate average metrics across all sequences
    avg_metrics = {}    # Length = 19 for the number of metrics
    for key in all_metrics[0].keys():
        if key != 'sequence_idx':
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
 
    return avg_metrics, all_metrics   # all_metrics, avg_metrics



# --- Extract melody notes from pianoroll ---

def extract_melody_notes(pianoroll):
    """
    Extract melody notes from pianoroll representation.
    
    :param pianoroll: Piano roll array of shape [88, time]
    :return: List of melody notes with pitch and duration.
    """
    melody_notes = []
    current_note = None
    
    # For each time step
    for t in range(pianoroll.shape[1]):
        # Get active notes at this time step
        active_notes = np.where(pianoroll[:, t] > 0)[0]
        
        if len(active_notes) > 0:
            # Assume melody is the highest note
            highest_note = np.max(active_notes) + 21  # Add 21 to convert to MIDI pitch
            
            if current_note is None or current_note['pitch'] != highest_note:
                # End previous note if exists
                if current_note is not None:
                    melody_notes.append(current_note)
                
                # Start new note
                current_note = {'pitch': highest_note, 'onset': t, 'duration': 1}
            else:
                # Continue current note
                current_note['duration'] += 1
        elif current_note is not None:
            # End current note on silence
            melody_notes.append(current_note)
            current_note = None
    
    # Add last note if exists
    if current_note is not None:
        melody_notes.append(current_note)
    
    return melody_notes


# --- Reconstruct chord symbol ---

def transform_indices_to_chord_symbols(key_idx, degree1_idx, degree2_idx, quality_idx, inversion_idx):
    """
    Transform numeric indices to chord symbols (root and tquality).
    """

    key_str = reverse_key_dict[key_idx]

    if degree2_idx == 'none':
        degree1_str = str(degree1_idx)
        degree2_str = 'none'
    else:
        degree1_str = str(degree1_idx)
        degree2_str = str(degree2_idx)

    quality_str = reverse_quality_dict[quality_idx]
    
    chord_data = {
        'key' : key_str,
        'degree1' : degree1_str,
        'degree2' : degree2_str,
        'quality' : quality_str,
        'rchord' : ''
    }

    temp = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    keys = {}

    # Setup the key dictionary (same as in derive_chordSymbol_from_romanNumeral in preprocessing)
    # This creates the mapping from keys to scales
    for i in range(11):
        majtonic = temp[(i * 4) % 7] + int(i / 7) * '+' + int(i % 7 > 5) * '+'
        mintonic = temp[(i * 4 - 2) % 7].lower() + int(i / 7) * '+' + int(i % 7 > 2) * '+'
        
        scale = list(temp)
        for j in range(i):
            scale[(j + 1) * 4 % 7 - 1] += '+'
        majscale = scale[(i * 4) % 7:] + scale[:(i * 4) % 7]
        minscale = scale[(i * 4 + 5) % 7:] + scale[:(i * 4 + 5) % 7]
        minscale[6] += '+'
        keys[majtonic] = majscale
        keys[mintonic] = minscale
    
    for i in range(1, 9):
        majtonic = temp[(i * 3) % 7] + int(i / 7) * '-' + int(i % 7 > 1) * '-'
        mintonic = temp[(i * 3 - 2) % 7].lower() + int(i / 7) * '-' + int(i % 7 > 4) * '-'
        scale = list(temp)
        for j in range(i):
            scale[(j + 2) * 3 % 7] += '-'
        majscale = scale[(i * 3) % 7:] + scale[:(i * 3) % 7]
        minscale = scale[(i * 3 + 5) % 7:] + scale[:(i * 3 + 5) % 7]
        if len(minscale[6]) == 1:
            minscale[6] += '+'
        else:
            minscale[6] = minscale[6][:-1]
        keys[majtonic] = majscale
        keys[mintonic] = minscale

    key = chord_data['key']
    degree1 = chord_data['degree1']
    degree2 = chord_data['degree2']

    # Check if key is in the dictionary
    if key not in keys:
        if key.lower() in keys:
            key = key.lower()
        elif key.upper() in keys:
            key = key.upper()
        else:
            # Default to C major
            key = 'C'
    
    # Check degree2='none' for regular chords
    if degree2 == 'none' or degree2 =='0':  # Regular chord (not secondary)
        try:
            degree = int(degree1)
            if 1 <= degree <= 7:
                root = keys[key][degree-1]
            else:
                # Default to tonic if degree is out of range
                root = keys[key][0]
        except ValueError:
            # Handle chromatic alterations like '+4' or '-6'
            if degree1.startswith('+') or degree1.startswith('-'):
                # Get the degree number
                degree = int(degree1[1:])
                if 1 <= degree <= 7:
                    root = keys[key][degree-1]
                    # Apply the alteration
                    if degree1.startswith('+'):
                        if '+' not in root:
                            root += '+'
                    else:   # degree1.startswith('-')
                        if '+' in root:
                            root = root[:-1]
                        else:
                            root += '-'
                else:
                    # Default to tonic if degree is out of range
                    root = keys[key][0]
            else:
                # Default to tonic for any othe rformat
                root = keys[key][0]
    else:   # Secondary chord
        try:
            d1 = int(degree1)
            d2 = int(degree2)

            if d1 > 0 and d1 <= 7:
                key2 = keys[key][d1-1]  # Secondary key
            else:
                key2 = keys[key][0] # Default to secondary tonic
            
            if d2 > 0 and d2 <= 7:
                root = keys[key][d2-1]  # Root in secondary key
            else:
                root = keys[key2][0]
        except ValueError:
            # Default handling for any parsing errors
            root = keys[key][0]
    

    # Handle double sharps and other enharmonic spellings
    if '++' in root:  # if root = x++
        # Convert F++ to G, C++ to D, etc.
        base_note = root[0]
        index_in_scale = temp.index(base_note)
        root = temp[(index_in_scale + 1) % 7]  # Next note
    elif '--' in root:  # if root = x--
        base_note = root[0]
        index_in_scale = temp.index(base_note)
        root = temp[(index_in_scale - 1) % 7]  # Previous note

    if '-' in root:  # case: root = x-
        if ('F' not in root) and ('C' not in root):  # case: root = x-, and x != F and C
            root = temp[((temp.index(root[0])) - 1) % 7] + '+'
        else:
            root = temp[((temp.index(root[0])) - 1) % 7]  # case: root = x-, and x == F or C
    elif ('+' in root) and ('E' in root or 'B' in root):  # case: root = x+, and x == E or B
        root = temp[((temp.index(root[0])) + 1) % 7]

    # Get quality
    quality_str = chord_data['quality']

    # Map to tquality using MIREX_Mm vocabulary 
    tquality_map = {
        'M': 'M', 'm': 'm', 'a': 'O', 'd': 'O', 
        'M7': 'M', 'D7': 'M', 'm7': 'm', 'h7': 'O', 'd7': 'O'
    }
    
    tquality = tquality_map.get(quality_str, 'M')  # Default to major if not found
    
    return {'root': root, 'tquality': tquality}    # Maybe 'quality', we'll see


#baroque_test_results = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_testing\eval_results\out_of_distribution\baroque_model_evaluation_results.pkl"
#eval_results = prepare_comparison_data(baroque_test_results)
#print(eval_results)