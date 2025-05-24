import numpy as np
import itertools
import numpy.lib.recfunctions as rfn
import math
import matplotlib.pyplot as plt
import openpyxl
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import euclidean
import pickle
from collections import Counter, OrderedDict
import os


### Adapted from HarmonyTransformerv2 https://github.com/Tsung-Ping/Harmony-Transformer-v2 ###

# --- Forward mapping ---
key_dict = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'a': 12, 'b': 13, 'C#': 14, 'C+': 14, 'D#': 15, 'D+': 15, 'E#': 16, 'E+': 16, 'F#': 17, 'F+': 17, 'G#': 18, 'G+': 18, 'A#': 19, 'A+': 19, 'B#': 20, 'B+': 20, 'c#': 21, 'c+': 21, 'd#': 22, 'd+': 22, 'e#': 23, 'e+': 23, 'f#': 24, 'f+': 24, 'g#': 25, 'g+': 25, 'a#': 26, 'a+': 26, 'b#': 27, 'b+': 27, 'Cb': 28, 'C-': 28, 'Db': 29, 'D-': 29, 'Eb': 30, 'E-': 30, 'Fb': 31, 'F-': 31, 'Gb': 32, 'G-': 32, 'Ab': 33, 'A-': 33, 'Bb': 34, 'B-': 34, 'cb': 35, 'c-': 35, 'db': 36, 'd-': 36, 'eb': 37, 'e-': 37, 'fb': 38, 'f-': 38, 'gb': 39, 'g-': 39, 'ab': 40, 'a-': 40, 'bb': 41, 'b-': 41, 'pad': 42}
quality_dict = {'M': 0, 'm': 1, 'a': 2, 'd': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9, 'pad': 10, 'a7': 2}
degree1_dict = {'none': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '-2': 8, '-7': 9, '+6': 10, 'pad': 11}
degree2_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '+1': 7, '+3': 8, '+4': 9, '-2': 10, '-3': 11, '-6': 12, '-7': 13, 'pad': 14}
inversion_dict = {'0': 0, '1': 1, '2': 2, '3': 3, 'pad': 4}
extra_info_dict = {'none': 0, '2': 1, '4': 2, '6': 3, '7': 4, '9': 5, '-2': 6, '-4': 7, '-6': 8, '-9': 9, '+2': 10, '+4': 11, '+5': 12, '+6': 13, '+7': 14, '+9': 15, '+72': 16, '72': 17, '62': 18, '42': 19, '64': 20, '94': 21, 'pad': 22}


def strided_axis1(a, window, hop):
    n_pad = window // 2
    b = np.lib.pad(a, ((0, 0), (n_pad, n_pad)), 'constant', constant_values=0)
    # Length of 3D output array along its axis=1
    nd1 = int((b.shape[1] - window) / hop) + 1
    # Store shape and strides info
    m, n = b.shape
    s0, s1 = b.strides
    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(b, shape=(nd1, m, window), strides=(s1 * hop, s0, s1))


def pianoroll2chromagram(pianoRoll, smoothing=False, window=17):
    """pianoRoll with shape = [88, time]"""
    pianoRoll_T = np.transpose(pianoRoll.astype(np.float32)) # [time, 88]
    pianoRoll_T_pad = np.pad(pianoRoll_T, [(0, 0), (9, 11)], 'constant') # [time, 108]
    pianoRoll_T = np.split(pianoRoll_T_pad, indices_or_sections=(pianoRoll_T_pad.shape[1]//12), axis=1) # [9, time, 12]
    chromagram_T = np.sum(pianoRoll_T, axis=0) # [time,  12]
    if smoothing:
        n_pad = window // 2
        chromagram_T_pad = np.pad(chromagram_T, ((n_pad, n_pad), (0, 0)), 'constant', constant_values=0)
        chromagram_T_smoothed = np.array([np.mean(chromagram_T_pad[(time+n_pad)-window//2:(time+n_pad)+window//2+1, :], axis=0) for time in range(chromagram_T.shape[0])])
        chromagram_T = chromagram_T_smoothed # [time,  12]
    L1_norm = chromagram_T.sum(axis=1) # [time]
    L1_norm[L1_norm == 0] = 1 # replace zeros with ones
    chromagram_T_norm = chromagram_T / L1_norm[:, np.newaxis] # L1 normalization, [time, 12]
    chromagram = np.transpose(chromagram_T_norm) # [12, time]
    return chromagram

# Edited
def load_pieces(resolution=4):
    """
    :param resolution: time resolution, default = 4 (16th note as 1unit in piano roll)
    :param representType: 'pianoroll' or 'onset_duration'
    :return: pieces, tdeviation
    """
    print('Message: load note data ...')
    dir = os.getcwd() + "\\Sonatas\\"
    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), 
          ('staffNum', 'int'), ('measure', 'int')] # datatype
    highest_pitch = 0
    lowest_pitch = 256
    pieces = {str(k): {'pianoroll': None, 'chromagram': None, 'start_time': None} for k in range(1,53)}
    for i in range(1,53):
        fileDir = dir + str(i) + "\\notes.csv"
        notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
        total_length = math.ceil((max(notes['onset'] + notes['duration']) - notes[0]['onset']) * resolution) # length of pianoroll
        start_time = notes[0]['onset']
        pianoroll = np.zeros(shape=[88, total_length], dtype=np.int32) # piano range: 21-108 (A0 to C8)
        for j, note in enumerate(notes):
            if note['duration'] == 0: # "Ornament"
                print(f"Skipping note {j} with duration 0")
                continue
            pitch = note['pitch']
            onset = int(math.floor((note['onset'] - start_time)*resolution))
            end = int(math.ceil((note['onset'] + note['duration'] - start_time)*resolution))
            if onset == end:
                print('no', i)
                print(f'Error: note onset = note end at {j}')
                exit(1)
            time = range(onset, end)
            pianoroll[pitch-21, time] = 1 # add note to representation
            if not 21 <= pitch <= 108:
                print(f"Skipping note {j} with pitch {pitch} outside range 21-108")
                continue
            if pitch > highest_pitch:
                highest_pitch = pitch
            if pitch < lowest_pitch:
                lowest_pitch = pitch

        pieces[str(i)]['pianoroll'] = pianoroll # [88, time]
        pieces[str(i)]['chromagram'] = pianoroll2chromagram(pianoroll) # [12, time]
        pieces[str(i)]['start_time'] = start_time
        print(f"Pianoroll shape: {pianoroll.shape}")
    #print('lowest pitch =', lowest_pitch, 'highest pitch = ', highest_pitch)
    #print(f"Piece: {i}, Note: {j}, Pitch: {pitch}, Onset: {onset}, End: {end}, Time: {time}")
    return pieces

# Edited
def load_chord_labels(vocabulary='MIREX_Mm'):
    print('Message: load chord labels...')
    dir = os.getcwd() + "\\Sonatas\\"
    # Update datatype to include a field for suspension/NC-tone as 'extra'
    dt = [('onset', 'float'), ('duration', 'float'), ('key', '<U10'), ('degree1', '<U10'), 
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), 
          ('rchord', '<U20'), ('extra_info', '<U10')] # datatype

    chord_labels = {str(k): None for k in range(1,53)}
    for i in range(1,53):
        fileDir = dir + str(i) + "\\chords.xlsx"
        workbook = openpyxl.load_workbook(fileDir) # Modified this for openpyxl
        sheet = workbook.active
        labels = []
        for row in sheet.iter_rows(min_row=1, values_only=True):  
            onset = row[0]
            duration = row[1] - row[0]
            key = row[2]
            quality = row[4]
            inversion = row[5]
            rchord = row[6]

            # Convert row[3] to string first to handle float values
            row3_str = str(row[3]) if row[3] is not None else ""
            if '/' not in row3_str:
                # For non-secondary chords, set degree1 to the actual value and degree2 to 'none'
                degree1 = row3_str
                degree2 = 'none'
            else:
                # For secondary chords, split by the slash
                parts = row3_str.split('/')
                degree2 = parts[0]
                degree1 = parts[1]

            # Extract suspension information from rchord
            extra_info = ''
            if '(' in rchord:
                start = rchord.find('(')
                end = rchord.find(')')
                if end > start:
                    extra_info = rchord[start+1:end]

            # Add suspension to the tuple
            labels.append((onset, duration, key, degree1, degree2, quality, inversion, rchord, extra_info))

        labels = np.array(labels, dtype=dt) # convert to structured array
        chord_labels[str(i)] = derive_chordSymbol_from_romanNumeral(labels, vocabulary) # translate rchords to tchords
    return chord_labels

# Edited
def get_framewise_labels(pieces, chord_labels, resolution=4):
    """
    :param pieces:
    :param chord_labels:
    :param resolution: time resolution, default=4 (16th note as 1 unit of a pianoroll)
    :return: images, image_labels
    """
    print("Message: get framewise labels ...")
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'), 
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U20'), 
          ('extra_info', '<U10'), ('root', '<U10'), ('tquality', '<U10')] # label datatype
    
    for p in range(1,53):
        # Split Piano Roll into frames of the same size (88, wsize)
        pianoroll = pieces[str(p)]['pianoroll'] # [88, time]
        labels = chord_labels[str(p)]
        start_time = pieces[str(p)]['start_time']

        # Calculate the end time of the last label
        if len(labels) > 0:
            last_label_end_time = max(labels['onset'] + labels['duration'])
            # Calculate how many frames we need based on the last label's end time
            # Add a small epsilon to ensure we include the final frame if it's at an exact boundary
            needed_frames = int(np.ceil((last_label_end_time - start_time) * resolution)) + 1
            # Ensure we don't exceed the actual pianoroll shape
            n_frames = min(needed_frames, pianoroll.shape[1])
        else:
            # If there are no labels, use the original shape
            n_frames = pianoroll.shape[1]

        #print(f"Number of frames (time steps) at {p}:", n_frames)
        frame_labels = []
        
        for n in range(n_frames):
            frame_time = n*(1/resolution) + start_time

            if n < len(labels):
                label_onset = labels[n]['onset']
                #print(f"Frame {n}: Frame Time = {frame_time}, Label Onset = {label_onset}")
            
            # NEW: try/except block that handles negative frame times
            try:
                # Find the label for this frame
                label = labels[(labels['onset'] <= frame_time) & (labels['onset'] + labels['duration'] >= frame_time)][0]
                
                # Extract row3_str (the field with degree information)
                row3_str = str(label['degree1'])
                
                # MODIFIED: Better extraction of degree1 and degree2
                if '/' not in row3_str:
                    # Not a secondary chord - but don't default to '1'
                    # Instead, extract the actual scale degree
                    degree1 = row3_str  # Replaced '1' with 'none' for non-secondary chords
                    degree2 = 'none'
                else:
                    # Secondary chord - extract both parts
                    parts = row3_str.split('/')
                    degree2 = parts[1]
                    degree1 = parts[0]
                
                # Create the frame label with the modified degree1
                frame_label = tuple([   str(p),
                                        frame_time,
                                        label['key'],
                                        label['degree1'],
                                        label['degree2'],
                                        label['quality'],
                                        label['inversion'],
                                        label['rchord'],
                                        label['extra_info'],
                                        label['root'],
                                        label['tquality']
                                    ])
                frame_labels.append(frame_label)
            
            except:
                print(f"Warning: No label found for frame {n} at time {frame_time}. Assigning default label.")
                default_label = tuple([str(p), frame_time, "rest", "none", "none", "none", 0, "none", "none", "none", "none"])
                frame_labels.append(default_label)
                
                print('Error: cannot get label !')
                print('piece =', p)
                print('frame time =', frame_time)
                exit(1)
                
        frame_labels = np.array(frame_labels, dtype=dt)
        actual_frames = len(frame_labels)
        chord_change = [1] + [0 if frame_labels[n]['root']+frame_labels[n]['tquality'] == frame_labels[n-1]['root']+frame_labels[n-1]['tquality'] else 1 for n in range(1, actual_frames)] # chord change labels
        chord_change = np.array([(cc) for cc in chord_change], dtype=[('chord_change', 'int')])
        pieces[str(p)]['label'] = rfn.merge_arrays([frame_labels, chord_change], flatten=True, usemask=False)
    return pieces


def load_dataset(resolution, vocabulary):
    pieces = load_pieces(resolution=resolution) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float}...}
    chord_labels = load_chord_labels(vocabulary=vocabulary) # {'no':  array}
    corpus = get_framewise_labels(pieces, chord_labels, resolution=resolution) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float, 'label': array, 'chord_change': array},  ...}
    pianoroll_lens = [x['pianoroll'].shape[1] for x in corpus.values()]
    print('max_length =', max(pianoroll_lens))
    print('min_length =', min(pianoroll_lens))
    print('keys in corpus[\'op\'] =', corpus['1'].keys())
    print('label fields = ', corpus['1']['label'].dtype)
    return corpus


def augment_data(corpus):
    print('Running Message: augment data...')
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'), 
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U20'),
          ('extra_info', '<U10'), ('root', '<U10'), ('tquality', '<U10'), ('chord_change', 'int')] # label datatype

    # Define note mappings
    note_to_number = {
        # Natural notes
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
        # Sharps
        'C+': 1, 'D+': 3, 'E+': 5, 'F+': 6, 'G+': 8, 'A+': 10, 'B+': 0,
        # Flats
        'C-': 11, 'D-': 1, 'E-': 3, 'F-': 4, 'G-': 6, 'A-': 8, 'B-': 10,
        # Lowercase versions (minor keys)
        'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11,
        'c+': 1, 'd+': 3, 'e+': 5, 'f+': 6, 'g+': 8, 'a+': 10, 'b+': 0,
        'c-': 11, 'd-': 1, 'e-': 3, 'f-': 4, 'g-': 6, 'a-': 8, 'b-': 10
    }
    
    number_to_note = {
        0: 'C', 1: 'C+', 2: 'D', 3: 'D+', 4: 'E', 5: 'F', 
        6: 'F+', 7: 'G', 8: 'G+', 9: 'A', 10: 'A+', 11: 'B'
    }
    
    number_to_note_minor = {
        0: 'c', 1: 'c+', 2: 'd', 3: 'd+', 4: 'e', 5: 'f', 
        6: 'f+', 7: 'g', 8: 'g+', 9: 'a', 10: 'a+', 11: 'b'
    }

    def shift_labels(label, shift):
        def shift_note(note, shift_amount):
            if note.upper() == 'NONE' or note.upper() == 'REST' or note.upper() == 'R' or note == 'pad':
                return note
            
            if '+' in note and '-' in note:
                # Normalize by keeping just the base note
                return note[0]
            
            # Determine if it's a minor key (lowercase first letter)
            is_minor = note[0].islower()
            
            # Convert to number, shift, then convert back
            try:
                note_num = note_to_number[note]
                shifted_num = (note_num + shift_amount) % 12
                
                # Use appropriate mapping based on major/minor
                if is_minor:
                    return number_to_note_minor[shifted_num]
                else:
                    return number_to_note[shifted_num]
            except KeyError:
                return note
        
        # Shift key and root
        original_key = label['key']
        original_root = label['root'] if 'root' in label.dtype.names else 'none'
        
        shifted_key = shift_note(original_key, shift)
        shifted_root = shift_note(original_root, shift)
        
        #print(f"Shifted: key from {original_key} to {shifted_key}, root from {original_root} to {shifted_root}")
        
        return (label['op'], 
                label['onset'], 
                shifted_key, 
                label['degree1'], 
                label['degree2'], 
                label['quality'], 
                label['inversion'], 
                label['rchord'], 
                label['extra_info'], 
                shifted_root, 
                label['tquality'] if 'tquality' in label.dtype.names else 'none', 
                label['chord_change'])

    # Rest of your augment_data function...
    corpus_aug = {}
    for shift in range(-3,7):
        shift_id = 'shift_' + str(shift)
        corpus_aug[shift_id] = {}
        for op in range(1,53):
            pianoroll_shift = np.roll(corpus[str(op)]['pianoroll'], shift=shift, axis=0)
            chromagram_shift = np.roll(corpus[str(op)]['chromagram'], shift=shift, axis=0)
            tonal_centroid = compute_Tonal_centroids(chromagram_shift)
            start_time = corpus[str(op)]['start_time']
            labels_shift = np.array([shift_labels(l, shift) for l in corpus[str(op)]['label']], dtype=dt)
            corpus_aug[shift_id][str(op)] = {'pianoroll': pianoroll_shift, 'tonal_centroid': tonal_centroid, 'start_time': start_time, 'label': labels_shift}
    print('keys in corpus_aug[\'shift_id\'][\'op\'] =', corpus_aug['shift_0']['1'].keys())
    return corpus_aug


def reshape_data(corpus_aug, n_steps=128, hop_size=16):
    print('Running Message: reshape data...')
    corpus_aug_reshape = copy.deepcopy(corpus_aug)
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'),
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U20'),
          ('extra_info', '<U10'), ('root', '<U10'), ('tquality', '<U10'), ('chord_change', 'int')]

    for shift_id, op_dict in corpus_aug.items():
        for op, piece in op_dict.items():
            #print(f"Shape of pianoroll in {shift_id}-{op}: {piece['pianoroll'].shape}")
            #print(f"Adding missing 'len' field to {shift_id}-{op}")
            
            # Initialize with proper empty arrays
            empty_pianoroll = np.zeros((1, n_steps, 88), dtype=np.float32)
            empty_tonal_centroid = np.zeros((1, n_steps, 6), dtype=np.float32)
            empty_label = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)
            empty_label = np.repeat(empty_label, n_steps).reshape(1, n_steps)

            corpus_aug_reshape[shift_id][op]['pianoroll'] = [empty_pianoroll.copy(), empty_pianoroll.copy()]
            corpus_aug_reshape[shift_id][op]['tonal_centroid'] = [empty_tonal_centroid.copy(), empty_tonal_centroid.copy()]
            corpus_aug_reshape[shift_id][op]['label'] = [empty_label.copy(), empty_label.copy()]
            corpus_aug_reshape[shift_id][op]['len'] = [np.array([n_steps], dtype=np.int32), np.array([n_steps], dtype=np.int32)]
            
            # Get data dimensions
            pianoroll = piece['pianoroll']
            tonal_centroid = piece['tonal_centroid']
            label_array = piece['label']
            array_size = label_array.shape[0]
            
            try:
                # IMPROVED RESHAPE APPROACH - CREATE SEQUENCES MANUALLY
                # Instead of reshaping the entire array at once, we'll create sequences one by one
                
                # Calculate how many full sequences we can make
                n_full_sequences = array_size // n_steps
                
                # If there's a remainder, we'll need one more sequence
                remainder = array_size % n_steps
                n_sequences = n_full_sequences + (1 if remainder > 0 else 0)
                
                # Create padding for the last sequence if needed
                label_padding = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)
                
                # Initialize arrays to store all our sequences
                non_overlapped_pianoroll = []
                non_overlapped_tc = []
                non_overlapped_label = []
                sequence_lengths = []
                
                # Create each sequence individually
                for i in range(n_sequences):
                    start_idx = i * n_steps
                    end_idx = min((i + 1) * n_steps, array_size)
                    
                    # Get slice of data for this sequence
                    pianoroll_slice = pianoroll[:, start_idx:end_idx].T  # Transpose to [time, 88]
                    tc_slice = tonal_centroid[:, start_idx:end_idx].T  # Transpose to [time, 6]
                    label_slice = label_array[start_idx:end_idx]
                    
                    # If this is the last sequence and it's not full, pad it
                    if end_idx - start_idx < n_steps:
                        pad_size = n_steps - (end_idx - start_idx)
                        
                        # Pad piano roll and tonal centroid
                        pianoroll_slice = np.pad(pianoroll_slice, [(0, pad_size), (0, 0)], 'constant')
                        tc_slice = np.pad(tc_slice, [(0, pad_size), (0, 0)], 'constant')
                        
                        # Pad label array
                        padding_array = np.array([label_padding[0]] * pad_size)
                        try:
                            label_slice = np.concatenate([label_slice, padding_array])
                        except Exception as e:
                            # If concatenation fails, create a new padded array
                            padded_label = np.zeros(n_steps, dtype=dt)
                            padded_label[:end_idx-start_idx] = label_slice
                            for j in range(end_idx-start_idx, n_steps):
                                padded_label[j] = label_padding[0]
                            label_slice = padded_label
                    
                    # Add to our sequence arrays
                    non_overlapped_pianoroll.append(pianoroll_slice)
                    non_overlapped_tc.append(tc_slice)
                    non_overlapped_label.append(label_slice)
                    
                    # Store the actual length (without padding)
                    sequence_lengths.append(min(n_steps, end_idx - start_idx))
                
                # Convert lists to arrays with proper shape
                non_overlapped_pianoroll = np.array(non_overlapped_pianoroll)
                non_overlapped_tc = np.array(non_overlapped_tc)
                non_overlapped_label = np.array(non_overlapped_label)
                
                # Store in our result structure
                corpus_aug_reshape[shift_id][op]['pianoroll'][0] = non_overlapped_pianoroll
                corpus_aug_reshape[shift_id][op]['tonal_centroid'][0] = non_overlapped_tc
                corpus_aug_reshape[shift_id][op]['label'][0] = non_overlapped_label
                corpus_aug_reshape[shift_id][op]['len'][0] = np.array(sequence_lengths, dtype=np.int32)
                
                # Handle overlapped sequences similarly
                if array_size >= n_steps + hop_size:
                    # We have enough data for overlapping
                    overlapped_pianoroll = []
                    overlapped_tc = []
                    overlapped_label = []
                    overlapped_lengths = []
                    
                    # Create overlapped sequences
                    for i in range(0, array_size - n_steps + 1, hop_size):
                        end_idx = i + n_steps
                        if end_idx <= array_size:
                            pianoroll_slice = pianoroll[:, i:end_idx].T
                            tc_slice = tonal_centroid[:, i:end_idx].T
                            label_slice = label_array[i:end_idx]
                            
                            overlapped_pianoroll.append(pianoroll_slice)
                            overlapped_tc.append(tc_slice)
                            overlapped_label.append(label_slice)
                            overlapped_lengths.append(n_steps)
                    
                    # Convert to arrays
                    overlapped_pianoroll = np.array(overlapped_pianoroll)
                    overlapped_tc = np.array(overlapped_tc)
                    overlapped_label = np.array(overlapped_label)
                    
                    # Store in result structure
                    corpus_aug_reshape[shift_id][op]['pianoroll'][1] = overlapped_pianoroll
                    corpus_aug_reshape[shift_id][op]['tonal_centroid'][1] = overlapped_tc
                    corpus_aug_reshape[shift_id][op]['label'][1] = overlapped_label
                    corpus_aug_reshape[shift_id][op]['len'][1] = np.array(overlapped_lengths, dtype=np.int32)
                else:
                    # Not enough data for overlapping, duplicate non-overlapped
                    corpus_aug_reshape[shift_id][op]['pianoroll'][1] = corpus_aug_reshape[shift_id][op]['pianoroll'][0]
                    corpus_aug_reshape[shift_id][op]['tonal_centroid'][1] = corpus_aug_reshape[shift_id][op]['tonal_centroid'][0]
                    corpus_aug_reshape[shift_id][op]['label'][1] = corpus_aug_reshape[shift_id][op]['label'][0]
                    corpus_aug_reshape[shift_id][op]['len'][1] = corpus_aug_reshape[shift_id][op]['len'][0]
                
            except Exception as e:
                print(f"Error processing sequences for {shift_id}-{op}: {e}")
                # Error handling is already in place with our initial empty arrays
            
    # Final check for any remaining dimension issues
    for shift_id in corpus_aug_reshape:
        for op in corpus_aug_reshape[shift_id]:
            for idx in [0, 1]:
                for field in ['pianoroll', 'tonal_centroid', 'label']:
                    # Check and fix dimensions if needed
                    if corpus_aug_reshape[shift_id][op][field][idx] is None:
                        if field == 'pianoroll':
                            corpus_aug_reshape[shift_id][op][field][idx] = np.zeros((1, n_steps, 88), dtype=np.float32)
                        elif field == 'tonal_centroid':
                            corpus_aug_reshape[shift_id][op][field][idx] = np.zeros((1, n_steps, 6), dtype=np.float32)
                        elif field == 'label':
                            empty_label = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)
                            corpus_aug_reshape[shift_id][op][field][idx] = np.repeat(empty_label, n_steps).reshape(1, n_steps)
                    
                    # Fix wrong dimensions
                    if field == 'label' and corpus_aug_reshape[shift_id][op][field][idx].ndim != 2:
                        print(f"Fixing dimension for {shift_id}-{op}-{field}-{idx}")
                        empty_label = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)
                        corpus_aug_reshape[shift_id][op][field][idx] = np.repeat(empty_label, n_steps).reshape(1, n_steps)
    
    return corpus_aug_reshape


def compute_Tonal_centroids(chromagram, filtering=True, sigma=8):
    # define transformation matrix - phi
    Pi = math.pi
    r1, r2, r3 = 1, 1, 0.5
    phi_0 = r1 * np.sin(np.array(range(12)) * 7 * Pi / 6)
    phi_1 = r1 * np.cos(np.array(range(12)) * 7 * Pi / 6)
    phi_2 = r2 * np.sin(np.array(range(12)) * 3 * Pi / 2)
    phi_3 = r2 * np.cos(np.array(range(12)) * 3 * Pi / 2)
    phi_4 = r3 * np.sin(np.array(range(12)) * 2 * Pi / 3)
    phi_5 = r3 * np.cos(np.array(range(12)) * 2 * Pi / 3)
    phi_ = [phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]
    phi = np.concatenate(phi_).reshape(6, 12) # [6, 12]
    phi_T = np.transpose(phi) # [12, 6]

    chromagram_T = np.transpose(chromagram) # [time, 12]
    TC_T = chromagram_T.dot(phi_T) # convert to tonal centiod representations, [time, 6]
    TC = np.transpose(TC_T) # [6, time]
    if filtering: # Gaussian filtering
        TC = gaussian_filter1d(TC, sigma=sigma, axis=1)
    return TC.astype(np.float32) # [6, time]


def rlabel_indexing(labels):
    def analyze_label(label):
        def analyze_degree(degree):
            if '/' not in degree:
                pri_degree = 1
                sec_degree = translate_degree(degree)
            else:
                sec_degree = degree.split('/')[0]
                pri_degree = degree.split('/')[1]
                sec_degree = translate_degree(sec_degree)
                pri_degree = translate_degree(pri_degree)
            return pri_degree, sec_degree
        key = label['key']
        degree = label['degree']
        quality = label['quality']
        inversion = label['inversion']

        key_idx = key_dict[key]
        pri_degree, sec_degree = analyze_degree(degree)
        pri_degree_idx = pri_degree - 1
        sec_degree_idx = sec_degree - 1
        quality_idx = quality_dict[quality]
        inversion_idx = inversion
        return (key_idx, pri_degree_idx, sec_degree_idx, quality_idx, inversion_idx)
    dt = [('key', int), ('pri_degree', int), ('sec_degree', int), ('quality', int), ('inversion', int)]
    return np.array([analyze_label(label) for label in labels], dtype=dt)


def split_dataset(input_features, input_TC, input_labels, input_cc_labels, input_lengths, sequence_info):
    print('Running Message: split dataset into training, validation and testing sets ...')

    # Updated indices to cover all 52 pieces (indices 0-51)
    s1 = [i for i in range(0, 52, 4)]   # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    s2 = [i for i in range(1, 52, 4)]   # [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
    s3 = [i for i in range(2, 52, 4)]   # [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]
    s4 = [i for i in range(3, 52, 4)]   # [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]
    train_indices = s1 + s2
    valid_indices = s3
    test_indices = s4

    feature_train = np.concatenate([input_features[m][p] for m in range(12) for p in train_indices], axis=0)
    feature_valid = np.concatenate([input_features[m][p] for m in range(12) for p in valid_indices], axis=0)
    feature_test = np.concatenate([input_features[0][p][::2] for p in test_indices], axis=0)

    TC_train = np.concatenate([input_TC[m][p] for m in range(12) for p in train_indices], axis=0)
    TC_valid = np.concatenate([input_TC[m][p] for m in range(12) for p in valid_indices], axis=0)
    TC_test = np.concatenate([input_TC[0][p][::2] for p in test_indices], axis=0)

    labels_train = np.concatenate([input_labels[m][p] for m in range(12) for p in train_indices], axis=0)
    labels_valid = np.concatenate([input_labels[m][p] for m in range(12) for p in valid_indices], axis=0)
    labels_test = np.concatenate([input_labels[0][p][::2] for p in test_indices], axis=0)

    cc_labels_train = np.concatenate([input_cc_labels[p] for m in range(12) for p in train_indices], axis=0)
    cc_labels_valid = np.concatenate([input_cc_labels[p] for m in range(12) for p in valid_indices], axis=0)
    cc_labels_test = np.concatenate([input_cc_labels[p][::2] for p in test_indices], axis=0)

    lens_train = list(itertools.chain.from_iterable([input_lengths[p] for m in range(12) for p in train_indices]))
    lens_valid = list(itertools.chain.from_iterable([input_lengths[p] for m in range(12) for p in valid_indices]))
    lens_test = list(itertools.chain.from_iterable([input_lengths[p][::2] for p in test_indices]))

    split_sets = {}
    split_sets['train'] = [sequence_info[p] for p in train_indices]
    split_sets['valid'] = [sequence_info[p] for p in valid_indices]
    split_sets['test'] = [(sequence_info[p][0], sequence_info[p][1]//2+1)  for p in test_indices]
    return feature_train, feature_valid, feature_test, \
           TC_train, TC_valid, TC_test, \
           labels_train, labels_valid, labels_test, \
           cc_labels_train, cc_labels_valid, cc_labels_test, \
           lens_train, lens_valid, lens_test, \
           split_sets

# Edited
def derive_chordSymbol_from_romanNumeral(labels, vocabulary):
    # Create scales of all keys
    temp = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    keys = {}
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

    # Translate chords
    tchords = []

    for rchord in labels:
        # print(str(rchord['key'])+': '+str(rchord['degree'])+', '+str(rchord['quality']))
        key = str(rchord['key'])
        degree1 = rchord['degree1']
        degree2 = rchord['degree2']

        # Extract parentheses content (e.g. '4' from 'V7(4)')
        rchord_str = str(rchord['rchord'])  # Convert rchord to string
        extra_info = ''

        if '(' in rchord_str and ')' in rchord_str:
            start = rchord_str.find('(')
            end = rchord_str.find(')')
            if end > start:
                extra_info = rchord_str[start+1:end]  # Extract '4'
            rchord_str = rchord_str[:start]  # Remove parentheses from main label
       
        # REVISED: Check degree2='none' for regular chords instead of degree1='1'
        if degree2 == 'none':  # Regular chord (not secondary)
            # Extract degree from degree1
            if len(degree1) == 1:  # Simple degree like '1', '4', '5'
                degree = int(degree1)
                root = keys[key][degree-1]
            else:  # Chromatic degree like '-2', '+6'
                if str(rchord['quality']) != 'a6':  # Normal chromatic chord
                    degree = int(degree1[1])
                    root = keys[key][degree-1]
                    if '+' not in root:
                        root += degree1[0]
                    else:
                        root = root[:-1]
                else:  # Augmented 6th chord
                    degree = 6
                    root = keys[key][degree-1]
                    if str(rchord['key'])[0].isupper():  # Major key
                        if '+' not in root:
                            root += '-'
                        else:
                            root = root[:-1]
        
        else:  # Secondary chord
            # Get the chord degree (from degree2) and tonicization target (from degree1)
            d2 = int(degree2) if degree2 != '+4' else 6  # The chord type in the secondary key
            d1 = int(degree1) if degree1[0] not in ['+', '-'] else \
                int(degree1[1]) * (-1 if degree1[0] == '-' else 1)  # The temporary tonic
            
            # Determine the secondary key
            if d1 > 0:
                key2 = keys[key][d1-1]  # Simple secondary key
            else:
                key2 = keys[key][abs(d1)-1]  # Handle negative degrees
                if '+' not in key2:
                    key2 += '-'
                else:
                    key2 = key2[:-1]
            
            # Find the root in the secondary key
            root = keys[key2][d2-1]
            
            # Special case for augmented fourth
            if degree2 == '+4':
                if key2.isupper():  # Major key
                    if '+' not in root:
                        root += '-'
                    else:
                        root = root[:-1]

            # Re-translate root for enharmonic equivalence
            if '++' in root:  # if root = x++
                # Convert F++ to G, C++ to D, etc.
                base_note = root[0]
                index_in_scale = temp.index(base_note)
                root = temp[(index_in_scale(root[0]) + 1) % 7] # Next note
            elif '--' in root:  # if root = x--
                base_note = root[0]
                index_in_scale = temp.index(base_note)
                root = temp[(index_in_scale(root[0]) - 1) % 7] # Previous note

            if '-' in root:  # case: root = x-
                if ('F' not in root) and ('C' not in root):  # case: root = x-, and x != F and C
                    root = temp[((temp.index(root[0])) - 1) % 7] + '+'
                else:
                    root = temp[((temp.index(root[0])) - 1) % 7]  # case: root = x-, and x == F or C
            elif ('+' in root) and ('E' in root or 'B' in root):  # case: root = x+, and x == E or B
                root = temp[((temp.index(root[0])) + 1) % 7]

        tquality = rchord['quality'] if rchord['quality'] != 'a6' else 'D7' # outputQ[rchord['quality']]

        # tquality mapping
        if vocabulary == 'MIREX_Mm':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'O', 'd': 'O', 'M7': 'M', 'D7': 'M', 'm7': 'm', 'h7': 'O', 'd7': 'O'} # 'O' stands for 'others'
        elif vocabulary == 'MIREX_7th':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'O', 'd': 'O', 'M7': 'M7', 'D7': 'D7', 'm7': 'm7', 'h7': 'O', 'd7': 'O'}
        elif vocabulary == 'triad':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'a', 'd': 'd', 'M7': 'M', 'D7': 'M', 'm7': 'm', 'h7': 'd', 'd7': 'd'}
        elif vocabulary == 'seventh':
            tquality_map_dict = {'M': 'M', 'm': 'm', 'a': 'a', 'd': 'd', 'M7': 'M7', 'D7': 'D7', 'm7': 'm7', 'h7': 'h7', 'd7': 'd7'}
        
        tquality = tquality_map_dict[tquality]
    
        # **MODIFICATION: Include extra_info in final chord representation**
        tchord = (root, tquality)
        tchords.append(tchord)

    # Define new dtype with an additional field for extra tones (if needed)
    tchords_dt = [('root', '<U10'), ('tquality', '<U10')] 

    tchords = np.array(tchords, dtype=tchords_dt)
    #print(f"  sample: {tchords[:2]}")
    rtchords = rfn.merge_arrays((labels, tchords), flatten=True, usemask=False) # merge rchords and tchords into one structured array
    return rtchords 


def translate_degree(degree_str):
    if ('+' not in degree_str and '-' not in degree_str) or ('+' in degree_str and degree_str[1] == '+'):
        degree_hot = int(degree_str[0])
    elif degree_str[0] == '-':
        degree_hot = int(degree_str[1]) + 14
    elif degree_str[0] == '+':
        degree_hot = int(degree_str[1]) + 7
    return degree_hot


def save_preprocessed_data(data, save_dir):
    with open(save_dir, 'wb') as save_file:
        pickle.dump(data, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    print('Preprocessed data saved.')


def main():
    vocabulary = 'MIREX_Mm'
    corpus = load_dataset(resolution=4, vocabulary=vocabulary) # {'no': {'pianoroll': 2d array, 'chromagram': 2d array, 'start_time': float, 'label': array},  ...}
    corpus_aug = augment_data(corpus) # {'shift_id': {'no': {'pianoroll': 2d array, 'chromagram': 2d array, tonal_centroid': 2d array, 'start_time': float, 'label': 1d array}, ...},  ...}
    corpus_aug_reshape = reshape_data(corpus_aug, n_steps=128, hop_size=16) # {'shift_id': {'no': {'pianoroll': 3d array, 'chromagram': 3d array, 'tonal_centroid': 3d array, 'start_time': float, 'label': 2d array, 'len': 2d array}, ...},  ...}

    # Add validation check here, before saving
    for shift_id in corpus_aug_reshape:
        for op in corpus_aug_reshape[shift_id]:
            for field in ['pianoroll', 'tonal_centroid', 'label', 'len']:
                for idx in [0, 1]:
                    assert corpus_aug_reshape[shift_id][op][field][idx] is not None, f"None value in {shift_id}-{op}-{field}-{idx}"
                    # Add dimension checks if desired
                    if field == 'pianoroll':
                        assert corpus_aug_reshape[shift_id][op][field][idx].ndim == 3, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'tonal_centroid':
                        assert corpus_aug_reshape[shift_id][op][field][idx].ndim == 3, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'label':
                        assert corpus_aug_reshape[shift_id][op][field][idx].ndim == 2, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'len':
                        assert corpus_aug_reshape[shift_id][op][field][idx].ndim == 1, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"

    # Save processed data
    dir = 'Sonatas_preprocessed_data_' + vocabulary + '.pickle'
    save_preprocessed_data(corpus_aug_reshape, save_dir=dir)

if __name__ == '__main__':
    main()