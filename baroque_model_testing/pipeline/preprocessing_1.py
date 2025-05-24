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


def load_pieces(resolution=4):
    """
    Load the test set note data
    
    :param resolution: time resolution, default = 4 (16th note as 1unit in piano roll)
    :return: pieces dictionary containing pianoroll and chromagram for each piece
    """
    print('Message: loading test set note data...')
    # Updated directory path for the test set
    base_dir = os.path.join(os.getcwd(), "cross_composer") 
    
    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), 
          ('staffNum', 'int'), ('measure', 'int')] # datatype
    highest_pitch = 0
    lowest_pitch = 256
    
    # Initialize pieces dictionary for the test set with 10 files
    pieces = {str(k): {'pianoroll': None, 'chromagram': None, 'start_time': None} for k in range(1, 11)}
    
    for i in range(1, 11):
        fileDir = os.path.join(base_dir, str(i), "notes.csv")
        # Check if file exists
        if not os.path.exists(fileDir):
            print(f"Warning: File {fileDir} not found. Skipping...")
            continue
            
        try:
            notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
            
            if len(notes) == 0:
                print(f"Warning: No notes found in file {fileDir}. Skipping...")
                continue
                
            total_length = math.ceil((max(notes['onset'] + notes['duration']) - notes[0]['onset']) * resolution) # length of pianoroll
            start_time = notes[0]['onset']
            pianoroll = np.zeros(shape=[88, total_length], dtype=np.int32) # piano range: 21-108 (A0 to C8)
            
            for j, note in enumerate(notes):
                if note['duration'] == 0: # "Ornament"
                    continue
                pitch = note['pitch']
                onset = int(math.floor((note['onset'] - start_time)*resolution))
                end = int(math.ceil((note['onset'] + note['duration'] - start_time)*resolution))
                
                if onset == end:
                    print(f'Warning: note onset = note end at file {i}, note {j}')
                    end = onset + 1  # Fix by adding a small duration
                    
                time = range(onset, end)
                # Check if pitch is in valid range
                if 21 <= pitch <= 108:
                    pianoroll[pitch-21, time] = 1 # add note to representation
                else:
                    print(f'Warning: Pitch {pitch} out of range in file {i}, note {j}')

                if pitch > highest_pitch:
                    highest_pitch = pitch
                if pitch < lowest_pitch:
                    lowest_pitch = pitch

            pieces[str(i)]['pianoroll'] = pianoroll # [88, time]
            pieces[str(i)]['chromagram'] = pianoroll2chromagram(pianoroll) # [12, time]
            pieces[str(i)]['start_time'] = start_time
            
        except Exception as e:
            print(f"Error processing file {fileDir}: {e}")
            
    print('lowest pitch =', lowest_pitch, 'highest pitch = ', highest_pitch)
    return pieces


def load_chord_labels(vocabulary='MIREX_Mm'):
    
    #Load chord labels from the test set
    
    #:param vocabulary: chord vocabulary type
    #:return: chord_labels dictionary containing chord annotations for each piece
    
    print('Message: loading test set chord labels...')
    # Updated directory path for the test set
    base_dir = os.path.join(os.getcwd(), "cross_composer") 
    
    # Update datatype to include a field for suspension/NC-tone as 'extra'
    dt = [('onset', 'float'), ('duration', 'float'), ('key', '<U10'), ('degree1', '<U10'), 
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), 
          ('rchord', '<U20'), ('extra_info', '<U10')] # datatype

    chord_labels = {str(k): None for k in range(1, 11)}
    
    for i in range(1, 11):
        fileDir = os.path.join(base_dir, str(i), "chords.xlsx")
        # Check if file exists
        if not os.path.exists(fileDir):
            print(f"Warning: File {fileDir} not found. Skipping...")
            continue
        
        try:
            workbook = openpyxl.load_workbook(fileDir)
            sheet = workbook.active
            labels = []
            
            for row in sheet.iter_rows(min_row=1, values_only=True):  
                # Skip rows with missing essential values
                if len(row) < 7 or row[0] is None or row[1] is None:
                    continue
                    
                onset = row[0]
                duration = row[1] - row[0]
                key = row[2] 
                quality = row[4]
                inversion = row[5]
                rchord = row[6]

                # Convert row[3] to string first to handle float values
                row3_str = str(row[3]) if row[3] is not None else ""
                if '/' not in row3_str:
                    degree1 = row3_str
                    degree2 = 'none'
                else:
                    parts = row3_str.split('/')
                    degree2 = parts[0]
                    degree1 = parts[1]

                # Extract suspension information from rchord
                extra_info = ''
                if rchord and '(' in rchord:
                    start = rchord.find('(')
                    end = rchord.find(')')
                    if end > start:
                        extra_info = rchord[start+1:end]

                # Add suspension to the tuple
                labels.append((onset, duration, key, degree1, degree2, quality, inversion, rchord, extra_info))

            if labels:
                labels = np.array(labels, dtype=dt) # convert to structured array
                chord_labels[str(i)] = derive_chordSymbol_from_romanNumeral(labels, vocabulary) # translate rchords to tchords
            else:
                print(f"Warning: No valid chord labels found in {fileDir}")
                
        except Exception as e:
            print(f"Error processing file {fileDir}: {e}")
            
    return chord_labels


def get_framewise_labels(pieces, chord_labels, resolution=4):
    """
    Convert chord labels to framewise labels (one label per time step)
    
    :param pieces: dictionary of pieces with pianoroll data
    :param chord_labels: dictionary of chord labels
    :param resolution: time resolution (16th note = 4)
    :return: updated pieces with framewise labels
    """
    print("Message: get framewise labels for test set...")
    dt = [('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'),
          ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U20'),
          ('extra_info', '<U10'), ('root', '<U10'), ('tquality', '<U10')]
    
    for p in range(1, 11):
        # Skip if this piece wasn't successfully loaded
        if str(p) not in pieces or pieces[str(p)]['pianoroll'] is None or str(p) not in chord_labels or chord_labels[str(p)] is None:
            print(f"Skipping framewise labels for piece {p} - missing data")
            continue
            
        # Split Piano Roll into frames of the same size (88, wsize)
        pianoroll = pieces[str(p)]['pianoroll'] # [88, time]
        labels = chord_labels[str(p)]
        start_time = pieces[str(p)]['start_time']

        # Calculate the end time of the last label
        if len(labels) > 0:
            last_label_end_time = max(labels['onset'] + labels['duration'])
            # Calculate how many frames we need based on the last label's end time
            needed_frames = int(np.ceil((last_label_end_time - start_time) * resolution)) + 1
            # Ensure we don't exceed the actual pianoroll shape
            n_frames = min(needed_frames, pianoroll.shape[1])
        else:
            # If there are no labels, use the original shape
            n_frames = pianoroll.shape[1]

        #print(f"Number of frames (time steps) for test piece {p}:", n_frames)
        frame_labels = []
        
        for n in range(n_frames):
            frame_time = n*(1/resolution) + start_time

            try:
                # Special handling for negative frame times
                if frame_time < 0:
                    # Look for any labels that might include this negative time
                    matching_labels = labels[(labels['onset'] <= frame_time) & (labels['onset'] + labels['duration'] >= frame_time)]
                    if len(matching_labels) > 0:
                        # We found a label that includes this negative time
                        label = matching_labels[0]
                    elif len(labels) > 0:
                        # Use the first label as a fallback for negative times
                        label = labels[0]
                    else:
                        # Force it to go to the except block if there are no labels
                        raise IndexError("No labels available")
                else:
                    # Original label matching for non-negative times
                    matching_labels = labels[(labels['onset'] <= frame_time) & (labels['onset'] + labels['duration'] >= frame_time)]
                    if len(matching_labels) > 0:
                        label = matching_labels[0]
                    else:
                        raise IndexError("No matching label found")
        
                #frame_label = tuple([str(p), frame_time] + list(label)[2:] + [label['key_idx'], label['quality_idx']])
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
            
            except Exception as e:
                print(f"No label found for piece {p}, frame {n} at time {frame_time}. Assigning default label. Error: {e}")
                default_label = tuple([str(p), frame_time, "C", "1", "1", "M", 0, "I", "", "C", "M"])
                frame_labels.append(default_label)
                
        frame_labels = np.array(frame_labels, dtype=dt)
        actual_frames = len(frame_labels)
        
        chord_change = [1] + [0 if frame_labels[n]['root']+frame_labels[n]['tquality'] == frame_labels[n-1]['root']+frame_labels[n-1]['tquality'] 
                             else 1 for n in range(1, actual_frames)] # chord change labels
        chord_change = np.array([(cc) for cc in chord_change], dtype=[('chord_change', 'int')])
        
        pieces[str(p)]['label'] = rfn.merge_arrays([frame_labels, chord_change], flatten=True, usemask=False)
        
    return pieces


def compute_Tonal_centroids(chromagram, filtering=True, sigma=8):
    """
    Compute the tonal centroid feature from a chromagram
    
    :param chromagram: chromagram [12, time]
    :param filtering: whether to apply Gaussian filtering
    :param sigma: sigma for Gaussian filtering
    :return: tonal centroid [6, time]
    """
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


# Dictionaries for mapping label values to indices


def derive_chordSymbol_from_romanNumeral(labels, vocabulary):
    """
    Translate Roman numeral chord labels to chord symbols.

    :param labels: structured array of Roman numeral labels
    :param vocabulary: chord vocabulary type
    :return: structured array with chord symbols added
    """
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

        rchord_str = str(rchord['rchord'])  # Convert rchord to string
        extra_info = ''

        # Extract parentheses content (e.g. '4' from 'V7(4)')
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
                try:
                    degree = int(degree1)
                    # Add this line to print the successful conversion for debugging
                    print(f"Successfully converted '{degree1}' to int {degree}")
                except ValueError as e:
                    print(f"Error converting '{degree1}' (type: {type(degree1)}) to int at row with onset {onset}")
                    raise  # Re-raise the error to see the full traceback
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
                root = temp[(temp.index(root[0]) + 1) % 7]
            elif '--' in root:  # if root = x--
                root = temp[(temp.index(root[0]) - 1) % 7]

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
    
        # Include extra_info in final chord representation
        tchord = (root, tquality)
        tchords.append(tchord)

    # Define dtype with fields for root and tquality
    tchords_dt = [('root', '<U10'), ('tquality', '<U10')] 

    tchords = np.array(tchords, dtype=tchords_dt)
    rtchords = rfn.merge_arrays((labels, tchords), flatten=True, usemask=False) # merge rchords and tchords
    return rtchords



def load_test_dataset(resolution, vocabulary):
    """
    Load and process the test dataset
    
    :param resolution: time resolution
    :param vocabulary: chord vocabulary
    :return: dictionary of test pieces with pianoroll, chromagram, and labels
    """
    pieces = load_pieces(resolution=resolution)
    chord_labels = load_chord_labels(vocabulary=vocabulary)
    corpus = get_framewise_labels(pieces, chord_labels, resolution=resolution)
    
    # Add tonal centroid to each piece
    for piece_id, piece_data in corpus.items():
        if piece_data['pianoroll'] is not None and 'chromagram' in piece_data:
            piece_data['tonal_centroid'] = compute_Tonal_centroids(piece_data['chromagram'])
            
    # Filter out any pieces that failed to load
    valid_pieces = {}
    for piece_id, piece_data in corpus.items():
        if piece_data['pianoroll'] is not None and 'label' in piece_data:
            valid_pieces[piece_id] = piece_data
            
    print(f"Successfully processed {len(valid_pieces)} out of 10 test pieces")
    
    if valid_pieces:
        pianoroll_lens = [x['pianoroll'].shape[1] for x in valid_pieces.values()]
        print('max_length =', max(pianoroll_lens))
        print('min_length =', min(pianoroll_lens))
        print('keys in test corpus[piece_id] =', list(list(valid_pieces.values())[0].keys()))
        if 'label' in list(valid_pieces.values())[0]:
            print('label fields = ', list(valid_pieces.values())[0]['label'].dtype.names)
    
    return valid_pieces
    

def reshape_test_data(corpus, n_steps=128, hop_size=16):
    print('Running Message: reshape test data...')
    corpus_reshape = {'shift_0': {}}
    dt = [
        ('op', '<U10'), ('onset', 'float'), ('key', '<U10'), ('degree1', '<U10'),
        ('degree2', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U20'),
        ('extra_info', '<U10'), ('root', '<U10'), ('tquality', '<U10'), ('chord_change', 'int')
    ]

    for op, piece in corpus.items():
        print(f"Shape of pianoroll in test piece {op}: {piece['pianoroll'].shape}")

        empty_pianoroll = np.zeros((1, n_steps, 88), dtype=np.float32)
        empty_tonal_centroid = np.zeros((1, n_steps, 6), dtype=np.float32)
        empty_label = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)
        empty_label = np.repeat(empty_label, n_steps).reshape(1, n_steps)

        corpus_reshape['shift_0'][op] = {
            'pianoroll': [empty_pianoroll.copy(), empty_pianoroll.copy()],
            'tonal_centroid': [empty_tonal_centroid.copy(), empty_tonal_centroid.copy()],
            'label': [empty_label.copy(), empty_label.copy()],
            'len': [np.array([n_steps], dtype=np.int32), np.array([n_steps], dtype=np.int32)]
        }

        pianoroll = piece['pianoroll']
        tonal_centroid = piece['tonal_centroid']
        label_array = piece['label']
        array_size = label_array.shape[0]

        # --- Non-Overlapping Sequences (Index [0]) ---
        non_overlapped_pianoroll_list = []
        non_overlapped_tc_list = []
        non_overlapped_label_list = []
        sequence_lengths = []

        n_full_sequences = array_size // n_steps
        remainder = array_size % n_steps
        n_sequences = n_full_sequences + (1 if remainder > 0 else 0)

        label_padding = np.array([(op, -1, 'pad', 'pad', 'pad', 'pad', -1, 'pad', 'pad', 'pad', 'pad', 0)], dtype=dt)

        for i in range(n_sequences):
            start_idx = i * n_steps
            end_idx = min((i + 1) * n_steps, array_size)

            pianoroll_slice = pianoroll[:, start_idx:end_idx].T
            tc_slice = tonal_centroid[:, start_idx:end_idx].T
            label_slice = label_array[start_idx:end_idx]
            seq_len = end_idx - start_idx

            if seq_len < n_steps:
                pad_size = n_steps - seq_len
                pianoroll_slice = np.pad(pianoroll_slice, [(0, pad_size), (0, 0)], 'constant')
                tc_slice = np.pad(tc_slice, [(0, pad_size), (0, 0)], 'constant')
                padding_array = np.array([label_padding[0]] * pad_size)
                label_slice = np.concatenate([label_slice, padding_array])

            non_overlapped_pianoroll_list.append(pianoroll_slice)
            non_overlapped_tc_list.append(tc_slice)
            non_overlapped_label_list.append(label_slice)
            sequence_lengths.append(seq_len)

        corpus_reshape['shift_0'][op]['pianoroll'][0] = np.array(non_overlapped_pianoroll_list)
        corpus_reshape['shift_0'][op]['tonal_centroid'][0] = np.array(non_overlapped_tc_list)
        corpus_reshape['shift_0'][op]['label'][0] = np.array(non_overlapped_label_list)
        corpus_reshape['shift_0'][op]['len'][0] = np.array(sequence_lengths, dtype=np.int32)

        # --- Overlapping Sequences (Index [1]) ---
        if array_size >= n_steps + hop_size:
            overlapped_pianoroll_list = []
            overlapped_tc_list = []
            overlapped_label_list = []
            overlapped_lengths = []

            for i in range(0, array_size - n_steps + 1, hop_size):
                end_idx = i + n_steps
                pianoroll_slice = pianoroll[:, i:end_idx].T
                tc_slice = tonal_centroid[:, i:end_idx].T
                label_slice = label_array[i:end_idx]

                overlapped_pianoroll_list.append(pianoroll_slice)
                overlapped_tc_list.append(tc_slice)
                overlapped_label_list.append(label_slice)
                overlapped_lengths.append(n_steps)

            corpus_reshape['shift_0'][op]['pianoroll'][1] = np.array(overlapped_pianoroll_list)
            corpus_reshape['shift_0'][op]['tonal_centroid'][1] = np.array(overlapped_tc_list)
            corpus_reshape['shift_0'][op]['label'][1] = np.array(overlapped_label_list)
            corpus_reshape['shift_0'][op]['len'][1] = np.array(overlapped_lengths, dtype=np.int32)
        else:
            # Duplicate non-overlapping if not enough data for overlapping
            corpus_reshape['shift_0'][op]['pianoroll'][1] = corpus_reshape['shift_0'][op]['pianoroll'][0]
            corpus_reshape['shift_0'][op]['tonal_centroid'][1] = corpus_reshape['shift_0'][op]['tonal_centroid'][0]
            corpus_reshape['shift_0'][op]['label'][1] = corpus_reshape['shift_0'][op]['label'][0]
            corpus_reshape['shift_0'][op]['len'][1] = corpus_reshape['shift_0'][op]['len'][0]

    return corpus_reshape
    

def save_preprocessed_data(data, save_dir):
    """
    Save the preprocessed data to a pickle file
    
    :param data: data to save
    :param save_dir: file path to save to
    """
    with open(save_dir, 'wb') as save_file:
        pickle.dump(data, save_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Preprocessed test data saved to {save_dir}')


def process_test_data_matching_train_format(vocabulary='MIREX_Mm', resolution=4, n_steps=128, save=True):
    """
    Process test data to match the training data format
    
    :param vocabulary: chord vocabulary type
    :param resolution: time resolution
    :param n_steps: sequence length
    :param save: whether to save the processed data
    :return: processed test data
    """
    print(f"Processing test data with vocabulary: {vocabulary}, resolution: {resolution}, sequence length: {n_steps}")
    
    # Step 1: Load the raw test data
    test_corpus = load_test_dataset(resolution=resolution, vocabulary=vocabulary)
    
    # Step 2: Reshape the data to match the training data format
    reshaped_corpus = reshape_test_data(test_corpus, n_steps=n_steps)
    
    # Save the processed data if requested
    if save:
        save_dir = f'test_data_preprocessed_{vocabulary}_train_format.pickle'
        save_preprocessed_data(reshaped_corpus, save_dir)
    
    return reshaped_corpus


def validate_test_data_format(data):
    """
    Validate that the test data is properly formatted
    
    :param data: test data to validate
    :return: True if valid, False otherwise
    """
    valid = True
    try:
        # Check the basic structure
        if 'shift_0' not in data:
            print("ERROR: Missing 'shift_0' key in data")
            valid = False
            return valid
            
        # Check each piece
        for op, piece_data in data['shift_0'].items():
            # Check required keys
            for key in ['pianoroll', 'tonal_centroid', 'label', 'len']:
                if key not in piece_data:
                    print(f"ERROR: Missing '{key}' key in piece {op}")
                    valid = False
                    continue
                    
                # Check list structure for each key
                if not isinstance(piece_data[key], list) or len(piece_data[key]) != 2:
                    print(f"ERROR: '{key}' for piece {op} should be a list of length 2")
                    valid = False
                    continue
            
            # Check dimensions
            try:
                # Pianoroll should be [sequences, n_steps, 88]
                for idx in [0, 1]:
                    if piece_data['pianoroll'][idx].ndim != 3 or piece_data['pianoroll'][idx].shape[2] != 88:
                        print(f"ERROR: Wrong dimensions for pianoroll[{idx}] in piece {op}")
                        valid = False
                    
                    # Tonal centroid should be [sequences, n_steps, 6]
                    if piece_data['tonal_centroid'][idx].ndim != 3 or piece_data['tonal_centroid'][idx].shape[2] != 6:
                        print(f"ERROR: Wrong dimensions for tonal_centroid[{idx}] in piece {op}")
                        valid = False
                    
                    # Label should be [sequences, n_steps]
                    if piece_data['label'][idx].ndim != 2:
                        print(f"ERROR: Wrong dimensions for label[{idx}] in piece {op}")
                        valid = False
                    
                    # Length should be [sequences]
                    if piece_data['len'][idx].ndim != 1:
                        print(f"ERROR: Wrong dimensions for len[{idx}] in piece {op}")
                        valid = False
                    
                    # Consistency check: pianoroll and tonal_centroid should have same batch and sequence length
                    if piece_data['pianoroll'][idx].shape[0] != piece_data['tonal_centroid'][idx].shape[0] or \
                       piece_data['pianoroll'][idx].shape[1] != piece_data['tonal_centroid'][idx].shape[1]:
                        print(f"ERROR: Inconsistent dimensions between pianoroll and tonal_centroid for piece {op}")
                        valid = False
                    
                    # Consistency check: pianoroll and label should have same batch and sequence length
                    if piece_data['pianoroll'][idx].shape[0] != piece_data['label'][idx].shape[0] or \
                       piece_data['pianoroll'][idx].shape[1] != piece_data['label'][idx].shape[1]:
                        print(f"ERROR: Inconsistent dimensions between pianoroll and label for piece {op}")
                        valid = False
                    
                    # Length array should match batch size
                    if piece_data['len'][idx].shape[0] != piece_data['pianoroll'][idx].shape[0]:
                        print(f"ERROR: Length array size doesn't match batch size for piece {op}")
                        valid = False
            except Exception as e:
                print(f"ERROR checking dimensions for piece {op}: {e}")
                valid = False
        
        if valid:
            print("Test data format validation PASSED!")
        else:
            print("Test data format validation FAILED!")
            
    except Exception as e:
        print(f"ERROR during validation: {e}")
        valid = False
        
    return valid


def inspect_test_pickle(pickle_file):
    """
    Load and inspect a test data pickle file
    
    :param pickle_file: path to the pickle file
    """
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Successfully loaded pickle file: {pickle_file}")
        
        # Check main structure
        if 'shift_0' in data:
            print(f"Contains 'shift_0' key with {len(data['shift_0'])} pieces")
            
            # Get a sample piece
            sample_op = next(iter(data['shift_0'].keys()))
            piece_data = data['shift_0'][sample_op]
            
            print(f"\nSample piece {sample_op} details:")
            for key in piece_data:
                if isinstance(piece_data[key], list):
                    print(f"  {key}: List of length {len(piece_data[key])}")
                    for idx, item in enumerate(piece_data[key]):
                        if hasattr(item, 'shape'):
                            print(f"    [{idx}] shape = {item.shape}, dtype = {item.dtype}")
                        else:
                            print(f"    [{idx}] type = {type(item)}")
                else:
                    print(f"  {key}: {type(piece_data[key])}")
            
            # Print label dtype if available
            if 'label' in piece_data and len(piece_data['label']) > 0 and len(piece_data['label'][0]) > 0:
                print(f"\nLabel dtype: {piece_data['label'][0].dtype}")
        else:
            print("WARNING: Data does not contain 'shift_0' key - not in training format!")
            print(f"Keys at root level: {list(data.keys())}")
            
        # Validate format
        print("\nValidating data format...")
        validate_test_data_format(data)
        
    except Exception as e:
        print(f"Error inspecting pickle file: {e}")


def main():
    vocabulary = 'MIREX_Mm'  # Use the same vocabulary as training
    resolution = 4  # Use the same resolution as training
    n_steps = 128  # Use the same sequence length as training
    
    # Load and process test data
    corpus = load_test_dataset(resolution=resolution, vocabulary=vocabulary)
    corpus_reshape = reshape_test_data(corpus, n_steps=n_steps)
    
    # Add validation check here, before saving
    for shift_id in corpus_reshape:
        for op in corpus_reshape[shift_id]:
            for field in ['pianoroll', 'tonal_centroid', 'label', 'len']:
                for idx in [0, 1]:
                    assert corpus_reshape[shift_id][op][field][idx] is not None, f"None value in {shift_id}-{op}-{field}-{idx}"
                    # Add dimension checks if desired
                    if field == 'pianoroll':
                        assert corpus_reshape[shift_id][op][field][idx].ndim == 3, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'tonal_centroid':
                        assert corpus_reshape[shift_id][op][field][idx].ndim == 3, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'label':
                        assert corpus_reshape[shift_id][op][field][idx].ndim == 2, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
                    elif field == 'len':
                        assert corpus_reshape[shift_id][op][field][idx].ndim == 1, f"Wrong dimension for {shift_id}-{op}-{field}-{idx}"
    
    # Save processed data
    save_dir = f'test_data_preprocessed_{vocabulary}_train_format.pickle'
    save_preprocessed_data(corpus_reshape, save_dir)
    
    # Additional validation with separate function if desired
    validate_test_data_format(corpus_reshape)
    
if __name__ == '__main__':
    main()
    # If you want to inspect an existing pickle file
    # inspect_test_pickle('test_data_preprocessed_MIREX_Mm_train_format.pickle')