import pickle
import numpy as np
import os
import json
from collections import defaultdict

def load_pickle_data(file_path):
    """Load data from a pickle file and return its structure."""
    print(f"Loading {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_item_structure(data, shift_key='shift_0'):
    """Analyze the structure of items in the data to identify file groupings."""
    if shift_key not in data:
        print(f"ERROR: '{shift_key}' key not found in data!")
        return None
    
    # Dictionary to store information about each item
    item_info = {}
    
    # Process each item
    for item_id, item_data in data[shift_key].items():
        if 'len' not in item_data:
            print(f"Item {item_id} doesn't have 'len' field, skipping.")
            continue
        
        lengths = item_data['len']
        sequence_count = 0
        sequence_lengths = []
        
        # Handle different possible structures of 'len'
        if isinstance(lengths, list) or isinstance(lengths, np.ndarray):
            for length_item in lengths:
                # Check if length_item is an array/list or a single value
                if hasattr(length_item, '__len__') and not isinstance(length_item, (str, bytes)):
                    # It's an array-like object
                    sequence_count += len(length_item)
                    if isinstance(length_item, np.ndarray):
                        sequence_lengths.extend(length_item.tolist())
                    else:
                        sequence_lengths.extend(length_item)
                else:
                    # It's a single value
                    sequence_count += 1
                    sequence_lengths.append(int(length_item))
        else:
            # Handle case where lengths is a single value
            sequence_count = 1
            sequence_lengths = [int(lengths)]
        
        # Store item information
        item_info[item_id] = {
            'sequence_count': sequence_count,
            'sequence_lengths': sequence_lengths,
            'total_chords': sum(sequence_lengths)
        }
    
    return item_info

def identify_file_signatures(train_data, test_data):
    """
    Compare train and test data to identify unique signatures for the 10 test files.
    """
    # Analyze both datasets
    train_info = analyze_item_structure(train_data)
    test_info = analyze_item_structure(test_data)
    
    if not train_info or not test_info:
        print("Failed to analyze one or both datasets.")
        return None
    
    print(f"\nTrain dataset: {len(train_info)} items")
    print(f"Test dataset: {len(test_info)} items")
    
    # Create signature for each item based on sequence lengths
    train_signatures = {}
    for item_id, info in train_info.items():
        signature = tuple(sorted(info['sequence_lengths']))
        train_signatures[signature] = item_id
    
    # Look for unique signatures in test data
    test_signatures = {}
    for item_id, info in test_info.items():
        signature = tuple(sorted(info['sequence_lengths']))
        test_signatures[item_id] = signature
    
    # Find which test signatures match training signatures
    matching_signatures = {}
    unique_test_signatures = {}
    
    for test_id, signature in test_signatures.items():
        if signature in train_signatures:
            matching_signatures[test_id] = train_signatures[signature]
        else:
            # This is likely one of our 10 test files
            unique_test_signatures[test_id] = signature
    
    print(f"\nFound {len(unique_test_signatures)} unique test signatures (potential test files)")
    print(f"Found {len(matching_signatures)} test items matching training signatures")
    
    return {
        'train_info': train_info,
        'test_info': test_info,
        'unique_test_signatures': unique_test_signatures,
        'matching_signatures': matching_signatures
    }

def extract_sequences_from_item(item_data):
    """
    Extract sequence information from item data, handling different data structures.
    Returns a list of sequence info dictionaries.
    """
    sequences = []
    
    if 'len' not in item_data:
        return sequences
    
    lengths = item_data['len']
    
    # Handle different possible structures of 'len'
    if isinstance(lengths, list) or isinstance(lengths, np.ndarray):
        for array_idx, length_item in enumerate(lengths):
            if hasattr(length_item, '__len__') and not isinstance(length_item, (str, bytes)):
                # It's an array-like object
                for local_seq_idx, seq_length in enumerate(length_item):
                    sequences.append({
                        'array_idx': array_idx,
                        'local_seq_idx': local_seq_idx,
                        'length': int(seq_length)
                    })
            else:
                # It's a single value
                sequences.append({
                    'array_idx': array_idx,
                    'local_seq_idx': 0,
                    'length': int(length_item)
                })
    else:
        # Handle case where lengths is a single value
        sequences.append({
            'array_idx': 0,
            'local_seq_idx': 0,
            'length': int(lengths)
        })
    
    return sequences

def cluster_test_sequences_into_files(test_data, signature_analysis, target_files=10):
    """
    Group the sequences in the test data into the 10 test files.
    """
    # First, extract all test sequences
    test_sequences = []
    seq_idx = 1  # Start sequence indexing at 1
    
    for item_id, item_data in test_data['shift_0'].items():
        item_sequences = extract_sequences_from_item(item_data)
        
        for seq_info in item_sequences:
            seq_info['item_id'] = item_id
            seq_info['global_seq_idx'] = seq_idx
            test_sequences.append(seq_info)
            seq_idx += 1
    
    print(f"\nTotal test sequences: {len(test_sequences)}")
    
    # Group sequences by item_id
    sequences_by_item = defaultdict(list)
    for seq in test_sequences:
        sequences_by_item[seq['item_id']].append(seq['global_seq_idx'])
    
    # Identify the 10 test files
    unique_test_ids = list(signature_analysis['unique_test_signatures'].keys())
    
    # If we have exactly 10 unique signatures, use those
    if len(unique_test_ids) == target_files:
        test_file_ids = unique_test_ids
    else:
        # Otherwise, we need to select 10 test files
        # First take all unique signatures
        test_file_ids = unique_test_ids.copy()
        
        # If we need more, add some from matching signatures
        remaining_needed = target_files - len(test_file_ids)
        if remaining_needed > 0:
            # Add items with the most sequences
            additional_candidates = []
            for item_id in signature_analysis['matching_signatures'].keys():
                if item_id not in test_file_ids:
                    seq_count = len(sequences_by_item[item_id])
                    additional_candidates.append((item_id, seq_count))
            
            # Sort by sequence count (descending)
            additional_candidates.sort(key=lambda x: x[1], reverse=True)
            additional_ids = [item_id for item_id, _ in additional_candidates[:remaining_needed]]
            test_file_ids.extend(additional_ids)
    
    # Map from sequence index to file index
    sequence_to_file = {}
    file_to_sequences = {}
    
    for file_idx, item_id in enumerate(test_file_ids, 1):  # 1-indexed file IDs
        sequence_list = sequences_by_item[item_id]
        file_to_sequences[file_idx] = sequence_list
        
        for seq_idx in sequence_list:
            sequence_to_file[seq_idx] = file_idx
    
    return {
        'sequence_to_file': sequence_to_file,
        'file_to_sequences': file_to_sequences,
        'test_file_ids': test_file_ids
    }

def write_mapping_to_file(mapping_result, output_file='test_sequence_to_file_mapping.txt'):
    """Write the test sequence to file mapping to a file."""
    sequence_to_file = mapping_result['sequence_to_file']
    file_to_sequences = mapping_result['file_to_sequences']
    
    with open(output_file, 'w') as f:
        f.write("# Mapping of test sequences to their original files\n\n")
        
        # First, list all sequences by file
        f.write("## Sequences Grouped by File\n\n")
        
        for file_idx, sequence_list in sorted(file_to_sequences.items()):
            f.write(f"File {file_idx}: {sorted(sequence_list)}\n")
        
        f.write("\n## Individual Sequence Mapping\n\n")
        
        # Then list individual sequence mappings
        for seq_idx, file_idx in sorted(sequence_to_file.items()):
            f.write(f"Sequence {seq_idx} -> File {file_idx}\n")
    
    print(f"\nMapping written to {output_file}")
    
    # Also save as JSON for programmatic use
    with open('test_sequence_mapping.json', 'w') as f:
        json.dump({
            'sequence_to_file': {str(k): v for k, v in sequence_to_file.items()},
            'file_to_sequences': {str(k): v for k, v in file_to_sequences.items()},
            'test_file_ids': mapping_result['test_file_ids']
        }, f, indent=2)
    
    print("JSON mapping saved to 'test_sequence_mapping.json'")

def main():
    # File paths - update with your actual paths
    train_pickle = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_model_training\Sonatas_preprocessed_data_MIREX_Mm.pickle"
    test_pickle = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\test_data\test_data_preprocessed_MIREX_Mm_train_format.pickle"
    # Check if files exist
    for file_path in [train_pickle, test_pickle]:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return
    
    # Load data
    train_data = load_pickle_data(train_pickle)
    test_data = load_pickle_data(test_pickle)
    
    # Identify file signatures
    signature_analysis = identify_file_signatures(train_data, test_data)
    
    if signature_analysis:
        # Cluster test sequences into 10 files
        mapping_result = cluster_test_sequences_into_files(test_data, signature_analysis)
        
        # Report results
        print("\nTest Sequences Mapped to Files:")
        for file_idx, sequence_list in sorted(mapping_result['file_to_sequences'].items()):
            print(f"File {file_idx}: {len(sequence_list)} sequences - {sorted(sequence_list)}")
        
        # Write mapping to file
        write_mapping_to_file(mapping_result)
        
        print("\nDone! You can now use the mapping to reconstruct your 10 test files from the 45 sequences.")
    else:
        print("Failed to analyze file signatures.")

if __name__ == "__main__":
    main()