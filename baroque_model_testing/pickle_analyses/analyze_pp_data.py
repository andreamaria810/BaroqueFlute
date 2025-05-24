import pickle
import numpy as np
import os


#### --- Map sequences to their respective test data piece --- ####


def analyze_test_data_structure(pickle_file):
    """Analyze the structure of the preprocessed test data"""
    with open(pickle_file, 'rb') as f:
        test_data = pickle.load(f)
    
    print("\nAnalyzing test data structure:")
    print(f"Keys at top level: {list(test_data.keys())}")
    
    # Assuming 'shift_0' is a key
    if 'shift_0' in test_data:
        pieces = list(test_data['shift_0'].keys())
        print(f"Test pieces found: {pieces}")
        
        # Count total sequences across pieces
        total_sequences = 0
        piece_to_sequences = {}
        
        for piece_id in pieces:
            piece_data = test_data['shift_0'][piece_id]
            print(f"\nPiece {piece_id} data keys: {list(piece_data.keys())}")
            
            # Check if 'label' is in the data and has the expected structure
            if 'label' in piece_data and isinstance(piece_data['label'], list) and len(piece_data['label']) > 0:
                # Count sequences in this piece
                n_sequences = piece_data['label'][0].shape[0]
                total_sequences += n_sequences
                
                # Store mapping from sequences to piece
                start_seq = sum(piece_to_sequences.values())
                piece_to_sequences[piece_id] = n_sequences
                
                # Get sequence lengths if available
                if 'len' in piece_data and isinstance(piece_data['len'], list) and len(piece_data['len']) > 0:
                    seq_lengths = piece_data['len'][0]
                    print(f"Piece {piece_id} has {n_sequences} sequences with lengths: {seq_lengths}")
                else:
                    print(f"Piece {piece_id} has {n_sequences} sequences")
        
        print(f"\nTotal sequences across all pieces: {total_sequences}")
        
        # Create mapping from sequence index to piece ID
        sequence_to_piece = {}
        sequence_idx = 0
        for piece_id, n_sequences in piece_to_sequences.items():
            for _ in range(n_sequences):
                sequence_to_piece[sequence_idx] = piece_id
                sequence_idx += 1
        
        return sequence_to_piece
    else:
        print("Data structure does not contain 'shift_0' key. Cannot analyze further.")
        return None

def create_mapping_file(sequence_to_piece, output_dir):
    """Create a mapping file showing which sequences belong to which test pieces"""
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'sequence_to_piece_mapping.txt'), 'w') as f:
        f.write("Sequence Index\tTest Piece ID\n")
        f.write("-" * 30 + "\n")
        
        for sequence_idx, piece_id in sequence_to_piece.items():
            f.write(f"{sequence_idx+1}\t{piece_id}\n")
    
    # Also create a reverse mapping for easier reference
    piece_to_sequences = {}
    for sequence_idx, piece_id in sequence_to_piece.items():
        if piece_id not in piece_to_sequences:
            piece_to_sequences[piece_id] = []
        piece_to_sequences[piece_id].append(sequence_idx+1)
    
    with open(os.path.join(output_dir, 'piece_to_sequences_mapping.txt'), 'w') as f:
        f.write("Test Piece ID\tSequence Indices\n")
        f.write("-" * 50 + "\n")
        
        for piece_id, sequence_indices in piece_to_sequences.items():
            f.write(f"{piece_id}\t{sequence_indices}\n")
    
    print(f"Mapping files created in {output_dir}")
    return piece_to_sequences

def main():
    # Path to your preprocessed test data
    test_file_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\test_data\test_data_preprocessed_MIREX_Mm_train_format.pickle"
    
    sequence_to_piece = analyze_test_data_structure(test_file_path)
    
    if sequence_to_piece:
        # Create mapping files
        piece_to_sequences = create_mapping_file(sequence_to_piece, 'chord_sequences')
        
        # Print a summary of the mapping
        print("\nSummary of piece to sequences mapping:")
        for piece_id, sequence_indices in piece_to_sequences.items():
            print(f"Test Piece {piece_id}: {len(sequence_indices)} sequences - {sequence_indices}")

if __name__ == "__main__":
    main()