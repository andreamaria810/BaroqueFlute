import pickle
import sys
import os
import numpy as np
from collections import defaultdict
import gc

def analyze_music_pickle(file_path):
    """Specifically analyze a music pickle file with shift keys"""
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"File: {file_path}")
    print(f"Total file size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    try:
        # Load the file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it has the expected structure
        if isinstance(data, dict) and any(k.startswith('shift_') for k in data.keys()):
            print("\n=== SHIFT KEY ANALYSIS ===")
            
            # Analyze each shift key
            total_size_accounted = 0
            
            for shift_key, shift_data in data.items():
                print(f"\nAnalyzing key: {shift_key}")
                
                # If shift_data is a dict, analyze its structure
                if isinstance(shift_data, dict):
                    print(f"  Dict with {len(shift_data)} keys")
                    
                    # Check the first few keys
                    for i, (key, val) in enumerate(shift_data.items()):
                        if i >= 5:
                            print(f"  ... and {len(shift_data)-5} more keys")
                            break
                            
                        # Check what's in each value
                        val_type = type(val).__name__
                        val_size = get_deep_size(val)
                        val_size_mb = val_size / (1024 * 1024)
                        total_size_accounted += val_size
                        
                        print(f"  - {key} ({val_type}): {val_size_mb:.2f} MB")
                        
                        # For arrays, show shape and dtype
                        if isinstance(val, np.ndarray):
                            print(f"    NumPy array: shape={val.shape}, dtype={val.dtype}")
                        
                        # For lists, show first item type and length
                        elif isinstance(val, list) and val:
                            print(f"    List of {len(val)} items, first item type: {type(val[0]).__name__}")
                            
                            # If first item is list or numpy array, go deeper
                            if isinstance(val[0], (list, np.ndarray)):
                                if isinstance(val[0], np.ndarray):
                                    print(f"    First item is array: shape={val[0].shape}, dtype={val[0].dtype}")
                                else:
                                    print(f"    First item is list of length {len(val[0])}")
                                    
                                # Check if all items have same structure
                                if len(val) > 1:
                                    if all(isinstance(x, type(val[0])) for x in val[:10]):
                                        print("    All sampled items have same type")
                                    else:
                                        print("    Items have mixed types")
                else:
                    # Handle non-dict shift_data
                    shift_size = get_deep_size(shift_data)
                    shift_size_mb = shift_size / (1024 * 1024)
                    total_size_accounted += shift_size
                    print(f"  Not a dict but {type(shift_data).__name__}: {shift_size_mb:.2f} MB")
            
            # Report total size accounted for
            total_size_accounted_mb = total_size_accounted / (1024 * 1024)
            percent_accounted = (total_size_accounted / file_size_bytes) * 100
            
            print(f"\nTotal size accounted for: {total_size_accounted_mb:.2f} MB ({percent_accounted:.2f}% of file)")
            
        else:
            print("Data doesn't match expected structure with shift keys")
            print(f"Top level type: {type(data).__name__}")
            
            if isinstance(data, dict):
                print(f"Keys: {list(data.keys())[:10]}")
            
    except Exception as e:
        print(f"Error analyzing file: {e}")


def get_deep_size(obj, seen=None):
    """More accurate size calculation that handles circular references"""
    if seen is None:
        seen = set()
        
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_deep_size(item, seen) for item in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
        
    return size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze a music pickle file with shift keys')
    parser.add_argument('file_path', help='Path to the pickle file')
    
    args = parser.parse_args()
    analyze_music_pickle(args.file_path)