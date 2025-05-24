import pickle
import numpy as np

evaluation_file_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\test_data\baroque_model_evaluation_results.pkl"

try:
    with open(evaluation_file_path, 'rb') as f:
        results = pickle.load(f)
        print("Evaluation results loaded successfully!")
        print("\nTop-level keys of the 'results' dictionary:")
        print(results.keys())

        # Let's inspect the keys related to predictions and test data
        if 'baroque_on_baroque_preds' in results:
            print("\nContents of 'baroque_on_baroque_preds' (first level keys):")
            print(results['baroque_on_baroque_preds'].keys())
            # Optionally, inspect the shape of one of the prediction arrays
            if 'key' in results['baroque_on_baroque_preds']:
                print("\nShape of 'baroque_on_baroque_preds']['key']:", results['baroque_on_baroque_preds']['key'].shape)

        if 'romantic_on_baroque_preds' in results:
            print("\nContents of 'romantic_on_baroque_preds' (first level keys):")
            print(results['romantic_on_baroque_preds'].keys())
            if 'key' in results['romantic_on_baroque_preds']:
                print("\nShape of 'romantic_on_baroque_preds']['key']:", results['romantic_on_baroque_preds']['key'].shape)

        if 'baroque_test_data' in results:
            print("\nContents of 'baroque_test_data' (first level keys):")
            print(results['baroque_test_data'].keys())
            if 'len' in results['baroque_test_data']:
                print("\nShape of 'baroque_test_data']['len']:", np.array(results['baroque_test_data']['len']).shape)
                print("\nFirst few sequence lengths:", results['baroque_test_data']['len'][:5])

except FileNotFoundError:
    print(f"Error: File not found at {evaluation_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")