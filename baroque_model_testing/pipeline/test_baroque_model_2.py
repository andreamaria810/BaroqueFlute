import os
import tensorflow.compat.v1 as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.test.is_gpu_available())
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
tf.disable_v2_behavior()


import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

# Import your model architectures
import chord_recognition_models as crm

# Import essential functions from your training script
from functional_harmony_recognition import load_data_functional, hyperparameters, key_dict, degree1_dict, degree2_dict, quality_dict, inversion_dict, extra_info_dict
    


# --- Labels to indices ---

# Forward mapping
key_dict = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6, 'c': 7, 'd': 8, 'e': 9, 'f': 10, 'g': 11, 'a': 12, 'b': 13, 'C#': 14, 'C+': 14, 'D#': 15, 'D+': 15, 'E#': 16, 'E+': 16, 'F#': 17, 'F+': 17, 'G#': 18, 'G+': 18, 'A#': 19, 'A+': 19, 'B#': 20, 'B+': 20, 'c#': 21, 'c+': 21, 'd#': 22, 'd+': 22, 'e#': 23, 'e+': 23, 'f#': 24, 'f+': 24, 'g#': 25, 'g+': 25, 'a#': 26, 'a+': 26, 'b#': 27, 'b+': 27, 'Cb': 28, 'C-': 28, 'Db': 29, 'D-': 29, 'Eb': 30, 'E-': 30, 'Fb': 31, 'F-': 31, 'Gb': 32, 'G-': 32, 'Ab': 33, 'A-': 33, 'Bb': 34, 'B-': 34, 'cb': 35, 'c-': 35, 'db': 36, 'd-': 36, 'eb': 37, 'e-': 37, 'fb': 38, 'f-': 38, 'gb': 39, 'g-': 39, 'ab': 40, 'a-': 40, 'bb': 41, 'b-': 41, 'pad': 42}
quality_dict = {'M': 0, 'm': 1, 'a': 2, 'd': 3, 'M7': 4, 'm7': 5, 'D7': 6, 'd7': 7, 'h7': 8, 'a6': 9, 'pad': 10, 'a7': 2}
degree1_dict = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '-2': 7, '-7': 8, '+6': 9, 'pad': 10}
degree2_dict = {'none': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '+1': 8, '+3': 9, '+4': 10, '-2': 11, '-3': 12, '-6': 13, '-7': 14, 'pad': 15}
inversion_dict = {'0': 0, '1': 1, '2': 2, '3': 3, 'pad': 4}
extra_info_dict = {'none': 0, '2': 1, '4': 2, '6': 3, '7': 4, '9': 5, '-2': 6, '-4': 7, '-6': 8, '-9': 9, '+2': 10, '+4': 11, '+5': 12, '+6': 13, '+7': 14, '+9': 15, '+72': 16, '72': 17, '62': 18, '42': 19, '64': 20, '94': 21, 'pad': 22}

# --- Indices to labels ---

# Reverse mapping
reverse_key_dict = {0: 'C', 1: 'D', 2: 'E', 3: 'F', 4: 'G', 5: 'A', 6: 'B', 7: 'c', 8: 'd', 9: 'e', 10: 'f', 11: 'g', 12: 'a', 13: 'b', 14: 'C+', 15: 'D+', 16: 'E+', 17: 'F+', 18: 'G+', 19: 'A+', 20: 'B+', 21: 'c+', 22: 'd+', 23: 'e+', 24: 'f+', 25: 'g+', 26: 'a+', 27: 'b+', 28: 'C-', 29: 'D-', 30: 'E-', 31: 'F-', 32: 'G-', 33: 'A-', 34: 'B-', 35: 'c-', 36: 'd-', 37: 'e-', 38: 'f-', 39: 'g-', 40: 'a-', 41: 'b-', 42: 'pad'}
reverse_quality_dict = {0: 'M', 1: 'm', 2: 'a7', 3: 'd', 4: 'M7', 5: 'm7', 6: 'D7', 7: 'd7', 8: 'h7', 9: 'a6', 10: 'pad'}
reverse_degree1_dict = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '-2', 8: '-7', 9: '+6', 10: 'pad'}
reverse_degree2_dict = {0: 'none', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '+1', 9: '+3', 10: '+4', 11: '-2', 12: '-3', 13: '-6', 14: '-7', 15: 'pad'}
reverse_inversion_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: 'pad'}
reverse_extra_info_dict = {0: 'none', 1: '2', 2: '4', 3: '6', 4: '7', 5: '9', 6: '-2', 7: '-4', 8: '-6', 9: '-9', 10: '+2', 11: '+4', 12: '+5', 13: '+6', 14: '+7', 15: '+9', 16: '+72', 17: '72', 18: '62', 19: '42', 20: '64', 21: '94', 22: 'pad'}

##################################################################################

def load_specific_test_data(pickle_file_path, test_op_ids):
    """Loads specific test data based on operation IDs."""
    with open(pickle_file_path, 'rb') as f:
        all_data = pickle.load(f)
    
    # Collect all sequences across all test pieces
    pianoroll_sequences = []
    tc_sequences = []
    label_sequences = []
    sequence_lengths = []

    for op_id in test_op_ids:
        if op_id in all_data['shift_0']:
            for i in range(len(all_data['shift_0'][op_id]['pianoroll'])):
                pianorolls = all_data['shift_0'][op_id]['pianoroll'][i]
                tonal_centroids = all_data['shift_0'][op_id]['tonal_centroid'][i]
                labels = all_data['shift_0'][op_id]['label'][i]
                lengths = all_data['shift_0'][op_id]['len'][i]

                if pianorolls.ndim == 3:
                    for seq_idx in range(pianorolls.shape[0]):
                        pianoroll_sequences.append(pianorolls[seq_idx])
                        tc_sequences.append(tonal_centroids[seq_idx])
                        label_sequences.append(labels[seq_idx])
                        sequence_lengths.append(lengths[seq_idx])
                elif pianorolls.ndim == 2:
                    # This case should ideally not happen if reshaping is correct
                    # But handle it defensively by reshaping to (1, 128)
                    pianoroll_sequences.append(pianorolls)
                    tc_sequences.append(tonal_centroids)
                    label_sequences.append(labels.reshape(1, -1)) # Reshape to ensure 2D
                    sequence_lengths.append(lengths)
                else:
                    print(f"Warning: Unexpected dimensions for op_id {op_id} at index {i}")

    # Combine all sequences
    combined_pianoroll = np.stack(pianoroll_sequences, axis=0) if pianoroll_sequences else np.array([])
    combined_tc = np.stack(tc_sequences, axis=0) if tc_sequences else np.array([])
    combined_labels = np.vstack(label_sequences) if label_sequences else np.array([]) # Use vstack for 2D stacking
    combined_lengths = np.array(sequence_lengths, dtype=np.int32)
   

    
    # Extract individual components from structured array
    test_data = {
        'pianoroll': combined_pianoroll,
        'tonal_centroid': combined_tc,
        'len': combined_lengths
    }

    # Initialize component arrays for labels
    n_sequences = combined_labels.shape[0]
    n_timesteps = combined_labels.shape[1] if n_sequences > 0 else 0
 

    test_data['key'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['degree1'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['degree2'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['quality'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['inversion'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['extra_info'] = np.zeros((n_sequences, n_timesteps), dtype=np.int32)
    test_data['label'] = {'chord_change': np.zeros((n_sequences, n_timesteps), dtype=np.int32)}

    # Extract data from structured array fields to component arrays
    if n_sequences > 0:
        for i in range(n_sequences):
            for j in range(n_timesteps):
                label = combined_labels[i, j]
                test_data['key'][i, j] = key_dict.get(label['key'], 0)
                test_data['degree1'][i, j] = degree1_dict.get(label['degree1'], 0)
                test_data['degree2'][i, j] = degree2_dict.get(label['degree2'], 0)
                test_data['quality'][i, j] = quality_dict.get(label['quality'], 0)
                test_data['inversion'][i, j] = label['inversion']
                test_data['label']['chord_change'][i, j] = label['chord_change']
                if 'extra_info' in label.dtype.names:
                    test_data['extra_info'][i, j] = extra_info_dict.get(label['extra_info'], 0)
    
    return test_data, None


def load_trained_model(graph, sess, checkpoint_path, hp, include_extra_info=False):
    """Loads a trained model into the provided graph and session."""
    with graph.as_default():
        # Create placeholders within the provided graph
        x = tf.placeholder(tf.float32, [None, 128, 88], name="pianoroll")
        source_mask = tf.placeholder(tf.float32, [None, 128], name="source_mask")
        target_mask = tf.placeholder(tf.float32, [None, 128], name="target_mask")
        slope = tf.placeholder(tf.float32, [], name='annealing_slope')
        dropout_rate = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")


        chord_change_logits, dec_output, enc_weights, dec_weights, _, _ = crm.HTv2(
            x, source_mask, target_mask, slope, dropout_rate, is_training, hp
        )

        # Apply dropout (set training to False for inference)
        dec_output_dropout = tf.layers.dropout(dec_output, rate=dropout_rate, training=False)

        # Add output layers within the 'output_projection' variable scope
        with tf.variable_scope("output_projection"):
            key_logits = tf.layers.dense(dec_output_dropout, 43)
            degree1_logits = tf.layers.dense(dec_output_dropout, 11)
            degree2_logits = tf.layers.dense(dec_output_dropout, 16)
            quality_logits = tf.layers.dense(dec_output_dropout, 11)
            inversion_logits = tf.layers.dense(dec_output_dropout, 5)
            if include_extra_info:
                extra_info_logits = tf.layers.dense(dec_output_dropout, 23)
                pred_ex = tf.argmax(extra_info_logits, axis=2, output_type=tf.int32)

            pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
            pred_d1 = tf.argmax(degree1_logits, axis=2, output_type=tf.int32)
            pred_d2 = tf.argmax(degree2_logits, axis=2, output_type=tf.int32)
            pred_q = tf.argmax(quality_logits, axis=2, output_type=tf.int32)
            pred_inv = tf.argmax(inversion_logits, axis=2, output_type=tf.int32)
            pred_cc = tf.cast(tf.round(tf.sigmoid(slope * chord_change_logits)), tf.int32)

            predictions = {
                'key': pred_k,
                'degree1': pred_d1,
                'degree2': pred_d2,
                'quality': pred_q,
                'inversion': pred_inv,
                'chord_change': pred_cc
            }
            if include_extra_info:
                predictions['extra_info'] = pred_ex

            logits = {
                'key': key_logits,
                'degree1': degree1_logits,
                'degree2': degree2_logits,
                'quality': quality_logits,
                'inversion': inversion_logits,
                'chord_change': chord_change_logits
            }
            if include_extra_info:
                logits['extra_info'] = extra_info_logits

        # Create a saver within the loaded graph
        saver = tf.train.Saver()
        return saver, x, source_mask, target_mask, slope, dropout_rate, is_training, predictions, logits


def evaluate_model(sess, x, source_mask, target_mask, slope, dropout_rate, is_training, 
                  predictions, logits, test_data, best_slope=1.0, has_extra_info=False):
    """Evaluate model on test data with component-specific metrics"""

    # Create placeholders for evaluation metrics
    accuracy = {
        'key': 0.0,
        'degree1': 0.0,
        'degree2': 0.0,
        'quality': 0.0,
        'inversion': 0.0,
        'components_avg': 0.0
    }
    
    # Add extra_info if necessary
    if has_extra_info and 'extra_info' in predictions and 'extra_info' in test_data:
        accuracy['extra_info'] = 0.0

    chord_change_metrics = {
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
    }
    
    segmentation_quality = {
        'degree1': 0.0,
        'degree2': 0.0,
        'quality': 0.0,
        'inversion': 0.0,
        'avg': 0.0
    }
    
    # Prepare the feed dict for evaluation
    feed_dict = {
        x: test_data['pianoroll'],
        source_mask: sess.run(tf.sequence_mask(test_data['len'], maxlen=128, dtype=tf.float32)),
        target_mask: sess.run(tf.sequence_mask(test_data['len'], maxlen=128, dtype=tf.float32)),
        slope: best_slope,
        dropout_rate: 0.0,
        is_training: False
    }
    
    # Run predictions
    pred_results = sess.run(predictions, feed_dict=feed_dict)

    # Convert numerical predictions to string representations
    predicted_keys = np.vectorize(lambda i: reverse_key_dict.get(i, 'Unknown'))(pred_results['key'])
    predicted_degree1s = np.vectorize(lambda i: reverse_degree1_dict.get(i, 'Unknown'))(pred_results['degree1'])
    predicted_degree2s = np.vectorize(lambda i: reverse_degree2_dict.get(i, 'Unknown'))(pred_results['degree2'])
    predicted_qualities = np.vectorize(lambda i: reverse_quality_dict.get(i, 'Unknown'))(pred_results['quality'])
    predicted_inversions = np.vectorize(lambda i: reverse_inversion_dict.get(i, 'Unknown'))(pred_results['inversion'])  # Corrected
    predicted_chord_changes = pred_results['chord_change']
    if has_extra_info and 'extra_info' in predictions:
        predicted_extra_infos = np.vectorize(lambda i: reverse_extra_info_dict.get(i, 'Unknown'))(pred_results['extra_info'])
    else:
        predicted_extra_infos = None

    """
    print("\n--- Raw Numerical Predictions (First few sequences, first 10 timesteps) ---")
    for key, value in pred_results.items():
        if value.ndim > 1:
            print(f"  {key}:")
            print(value[:2, :10])  # Print first 2 sequences, first 10 timesteps
        elif value.ndim == 1 and len(value) > 10:
            print(f"  {key}:")
            print(value[:10])
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
    print("---")
    """
    # Calculate component-specific accuracies
    total_tokens = np.sum(test_data['len'])
    
    # Apply masking
    mask = sess.run(tf.sequence_mask(test_data['len'], maxlen=128))

    # Key accuracy
    key_correct = np.equal(pred_results['key'], test_data['key'])
    key_masked = np.logical_and(key_correct, mask)
    accuracy['key'] = np.sum(key_masked) / total_tokens
    
    # Degree1 accuracy
    d1_correct = np.equal(pred_results['degree1'], test_data['degree1'])
    d1_masked = np.logical_and(d1_correct, mask)
    accuracy['degree1'] = np.sum(d1_masked) / total_tokens
    
    # Degree2 accuracy
    d2_correct = np.equal(pred_results['degree2'], test_data['degree2'])
    d2_masked = np.logical_and(d2_correct, mask)
    accuracy['degree2'] = np.sum(d2_masked) / total_tokens
    
    # Quality accuracy
    q_correct = np.equal(pred_results['quality'], test_data['quality'])
    q_masked = np.logical_and(q_correct, mask)
    accuracy['quality'] = np.sum(q_masked) / total_tokens
    
    # Inversion accuracy
    inv_correct = np.equal(pred_results['inversion'], test_data['inversion'])
    inv_masked = np.logical_and(inv_correct, mask)
    accuracy['inversion'] = np.sum(inv_masked) / total_tokens
    
    # Extra info accuracy (if available)
    if has_extra_info and 'extra_info' in predictions and 'extra_info' in test_data:
        ex_correct = np.equal(pred_results['extra_info'], test_data['extra_info'])
        ex_masked = np.logical_and(ex_correct, mask)
        accuracy['extra_info'] = np.sum(ex_masked) / total_tokens
    
    # Components average
    if has_extra_info and 'extra_info' in predictions and 'extra_info' in test_data:
        accuracy['components_avg'] = (accuracy['degree1'] + accuracy['degree2'] + 
                                     accuracy['quality'] + accuracy['inversion'] + 
                                     accuracy['extra_info']) / 5.0
    else:
        accuracy['components_avg'] = (accuracy['degree1'] + accuracy['degree2'] + 
                                     accuracy['quality'] + accuracy['inversion']) / 4.0
    
    # Chord change F1 score
    cc_pred = pred_results['chord_change']
    
    # Get chord_change from test data - might be in different formats
    if 'chord_change' in test_data:
        cc_true = test_data['chord_change']
    elif isinstance(test_data.get('label', None), dict) and 'chord_change' in test_data['label']:
        cc_true = test_data['label']['chord_change']
    else:
        # Try to extract from structured array if available
        try:
            cc_true = np.array([[label['chord_change'] for label in seq] for seq in test_data['label_array']])
        except:
            # Fallback - create zeros array of same shape as predictions
            print("Warning: Could not find chord_change in test data, using zeros.")
            cc_true = np.zeros_like(cc_pred)
    
    # Apply masking
    cc_pred_masked = cc_pred[mask]
    cc_true_masked = cc_true[mask]
    
    # Calculate precision, recall, F1
    true_positives = np.sum(np.logical_and(cc_pred_masked == 1, cc_true_masked == 1))
    false_positives = np.sum(np.logical_and(cc_pred_masked == 1, cc_true_masked == 0))
    false_negatives = np.sum(np.logical_and(cc_pred_masked == 0, cc_true_masked == 1))
    
    chord_change_metrics['precision'] = true_positives / (true_positives + false_positives + 1e-8)
    chord_change_metrics['recall'] = true_positives / (true_positives + false_negatives + 1e-8)
    chord_change_metrics['f1'] = 2 * (chord_change_metrics['precision'] * chord_change_metrics['recall']) / (chord_change_metrics['precision'] + chord_change_metrics['recall'] + 1e-8)
    
    # Calculate segmentation quality
    segmentation_quality['degree1'] = crm.segmentation_quality(test_data['degree1'], pred_results['degree1'], test_data['len'])
    segmentation_quality['degree2'] = crm.segmentation_quality(test_data['degree2'], pred_results['degree2'], test_data['len'])
    segmentation_quality['quality'] = crm.segmentation_quality(test_data['quality'], pred_results['quality'], test_data['len'])
    segmentation_quality['inversion'] = crm.segmentation_quality(test_data['inversion'], pred_results['inversion'], test_data['len'])
    
    if has_extra_info and 'extra_info' in predictions and 'extra_info' in test_data:
        segmentation_quality['extra_info'] = crm.segmentation_quality(
            test_data['extra_info'], pred_results['extra_info'], test_data['len']
        )
        # Calculate average including extra_info
        segmentation_quality['avg'] = (
            segmentation_quality['degree1'] + 
            segmentation_quality['degree2'] + 
            segmentation_quality['quality'] + 
            segmentation_quality['inversion'] + 
            segmentation_quality['extra_info']
        ) / 5.0
    else:
        # Calculate average without extra_info
        segmentation_quality['avg'] = (
            segmentation_quality['degree1'] + 
            segmentation_quality['degree2'] + 
            segmentation_quality['quality'] + 
            segmentation_quality['inversion']
        ) / 4.0
    
    return accuracy, chord_change_metrics, segmentation_quality, pred_results


"""
def predict_on_test_data(sess, x, source_mask, target_mask, slope, dropout_rate, 
                        is_training, predictions, test_data, hp, best_slope=1.0):
    # Run prediction on test data
    all_predictions = []
    
    # Process batches of test data
    feed_dict = {
        x: test_data['pianoroll'],
        source_mask: tf.sequence_mask(test_data['len'], maxlen=hp.n_steps, dtype=tf.float32).eval(session=sess),
        target_mask: tf.sequence_mask(test_data['len'], maxlen=hp.n_steps, dtype=tf.float32).eval(session=sess),
        slope: best_slope,
        dropout_rate: 0.0,
        is_training: False
    }
    
    # Run predictions for all components
    pred_results = {}
    for component, pred_tensor in predictions.items():
        pred_results[component] = sess.run(pred_tensor, feed_dict=feed_dict)
    
    return pred_results
"""


def prepare_test_batches(test_data, hp):
    """Prepare batches of test data"""
    n_test_sequences = test_data['pianoroll'].shape[0]
    source_mask = tf.sequence_mask(test_data['len'], maxlen=128, dtype=tf.float32) # [n_batches, n_steps]
    target_mask = source_mask

    batch_size = 20

    for i in range(0, n_test_sequences, batch_size):
        end_idx = min(i + batch_size, n_test_sequences)
        batch_indices = slice(i, end_idx)

        # Extract data from this batch
        batch_x = test_data['pianoroll'][batch_indices]
        batch_len = test_data['len'][batch_indices]
        batch_source_mask = source_mask[batch_indices]
        batch_target_mask = target_mask[batch_indices]

        # Ground truth labels
        batch_y = {
            'key': test_data['key'][batch_indices],
            'degree1': test_data['degree1'][batch_indices],
            'degree2': test_data['degree2'][batch_indices],
            'quality': test_data['quality'][batch_indices],
            'inversion': test_data['inversion'][batch_indices],
            'chord_change': test_data['label']['chord_change'][batch_indices]
        }


        yield batch_x, batch_source_mask, batch_target_mask, batch_y, batch_len


def main():
    # Hyperparameters for the baroque model
    baroque_hp = hyperparameters(
        dataset='in_distribution',
        test_set_id=4,
        graph_location='../baroque_model_training/model',
        n_steps=128,
        input_embed_size=128,
        n_layers=2,
        n_heads=4,
        train_sequence_with_overlap=True,
        initial_learning_rate=1e-4,
        drop=0.1,
        n_batches=20,
        n_training_steps=100000,
        n_in_succession=10,
        annealing_rate=1.1
    )

    # Paths to best model checkpoints
    baroque_model_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_model_training\model\HT_functional_harmony_recognition_Sonatas_4.ckpt"
    test_file_path = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_testing\pipeline\cross_composer\test_data_preprocessed_MIREX_Mm_train_format.pickle"
    # Define operation IDs for each test piece
    test_op_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    baroque_test_data, _ = load_specific_test_data(test_file_path, test_op_ids)

    # Check for extra_info component
    has_baroque_extra_info = 'extra_info' in baroque_test_data

    # --- Load and Evaluate Baroque model ---
    baroque_graph = tf.Graph()
    with tf.Session(graph=baroque_graph) as sess_baroque:
        print("Loading Baroque-trained model...")
        saver_b, x_b, source_mask_b, target_mask_b, slope_b, dropout_rate_b, is_training_b, predictions_b, logits_b = load_trained_model(
            baroque_graph, sess_baroque, baroque_model_path, baroque_hp, include_extra_info=True
        )

        saver_b.restore(sess_baroque, baroque_model_path)

        # --- Evaluate Baroque model on Baroque test data ---
        print("Evaluating Baroque model on Baroque data...")
        baroque_on_baroque_acc, baroque_on_baroque_cc, baroque_on_baroque_sq, baroque_on_baroque_preds = evaluate_model(
            sess_baroque, x_b, source_mask_b, target_mask_b, slope_b, dropout_rate_b, is_training_b,
            predictions_b, logits_b, baroque_test_data, best_slope=8.14, has_extra_info=has_baroque_extra_info
        )

        # Convert the TensorFlow mask to NumPy array inside the session
        baroque_mask_tensor = tf.sequence_mask(baroque_test_data['len'], maxlen=128)
        baroque_mask = sess_baroque.run(baroque_mask_tensor)


    # --- Binary Extra Info Evaluation ---
    if 'extra_info' in baroque_test_data and 'extra_info' in baroque_on_baroque_preds:
        # Get the ground truth and predictions
        extra_info_true = baroque_test_data['extra_info']
        extra_info_pred = baroque_on_baroque_preds['extra_info']

        # Convert to binary (0 = no extra info, 1 = has extra info)
        extra_info_true_binary = (extra_info_true > 0).astype(int)
        extra_info_pred_binary = (extra_info_pred > 0).astype(int)

        # Calculate accuracy with masking
        binary_correct = np.equal(extra_info_true_binary, extra_info_pred_binary)
        binary_masked = np.logical_and(binary_correct, baroque_mask)
        binary_accuracy = np.sum(binary_masked) / np.sum(baroque_mask)

        print(f"\nBinary Extra Info Detection Accuracy: {binary_accuracy:.4f}")

        # Calculate precision, recall, F1 for detecting extra info
        # Apply mask to focus only on valid timesteps
        valid_preds = extra_info_pred_binary[baroque_mask]
        valid_true = extra_info_true_binary[baroque_mask]

        true_positives = np.sum(np.logical_and(valid_preds == 1, valid_true == 1))
        false_positives = np.sum(np.logical_and(valid_preds == 1, valid_true == 0))
        false_negatives = np.sum(np.logical_and(valid_preds == 0, valid_true == 1))

        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")


    # --- Display evaluation results ---
    print("\n=== Model Comparison on Baroque Test Data ===")
    print("\nComponent accuracies:")
    components = ['key', 'degree1', 'degree2', 'quality', 'inversion', 'components_avg']
    if has_baroque_extra_info:
        components.insert(5, 'extra_info')  # Add extra_info before components_avg
    print(f"{'Component':<15} | {'Baroque→Baroque':<15}")
    print("-" * 30)
    for comp in components:
        b_acc = baroque_on_baroque_acc.get(comp, 'N/A')
        if b_acc != 'N/A':
            print(f"{comp:<15} | {b_acc:<15.4f}")
        else:
            b_acc_str = f"{b_acc:.4f}" if isinstance(b_acc, float) else str(b_acc)
            print(f"{comp:<15} | {b_acc_str:<15}")


    # --- Display chord change metrics ---
    print("\nChord change detection metrics:")
    metrics = ['precision', 'recall', 'f1']
    print(f"{'Metric':<15} | {'Baroque Model':<15}")
    print("-" * 30)
    for metric in metrics:
        b_metric = baroque_on_baroque_cc.get(metric, 'N/A')
        if b_metric != 'N/A':
            print(f"{metric:<15} | {b_metric:<15.4f}")


    # --- Display segmentation quality ---
    print("\nSegmentation quality:")
    sq_components = ['degree1', 'degree2', 'quality', 'inversion', 'avg']
    if has_baroque_extra_info:
        sq_components.insert(4, 'extra_info')  # Add extra_info before avg
    print(f"{'Component':<15} | {'Baroque Model':<15}")
    print("-" * 30)
    for comp in sq_components:
        b_sq = baroque_on_baroque_sq.get(comp, 'N/A')
        if b_sq != 'N/A':
            print(f"{comp:<15} | {b_sq:<15.4f}")
        else:
            b_sq_str = f"{b_sq:.4f}" if isinstance(b_sq, float) else str(b_sq)
            print(f"{comp:<15} | {b_sq_str:<15}")

    output_data = {
        'baroque_on_baroque_preds': baroque_on_baroque_preds,
        'baroque_test_data': baroque_test_data
    }
    with open('baroque_model_evaluation_results.pkl', 'wb') as f:
        pickle.dump(output_data, f)

    print("\nBaroque model evaluation results saved to 'baroque_model_evaluation_results.pkl'")

    if 'sess_baroque' in locals():
        sess_baroque.close()
if __name__ == "__main__":
    main()