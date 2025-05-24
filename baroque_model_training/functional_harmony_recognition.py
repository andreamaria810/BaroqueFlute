# Disables AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(os.environ["PATH"])

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Enable TensorFlow 1.x behavior (original code used TensorFlow 1.8.0)
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib
print("GPU available:", any(x.device_type == "GPU" for x in device_lib.list_local_devices()))
print("GPU devices:", [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"])

import time
import random
import math
import pickle
from collections import Counter, namedtuple
import chord_recognition_models as crm
from datetime import datetime

print(os.environ["PATH"])  # Print system PATH

# Configure TensorFlow to use GPU and optimize CPU
config = tf.compat.v1.ConfigProto()
# GPU settings
config.gpu_options.allow_growth = True  # Allocate only as much GPU memory as needed


# CPU settings (will apply to CPU operations even when GPU is active)
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 4

# Create and set session
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # Set session for TF1.x compatibility


# Mappings of functional harmony
'''key: 7 degrees * 3 accidentals * 2 modes + 1 padding= 43'''
key_dict = {}
for i_a, accidental in enumerate(['', '#', 'b']):
    for i_t, tonic in enumerate(['C', 'D', 'E', 'F', 'G', 'A', 'B', 'c', 'd', 'e', 'f', 'g', 'a', 'b']):
        key_dict[tonic + accidental] = i_a * 14 + i_t
        if accidental == '#':
            key_dict[tonic + '+'] = i_a * 14 + i_t
        elif accidental == 'b':
            key_dict[tonic + '-'] = i_a * 14 + i_t
key_dict['pad'] = 42
print(key_dict)

'''degree1: 11 (['1', '2', '3', '4', '5', '6', '7', '-2', '-7', +6, 'pad'])'''
degree1_dict = {d1: i for i, d1 in enumerate(['1', '2', '3', '4', '5', '6', '7', '-2', '-7', '+6', 'pad'])}

'''degree2: 16 ['none', '1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])'''
degree2_dict = {d2: i for i, d2 in enumerate(['none', '1', '2', '3', '4', '5', '6', '7', '+1', '+3', '+4', '-2', '-3', '-6', '-7', 'pad'])}

'''quality: 11 (['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])'''
quality_dict = {q: i for i, q in enumerate(['M', 'm', 'a', 'd', 'M7', 'm7', 'D7', 'd7', 'h7', 'a6', 'pad'])}
quality_dict['a7'] = [v for k, v in quality_dict.items() if k == 'a'][0]

'''inversion: 5 (['0', '1', '2', '3', 'pad'])'''
inversion_dict = {i: i for i in range(4)}  # NEW
inversion_dict['pad'] = 4

'''extra_info: 23 (['none', '2', '4', '6', '7', '9', '-2', '-4', '-6', '-9', '+2', '+4', '+5', '+6', '+7', '+9', '+72', '72', '62', '42', '64', '94', 'pad')'''
extra_info_dict = {ex: i for i, ex in enumerate(['none', '2', '4', '6', '7', '9', '-2', '-4', '-6', '-9', '+2', '+4', '+5', '+6', '+7', '+9', '+72', '72', '62', '42', '64', '94', 'pad'])} # NEW



def load_data_functional(dir, test_set_id=1, sequence_with_overlap=True):
    if test_set_id not in [1, 2, 3, 4, 5]:
        print('Invalid testing_set_id.')
        exit(1)

    print("Load functional harmony data ...")
    print('test_set_id =', test_set_id)
    with open(dir, 'rb') as file:
        corpus_aug_reshape = pickle.load(file)
    print('keys in corpus_aug_reshape[\'shift_id\'][\'op\'] =', corpus_aug_reshape['shift_0']['1'].keys())

    shift_list = sorted(corpus_aug_reshape.keys())
    number_of_pieces = len(corpus_aug_reshape['shift_0'].keys())
    train_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 5 + 1 != test_set_id]
    test_op_list = [str(i + 1) for i in range(number_of_pieces) if i % 5 + 1 == test_set_id]
    print('shift_list =', shift_list)
    print('train_op_list =', train_op_list)
    print('test_op_list =', test_op_list)

    overlap = int(sequence_with_overlap)

    fixed_labels = []
    n_steps = 128

    for shift_id in shift_list:
        for op in train_op_list:
            label = corpus_aug_reshape[shift_id][op]['label'][overlap]
            if label.ndim == 1:
                # Reshape to 2D if it's 1D
                label = label.reshape(1, -1)

            # Ensure consistent width (n_steps)
            current_width = label.shape[1]
            if current_width < n_steps:
                # Create padding
                padding = np.zeros((label.shape[0], n_steps - current_width), dtype=label.dtype)
                # For structured arrays, we need to fill with a padding value
                for field in label.dtype.names:
                    if field == 'op':
                        padding[field] = op
                    else:
                        padding[field] = 'pad'  # Use appropriate padding value
                # Concatenate along width
                label = np.concatenate([label, padding], axis=1)
            elif current_width > n_steps:
                # Truncate to n_steps
                label = label[:, :n_steps]

            fixed_labels.append(label)

    # Training set
    train_data = {'pianoroll': np.concatenate([corpus_aug_reshape[shift_id][op]['pianoroll'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'tonal_centroid': np.concatenate([corpus_aug_reshape[shift_id][op]['tonal_centroid'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'len': np.concatenate([corpus_aug_reshape[shift_id][op]['len'][overlap] for shift_id in shift_list for op in train_op_list], axis=0),
                  'label': np.concatenate(fixed_labels, axis=0)}

    # Create int32 arrays 
    train_data_label_key = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_degree1 = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_degree2 = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_quality = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_inversion = np.zeros_like(train_data['label'], dtype=np.int32)
    train_data_label_extra_info = np.zeros_like(train_data['label'], dtype=np.int32) # NEW
    
    # Functional harmony labels
    '''key: 42'''
    for k, v in key_dict.items():
        train_data_label_key[train_data['label']['key'] == k] = v
    '''degree1: 10'''
    for k, v in degree1_dict.items():
        train_data_label_degree1[train_data['label']['degree1'] == k] = v
    '''degree2: 14'''
    for k, v in degree2_dict.items():
        train_data_label_degree2[train_data['label']['degree2'] == k] = v
    '''quality: 10'''
    for k, v in quality_dict.items():
        train_data_label_quality[train_data['label']['quality'] == k] = v
    
    # ADD THESE PRINT STATEMENTS HERE
    print("Sample inversion values:", train_data['label']['inversion'][0, :10])
    print("Sample inversion data type:", type(train_data['label']['inversion'][0, 0]))

    '''inversion: 5'''
    for k, v in inversion_dict.items():
        train_data_label_inversion[train_data['label']['inversion'] == k] = v
    '''extra_info: 22''' # NEW
    for k, v in extra_info_dict.items():
        train_data_label_extra_info[train_data['label']['extra_info'] == k] = v
    #'''roman numeral: (degree1, degree2, quality, extra_info, inversion)'''
    #train_data_label_roman = train_data_label_degree1 * 14 * 10 * 22 * 4 + train_data_label_degree2 * 10 * 22 * 4 + train_data_label_quality * 22 * 4 + train_data_label_extra_info * 4 + train_data_label_inversion
    #train_data_label_roman[train_data['label']['key'] == 'pad'] = 10 * 14 * 10 * 22 * 4

    train_data['key'] = train_data_label_key
    train_data['degree1'] = train_data_label_degree1
    train_data['degree2'] = train_data_label_degree2
    train_data['quality'] = train_data_label_quality
    train_data['inversion'] = train_data_label_inversion
    train_data['extra_info'] = train_data_label_extra_info
    #train_data['roman'] = train_data_label_roman

    # Test set
    test_data = {'pianoroll': np.concatenate([corpus_aug_reshape['shift_0'][op]['pianoroll'][0] for op in test_op_list], axis=0),
                 'tonal_centroid': np.concatenate([corpus_aug_reshape['shift_0'][op]['tonal_centroid'][0] for op in test_op_list], axis=0),
                 'len': np.concatenate([corpus_aug_reshape['shift_0'][op]['len'][0] for op in test_op_list], axis=0),
                 'label': np.concatenate([corpus_aug_reshape['shift_0'][op]['label'][0] for op in test_op_list], axis=0)}

    # Create int32 arrays
    test_data_label_key = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_degree1 = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_degree2 = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_quality = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_inversion = np.zeros_like(test_data['label'], dtype=np.int32)
    test_data_label_extra_info = np.zeros_like(test_data['label'], dtype=np.int32)


    # Functional harmony labels
    '''key: 42'''
    for k, v in key_dict.items():
        test_data_label_key[test_data['label']['key'] == k] = v
    '''degree1: 11'''
    for k, v in degree1_dict.items():
        test_data_label_degree1[test_data['label']['degree1'] == k] = v
    '''degree2: 16'''
    for k, v in degree2_dict.items():
        test_data_label_degree2[test_data['label']['degree2'] == k] = v
    '''quality: 11'''
    for k, v in quality_dict.items():
        test_data_label_quality[test_data['label']['quality'] == k] = v

    # ADD THESE PRINT STATEMENTS HERE TOO
    print("Sample test inversion values:", test_data['label']['inversion'][0, :10])
    print("Sample test inversion data type:", type(test_data['label']['inversion'][0, 0]))

    '''inversion: 5'''
    for k, v in inversion_dict.items():
        test_data_label_inversion[test_data['label']['inversion'] == k] = v    
    '''extra_info: 23'''
    for k, v in extra_info_dict.items():
        test_data_label_extra_info[test_data['label']['extra_info'] == k] = v
    #'''roman numeral'''
    #test_data_label_roman = test_data_label_degree1 * 14 * 10 * 22 * 4 + test_data_label_degree2 * 10 * 22 * 4 + test_data_label_quality * 22 * 4 + test_data_label_extra_info * 4 + test_data_label_inversion
    #test_data_label_roman[test_data['label']['key'] == 'pad'] = 10 * 14 * 10 * 22 * 4

    test_data['key'] = test_data_label_key
    test_data['degree1'] = test_data_label_degree1
    test_data['degree2'] = test_data_label_degree2
    test_data['quality'] = test_data_label_quality
    test_data['inversion'] = test_data_label_inversion
    test_data['extra_info'] = test_data_label_extra_info
    #test_data['roman'] = test_data_label_roman

    print('keys in train/test_data =', train_data.keys())

    required_keys = ['pianoroll', 'tonal_centroid', 'len', 'label', 'key', 'degree1', 'degree2', 
                'quality', 'inversion', 'extra_info']  # Removed 'roman'
    for key in required_keys:
        if key not in train_data:
            print(f"Error: Missing '{key}' in train_data")
    
    #print("Train roman range:", np.min(train_data['roman']), "to", np.max(train_data['roman']))
    #print("Test roman range:", np.min(test_data['roman']), "to", np.max(test_data['roman']))
    print("Train key range:", np.min(train_data['key']), "to", np.max(train_data['key']))
    print("Train degree1 range:", np.min(train_data['degree1']), "to", np.max(train_data['degree1']))
    print("Train degree2 range:", np.min(train_data['degree2']), "to", np.max(train_data['degree2']))
    print("Train quality range:", np.min(train_data['quality']), "to", np.max(train_data['quality']))
    print("Train inversion range:", np.min(train_data['inversion']), "to", np.max(train_data['inversion']))
    print("Train extra_info range:", np.min(train_data['extra_info']), "to", np.max(train_data['extra_info']))

    return train_data, test_data


def compute_pre_PRF(predicted, actual):
    predicted = tf.cast(predicted, tf.float32)
    actual = tf.cast(actual, tf.float32)
    TP = tf.count_nonzero(predicted * actual, dtype=tf.float32)
    # TN = tf.count_nonzero((predicted - 1) * (actual - 1), dtype=tf.float32)
    FP = tf.count_nonzero(predicted * (actual - 1), dtype=tf.float32)
    FN = tf.count_nonzero((predicted - 1) * actual, dtype=tf.float32)
    return TP, FP, FN


def comput_PRF_with_pre(TP, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    precision = tf.cond(tf.is_nan(precision), lambda: tf.constant(0.0), lambda: precision)
    recall = tf.cond(tf.is_nan(recall), lambda: tf.constant(0.0), lambda: recall)
    F1 = tf.cond(tf.is_nan(F1), lambda: tf.constant(0.0), lambda: F1)
    return precision, recall, F1


def train_HT():
    print('Run HT functional harmony recognition on %s-%d...' % (hp.dataset, hp.test_set_id))

    # Load training and testing data
    train_data, test_data = load_data_functional(dir=hp.dataset + '_preprocessed_data_MIREX_Mm.pickle', 
                                                 test_set_id=hp.test_set_id, sequence_with_overlap=hp.train_sequence_with_overlap)
    
    # Initialize logging results
    log_file = setup_logging(hp)
    
    n_train_sequences = train_data['pianoroll'].shape[0]
    n_test_sequences = test_data['pianoroll'].shape[0]
    n_iterations_per_epoch = int(math.ceil(n_train_sequences/hp.n_batches))
    print('n_train_sequences =', n_train_sequences)
    print('n_test_sequences =', n_test_sequences)
    print('n_iterations_per_epoch =', n_iterations_per_epoch)
    print(hp) 

    log_and_print(f"n_train_sequences = {n_train_sequences}", log_file)
    log_and_print(f"n_test_sequences = {n_test_sequences}", log_file)
    log_and_print(f"n_iterations_per_epoch = {n_iterations_per_epoch}", log_file)
    log_and_print(f"hyperparameters({hp})", log_file)

    with tf.name_scope('placeholder'):
        x_p = tf.placeholder(tf.int32, [None, hp.n_steps, 88], name="pianoroll")
        x_len = tf.placeholder(tf.int32, [None], name="seq_lens")
        y_k = tf.placeholder(tf.int32, [None, hp.n_steps], name="key") # 7 degrees * 3 accidentals * 2 modes = 42
        #y_r = tf.placeholder(tf.int32, [None, hp.n_steps], name="roman_numeral")
        y_cc = tf.placeholder(tf.int32, [None, hp.n_steps], name="chord_change")
        y_d1 = tf.placeholder(tf.int32, [None, hp.n_steps], name="degree1")
        y_d2 = tf.placeholder(tf.int32, [None, hp.n_steps], name="degree2")
        y_q = tf.placeholder(tf.int32, [None, hp.n_steps], name="quality")
        y_inv = tf.placeholder(tf.int32, [None, hp.n_steps], name="inversion")
        y_ex = tf.placeholder(tf.int32, [None, hp.n_steps], name="extra_info")
        dropout = tf.placeholder(dtype=tf.float32, name="dropout_rate")
        is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        global_step = tf.placeholder(dtype=tf.int32, name='global_step')
        slope = tf.placeholder(dtype=tf.float32, name='annealing_slope')

    with tf.name_scope('model'):
        x_in = tf.cast(x_p, tf.float32)
        source_mask = tf.sequence_mask(lengths=x_len, maxlen=hp.n_steps, dtype=tf.float32) # [n_batches, n_steps]
        target_mask = source_mask
        # chord_change_logits, dec_input_embed, enc_weights, dec_weights = crm.HT(x_in, source_mask, target_mask, slope, dropout, is_training, hp)
        chord_change_logits, dec_input_embed, enc_weights, dec_weights, _, _ = crm.HTv2(x_in, source_mask, target_mask, slope, dropout, is_training, hp)

    '''
    with tf.variable_scope("output_projection"):
        n_key_classes = 42 + 1
        n_roman_classes = 10 * 14 * 10 * 22 * 4 + 1
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout, training=is_training)
        key_logits = tf.layers.dense(dec_input_embed, n_key_classes)
        roman_logits = tf.layers.dense(dec_input_embed, n_roman_classes)
    '''  

    with tf.variable_scope("output_projection"):
        # Apply dropout to the decoder output
        dec_input_embed = tf.layers.dropout(dec_input_embed, rate=dropout, training=is_training)
        
        # Separate prediction heads for each component
        key_logits = tf.layers.dense(dec_input_embed, 43)  # 43 classes for key
        degree1_logits = tf.layers.dense(dec_input_embed, 11)  # 11 classes for degree1
        degree2_logits = tf.layers.dense(dec_input_embed, 16)  # 16 classes for degree2
        quality_logits = tf.layers.dense(dec_input_embed, 11)  # 11 classes for quality
        inversion_logits = tf.layers.dense(dec_input_embed, 5)  # 5 classes for inversion (0-3 + padding)
        extra_info_logits = tf.layers.dense(dec_input_embed, 23)  # 23 classes for extra_info


    with tf.name_scope('loss'):
        # Chord change
        loss_cc = 4 * tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_cc, tf.float32), logits=slope*chord_change_logits, weights=source_mask)
        # Key
        loss_k = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, 43), logits=key_logits, weights=target_mask, label_smoothing=0.01)
        # Component losses
        loss_d1 = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_d1, 11), logits=degree1_logits, weights=target_mask)
        loss_d2 = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_d2, 16), logits=degree2_logits, weights=target_mask)
        loss_q = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_q, 11), logits=quality_logits, weights=target_mask)
        loss_inv = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_inv, 5), logits=inversion_logits, weights=target_mask)
        loss_ex = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_ex, 23), logits=extra_info_logits, weights=target_mask)
        # Combined roman component loss
        loss_components = 0.5 * (loss_d1 + loss_d2 + loss_q + loss_ex + loss_inv)
        # Total loss
        loss = loss_cc + loss_k + loss_components

    valid = tf.reduce_sum(target_mask)
    # Update to include 8 loss values: total, cc, key, components, d1, d2, q, ex, inv
    summary_loss = tf.Variable([0.0 for _ in range(9)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_cc, loss_k, loss_components, loss_d1, loss_d2, loss_q, loss_ex, loss_inv])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_chord_change', summary_loss[1])
    tf.summary.scalar('Loss_key', summary_loss[2])
    tf.summary.scalar('Loss_components', summary_loss[3])
    tf.summary.scalar('Loss_degree1', summary_loss[4])
    tf.summary.scalar('Loss_degree2', summary_loss[5])
    tf.summary.scalar('Loss_quality', summary_loss[6])
    tf.summary.scalar('Loss_inversion', summary_loss[7])
    tf.summary.scalar('Loss_extra_info', summary_loss[8])

    '''
        # Chord change
        loss_cc = 4 * tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(y_cc, tf.float32), logits=slope*chord_change_logits, weights=source_mask)
        # Key
        loss_k = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_k, n_key_classes), logits=key_logits, weights=target_mask, label_smoothing=0.01)
        # Roman numeral
        loss_r = 0.5 * tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(y_r, n_roman_classes), logits=roman_logits, weights=target_mask, label_smoothing=0.0)
        # Total loss
        loss = loss_cc + loss_k + loss_r
    valid = tf.reduce_sum(target_mask)
    summary_loss = tf.Variable([0.0 for _ in range(4)], trainable=False, dtype=tf.float32)
    summary_valid = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    update_loss = tf.assign(summary_loss, summary_loss + valid * [loss, loss_cc, loss_k, loss_r])
    update_valid = tf.assign(summary_valid, summary_valid + valid)
    mean_loss = tf.assign(summary_loss, summary_loss / summary_valid)
    clr_summary_loss = summary_loss.initializer
    clr_summary_valid = summary_valid.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_chord_change', summary_loss[1])
    tf.summary.scalar('Loss_key', summary_loss[2])
    tf.summary.scalar('Loss_roman', summary_loss[3])
    '''

    with tf.name_scope('evaluation'):
        eval_mask = tf.cast(target_mask, tf.bool)
        # Chord change
        pred_cc = tf.cast(tf.round(tf.sigmoid(slope*chord_change_logits)), tf.int32)
        pred_cc_mask = tf.boolean_mask(pred_cc, tf.cast(source_mask, tf.bool))
        y_cc_mask = tf.boolean_mask(y_cc, tf.cast(source_mask, tf.bool))
        TP_cc, FP_cc, FN_cc = compute_pre_PRF(pred_cc_mask, y_cc_mask)
        # Key
        pred_k = tf.argmax(key_logits, axis=2, output_type=tf.int32)
        pred_k_correct = tf.equal(pred_k, y_k)
        pred_k_correct_mask = tf.boolean_mask(tensor=pred_k_correct, mask=eval_mask)
        n_correct_k = tf.reduce_sum(tf.cast(pred_k_correct_mask, tf.float32))
        
        # Component predictions
        pred_d1 = tf.argmax(degree1_logits, axis=2, output_type=tf.int32)
        pred_d1_correct = tf.equal(pred_d1, y_d1)
        pred_d1_correct_mask = tf.boolean_mask(tensor=pred_d1_correct, mask=eval_mask)
        n_correct_d1 = tf.reduce_sum(tf.cast(pred_d1_correct_mask, tf.float32))
        
        pred_d2 = tf.argmax(degree2_logits, axis=2, output_type=tf.int32)
        pred_d2_correct = tf.equal(pred_d2, y_d2)
        pred_d2_correct_mask = tf.boolean_mask(tensor=pred_d2_correct, mask=eval_mask)
        n_correct_d2 = tf.reduce_sum(tf.cast(pred_d2_correct_mask, tf.float32))
        
        pred_q = tf.argmax(quality_logits, axis=2, output_type=tf.int32)
        pred_q_correct = tf.equal(pred_q, y_q)
        pred_q_correct_mask = tf.boolean_mask(tensor=pred_q_correct, mask=eval_mask)
        n_correct_q = tf.reduce_sum(tf.cast(pred_q_correct_mask, tf.float32))
        
        pred_inv = tf.argmax(inversion_logits, axis=2, output_type=tf.int32)
        pred_inv_correct = tf.equal(pred_inv, y_inv)
        pred_inv_correct_mask = tf.boolean_mask(tensor=pred_inv_correct, mask=eval_mask)
        n_correct_inv = tf.reduce_sum(tf.cast(pred_inv_correct_mask, tf.float32))

        pred_ex = tf.argmax(extra_info_logits, axis=2, output_type=tf.int32)
        pred_ex_correct = tf.equal(pred_ex, y_ex)
        pred_ex_correct_mask = tf.boolean_mask(tensor=pred_ex_correct, mask=eval_mask)
        n_correct_ex = tf.reduce_sum(tf.cast(pred_ex_correct_mask, tf.float32))
        
        # Roman numeral
        #pred_r = tf.argmax(roman_logits, axis=2, output_type=tf.int32)
        #pred_r_correct = tf.equal(pred_r, y_r)
        #pred_r_correct_mask = tf.boolean_mask(tensor=pred_r_correct, mask=eval_mask)
        #n_correct_r = tf.reduce_sum(tf.cast(pred_r_correct_mask, tf.float32))
        #n_total = tf.cast(tf.size(pred_k_correct_mask), tf.float32)

        # Calculate total number of tokens for accuracy
        n_total = tf.cast(tf.size(pred_k_correct_mask), tf.float32)


    # Define summary variables - one single definition with the correct size
    summary_count = tf.Variable([0.0 for _ in range(10)], trainable=False, dtype=tf.float32)  # For key, d1, d2, q, ex, inv, total, TP_cc, FP_cc, FN_cc
    summary_score = tf.Variable([0.0 for _ in range(10)], trainable=False, dtype=tf.float32)  # For key, components, d1, d2, q, ex, inv, P_cc, R_cc, F1_cc
    
    # Define update_count operation
    update_count = tf.assign(summary_count, summary_count + [n_correct_k, n_correct_d1, n_correct_d2, n_correct_q, n_correct_ex, n_correct_inv, n_total, TP_cc, FP_cc, FN_cc])

    # Define the new calculation
    acc_k = summary_count[0] / summary_count[6]
    acc_d1 = summary_count[1] / summary_count[6]
    acc_d2 = summary_count[2] / summary_count[6]
    acc_q = summary_count[3] / summary_count[6]
    acc_inv = summary_count[5] / summary_count[6]
    acc_ex = summary_count[4] / summary_count[6]
    P_cc, R_cc, F1_cc = comput_PRF_with_pre(summary_count[7], summary_count[8], summary_count[9])

    # Calculate combined component accuracy (average of all components)
    acc_components = (acc_d1 + acc_d2 + acc_q + acc_ex + acc_inv) / 5.0
    
    # Now define update_score after the accuracy calculations
    update_score = tf.assign(summary_score, summary_score + [acc_k, acc_components, acc_d1, acc_d2, acc_q, acc_ex, acc_inv, P_cc, R_cc, F1_cc])

    #acc_k = summary_count[0] / summary_count[2]
    #acc_r = summary_count[1] / summary_count[2]
    #P_cc, R_cc, F1_cc = comput_PRF_with_pre(summary_count[3], summary_count[4], summary_count[5])
    
    #update_score = tf.assign(summary_score, summary_score + [acc_k, acc_r, P_cc, R_cc, F1_cc])
    update_score = tf.assign(summary_score, summary_score + [acc_k, acc_components, acc_d1, acc_d2, acc_q, acc_inv, acc_ex, P_cc, R_cc, F1_cc])
    clr_summary_count = summary_count.initializer
    clr_summary_score = summary_score.initializer
    tf.summary.scalar('Loss_total', summary_loss[0])
    tf.summary.scalar('Loss_chord_change', summary_loss[1]) 
    tf.summary.scalar('Loss_key', summary_loss[2])
    tf.summary.scalar('Loss_components', summary_loss[3])
    tf.summary.scalar('Loss_degree1', summary_loss[4])
    tf.summary.scalar('Loss_degree2', summary_loss[5])
    tf.summary.scalar('Loss_quality', summary_loss[6])
    tf.summary.scalar('Loss_inversion', summary_loss[7])
    tf.summary.scalar('Loss_extra_info', summary_loss[8])

    tf.summary.scalar('Accuracy_key', summary_score[0])
    tf.summary.scalar('Accuracy_components', summary_score[1])
    tf.summary.scalar('Accuracy_degree1', summary_score[2])
    tf.summary.scalar('Accuracy_degree2', summary_score[3])
    tf.summary.scalar('Accuracy_quality', summary_score[4])
    tf.summary.scalar('Accuracy_inversion', summary_score[5])
    tf.summary.scalar('Accuracy_extra_info', summary_score[6])
    tf.summary.scalar('Precision_cc', summary_score[7])
    tf.summary.scalar('Recall_cc', summary_score[8])
    tf.summary.scalar('F1_cc', summary_score[9])


    with tf.name_scope('optimization'):
        # Apply warn-up learning rate
        warm_up_steps = tf.constant(4000, dtype=tf.float32)
        gstep = tf.cast(global_step, dtype=tf.float32)
        learning_rate = pow(hp.input_embed_size, -0.5) * tf.minimum(tf.pow(gstep, -0.5), gstep * tf.pow(warm_up_steps, -1.5))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=0.9,
                                           beta2=0.98,
                                           epsilon=1e-9)
        train_op = optimizer.minimize(loss)
    # Graph location and summary writers
    print('Saving graph to: %s' % hp.graph_location)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(hp.graph_location + '\\train')
    test_writer = tf.summary.FileWriter(hp.graph_location + '\\test')
    train_writer.add_graph(tf.get_default_graph())
    test_writer.add_graph(tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=1)

    # Log that we're saving the graph
    log_and_print(f"Saving graph to: {hp.graph_location}", log_file)
 

    # Training
    print('Train the model...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        startTime = time.time() # start time of training
        best_score = [0.0 for _ in range(6)]
        in_succession = 0
        best_epoch = 0
        annealing_slope = 1.0
        best_slope = 0.0
        for step in range(hp.n_training_steps):
            # Training
            if step == 0:
                indices = range(n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            if step > 0 and step % n_iterations_per_epoch == 0:
                annealing_slope *= hp.annealing_rate

            if step >= 2*n_iterations_per_epoch and step % n_iterations_per_epoch == 0:
                # Shuffle training data
                indices = random.sample(range(n_train_sequences), n_train_sequences)
                batch_indices = [indices[x:x + hp.n_batches] for x in range(0, len(indices), hp.n_batches)]

            batch = (train_data['pianoroll'][batch_indices[step % len(batch_indices)]],
                     train_data['len'][batch_indices[step % len(batch_indices)]],
                     train_data['label']['chord_change'][batch_indices[step % len(batch_indices)]],
                     train_data['key'][batch_indices[step % len(batch_indices)]],
                     train_data['degree1'][batch_indices[step % len(batch_indices)]],
                     train_data['degree2'][batch_indices[step % len(batch_indices)]],
                     train_data['quality'][batch_indices[step % len(batch_indices)]],
                     train_data['inversion'][batch_indices[step % len(batch_indices)]],
                     train_data['extra_info'][batch_indices[step % len(batch_indices)]])
                    # Removed 'train_data['roman'][batch_indices[step % len(batch_indices)]]'
                     

            #train_run_list = [train_op, update_valid, update_loss, update_count, loss, loss_cc, loss_k, loss_r, pred_cc, pred_k, pred_r, eval_mask, enc_weights, dec_weights]
            train_run_list = [
                train_op, 
                update_valid, 
                update_loss, 
                update_count, 
                loss, 
                loss_cc, 
                loss_k, 
                loss_components,  # Instead of loss_r
                loss_d1,          # Individual component losses
                loss_d2, 
                loss_q, 
                loss_inv,
                loss_ex, 
                pred_cc, 
                pred_k, 
                pred_d1,         # Component predictions
                pred_d2, 
                pred_q, 
                pred_inv,
                pred_ex,
                eval_mask, 
                enc_weights, 
                dec_weights
            ]

            """
            # Print out dictionaries
            print("\nDictionaries for conversion:")
            print("Key dictionary:", key_dict)
            print("Degree1 dictionary:", degree1_dict)
            print("Degree2 dictionary:", degree2_dict)
            print("Quality dictionary:", quality_dict)
            print("Inversion dictionary", inversion_dict)
            print("Extra_info dictionary:", extra_info_dict)

            # Check batch[3] (key data)
            print("\nChecking batch[3] data (key):")
            if isinstance(batch[3], np.ndarray):
                flat_data = batch[3].flatten()
                for i, val in enumerate(flat_data[:20]):  # Print first 20 values
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")

            # Check batch[4] (degree1 data)
            print("\nChecking batch[4] data (degree1):")
            if isinstance(batch[4], np.ndarray):
                flat_data = batch[4].flatten()
                for i, val in enumerate(flat_data[:20]):
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")

            # Check batch[5] (degree2 data)
            print("\nChecking batch[5] data (degree2):")
            if isinstance(batch[5], np.ndarray):
                flat_data = batch[5].flatten()
                for i, val in enumerate(flat_data[:20]):
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")

            # Check batch[6] (quality data)
            print("\nChecking batch[6] data (quality):")
            if isinstance(batch[6], np.ndarray):
                flat_data = batch[6].flatten()
                for i, val in enumerate(flat_data[:20]):
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")

            # Check batch[7] (inversion data)
            print("\nChecking batch[7] data (inversion):")
            if isinstance(batch[7], np.ndarray):
                flat_data = batch[7].flatten()
                for i, val in enumerate(flat_data[:20]):
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")

            # Check batch[8] (extra_info data)
            print("\nChecking batch[8] data (extra_info):")
            if isinstance(batch[8], np.ndarray):
                flat_data = batch[8].flatten()
                for i, val in enumerate(flat_data[:20]):
                    print(f"Position {i}: Value '{val}', Type: {type(val)}")


              
            # Convert string values to integers
            batch = list(batch)  # Convert tuple to list so we can modify it

            # Check and convert key data (y_k)
            if isinstance(batch[3], np.ndarray):
                batch[3] = np.array([key_dict.get(x, 0) if isinstance(x, str) else x for x in batch[3].flatten()]).reshape(batch[3].shape)

            # Check and convert degree1 data (y_d1)
            if isinstance(batch[5], np.ndarray):
                batch[5] = np.array([int(x) if isinstance(x, str) and x.isdigit() else 0 for x in batch[5].flatten()]).reshape(batch[5].shape)

            # Check and convert degree2 data (y_d2)
            if isinstance(batch[6], np.ndarray):
                batch[6] = np.array([int(x) if isinstance(x, str) and x.isdigit() else 0 for x in batch[6].flatten()]).reshape(batch[6].shape)
            
            # Check and convert quality data (y_q)
            if isinstance(batch[7], np.ndarray):
                batch[7] = np.array([int(x) if isinstance(x, str) and x.isdigit() else 0 for x in batch[7].flatten()]).reshape(batch[7].shape)
            
            # Check and convert extra_info data (y_ex)
            if isinstance(batch[8], np.ndarray):
                batch[8] = np.array([int(x) if isinstance(x, str) and x.isdigit() else 0 for x in batch[8].flatten()]).reshape(batch[8].shape)
            """

            train_feed_fict = {x_p: batch[0],            # pianoroll
                               x_len: batch[1],          # sequence lengths
                               y_cc: batch[2],           # chord change
                               y_k: batch[3],            # key
                               y_d1: batch[4],           # degree1
                               y_d2: batch[5],           # degree2
                               y_q: batch[6],            # quality 
                               y_inv: batch[7],          # inversion
                               y_ex: batch[8],           # extra_info
                               dropout: hp.drop,
                               is_training: True,
                               global_step: step + 1,
                               slope: annealing_slope}

            _, _, _, _, train_loss, train_loss_cc, train_loss_k, train_loss_components, \
            train_loss_d1, train_loss_d2, train_loss_q, train_loss_ex, train_loss_inv, \
            train_pred_cc, train_pred_k, train_pred_d1, train_pred_d2, train_pred_q, \
            train_pred_ex, train_pred_inv, train_eval_mask, enc_w, dec_w = sess.run(train_run_list, feed_dict=train_feed_fict)

            if step == 0:
                print('*~ loss_cc %.4f, loss_k %.4f' % (train_loss_cc, train_loss_k))   # Remove 'train_loss_r'

            # Display training log & Testing
            if step > 0 and step % n_iterations_per_epoch == 0:
                sess.run([mean_loss, update_score])
                train_summary, train_loss, train_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                train_writer.add_summary(train_summary, step)
                print("---- step %d, epoch %d: train_loss: total %.4f (cc %.4f, k %.4f, components %.4f), evaluation: k %.4f, components %.4f (d1 %.4f, d2 %.4f, q %.4f, ex %.4f, inv %.4f), cc (P %.4f, R %.4f, F1 %.4f) ----"
                        % (step, step // n_iterations_per_epoch, 
                           train_loss[0], train_loss[1], train_loss[2], train_loss[3],
                           train_score[0], train_score[1], train_score[2], train_score[3], train_score[4], train_score[5], train_score[6],
                           train_score[7], train_score[8], train_score[9]))

                # Log result
                log_and_print(f"---- step {step}, epoch {step // n_iterations_per_epoch}: train_loss: total {train_loss[0]:.4f} (cc {train_loss[1]:.4f}, k {train_loss[2]:.4f}, components {train_loss[3]:.4f}), evaluation: k {train_score[0]:.4f}, components {train_score[1]:.4f}, cc (P {train_score[7]:.4f}, R {train_score[8]:.4f}, F1 {train_score[9]:.4f}) ----", log_file)
                print('enc_w =', enc_w, 'dec_w =', dec_w)
                # Log result
                log_and_print(f"enc_w = {enc_w} dec_w = {dec_w}", log_file)
                display_len = 32
                n_just = 5
                print('len =', batch[1][0])
                log_and_print(f"len = {batch[1][0]}", log_file)

                print('y_k'.ljust(7, ' '), ''.join([[k for k, v in key_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[3][0, :display_len]]))
                line = 'y_k'.ljust(7, ' ') + ''.join([[k for k, v in key_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[3][0, :display_len]])
                log_and_print(line, log_file)
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
                line = 'y_d1'.ljust(7, ' ') + ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[4][0, :display_len]])
                log_and_print(line, log_file)
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]]))
                line = 'y_d2'.ljust(7, ' ') + ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[5][0, :display_len]])
                log_and_print(line, log_file)
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]]))
                line = 'y_q'.ljust(7, ' ') + ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[6][0, :display_len]])
                log_and_print(line, log_file)
                print('y_inv'.ljust(7, ' '), ''.join([str([k for k, v in inversion_dict.items() if v == b][0]).rjust(n_just, ' ') for b in batch[7][0, :display_len]]))
                line = 'y_inv'.ljust(7, ' ') + ''.join([str([k for k, v in inversion_dict.items() if v == b][0]).rjust(n_just, ' ') for b in batch[7][0, :display_len]])
                log_and_print(line, log_file)
                print('y_ex'.ljust(7, ' '), ''.join([[k for k, v in extra_info_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[8][0, :display_len]]))
                line = 'y_ex'.ljust(7, ' ') + ''.join([[k for k, v in extra_info_dict.items() if v == b][0].rjust(n_just, ' ') for b in batch[8][0, :display_len]])
                log_and_print(line, log_file)
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]]))
                line = 'valid'.ljust(7, ' ') + ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in train_eval_mask[0, :display_len]])
                log_and_print(line, log_file)
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]]))
                line = 'y_cc'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in batch[2][0, :display_len]])
                log_and_print(line, log_file)
                # Component predictions
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_cc[0, :display_len]]))
                line = 'pred_cc'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_cc[0, :display_len]])
                log_and_print(line, log_file)
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]]))
                line = 'pred_k'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_k[0, :display_len]])
                log_and_print(line, log_file)
                # NEW 
                print('pred_d1'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_d1[0, :display_len]]))
                line = 'pred_d1'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_d1[0, :display_len]])
                log_and_print(line, log_file)
                print('pred_d2'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_d2[0, :display_len]]))
                line = 'pred_d2'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_d2[0, :display_len]])
                log_and_print(line, log_file)
                print('pred_q'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_q[0, :display_len]]))
                line = 'pred_q'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_q[0, :display_len]])
                log_and_print(line, log_file)
                print('pred_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_inv[0, :display_len]]))
                line = 'pred_inv'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_inv[0, :display_len]])
                log_and_print(line, log_file)
                print('pred_ex'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_ex[0, :display_len]]))
                line = 'pred_ex'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_ex[0, :display_len]])
                log_and_print(line, log_file)
                
                #print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in batch[4][0, :display_len]]))
                #line = 'y_r'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in batch[4][0, :display_len]])
                #log_and_print(line, log_file)
                #print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]]))
                #line = 'pred_r'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in train_pred_r[0, :display_len]])
                #log_and_print(line, log_file)

                # Testing
                #test_run_list = [update_valid, update_loss, update_count, pred_cc, pred_k, pred_r, eval_mask]
                test_run_list = [
                    update_valid, 
                    update_loss, 
                    update_count, 
                    pred_cc, 
                    pred_k, 
                    pred_d1,         # Component predictions
                    pred_d2, 
                    pred_q, 
                    pred_ex, 
                    pred_inv,
                    eval_mask
                ]
                
                test_feed_fict = {x_p: test_data['pianoroll'],
                                  x_len: test_data['len'],
                                  y_cc: test_data['label']['chord_change'],
                                  y_k: test_data['key'],
                                  y_d1: test_data['degree1'],
                                  y_d2: test_data['degree2'],
                                  y_q: test_data['quality'],
                                  y_inv: test_data['inversion'],  # Inversion before extra_info
                                  y_ex: test_data['extra_info'],
                                  dropout: 0.0,
                                  is_training: False,
                                  slope: annealing_slope}


                _, _, _, test_pred_cc, test_pred_k, test_pred_d1, test_pred_d2, test_pred_q, test_pred_ex, test_pred_inv, test_eval_mask = sess.run(test_run_list, feed_dict=test_feed_fict)
                sess.run([mean_loss, update_score])
                test_summary, test_loss, test_score = sess.run([merged, summary_loss, summary_score])
                sess.run([clr_summary_valid, clr_summary_loss, clr_summary_count, clr_summary_score])
                test_writer.add_summary(test_summary, step)

                """
                # Create component-wise segmentation quality for training data
                train_sq_d1 = crm.segmentation_quality(train_data['degree1'][batch_indices[step % len(batch_indices)]], train_pred_d1, batch[1])
                train_sq_d2 = crm.segmentation_quality(train_data['degree2'][batch_indices[step % len(batch_indices)]], train_pred_d2, batch[1])
                train_sq_q = crm.segmentation_quality(train_data['quality'][batch_indices[step % len(batch_indices)]], train_pred_q, batch[1])
                train_sq_ex = crm.segmentation_quality(train_data['extra_info'][batch_indices[step % len(batch_indices)]], train_pred_ex, batch[1])
                train_sq_inv = crm.segmentation_quality(train_data['inversion'][batch_indices[step % len(batch_indices)]], train_pred_inv, batch[1])

                # Calculate average segmentation quality
                train_sq = (train_sq_d1 + train_sq_d2 + train_sq_q + train_sq_ex + train_sq_inv) / 5.0
                #sq = crm.segmentation_quality(test_data['roman'], test_pred_r, test_data['len'])
                """

                # Create component-wise segmentation quality for test data
                sq_d1 = crm.segmentation_quality(test_data['degree1'], test_pred_d1, test_data['len'])
                sq_d2 = crm.segmentation_quality(test_data['degree2'], test_pred_d2, test_data['len'])
                sq_q = crm.segmentation_quality(test_data['quality'], test_pred_q, test_data['len'])
                sq_inv = crm.segmentation_quality(test_data['inversion'], test_pred_inv, test_data['len'])
                sq_ex = crm.segmentation_quality(test_data['extra_info'], test_pred_ex, test_data['len'])
                
                # Calculate average segmentation quality
                sq = (sq_d1 + sq_d2 + sq_q + sq_inv + sq_ex) / 5.0

                print("==== step %d, epoch %d: test_loss: total %.4f (cc %.4f, k %.4f, components %.4f), evaluation: k %.4f, components %.4f (d1 %.4f, d2 %.4f, q %.4f, inv %.4f, ex %.4f), cc (P %.4f, R %.4f, F1 %.4f), sq %.4f ===="
                        % (step, step // n_iterations_per_epoch, 
                            test_loss[0], test_loss[1], test_loss[2], test_loss[3],
                            test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], test_score[5], test_score[6],
                            test_score[7], test_score[8], test_score[9], sq))
                sample_id = random.randint(0, n_test_sequences - 1)
                print('len =', test_data['len'][sample_id])
                log_and_print(f"len = {test_data['len'][sample_id]}", log_file)

                print('y_k'.ljust(7, ' '), ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]]))
                line = 'y_k'.ljust(7, ' ') + ''.join([b.rjust(n_just, ' ') for b in test_data['label']['key'][sample_id, :display_len]])
                log_and_print(line, log_file)                 
                print('y_d1'.ljust(7, ' '), ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]]))
                line = 'y_d1'.ljust(7, ' ') + ''.join([[k for k, v in degree1_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree1'][sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('y_d2'.ljust(7, ' '), ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]]))
                line = 'y_d2'.ljust(7, ' ') + ''.join([[k for k, v in degree2_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['degree2'][sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('y_q'.ljust(7, ' '), ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]]))
                line = 'y_q'.ljust(7, ' ') + ''.join([[k for k, v in quality_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['quality'][sample_id, :display_len]])
                log_and_print(line, log_file)    
                print('y_inv'.ljust(7, ' '), ''.join([str([k for k, v in inversion_dict.items() if v == b][0]).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]]))
                line = 'y_inv'.ljust(7, ' ') + ''.join([str([k for k, v in inversion_dict.items() if v == b][0]).rjust(n_just, ' ') for b in test_data['inversion'][sample_id, :display_len]])
                log_and_print(line, log_file)           
                print('y_ex'.ljust(7, ' '), ''.join([[k for k, v in extra_info_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['extra_info'][sample_id, :display_len]]))
                line = 'y_ex'.ljust(7, ' ') + ''.join([[k for k, v in extra_info_dict.items() if v == b][0].rjust(n_just, ' ') for b in test_data['extra_info'][sample_id, :display_len]])
                log_and_print(line, log_file)                   
                print('valid'.ljust(7, ' '), ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]]))
                line = 'valid'.ljust(7, ' ') + ''.join(['y'.rjust(n_just, ' ') if b else 'n'.rjust(n_just, ' ') for b in test_eval_mask[sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('y_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['label']['chord_change'][sample_id, :display_len]]))
                line = 'y_cc'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_data['label']['chord_change'][sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('pred_cc'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_cc[sample_id, :display_len]]))
                line = 'pred_cc'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_cc[sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('y_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]]))
                line = 'y_k'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_data['key'][sample_id, :display_len]])
                log_and_print(line, log_file)                
                print('pred_k'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]]))
                line = 'pred_k'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_k[sample_id, :display_len]])
                log_and_print(line, log_file)    
                # Add new component predictions
                print('pred_d1'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_d1[sample_id, :display_len]]))
                line = 'pred_d1'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_d1[sample_id, :display_len]])
                log_and_print(line, log_file)
                print('pred_d2'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_d2[sample_id, :display_len]]))
                line = 'pred_d2'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_d2[sample_id, :display_len]])
                log_and_print(line, log_file)
                print('pred_q'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_q[sample_id, :display_len]]))
                line = 'pred_q'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_q[sample_id, :display_len]])
                log_and_print(line, log_file)
                print('pred_inv'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_inv[sample_id, :display_len]]))
                line = 'pred_inv'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_inv[sample_id, :display_len]])
                log_and_print(line, log_file)  
                print('pred_ex'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_ex[sample_id, :display_len]]))
                line = 'pred_ex'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_ex[sample_id, :display_len]])
                log_and_print(line, log_file)
                

                #print('y_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]]))
                #line = 'y_r'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_data['roman'][sample_id, :display_len]])
                #log_and_print(line, log_file)  
                #print('pred_r'.ljust(7, ' '), ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]]))
                #line = 'pred_r'.ljust(7, ' ') + ''.join([str(b).rjust(n_just, ' ') for b in test_pred_r[sample_id, :display_len]])
                #log_and_print(line, log_file)

                # Calculate combined component accuracy (average of component accuracies)
                component_acc = (test_score[2] + test_score[3] + test_score[4] + test_score[5] + test_score[6]) / 5.0

                if step > 0 and (test_score[0] + component_acc) > (best_score[0] + best_score[1]):
                    # Store all metrics plus segmentation quality
                    best_score = np.concatenate([test_score, [sq]], axis=0)
                    best_epoch = step // n_iterations_per_epoch
                    best_slope = annealing_slope
                    in_succession = 0
                    # Save variables of the model
                    print('*saving variables...\n')
                    # Log result
                    log_and_print('*saving variables...\n', log_file)
                    saver.save(sess, hp.graph_location + '\\HT_functional_harmony_recognition_' + hp.dataset + '_' + str(hp.test_set_id) + '.ckpt')
                else:
                    in_succession += 1
                    if in_succession > hp.n_in_succession:
                        print('Early stopping.')
                        # Log result
                        log_and_print('Early stopping.', log_file)
                        break

        elapsed_time = time.time() - startTime
        print('\nHT functional harmony recognition on %s-%d:' % (hp.dataset, hp.test_set_id))
        print('training time = %.2f hr' % (elapsed_time / 3600))
        print('best epoch = ', best_epoch)
        print('best score =', np.round(best_score, 4))
        print('best slope =', best_slope)

        log_and_print(f"\nHT functional harmony recognition on {hp.dataset}-{hp.test_set_id}:", log_file)
        log_and_print(f"training time = {elapsed_time / 3600:.2f} hr", log_file)
        log_and_print(f"best epoch = {best_epoch}", log_file)
        log_and_print(f"best score = {np.round(best_score, 4)}", log_file)
        log_and_print(f"best slope = {best_slope}", log_file)

        log_file.close()



# Add this function to your code
def setup_logging(hp):
    """Set up logging to a file for training results"""
    # Create a logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique filename with timestamp and test set info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/training_log_testset{hp.test_set_id}_{timestamp}.txt"
    
    # Open the file and write header information
    log_file = open(log_filename, 'w')
    log_file.write(f"Training Log for Harmony Transformer\n")
    log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Dataset: {hp.dataset}, Test Set: {hp.test_set_id}\n")
    log_file.write(f"Hyperparameters: {str(hp)}\n\n")
    log_file.write("=" * 80 + "\n\n")
    
    return log_file


# Modify print statement for each log_file
def log_and_print(message, log_file):
    """Print to console and log file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush() 


def main():
    # Initialize model
    # Functional harmony recognition
    train_HT() # Harmony Transformer
    # train_BTC() # Bi-directional Transformer for Chord Recognition
    # train_CRNN() # Convolutional Recurrent Neural Network

 
if __name__ == '__main__':
    # Hyperparameters
    hyperparameters = namedtuple('hyperparameters',
                                 ['dataset',
                                  'test_set_id',
                                  'graph_location',
                                  'n_steps',
                                  'input_embed_size',
                                  'n_layers',
                                  'n_heads',
                                  'train_sequence_with_overlap',
                                  'initial_learning_rate',
                                  'drop',
                                  'n_batches',
                                  'n_training_steps',
                                  'n_in_succession',
                                  'annealing_rate'])

    hp = hyperparameters(dataset='Sonatas', # {'Baroque_Flute', 'Sonatas'}
                         test_set_id=5,
                         graph_location='model',
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
                         annealing_rate=1.1)

    main() 