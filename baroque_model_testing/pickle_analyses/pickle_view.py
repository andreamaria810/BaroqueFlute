import pickle


file = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_testing\pipeline\test_data_preprocessed_MIREX_Mm_train_format.pickle"

with open(file, 'rb') as f:
    test_data = pickle.load(f)
    
# Validate the data
for piece_id in test_data['shift_0']:
    for idx in [0, 1]:
        pianoroll = test_data['shift_0'][piece_id]['pianoroll'][idx]
        if pianoroll.shape[0] == 0 or np.all(pianoroll == 0):
            print(f"Warning: Empty pianoroll for piece {piece_id}, idx {idx}")
            # Create dummy data to avoid the error
            test_data['shift_0'][piece_id]['pianoroll'][idx] = np.zeros((1, 128, 88))

print(test_data)