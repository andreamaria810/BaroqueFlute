import pandas as pd



df = pd.read_csv(r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\test_metadata.csv", encoding="ISO-8859-1")
print(df.head())

"""
training_movement_types = {
    'J.S. Bach': {
        'Allegro': ['2/4', '3/4'], 'Menuet': ['3/4'], 'Siciliano': ['6/8']
    },
    'G.F. Handel': {
        'Grave': ['4/4'], 'Adagio': ['3/4', '2/2'], 'Andante': ['4/4'], 'Menuet': ['6/8'], 
        'Allegro': ['3/4', '3/8'], 'Largo': ['3/4'], 
    },
    'J.M. Leclair': {
        'Adagio': ['4/4'], 'Corrente': ['3/4'], 'Gavotta': ['2/2'], 'Giga': ['6/8'], 'Aria': ['2/2'],
        'Allegro': ['2/4'], 'Altro': ['2/4']
    },
    'L. Vinci': {
        'Allegro': ['2/4'], 'Pastorella': ['3/8']
    },
    'M. Blavet': {
        'Adagio': ['4/4'], 'Allegro': ['2/4'], 'Aria': ['6/8', '2/4'], 'Allemanda': ['4/4'], 'Sarabanda': ['3/4'],
        'Tambourins': ['4/8', '2/4']
    },
    'N. Chedeville': {
        'Gavotte': ['2/4'], 'Allegro': ['2/4', '3/8'], 'Preludio': ['4/8'], 'Sarabanda': ['3/4'], 'Corrente': ['3/4']
    },
    'Frederick the Great': {
        'Grave e spirituoso': ['4/4'], 'Allegro': ['2/4'], 'Presto': ['2/4']
    },
    'G.P. Telemann': {'Cantabile': ['4/4']},
    'C.P.E. Bach': {'Allegro': ['2/4'], 'Allegretto': ['2/4']}
}
"""

training_movements = {
    'J.S. Bach': ['Allegro', 'Menuet', 'Siciliano'],
    'G.F. Handel': ['Grave', 'Adagio', 'Andante', 'Menuet', 'Allegro', 'Largo'],
    'J.M. Leclair': ['Adagio', 'Corrente', 'Gavotta', 'Giga', 'Aria', 'Allegro', 'Altro'],
    'L. Vinci': ['Allegro', 'Pastorella'],
    'M. Blavet': ['Adagio', 'Allegro', 'Aria', 'Allemanda', 'Sarabanda', 'Tambourins'],
    'N. Chedeville': ['Gavotte', 'Allegro', 'Preludio', 'Sarabanda', 'Corrente'],
    'Frederick the Great': ['Grave e spirituoso', 'Allegro', 'Presto'],
    'G.P. Telemann': ['Cantabile'],
    'C.P.E. Bach': ['Allegro', 'Allegretto']
}



def categorize_test_type(row):
    composer = row['Composer']
        
    movement = row['Movement Type']
    in_training = row['In Training?']
    
    # If marked as in training
    if in_training == 'Yes':
        # Check if the composer exists in training data
        if composer in training_movements:
            # Check if this movement type exists for this composer
            if movement in training_movements[composer]:
                return 'In-Distribution'
            else:
                return 'Out-of-Distribution'
        else:
            return 'Out-of-Distribution'
    
    # If marked as not in training
    else:  # in_training == 'No'
        # Check if any other composer has this movement type
        for other_composer, movements in training_movements.items():
            if other_composer != composer and movement in movements:
                return 'Cross-Composer'
        
        # If no other composer has this movement type
        return 'Out-of-Distribution'

# Here's how you would use this function
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\test_metadata.csv", encoding="ISO-8859-1")
    
    # Apply the categorization function
    df['Test Category'] = df.apply(categorize_test_type, axis=1)
    
    # Check for unknowns (there should be none)
    unknown_count = df[df['Test Category'] == 'Unknown']
    if not unknown_count.empty:
        print(f"Total Unknowns: {len(unknown_count)}")
        print(unknown_count)
    else:
        print("No unknowns found!")
    
    # Save the categorized data
    df.to_excel(r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\categorized_test_set.xlsx", index=False)
    
    # Display the resulting DataFrame with categories
    print(df.head())