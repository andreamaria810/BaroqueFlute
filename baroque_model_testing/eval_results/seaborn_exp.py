import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Δ values for Melody/Harmonicity Metrics
melody_data = {
    'CTnCTR': [0.0189, 0.0143, 0.0452],
    'PCS': [0.0212, 0.0187, 0.0266],
    'MCTD': [-0.0475, -0.0691, -0.2401]
}
melody_index = ['In-Distribution', 'Cross-Composer', 'Out-of-Distribution']
melody_df = pd.DataFrame(melody_data, index=melody_index)

# Δ values for Chord Progression Metrics
chord_data = {
    'CHE': [0.1695, 0.4492, 0.4492],
    'CC': [-0.3412, 0.9716, 0.9716],
    'CTD': [0.1122, 0.0612, 0.0612]
}
chord_index = ['In-Distribution', 'Cross-Composer', 'Out-of-Distribution']
chord_df = pd.DataFrame(chord_data, index=chord_index)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(melody_df, annot=True, cmap='coolwarm', center=0, ax=axes[0])
axes[0].set_title('Δ Melody/Harmonicity Metrics')
axes[0].set_xlabel('Metric')
axes[0].set_ylabel('Test Set')

sns.heatmap(chord_df, annot=True, cmap='coolwarm', center=0, ax=axes[1])
axes[1].set_title('Δ Chord Progression Metrics')
axes[1].set_xlabel('Metric')
axes[1].set_ylabel('Test Set')

plt.tight_layout()
plt.show()
