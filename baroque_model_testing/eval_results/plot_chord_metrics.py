import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns

"""
# Melody/Harmonicity Metrics Data
melody_data = {
    "Test Set": [
        "In-Distribution", "GT", "Model", "Δ",
        "Cross-Composer", "GT", "Model", "Δ",
        "Out-of-Distribution", "GT", "Model", "Δ"
    ],
    "CTnCTR": [
        "", 0.425, 0.440, 0.103,
        "", 0.399, 0.418, 0.098,
        "", 0.380, 0.390, 0.098
    ],
    "PCS": [
        "", 0.094, 0.113, 0.172,
        "", 0.061, 0.090, 0.175,
        "", 0.092, 0.110, 0.152
    ],
    "MCTD": [
        "", 3.319, 3.272, 0.194,
        "", 3.340, 3.271, 0.208,
        "", 3.292, 3.265, 0.188
    ]
}

melody_df = pd.DataFrame(melody_data)

# Chord Progression Metrics Data
chord_data = {
    "Test Set": [
        "In-Distribution", "GT", "Model", "Δ",
        "Cross-Composer", "GT", "Model", "Δ",
        "Out-of-Distribution", "GT", "Model", "Δ"
    ],
    "CHE": [
        "", 1.563, 1.732, 0.374,
        "", 1.240, 1.689, 0.702,
        "", 1.542, 1.552, 0.599
    ],
    "CC": [
        "", 9.265, 8.924, 2.847,
        "", 7.313, 8.284, 4.105,
        "", 9.352, 7.563, 4.033
    ],
    "CTD": [
        "", 0.886, 0.998, 0.278,
        "", 0.686, 0.747, 0.476,
        "", 0.846, 0.568, 0.430
    ]
}

chord_df = pd.DataFrame(chord_data)

# Plotting Melody/Harmonicity Metrics
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
tbl = table(ax, melody_df, loc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title("Melody/Harmonicity Metrics")

# Plotting Chord Progression Metrics
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
tbl = table(ax, chord_df, loc='center', colWidths=[0.2, 0.2, 0.2, 0.2])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)
plt.title("Chord Progression Metrics")

plt.show()

"""


# Data dictionary
data = {
    "Test Context": [
        "Baroque (In-Distribution)",
        "Baroque (Out-of-Distribution)",
        "Baroque (Cross-Composer)"
    ],
    "Component Accuracy (%)": ["67.68±2.28", "67.37±2.39", "66.98±2.07"],
    "Chord Change F1 (%)": ["22.65±3.03", "21.79±3.72", "33.56±3.18"],
    "Segmentation Quality (%)": ["53.89±2.07", "52.45±1.73", "58.55±1.49"],
    "Key Accuracy (%)": ["43.91±0.00", "39.14±0.00", "46.65±0.00"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
tbl = table(ax, df, loc='center', cellLoc='center', colWidths=[0.2] + [0.15]*4)

# Styling
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.5)
plt.title("Music Structure Analysis Metrics", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()


"""

data = {
    "Component": [
        "key", "degree1", "degree2", "quality", "inversion", "extra_info", "components_avg"
    ],
    "In-Distribution": [43.91, 37.89, 94.41, 53.12, 61.64, 91.36, 63.72],
    "Out-of-Distribution": [39.14, 41.10, 94.08, 54.28, 58.54, 88.87, 62.67],
    "Cross-Composer": [46.65, 45.13, 91.78, 45.42, 60.75, 91.84, 63.60],
    "Best Test Set": [
        "Cross-Composer", "Cross-Composer", "In-Distribution",
        "Out-of-Distribution", "In-Distribution", "Cross-Composer", "In-Distribution"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Melt for seaborn
df_melted = df.melt(id_vars=["Component", "Best Test Set"],
                    value_vars=["In-Distribution", "Out-of-Distribution", "Cross-Composer"],
                    var_name="Test Set", value_name="Accuracy")

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Component", y="Accuracy", hue="Test Set", palette="muted")

# Highlight best test set with an asterisk or annotation
for i, row in df.iterrows():
    best = row["Best Test Set"]
    best_val = row[best]
    plt.text(i, best_val + 1, "*", ha='center', va='bottom', fontsize=14, color='black')

plt.title("Component-wise Accuracy by Test Set")
plt.ylim(0, 105)
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.legend(title="Test Set")
plt.tight_layout()
plt.show()
"""