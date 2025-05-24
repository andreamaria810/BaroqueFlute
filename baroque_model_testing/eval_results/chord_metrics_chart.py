import pandas as pd

# Provided average metric dictionaries
ID_avg_metrics = {
    "predicted_CHE": 1.7321, "ground_truth_CHE": 1.5626, "CHE_difference": 0.1695,
    "predicted_CC": 8.9235, "ground_truth_CC": 9.2647, "CC_difference": -0.3412,
    "predicted_CTD": 0.9982, "ground_truth_CTD": 0.8860, "CTD_difference": 0.1122,
    "predicted_CTnCTR": 0.4462, "ground_truth_CTnCTR": 0.4273, "CTnCTR_difference": 0.0189,
    "predicted_PCS": 0.1218, "ground_truth_PCS": 0.1006, "PCS_difference": 0.0212,
    "predicted_MCTD": 3.2719, "ground_truth_MCTD": 3.3194, "MCTD_difference": -0.0475
}

OOD_avg_metrics = {
    "predicted_CHE": 1.6893, "ground_truth_CHE": 1.2401, "CHE_difference": 0.4492,
    "predicted_CC": 8.2845, "ground_truth_CC": 7.3129, "CC_difference": 0.9716,
    "predicted_CTD": 0.7468, "ground_truth_CTD": 0.6856, "CTD_difference": 0.0612,
    "predicted_CTnCTR": 0.4317, "ground_truth_CTnCTR": 0.3865, "CTnCTR_difference": 0.0452,
    "predicted_PCS": 0.1110, "ground_truth_PCS": 0.0844, "PCS_difference": 0.0266,
    "predicted_MCTD": 3.2036, "ground_truth_MCTD": 3.4437, "MCTD_difference": -0.2401
}

CC_avg_metrics = {
    "predicted_CHE": 1.6893, "ground_truth_CHE": 1.2401, "CHE_difference": 0.4492,
    "predicted_CC": 8.2845, "ground_truth_CC": 7.3129, "CC_difference": 0.9716,
    "predicted_CTD": 0.7468, "ground_truth_CTD": 0.6856, "CTD_difference": 0.0612,
    "predicted_CTnCTR": 0.4158, "ground_truth_CTnCTR": 0.4015, "CTnCTR_difference": 0.0143,
    "predicted_PCS": 0.0945, "ground_truth_PCS": 0.0758, "PCS_difference": 0.0187,
    "predicted_MCTD": 3.2710, "ground_truth_MCTD": 3.3401, "MCTD_difference": -0.0691
}

# Helper function to extract values
def extract_metrics(metrics, keys):
    return [
        [metrics[f"ground_truth_{k}"] for k in keys],
        [metrics[f"predicted_{k}"] for k in keys],
        [metrics[f"{k}_difference"] for k in keys]
    ]

# Labels and columns
row_labels = [
    ("In-Distribution", "Ground Truth"),
    ("In-Distribution", "Model Output"),
    ("In-Distribution", "Δ"),
    ("Cross-Composer", "Ground Truth"),
    ("Cross-Composer", "Model Output"),
    ("Cross-Composer", "Δ"),
    ("Out-of-Distribution", "Ground Truth"),
    ("Out-of-Distribution", "Model Output"),
    ("Out-of-Distribution", "Δ"),
]

# Order of metrics
melody_cols = ["CTnCTR", "PCS", "MCTD"]
chord_cols = ["CHE", "CC", "CTD"]

# Build data for melody/harmonicity
melody_data = extract_metrics(ID_avg_metrics, melody_cols)
melody_data += extract_metrics(CC_avg_metrics, melody_cols)
melody_data += extract_metrics(OOD_avg_metrics, melody_cols)
melody_df = pd.DataFrame(melody_data, index=row_labels, columns=melody_cols)

# Build data for chord progression
chord_data = extract_metrics(ID_avg_metrics, chord_cols)
chord_data += extract_metrics(CC_avg_metrics, chord_cols)
chord_data += extract_metrics(OOD_avg_metrics, chord_cols)
chord_df = pd.DataFrame(chord_data, index=row_labels, columns=chord_cols)

# Export to LaTeX
melody_latex = melody_df.to_latex(float_format="%.4f", multirow=True, caption="Melody/Chord Harmonicity Metrics", label="tab:melody_metrics")
chord_latex = chord_df.to_latex(float_format="%.4f", multirow=True, caption="Chord Progression Metrics", label="tab:chord_metrics")

# Save LaTeX to files
with open("melody_metrics.tex", "w", encoding="utf-8") as f:
    f.write(melody_latex)

with open("chord_metrics.tex", "w", encoding="utf-8") as f:
    f.write(chord_latex)

print("✅ LaTeX tables saved: 'melody_metrics.tex' and 'chord_metrics.tex'")

# Show output
print("\nMelody/Chord Harmonicity Metrics")
print(melody_df.to_string(float_format="%.4f"))

print("\nChord Progression Metrics")
print(chord_df.to_string(float_format="%.4f"))

# Optional: export to CSV or LaTeX
# melody_df.to_csv("melody_metrics.csv")
# chord_df.to_csv("chord_metrics.csv")
