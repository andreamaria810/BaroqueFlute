import matplotlib.pyplot as plt
import json
import csv
from postprocessing_3 import prepare_comparison_data


def report_evaluation_results(avg_metrics, all_metrics, output_file=None):
    """
    Report the evaluation results
    
    :param avg_metrics: Dictionary of average metrics
    :param all_metrics: List of dictionaries with metrics for each sequence
    :param output_file: Optional path to save results
    """

    # Analyze per-metric performance
    print("\n===== DETAILED ANALYSIS =====")
    
    # Chord progression metrics
    print("\nChord Progression Metrics:")
    print(f"CHE difference (lower is better): {avg_metrics['CHE_difference']:.4f}")
    print(f"CC difference (lower is better): {avg_metrics['CC_difference']:.4f}")
    print(f"CTD difference (lower is better): {avg_metrics['CTD_difference']:.4f}")
    
    # Melody-chord harmonicity metrics (if available)
    if 'CTnCTR_difference' in avg_metrics:
        print("\nMelody-Chord Harmonicity Metrics:")
        print(f"CTnCTR difference: {avg_metrics['CTnCTR_difference']:.4f}")
        print(f"PCS difference: {avg_metrics['PCS_difference']:.4f}")
        print(f"MCTD difference: {avg_metrics['MCTD_difference']:.4f}")
    
    # Overall accuracy
    if 'chord_accuracy' in avg_metrics:
        print(f"\nOverall Chord Accuracy: {avg_metrics['chord_accuracy']*100:.2f}%")
    
    # Find best and worst performing sequences
    accuracies = []
    for seq in all_metrics:
        if isinstance(seq, dict) and 'sequence_idx' in seq and 'chord_accuracy' in seq:
            accuracies.append((seq['sequence_idx'], seq['chord_accuracy']))
    if accuracies:
        best_seq = max(accuracies, key=lambda x: x[1])
        worst_seq = min(accuracies, key=lambda x: x[1])
        
        print(f"\nBest performing sequence: {best_seq[0]} (Accuracy: {best_seq[1]*100:.2f}%)")
        print(f"Worst performing sequence: {worst_seq[0]} (Accuracy: {worst_seq[1]*100:.2f}%)")
    

    if output_file:
        # Save detailed metrics as JSON
        try:
            with open(f"{output_file}.json", 'w') as f:
                json.dump({
                    'average_metrics': avg_metrics,
                    'sequence_metrics': [
                        {**m, 'sequence_idx': m['sequence_idx']} 
                        for m in all_metrics
                    ]
                }, f, indent=2)
        
            # Save summary as CSV for easy import into spreadsheets
            with open(f"{output_file}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for metric, value in avg_metrics.items():
                    writer.writerow([metric, value])
            
            print(f"\nResults saved to {output_file}.json and {output_file}.csv")
        except Exception as e:
            print(f"Error saving results: {e}")


def visualize_results(avg_metrics, all_metrics):
    """
    Create visualizations of the evaluation results
    """

    accuracies = []
    for seq in all_metrics:
        if isinstance(seq, dict) and 'sequence_idx' in seq and 'chord_accuracy' in seq:
            accuracies.append((seq['sequence_idx'], seq['chord_accuracy']))
            
    # Example: Plot accuracy by sequence
    if accuracies:
        # Unpack the tuples for plotting
        seq_indices = [a[0] for a in accuracies] 
        accuracy_values = [a[1] for a in accuracies]
        
        plt.figure(figsize=(12, 6))
        plt.bar(seq_indices, accuracy_values)
        plt.xlabel('Sequence Index')
        plt.ylabel('Accuracy')
        plt.title('Chord Prediction Accuracy by Sequence')

        # Calculate and show average
        avg_accuracy = sum(accuracy_values) / len(accuracy_values)
        plt.axhline(y=avg_accuracy, color='r', linestyle='--', 
                label=f'Average: {avg_accuracy:.4f}')
        plt.legend()
        plt.savefig('accuracy_by_sequence.png')
        plt.close()
    
        
        # Example: Compare metric differences
        metrics_diff = ['CHE_difference', 'CC_difference', 'CTD_difference']
        if 'CTnCTR_diff' in avg_metrics:
            metrics_diff.extend(['CTnCTR_difference', 'PCS_difference', 'MCTD_difference'])
        
        values = [avg_metrics[m] for m in metrics_diff if m in avg_metrics]  # FIXED: Check if key exists
        
        if values:  # FIXED: Only create plot if we have values
            plt.figure(figsize=(10, 6))
            plt.bar(metrics_diff[:len(values)], values)  # FIXED: Make sure labels and values match in length
            plt.xlabel('Metric')
            plt.ylabel('Difference (lower is better)')
            plt.title('Metric Differences between Predictions and Ground Truth')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('metric_differences.png')
            plt.close()
            
        print("Visualizations saved as 'accuracy_by_sequence.png' and 'metric_differences.png'")



def main():
    # Load data
    baroque_test_results = r"C:\Users\amari\OneDrive\Documents\Master's Thesis\Git\BaroqueFlute\baroque_testing\eval_results\cross_composer\baroque_model_evaluation_results.pkl"
    avg_metrics, all_metrics = prepare_comparison_data(baroque_test_results)
    
    # Report results
    report_evaluation_results(
        avg_metrics, 
        all_metrics,
        output_file='evaluation_results'
    )
    
    # Optionally, visualize specific aspects
    visualize_results(avg_metrics, all_metrics)

if __name__ == "__main__":
    main()