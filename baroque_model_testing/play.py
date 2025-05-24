butt    =     [{'predicted_CHE': 1.9026797202032986, 
                'ground_truth_CHE': 2.0654840162640666, 
                'CHE_difference': 0.16280429606076807, 
                'predicted_CC': 11, 'ground_truth_CC': 11, 
                'CC_difference': 0, 'predicted_CTD': 1.403847316356054, 
                'ground_truth_CTD': 1.3581876147468508, 
                'CTD_difference': 0.045659701609203296, 
                'predicted_CTnCTR': 0.5, 
                'ground_truth_CTnCTR': 0.35714285714285715, 
                'CTnCTR_difference': 0.14285714285714285, 
                'predicted_PCS': 0.023809523809523832, 
                'ground_truth_PCS': -0.047619047619047596, 
                'PCS_difference': 0.07142857142857142, 
                'predicted_MCTD': 3.391095826629819, 
                'ground_truth_MCTD': 3.560981539152277, 
                'MCTD_difference': 0.1698857125224582, 
                'chord_accuracy': 0.1953125, 
                'sequence_idx': 0}, {'predicted_CHE': 1.9026797202032986, 
                'ground_truth_CHE': 2.0654840162640666, 
                'CHE_difference': 0.16280429606076807, 
                'predicted_CC': 11, 'ground_truth_CC': 11, 
                'CC_difference': 0, 'predicted_CTD': 1.403847316356054, 
                'ground_truth_CTD': 1.3581876147468508, 
                'CTD_difference': 0.045659701609203296, 
                'predicted_CTnCTR': 0.5, 
                'ground_truth_CTnCTR': 0.35714285714285715, 
                'CTnCTR_difference': 0.14285714285714285, 
                'predicted_PCS': 0.023809523809523832, 
                'ground_truth_PCS': -0.047619047619047596, 
                'PCS_difference': 0.07142857142857142, 
                'predicted_MCTD': 3.391095826629819, 
                'ground_truth_MCTD': 3.560981539152277, 
                'MCTD_difference': 0.1698857125224582, 
                'chord_accuracy': 0.1953125, 
                'sequence_idx': 0}, {'predicted_CHE': 1.9026797202032986, 
                'ground_truth_CHE': 2.0654840162640666, 
                'CHE_difference': 0.16280429606076807, 
                'predicted_CC': 11, 'ground_truth_CC': 11, 
                'CC_difference': 0, 'predicted_CTD': 1.403847316356054, 
                'ground_truth_CTD': 1.3581876147468508, 
                'CTD_difference': 0.045659701609203296, 
                'predicted_CTnCTR': 0.5, 
                'ground_truth_CTnCTR': 0.35714285714285715, 
                'CTnCTR_difference': 0.14285714285714285, 
                'predicted_PCS': 0.023809523809523832, 
                'ground_truth_PCS': -0.047619047619047596, 
                'PCS_difference': 0.07142857142857142, 
                'predicted_MCTD': 3.391095826629819, 
                'ground_truth_MCTD': 3.560981539152277, 
                'MCTD_difference': 0.1698857125224582, 
                'chord_accuracy': 0.1953125, 
                'sequence_idx': 0}
            ]

for i in butt:
    print(i['predicted_CHE'])



#for i, (key, value) in enumerate(butt[0].items()):
#    print(f"Key: {key}, Index: {i}")

accuracies = [(m['sequence_idx'], m['chord_accuracy']) for m in butt]
print(accuracies)
"""
Key: predicted_CHE, Index: 0
Key: ground_truth_CHE, Index: 1
Key: CHE_difference, Index: 2
Key: predicted_CC, Index: 3
Key: ground_truth_CC, Index: 4
Key: CC_difference, Index: 5
Key: predicted_CTD, Index: 6
Key: ground_truth_CTD, Index: 7
Key: CTD_difference, Index: 8
Key: predicted_CTnCTR, Index: 9
Key: ground_truth_CTnCTR, Index: 10
Key: CTnCTR_difference, Index: 11
Key: predicted_PCS, Index: 12
Key: ground_truth_PCS, Index: 13
Key: PCS_difference, Index: 14
Key: predicted_MCTD, Index: 15
Key: ground_truth_MCTD, Index: 16
Key: MCTD_difference, Index: 17
Key: chord_accuracy, Index: 18
Key: sequence_idx, Index: 19
"""