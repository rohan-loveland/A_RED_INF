import numpy as np

def calculate_precision_recall_all_classes(confusion_matrix):
    """
    Calculate precision and recall for each class from an nxn confusion matrix.

    Args:
        confusion_matrix: nxn numpy array where rows are true labels and columns are predicted labels

    Returns:
        tuple: (precision, recall) where each is an nx1 numpy array
    """
    # Ensure input is a numpy array
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]

    # Initialize arrays for precision and recall
    precision = np.zeros((n_classes,))
    recall = np.zeros((n_classes,))

    for i in range(n_classes):
        # True positives: diagonal element
        true_positives = cm[i, i]

        # False positives: sum of column i, excluding true positives
        false_positives = np.sum(cm[:, i]) - true_positives

        # False negatives: sum of row i, excluding true positives
        false_negatives = np.sum(cm[i, :]) - true_positives

        # Precision = TP / (TP + FP)
        precision[i] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall[i] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    return precision, recall

# Example usage
if __name__ == "__main__":
    # Sample 3x3 confusion matrix
    sample_cm = np.array([
        [10, 2, 1],
        [1, 15, 3],
        [2, 1, 12]
    ])

    precision, recall = calculate_precision_recall_all_classes(sample_cm)

    print("Precision per class:")
    print(precision)
    print("\nRecall per class:")
    print(recall)