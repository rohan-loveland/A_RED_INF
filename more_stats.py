import numpy as np
import matplotlib.pyplot as plt

def calculate_single_rel_recall(confusion_matrix,rel_classes,ared):
    """
    NOTE: this is tricky because we the confusion matrix doesn't record which points were queried
    However, we know the off-diagonal elements were not queried, so recall still = TP / (TP + FN)
    Args:
        confusion_matrix: nxn numpy array where rows are true labels and columns are predicted labels
    Returns:
        scalar: relevant recall
    """
    # Ensure input is a numpy array
    cm = np.array(confusion_matrix)
    n_classes = cm.shape[0]

    num_rel_pts_queried = 0.0
    num_rel_pts_streamed = 0.0

    for k,c in enumerate(rel_classes):
        i = ared.oracle.int_str_label_bidict[c]

        # True positives: diagonal element
        num_rel_pts_queried += cm[i, i]
        num_rel_pts_streamed += np.sum(cm[i, :])

    if num_rel_pts_queried == 0:
        rel_recall = 0.0
    else:
        rel_recall = num_rel_pts_queried/num_rel_pts_streamed

    return rel_recall, num_rel_pts_streamed

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


def calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared,
                                    num_correct_queries, num_queries,
                                    plot_flag, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS):
    """
    Per-batch A/RED metrics on a SINGLE 0–1 y-axis.
    - Query Rate = queries / GRAPH_BATCH_SIZE
    - Relevant Rate = relevant points / GRAPH_BATCH_SIZE
    - All values in [0, 1]
    - Compact, publication-ready
    """

    # ------------------------------------------------------------------
    # Initialize lists
    # ------------------------------------------------------------------
    single_rel_recall_list = []
    query_precision_list = []
    query_rate_list = []        # now in [0,1]
    rel_rate_list = []          # now in [0,1]
    rel_individual_recalls = [[] for _ in rel_classes]

    batch_num_pts = list(range(GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS + 1, GRAPH_BATCH_SIZE))
    conf_array = np.array(conf_matrices)

    # ------------------------------------------------------------------
    # First batch
    # ------------------------------------------------------------------
    first_cm = conf_matrices[0]
    first_correct = num_correct_queries[0]
    first_queries = num_queries[0]

    query_precision_list.append(first_correct / first_queries if first_queries > 0 else 0.0)
    query_rate_list.append(first_queries / GRAPH_BATCH_SIZE)

    rel_recall, rel_streamed = calculate_single_rel_recall(first_cm, rel_classes, ared)
    single_rel_recall_list.append(rel_recall)
    rel_rate_list.append(rel_streamed / GRAPH_BATCH_SIZE)

    _, recall_per_class = calculate_precision_recall_all_classes(first_cm)
    for i, c in enumerate(rel_classes):
        class_idx = ared.oracle.int_str_label_bidict[c]
        rel_individual_recalls[i].append(recall_per_class[class_idx])

    # ------------------------------------------------------------------
    # Subsequent batches
    # ------------------------------------------------------------------
    for b in range(1, len(conf_matrices)):
        batch_cm = conf_array[b] - conf_array[b - 1]

        # Query precision
        correct_in_batch = num_correct_queries[b] - num_correct_queries[b - 1]
        queries_in_batch = num_queries[b] - num_queries[b - 1]
        precision_batch = correct_in_batch / queries_in_batch if queries_in_batch > 0 else 0.0
        query_precision_list.append(precision_batch)

        # Query rate (normalized)
        query_rate_list.append(queries_in_batch / GRAPH_BATCH_SIZE)

        # Relevant recall & rate
        rel_recall, rel_streamed = calculate_single_rel_recall(batch_cm, rel_classes, ared)
        single_rel_recall_list.append(rel_recall)
        rel_rate_list.append(rel_streamed / GRAPH_BATCH_SIZE)

        # Individual recalls
        _, recall_per_class = calculate_precision_recall_all_classes(batch_cm)
        for i, c in enumerate(rel_classes):
            class_idx = ared.oracle.int_str_label_bidict[c]
            rel_individual_recalls[i].append(recall_per_class[class_idx])

    # ------------------------------------------------------------------
    # SINGLE PLOT: All metrics on 0–1 y-axis
    # ------------------------------------------------------------------
    if plot_flag and len(batch_num_pts) == len(single_rel_recall_list):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)

        ax.set_xlabel("Processed Points", fontsize=8)
        ax.set_ylabel("Value (0–1)", fontsize=8)
        ax.set_ylim(0, 1.05)

        # Plot all four lines
        ax.plot(batch_num_pts, query_precision_list,
                'o-', color='tab:blue', linewidth=1.5, markersize=5, label="Query Precision")
        ax.plot(batch_num_pts, single_rel_recall_list,
                's-', color='tab:cyan', linewidth=1.5, markersize=5, label="Relevant Recall")
        ax.plot(batch_num_pts, query_rate_list,
                '^-', color='tab:red', linewidth=1.5, markersize=5, label="Query Rate")
        ax.plot(batch_num_pts, rel_rate_list,
                'd--', color='tab:orange', linewidth=1.5, markersize=5, label="Relevant Rate")

        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.6)

        # Legend: middle-right
        ax.legend(loc="center right", fontsize=6.5, frameon=True,
                  fancybox=False, edgecolor='black', handlelength=1.2)

        plt.title("A/RED Performance per Batch", fontsize=9, pad=8)
        plt.tight_layout()
        plt.subplots_adjust(right=0.78, top=0.90)
        plt.show()

    # ------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------
    return (single_rel_recall_list,
            query_precision_list,
            rel_individual_recalls,
            query_rate_list)
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