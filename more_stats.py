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
    Final fixed version:
    - bar_width defined in the right place
    - x-axis from actual batches
    - Legend below plot
    - No clipping
    """

    # Convert to numpy for easy diff
    conf_array = np.array(conf_matrices)
    num_correct = np.array(num_correct_queries)
    num_quer = np.array(num_queries)

    # ------------------------------------------------------------------
    # Per-batch metrics
    # ------------------------------------------------------------------
    query_precision_list = []
    query_rate_list = []
    rel_rate_list = []
    single_rel_recall_list = []
    precision_ratio_list = []
    recall_ratio_list = []

    # First batch
    cm0 = conf_matrices[0]
    q0 = num_quer[0]
    c0 = num_correct[0]

    query_precision_list.append(c0 / q0 if q0 > 0 else 0.0)
    query_rate_list.append(q0 / GRAPH_BATCH_SIZE)

    recall0, streamed0 = calculate_single_rel_recall(cm0, rel_classes, ared)
    single_rel_recall_list.append(recall0)
    rel_rate_list.append(streamed0 / GRAPH_BATCH_SIZE)

    precision_ratio_list.append(query_precision_list[0] / rel_rate_list[0] if rel_rate_list[0] > 0 else 0.0)
    recall_ratio_list.append(recall0 / query_rate_list[0] if query_rate_list[0] > 0 else 0.0)

    # Subsequent batches
    for b in range(1, len(conf_matrices)):
        batch_cm = conf_array[b] - conf_array[b-1]

        queries_b = num_quer[b] - num_quer[b-1]
        correct_b = num_correct[b] - num_correct[b-1]

        query_precision_list.append(correct_b / queries_b if queries_b > 0 else 0.0)
        query_rate_list.append(queries_b / GRAPH_BATCH_SIZE)

        recall_b, streamed_b = calculate_single_rel_recall(batch_cm, rel_classes, ared)
        single_rel_recall_list.append(recall_b)
        rel_rate_list.append(streamed_b / GRAPH_BATCH_SIZE)

        precision_ratio_list.append(query_precision_list[-1] / rel_rate_list[-1] if rel_rate_list[-1] > 0 else 0.0)
        recall_ratio_list.append(recall_b / query_rate_list[-1] if query_rate_list[-1] > 0 else 0.0)

    # ------------------------------------------------------------------
    # Fix x-axis and list lengths
    # ------------------------------------------------------------------
    num_batches = len(conf_matrices)
    batch_num_pts = np.arange(GRAPH_BATCH_SIZE, GRAPH_BATCH_SIZE * num_batches + 1, GRAPH_BATCH_SIZE)

    query_precision_list = query_precision_list[:num_batches]
    query_rate_list = query_rate_list[:num_batches]
    rel_rate_list = rel_rate_list[:num_batches]
    single_rel_recall_list = single_rel_recall_list[:num_batches]
    precision_ratio_list = precision_ratio_list[:num_batches]
    recall_ratio_list = recall_ratio_list[:num_batches]

    # ------------------------------------------------------------------
    # Define bar_width here (available for plotting)
    # ------------------------------------------------------------------
    bar_width = GRAPH_BATCH_SIZE * 0.7

    # --------------------- Plot 1: Query Precision + Ratio ---------------------
    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=300)  # Taller figure for safety

    ax1.set_xlabel("Processed Points", fontsize=12)
    ax1.set_ylabel("Precision", color='black', fontsize=12)

    ax1.plot(batch_num_pts, query_precision_list, 'o-', color='tab:blue',
             linewidth=2.5, markersize=6, label="Query Precision")
    ax1.plot(batch_num_pts, rel_rate_list, 'd--', color='tab:orange',
             linewidth=2.2, markersize=6, label="Random Precision (Rel. Rate)")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    ax2.bar(batch_num_pts, precision_ratio_list, width=bar_width,
            color='tab:purple', alpha=0.55, edgecolor='tab:purple', linewidth=1.2,
            label="Precision Ratio (Query / Random)")
    ax2.set_ylabel("Precision Ratio", color='tab:purple', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    max_ratio = max(precision_ratio_list) if precision_ratio_list else 1
    ax2.set_ylim(0, max_ratio * 1.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', bbox_to_anchor=(0.5, 0.05),  # Tighter position
               ncol=3, fontsize=11, frameon=True, fancybox=False, edgecolor='black')

    plt.title("A/RED Query Precision & Improvement Ratio", fontsize=13, pad=20)

    # Adjusted margins for compact layout
    plt.subplots_adjust(left=0.11, right=0.85, top=0.94, bottom=0.18)

    plt.savefig('ared_query_precision_ratio.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    # --------------------- Plot 2: Relevant Recall + Ratio ---------------------
    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=300)  # Taller

    ax1.set_xlabel("Processed Points", fontsize=12)
    ax1.set_ylabel("Recall", color='black', fontsize=12)

    ax1.plot(batch_num_pts, single_rel_recall_list, 's-', color='tab:cyan',
             linewidth=2.5, markersize=7, label="Relevant Recall")
    ax1.plot(batch_num_pts, query_rate_list, '^-', color='tab:red',
             linewidth=2.2, markersize=6, label="Random Recall (Query Rate)")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = ax1.twinx()
    ax2.bar(batch_num_pts, recall_ratio_list, width=bar_width,
            color='tab:green', alpha=0.55, edgecolor='tab:green', linewidth=1.2,
            label="Recall Ratio (Recall / Query Rate)")
    ax2.set_ylabel("Recall Ratio", color='tab:green', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='tab:green')
    max_ratio = max(recall_ratio_list) if recall_ratio_list else 1
    ax2.set_ylim(0, max_ratio * 1.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2,
               loc='lower center', bbox_to_anchor=(0.5, 0.05),
               ncol=3, fontsize=11, frameon=True, fancybox=False, edgecolor='black')

    plt.title("A/RED Relevant Recall & Improvement Ratio", fontsize=13, pad=20)

    # Adjusted margins for compact layout
    plt.subplots_adjust(left=0.11, right=0.85, top=0.94, bottom=0.18)

    plt.savefig('ared_relevant_recall_ratio.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.show()
    # ------------------------------------------------------------------
    # Return values
    # ------------------------------------------------------------------
    return (single_rel_recall_list,
            query_precision_list,
            query_rate_list,
            rel_rate_list,
            precision_ratio_list,
            recall_ratio_list)
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