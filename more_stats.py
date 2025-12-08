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
    Final clean version:
    - All metrics vs. total processed points
    - Green bars = relevant points per batch (second y-axis)
    - Two plots: Precision + Recall, each with difficulty bars
    """

    # ------------------------------------------------------------------
    # Setup x-axis: points at the end of each batch
    # ------------------------------------------------------------------
    batch_num_pts = np.arange(GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS + 1, GRAPH_BATCH_SIZE)

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
    rel_per_batch = []          # This goes on the second axis

    # First batch
    cm0 = conf_matrices[0]
    q0 = num_quer[0]
    c0 = num_correct[0]

    query_precision_list.append(c0 / q0 if q0 > 0 else 0.0)
    query_rate_list.append(q0 / GRAPH_BATCH_SIZE)

    recall0, streamed0 = calculate_single_rel_recall(cm0, rel_classes, ared)
    single_rel_recall_list.append(recall0)
    rel_rate_list.append(streamed0 / GRAPH_BATCH_SIZE)
    rel_per_batch.append(streamed0)

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
        rel_per_batch.append(streamed_b)

    # ------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------
    if plot_flag:
        # --------------------- Plot 1: Query Precision ---------------------
        fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=300)

        ax1.set_xlabel("Processed Points", fontsize=10)
        ax1.set_ylabel("Query Precision & Relevant Rate", color='tab:blue', fontsize=10)
        ln1 = ax1.plot(batch_num_pts, query_precision_list, 'o-', color='tab:blue',
                       linewidth=2.2, markersize=5, label="Query Precision")
        ln2 = ax1.plot(batch_num_pts, rel_rate_list, 'd--', color='tab:orange',
                    linewidth=2, markersize=5, label="Random Precision (Rel. Rate)")
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.3)

        # Second axis: relevant points per batch
        ax2 = ax1.twinx()
        bars = ax2.bar(batch_num_pts, rel_per_batch, width=GRAPH_BATCH_SIZE*0.75,
                       alpha=0.4, color='tab:green', edgecolor='tab:green', linewidth=0.8,
                       label="Relevant Points / Batch")
        ax2.set_ylabel("Relevant Points per Batch", color='tab:green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylim(0, max(rel_per_batch)*1.4 if rel_per_batch else 10)

        # Legends
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        leg1 = ax1.legend(lns, labs, loc="upper left", fontsize=8.5, frameon=True,
                          fancybox=False, edgecolor='black')
        leg2 = ax2.legend([bars], ["Relevant Points / Batch"], loc="upper right", fontsize=8.5,
                          frameon=True, fancybox=False, edgecolor='black')
        ax1.add_artist(leg1)

        plt.title("A/RED Query Precision vs Batch Difficulty", fontsize=11, pad=12)
        plt.tight_layout()
        plt.show()

        # --------------------- Plot 2: Relevant Recall ---------------------
        fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=300)

        ax1.set_xlabel("Processed Points", fontsize=10)
        ax1.set_ylabel("Relevant Recall & Query Rate", color='tab:cyan', fontsize=10)
        ln1 = ax1.plot(batch_num_pts, single_rel_recall_list, 's-', color='tab:cyan',
                       linewidth=2.2, markersize=6, label="Relevant Recall")
        ln2 = ax1.plot(batch_num_pts, query_rate_list, '^-', color='tab:red',
                       linewidth=2, markersize=5, label="Random Recall (Query Rate)")
        ax1.set_ylim(0, 1.05)
        ax1.tick_params(axis='y', labelcolor='tab:cyan')
        ax1.grid(True, alpha=0.3)

        # Same bars
        ax2 = ax1.twinx()
        ax2.bar(batch_num_pts, rel_per_batch, width=GRAPH_BATCH_SIZE*0.75,
                alpha=0.4, color='tab:green', edgecolor='tab:green',
                label="Relevant Points / Batch")
        ax2.set_ylabel("Relevant Points per Batch", color='tab:green', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.set_ylim(0, max(rel_per_batch)*1.4 if rel_per_batch else 10)

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        leg1 = ax1.legend(lns, labs, loc="upper left", fontsize=8.5, frameon=True,
                          fancybox=False, edgecolor='black')
        leg2 = ax2.legend([bars], ["Relevant Points / Batch"], loc="upper right", fontsize=8.5,
                          frameon=True, fancybox=False, edgecolor='black')
        ax1.add_artist(leg1)

        plt.title("A/RED Relevant Recall vs Batch Difficulty", fontsize=11, pad=12)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Return values (optional, for further analysis)
    # ------------------------------------------------------------------
    return (single_rel_recall_list,
            query_precision_list,
            rel_per_batch,           # now also returning this!
            query_rate_list,
            rel_rate_list)


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