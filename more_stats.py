import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

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
                                    plot_flag, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS,
                                    anom_only_queries, rel_only_queries, both_a_and_r_queries, cumulative_relevants):
    """
    Final version: Combined multi-panel figure (a,b,c) with Query Breakdown integrated
    All plots in one PDF, vector graphics, paper-ready
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    # Convert to numpy for easy diff
    conf_array = np.array(conf_matrices)
    num_correct = np.array(num_correct_queries)
    num_quer = np.array(num_queries)

    # ------------------------------------------------------------------
    # Per-batch metrics for panels (a) and (b)
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

    bar_width = GRAPH_BATCH_SIZE * 0.7

    if plot_flag:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # --------------------------------------------------------------
        # FIGURE 1: Panels (a) and (b) side-by-side
        # --------------------------------------------------------------
        fig1, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.50, 4.50))

        # Panel (a): Query Precision + Ratio
        ax_a.plot(batch_num_pts, query_precision_list, 'o-', color='tab:blue',
                  linewidth=3, markersize=8, label="Query Precision")
        ax_a.plot(batch_num_pts, rel_rate_list, 'd--', color='tab:orange',
                  linewidth=2.5, markersize=7, label="Random Precision (Rel. Rate)")
        ax_a.set_xlabel("Processed Points", fontsize=14)
        ax_a.set_ylabel("Precision", fontsize=14)
        ax_a.set_ylim(0, 1.05)
        ax_a.grid(True, alpha=0.3, axis='y')
        ax_a.tick_params(axis='both', labelsize=12)

        ax_a2 = ax_a.twinx()
        ax_a2.bar(batch_num_pts, precision_ratio_list, width=bar_width,
                  color='tab:purple', alpha=0.6, edgecolor='tab:purple', linewidth=1.5,
                  label="Precision Ratio")
        ax_a2.set_ylabel("Precision Ratio", color='tab:purple', fontsize=12)
        ax_a2.tick_params(axis='y', labelcolor='tab:purple', labelsize=12)
        max_ratio_a = max(precision_ratio_list) if precision_ratio_list else 1
        ax_a2.set_ylim(0, max_ratio_a * 1.25)

        ax_a.set_title("(a) A/RED Query Precision & Improvement Ratio", fontsize=14, pad=15)

        # Panel (b): Relevant Recall + Ratio
        ax_b.plot(batch_num_pts, single_rel_recall_list, 's-', color='tab:cyan',
                  linewidth=3, markersize=9, label="Relevant Recall")
        ax_b.plot(batch_num_pts, query_rate_list, '^-', color='tab:red',
                  linewidth=2.5, markersize=7, label="Random Recall (Query Rate)")
        ax_b.set_xlabel("Processed Points", fontsize=14)
        ax_b.set_ylabel("Recall", fontsize=14)
        ax_b.set_ylim(0, 1.05)
        ax_b.grid(True, alpha=0.3, axis='y')
        ax_b.tick_params(axis='both', labelsize=12)

        ax_b2 = ax_b.twinx()
        ax_b2.bar(batch_num_pts, recall_ratio_list, width=bar_width,
                  color='tab:green', alpha=0.6, edgecolor='tab:green', linewidth=1.5,
                  label="Recall Ratio")
        ax_b2.set_ylabel("Recall Ratio", color='tab:green', fontsize=12)
        ax_b2.tick_params(axis='y', labelcolor='tab:green', labelsize=12)
        max_ratio_b = max(recall_ratio_list) if recall_ratio_list else 1
        ax_b2.set_ylim(0, max_ratio_b * 1.25)

        ax_b.set_title("(b) A/RED Relevant Recall & Improvement Ratio", fontsize=14, pad=15)

        # Shared legend below Figure 1
        handles1, labels1 = [], []
        for ax in [ax_a, ax_a2, ax_b, ax_b2]:
            h, l = ax.get_legend_handles_labels()
            handles1.extend(h)
            labels1.extend(l)
        fig1.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, -0.08),
                    ncol=3, fontsize=12, frameon=True, edgecolor='black')

        fig1.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.22, wspace=0.4)
        fig1.savefig('./Figures/ared_precision_recall_figures.pdf', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.show()

        # --------------------------------------------------------------
        # FIGURE 2: Panel (c) — Query Breakdown & Relevant Points
        # --------------------------------------------------------------
        fig2 = plt.figure(figsize=(9.00, 6.00))

        ax_c = fig2.add_subplot(111)

        num_batches_c = len(num_queries)
        batch_points_c = np.arange(GRAPH_BATCH_SIZE, GRAPH_BATCH_SIZE * num_batches_c + 1, GRAPH_BATCH_SIZE)

        anom_only_c = np.diff(anom_only_queries, prepend=0)[:num_batches_c]
        rel_only_c = np.diff(rel_only_queries, prepend=0)[:num_batches_c]
        both_c = np.diff(both_a_and_r_queries, prepend=0)[:num_batches_c]
        total_c = np.diff(num_queries, prepend=0)[:num_batches_c]
        relevant_per_batch = np.diff(cumulative_relevants, prepend=0)[:num_batches_c]

        query_bar_width = GRAPH_BATCH_SIZE * 0.8
        rel_bar_width = GRAPH_BATCH_SIZE * 0.3
        offset = GRAPH_BATCH_SIZE * 0.15

        # Stacked query bars
        ax_c.bar(batch_points_c, anom_only_c, width=query_bar_width,
                 label='Anomalous Only Queries', color='#E74C3C', edgecolor='black', linewidth=1.2, alpha=0.95)
        ax_c.bar(batch_points_c, rel_only_c, bottom=anom_only_c,
                 width=query_bar_width, label='Relevant Only Queries', color='#3498DB', edgecolor='black',
                 linewidth=1.2, alpha=0.95)
        ax_c.bar(batch_points_c, both_c, bottom=anom_only_c + rel_only_c,
                 width=query_bar_width, label='Both Triggers Queries', color='#9B59B6', edgecolor='black',
                 linewidth=1.2, alpha=0.95)

        # Interleaved relevant points bars
        ax_c.bar(batch_points_c + offset, relevant_per_batch, width=rel_bar_width,
                 label='Relevant Points / Batch', color='tab:green', edgecolor='black', linewidth=1.5, alpha=0.85)

        # Total queries line
        ax_c.plot(batch_points_c, total_c, 'k-o', markersize=9, linewidth=3.5,
                  label='Total Queries', markerfacecolor='white', markeredgewidth=2.2)

        ax_c.set_xlabel('Processed Points', fontsize=16)
        ax_c.set_ylabel('Count per Batch', fontsize=16)
        ax_c.tick_params(axis='both', labelsize=14)
        ax_c.grid(True, axis='y', alpha=0.4, linestyle='--', linewidth=1.2)

        max_val = max(np.maximum(total_c, relevant_per_batch)) if len(total_c) > 0 else 1
        for x, total in zip(batch_points_c, total_c):
            if total > 0:
                ax_c.text(x, total + max_val * 0.03, str(int(total)),
                          ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')

        ax_c.set_title("A/RED Query Breakdown & Relevant Points Over Time", fontsize=18, pad=25)

        # Legend below Figure 2
        ax_c.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                    ncol=3, fontsize=14, frameon=True, edgecolor='black')

        fig2.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.25)
        fig2.savefig('./Figures/ared_query_breakdown.pdf', dpi=300, bbox_inches='tight', pad_inches=0.8)
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