import numpy as np
import matplotlib.pyplot as plt

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

def calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared, num_correct_queries, \
                                     num_queries, plot_flag, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS):
    rel_recall_ave_list = []
    query_precision_list = []
    rel_class_info = [None]*len(rel_classes)
    rel_individual_recalls = [None]*len(rel_classes)

    sparsity_labels = [l for l, _ in sparsity_levels]
    sparsity_numbers = [n for _, n in sparsity_levels]
    precision, recall = calculate_precision_recall_all_classes(conf_matrices[0])
    rel_recall_ave = 0
    for i,c in enumerate(rel_classes):
        n = ared.oracle.int_str_label_bidict[c]
        rel_class_info[i] = [(c, n, sparsity_numbers[n],)]
        rel_individual_recalls[i]=[recall[n]]
        rel_recall_ave += recall[n]
    rel_recall_ave /= len(rel_classes)
    rel_recall_ave_list.append(rel_recall_ave)
    query_precision_list.append(num_correct_queries[0] / num_queries[0])

    conf_array = np.array(conf_matrices)  # final matrix

    for b in range(1, len(conf_matrices)):
        this_batch_conf_matrix = conf_array[b] - conf_array[b - 1]
        precision, recall = calculate_precision_recall_all_classes(this_batch_conf_matrix)
        rel_recall_ave = 0
        for i,c in enumerate(rel_classes):
            n = ared.oracle.int_str_label_bidict[c]
            rel_individual_recalls[i].append(recall[n])
            rel_recall_ave += recall[n]
        rel_recall_ave /= len(rel_classes)
        rel_recall_ave_list.append(rel_recall_ave)
        query_precision_list.append(num_correct_queries[b] / num_queries[b])

    if plot_flag:
        # would be nice at some point to show individual recalls, with line widths indicating sparsity level
        batch_num_pts = list(range(GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS + 1, GRAPH_BATCH_SIZE))
        plt.figure(figsize=(10, 5))
        plt.plot(batch_num_pts, rel_recall_ave_list)
        # for n in range(len(rel_individual_recalls)):
        #     plt.plot(batch_num_pts, rel_individual_recalls[n])
        plt.plot(batch_num_pts, query_precision_list)
        plt.grid()
        plt.legend(("average_relevant_recall","relevant_recall_0","relevant_recall_1", \
                    "relevant_recall_2","relevant_recall_3", "query_precision"))

    return rel_recall_ave_list, query_precision_list, rel_individual_recalls




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