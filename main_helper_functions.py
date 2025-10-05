import time
from MNIST_Data_Processing import *
from EMNIST_Data_Processing import *
from NICE_Data_Processing import *

def get_data(data_source,N_REL_CLASSES, VERBOSE_FLAGS, seed):
    if data_source == "MNIST":
        X_skewed, y_w_rel, sparsity_levels = MNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "EMNIST":
        X_skewed, y_w_rel, sparsity_levels = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)
    elif data_source == "NICE":
        X_skewed, y_w_rel, sparsity_levels = generate_synthetic_dataset_with_relevance(N_REL_CLASSES, seed)

    return X_skewed, y_w_rel, sparsity_levels

def set_up_stats(ared,y_w_rel):
    start_time = time.time()
    # set up confusion matrix
    # NOTE: the confusion matrix is organized starting at integer label 0 - NOT (necessarily) in sparsity order

    num_queries_last_batch = 0

    times = [start_time]
    num_queries = [ared.num_queries]
    num_clusters = [len(ared.subspace_partition.cluster_dict)]
    num_labels = [len(ared.subspace_partition.set_of_known_labels)]
    recall = []
    precision = []

    # DEBUG ONLY
    pt_dists = []
    num_pts_searched_list = []

    return start_time, times, num_queries, num_clusters, num_labels, recall, precision, \
            pt_dists, num_pts_searched_list, num_queries_last_batch