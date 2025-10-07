import time
from MNIST_Data_Processing import *
from EMNIST_Data_Processing import *
from NICE_Data_Processing import *
from Parking_Lot_Data_Processing import *

def get_data(data_source,N_REL_CLASSES, VERBOSE_FLAGS, seed):
    if data_source == "MNIST":
        X_skewed, y_w_rel, sparsity_levels, rel_classes = MNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)
    elif data_source == "EMNIST":
        X_skewed, y_w_rel, sparsity_levels, rel_classes = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)
    elif data_source == "NICE":
        X_skewed, y_w_rel, sparsity_levels, rel_classes = generate_synthetic_dataset_with_relevance(N_REL_CLASSES, seed)
        X_skewed, y_w_rel, sparsity_levels = generate_synthetic_dataset_with_relevance(N_REL_CLASSES, seed)
    elif data_source == "PARKING_LOT":
        X_skewed, y_w_rel, sparsity_levels = parking_lot_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed)

    return X_skewed, y_w_rel, sparsity_levels, rel_classes

def set_up_stats(ared):
    start_time = time.time()

    num_queries_last_batch = 0

    times = [start_time]
    num_queries = [ared.num_queries]
    num_clusters = [len(ared.subspace_partition.cluster_dict)]
    num_labels = [len(ared.subspace_partition.set_of_known_labels)]
    conf_matrices = []


    # DEBUG ONLY
    pt_dists = []
    num_pts_searched_list = []

    return start_time, times, num_queries, num_clusters, num_labels, conf_matrices, pt_dists, num_pts_searched_list, num_queries_last_batch