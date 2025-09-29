'''
DATA_SOURCE: Dataset to run ARED on
|- MNIST: MNIST dataset
|- EMNIST: EMNIST dataset
|- P_LOT
'''

# goes with...

'''
N_REL_CLASSES: Specified number of relevant classes
|- This is the number of classes (starting from sparsest) that are to be considered relevant
|- MNIST settings:
|=== High relevance: 8 relevant classes ~25% of data as relevant
|=== Low relevance: 4 relevant classes ~1.4% of data as relevant`
|- EMNIST settings:
|=== High relevance: 40 relevant classes ~25% of data as relevant
|=== Low relevance: 10 relevant classes ~1.4% of data as relevant`
|- EASY_MODE settings:
|=== Low relevance: 4 relevant classes ~1.4% of data as relevant`
'''
DATA_SOURCE = "MNIST" # NOTE: currently multiplied by 10x to get ~130,000 samples
N_REL_CLASSES = 4

# DATA_SOURCE = "EMNIST"
# N_REL_CLASSES = 10

# DATA_SOURCE = "NICE"
# N_REL_CLASSES = 4

'''
KAPPAS: Paranoia Parameter
|- Array of Kappas to run ARED on
|- Run more than one for graphing purposes
'''
KAPPAS = [0.5] #0.5, , 1.4, 10

'''
K_COMP_PTS: Number of points to compare to when looking for relevance
|- 1: Standard ARED
|- 2 or more: k ARED
@WARNING: must be 1 or greater
'''
K_COMP_PTS = 2 # use 2 for baseline

'''
QS_VAR: Query Strategy Variants
|- 0: Diameter check
|- 1: Approx. Ave Single Linkage Average 
'''
QS_VAR = 1

'''
SM_VAR: Split Method Var 
|- 0: Voronoi Split
|- 1: eventually 2 component GMM...
'''
SM_VAR = 0

'''
REL_PROC_VAR: Relevance Processing Variants
|- 0: No relevance processing
|- 1: Single
'''
REL_PROC_VAR = 0

'''
window_size: size of the data_window window saved by ARED
|- int: larger window size means it remembers more data
|- WARNING: value must be larger than 0
'''
DATA_WINDOW_SIZE = 1000 # ultimately needs to be driven by anomaly ratio

'''
NUM_POINTS_TO_PROCESS: Number of points in dataset to process
|- -1: process all the data
|-  0 to inf: process up to that number if data is available
'''
NUM_POINTS_TO_PROCESS = -1

'''
NUM_RUN_TO_AVE: number of runs to average.
|- INT, higher numbers means more runs to average for graphs. 
|- @WARNING MUST BE GREATER THAN 1.
'''
NUM_RUNS_TO_AVE = 1

'''
GRAPH_BATCH_SIZE: number of points in batch for stats purposes.
'''
GRAPH_BATCH_SIZE = 1000

'''
VERBOSE_FLAGS: Array of control flags to make ARED loud or quite
|- Array, containing verbose flags for different types of messages
|- Example: VERBOSE_FLAGS = [0, 1, 2]
|- Put these numbers in the array to change which parts of ARED are very loud
|- 0: Prints dataset info and every 1000th loop of data processing
|- 1: Prints when new clusters are created and where clusters are inserted
|- 2: Prints split information, which cluster id and o_pt movements 
|- 3: Prints the l_buf cluster_id_array and abs_index_array and data windows assigned_cluster_id buffer
|- 4: Prints the forgotten_abs_index and forgotten_point_cluster_id during subspace_partition_maintenance 
|- 5: Prints information about cluster merging 
'''
VERBOSE_FLAGS = [0] #example setting [1, 2] for two verbose level control flags

'''
MAKE_GRAPHS
|- True: make graphs
|- False: do not make graphs
'''
MAKE_GRAPHS = True

'''
RANDOM_SEED_OFFSET
|- any integer
|- a way to vary seed sequence
'''
RANDOM_SEED_OFFSET = 25

# Imports ===================================
from MNIST_Data_Processing import *
from EMNIST_Data_Processing import *
from NICE_Data_Processing import *
from Data_Stream import *
from Oracle import *
from A_REDIN import *
from Stats import *

import time

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'wxAgg' depending on your system
from data_visualization import *

from cluster_visualization import plot_clusters_colored_by_label

if __name__ == '__main__':
    stats = Stats()

    for kappa in KAPPAS:

        stats.init_for_kappa_loop(kappa)

        for seed in range(NUM_RUNS_TO_AVE):
            seed = seed + RANDOM_SEED_OFFSET

            # Get data and skew and add relevance
            if DATA_SOURCE == "MNIST":
                X_skewed, y_w_rel = MNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS,seed)
            elif DATA_SOURCE == "EMNIST":
                X_skewed, y_w_rel = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)
            elif DATA_SOURCE == "NICE":
                X_skewed, y_w_rel = generate_synthetic_dataset_with_relevance(N_REL_CLASSES,seed)


            # set up confusion matrix
            num_all_classes = len(np.unique(np.array(y_w_rel)[:, 0]))
            conf_matrix = np.zeros((num_all_classes, num_all_classes),dtype=int)

            # Initialize Oracle and ARED ===================================
            data_stream = Data_Stream(X_skewed, y_w_rel)
            oracle = Oracle(X_skewed, y_w_rel)

            ared = ARED(oracle, conf_matrix, kappa, DATA_WINDOW_SIZE, K_COMP_PTS, QS_VAR, REL_PROC_VAR, SM_VAR, VERBOSE_FLAGS)

            if NUM_POINTS_TO_PROCESS == -1:
                num_points_to_process = data_stream.get_remaining_num_points()
            else:
                num_points_to_process = NUM_POINTS_TO_PROCESS - 1

            start_time = time.time()
            # Stream and Process data =========================================
            ared.process_first_point(data_stream.stream_new_data_point())
            ared.num_queries = 1 # already processed first point

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

            for i in range(1, num_points_to_process):

                # save and print per batch ---------------------------------------------------------------------
                if i % GRAPH_BATCH_SIZE == 0:
                    j = i//GRAPH_BATCH_SIZE # count of number of batches
                    times.append(time.time())
                    time_elapsed =  times[j]- times[j-1]
                    num_queries.append(ared.num_queries)
                    num_queries_this_batch = num_queries[j] - num_queries[j-1]
                    num_clusters.append(len(ared.subspace_partition.cluster_dict))
                    num_labels.append(len(ared.subspace_partition.set_of_known_labels))
                    # precision.append(precision_this_batch)

                    # num_rel_this_batch_TP = ?????
                    # num_rel_this_batch_P = ????
                    # precision_this_batch = num_rel_this_batch_TP/num_rel_this_batch_P
                    # num_rel_this_batch_streamed = ????????????????
                    if 0 in VERBOSE_FLAGS:
                        print(f"Processing point {i}... (last {GRAPH_BATCH_SIZE} points took {time_elapsed:.2f} seconds)")
                        print(f"Points queried in this batch: {num_queries_this_batch}, Query Rate: {num_queries_this_batch / GRAPH_BATCH_SIZE * 100}%")
                        print(f"Number of clusters: {num_clusters[j-1]}")  # Add cluster count
                        # print(f"Precision: {precision_this_batch}")  # Add cluster count

                    # plot_clusters_colored_by_label(ared, X_skewed, y_w_rel, title="Cluster Visualization by Label")
                # end save and print -------------------------------------------------------------

                pt_dist, num_pts_searched = ared.process_point(data_stream.stream_new_data_point())
                pt_dists.append(pt_dist)
                num_pts_searched_list.append(num_pts_searched)


            print(ared.conf_matrix)
            print(np.sum(ared.conf_matrix[:]))

            current_time = time.time()
            time_elapsed = current_time - start_time
            print(f"Run took {time_elapsed:.2f} seconds")
            print("ARED COMPLETE")

            batch_times = np.diff(np.array(times))
            batch_queries = np.diff(np.array(num_queries))
            plt.figure()
            plt.plot(batch_times/np.max(batch_times))
            plt.plot(np.array(batch_queries)/max(batch_queries))
            plt.plot(np.array(num_clusters)/max(num_clusters))
            plt.plot(np.array(num_labels))
            # plt.plot(np.array(precision)/1.0)
            plt.legend(["time per batch", "num queries per batch", "num_clusters,num_labels"])
            plt.figure()
            #DEBUG ONLY------------------------------------------------
            pt_dists = np.array(pt_dists)
            num_pts_searched_list = np.array(num_pts_searched_list)
            total_num_pts_searched = np.sum(num_pts_searched_list,axis=1)
            plt.plot(pt_dists/max(pt_dists)*max(total_num_pts_searched),'o-') # don't care about scale of this
            plt.plot(num_pts_searched_list,'o-') # do care about scale of this!
            plt.legend(["closest pt dists", "num pts searched in l_buf"])
            plt.figure()
            plot_stacked_area(num_pts_searched_list[:,0], num_pts_searched_list[:,1], num_pts_searched_list[:,2], \
                              legend_labels=['# tail_pts', '# ball_trees_pts', '# head_pts'])
            plt.show()


