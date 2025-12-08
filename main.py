'''
DATA_SOURCE: Dataset to run ARED on
|- NICE: Nice dataset - synthetic data with well separated Gaussian clusters
|- MNIST: MNIST dataset
|- EMNIST: EMNIST dataset
|- P_LOT

and ...

N_REL_CLASSES: Specified number of relevant classes
|- This is the number of classes (starting from sparsest) that are to be considered relevant
|- MNIST settings:
|=== High Relevant Class Representation (HRCR): 8 relevant classes ~25% of data as relevant
|=== Low Relevant Class Representation (LRCR): 4 relevant classes ~1.4% of data as relevant`
|- EMNIST settings:
|=== HRCR 40 relevant classes ~25% of data as relevant
|=== LRCR: 3 relevant classes ~1% of data as relevant`
|- NICE settings:
|=== Low relevance: 4 relevant classes ~1.4% of data as relevant`
'''
# DATA_SOURCE = "MNIST" # NOTE: currently multiplied by 10x to get ~130,000 samples
# N_REL_CLASSES = 4

# DATA_SOURCE = "MNIST_2D"
# N_REL_CLASSES = 3

# DATA_SOURCE = "EMNIST"
# N_REL_CLASSES = 3

# DATA_SOURCE = "NICE"
# N_REL_CLASSES = 4

DATA_SOURCE = "PARKING_LOT"
N_REL_CLASSES = 8

# DATA_SOURCE = "PARKING_LOT_DAGMM"
# N_REL_CLASSES = 8

'''
KAPPA: Paranoia Parameter
(single value for now)
'''
KAPPA = 0.5
# # KAPPAS = [0.5] #0.5, , 1.4, 10
# |- Array of Kappas to run ARED on
# |- Run more than one for graphing purposes

'''
QS_VAR: Query Strategy Variants
|- 0: Diameter check
|- 1: Approx. Ave Single Linkage Average 
'''
QS_VAR = 1

'''
K_COMP_PTS: Number of points to compare to when looking for relevance
|- 1: Standard ARED
|- 2 or more: k ARED
@WARNING: must be 1 or greater
'''
K_COMP_PTS = 5

'''
NGHBHOOD_MERGE: Neighborhood Merge Variants
|- if selected, if a split is going to occur, then check if 2nd closest neighbor cluster
|- has same label as newly queried point, and merge with that cluster instead
|- NOTE: K_COMP_PTS must be >= 2 for this
|- NOTE: currently this is only implemented for top 2
|- False: No neighborhood merge
|- True: Neighborhood merge
'''
NGHBHOOD_MERGE = True

'''
SINGLETON_MERGE: Neighborhood Merge Variants
|- if selected, periodically (atm every GRAPH_BATCH_SIZE) merge singleton clusters w/ K_COMP_PTS nearest neighbor clusters
|- NOTE: K_COMP_PTS must be >= 2 for this
|- False: No singleton merge
|- True: singleton merge
'''
SINGLETON_MERGE = True

# ------------------------------------------------------------------
# NEW: threshold for “small” clusters that will be forcibly merged
# ------------------------------------------------------------------
SMALL_CLUSTER_THRESHOLD = 3      # clusters with < 3 points are merged

'''
window_size: size of the data_window window saved by ARED
|- int: larger window size means it remembers more data
|- WARNING: value must be larger than 0
'''
DATA_WINDOW_SIZE = 10000 # ultimately needs to be driven by anomaly ratio

'''
NUM_POINTS_TO_PROCESS: Number of points in dataset to process
|- -1: process all the data
|-  0 to inf: process up to that number if data is available
'''
NUM_POINTS_TO_PROCESS = 100000#-1

# '''
# NUM_RUN_TO_AVE: number of runs to average.
# |- INT, higher numbers means more runs to average for graphs.
# |- @WARNING MUST BE GREATER THAN 1.
# '''
# NUM_RUNS_TO_AVE = 1

'''
GRAPH_BATCH_SIZE: number of points in batch for stats purposes.
'''
GRAPH_BATCH_SIZE = 500

'''
VERBOSE_FLAGS: Array of control flags to make ARED loud or quiet
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
MAKE_EVO_GRAPHS = False

'''
RANDOM_SEED_OFFSET
|- any integer
|- a way to vary seed sequence
'''
RANDOM_SEED_OFFSET = 25

# Imports ===================================

from Data_Stream import *
from Oracle import *
from A_REDIN import *
import sys
from Stats import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'wxAgg' depending on your system
from data_visualization import *
from more_stats import *
from main_helper_functions import *
from sklearn.metrics import ConfusionMatrixDisplay

from cluster_visualization import ClusterEvolutionPlotter

if __name__ == '__main__':

        # Get data ===================================================
        X_skewed, y_w_rel, sparsity_levels,rel_classes = get_data(DATA_SOURCE,N_REL_CLASSES, VERBOSE_FLAGS, RANDOM_SEED_OFFSET)

        # Initialize Data Stream, Oracle and ARED ===================================
        data_stream = Data_Stream(X_skewed, y_w_rel)
        oracle = Oracle(X_skewed, y_w_rel)
        ared = ARED(oracle, KAPPA, DATA_WINDOW_SIZE, K_COMP_PTS, QS_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE, VERBOSE_FLAGS)
        start_time, times, num_correct_queries, num_queries, num_clusters, num_labels, pt_dists, num_pts_searched_list, conf_matrices, \
            num_queries_last_batch, cumulative_relevants = set_up_stats(ared)
        buffer_fill_percents = []

        if MAKE_EVO_GRAPHS:
            evo_plotter = ClusterEvolutionPlotter()
            build_up_flag_evo_plotter = False # Flag used to only add the full l_buf to the subgraph once

        # Stream and Process data =========================================
        ared.process_first_point(data_stream.stream_new_data_point())

        dataset_size = len(X_skewed)
        if NUM_POINTS_TO_PROCESS == -1:
            NUM_POINTS_TO_PROCESS = data_stream.get_remaining_num_points()
        else:
            NUM_POINTS_TO_PROCESS = min(NUM_POINTS_TO_PROCESS, dataset_size-1)

        for i in range(1, NUM_POINTS_TO_PROCESS+1): # the +1 gives us an extra point, but makes the batch
            # arithmetic work out to include last batch by having last sample # end in 0

            if MAKE_EVO_GRAPHS:
                if i == 100:
                    evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "a) First 100 Points")
                elif i > 1000 and not ared.l_buf.build_up_period and not build_up_flag_evo_plotter:
                    evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "b) Labeled Buffer Full")
                    build_up_flag_evo_plotter = True

            # save and print per batch ---------------------------------------------------------------------
            if i % GRAPH_BATCH_SIZE == 0:
                j = i//GRAPH_BATCH_SIZE # count of number of batches
                print(i)
                times.append(time.time())
                num_correct_queries.append(ared.num_correct_queries)
                num_queries.append(ared.num_queries)
                num_clusters.append(len(ared.subspace_partition.cluster_dict))
                num_labels.append(len(ared.subspace_partition.set_of_known_labels))
                conf_matrices.append(ared.conf_matrix.copy())
                fill_pct = ared.l_buf.data_circular_buffer.count / ared.l_buf.data_circular_buffer.size * 100
                buffer_fill_percents.append(fill_pct)
                cumulative_relevants.append(ared.cumulative_relevant_seen)
                print(f"fill % of buffer: {fill_pct:.2f}%")
                if 0 in VERBOSE_FLAGS:
                    if j > 1:
                        print(f"Processing point {i}... (last {GRAPH_BATCH_SIZE} points took {times[j]- times[j-1]:.2f} seconds)")
                        print(f"# correct queries in this batch: {num_correct_queries[j-1] - num_correct_queries[j-2]}")
                        print(f"# queries in this batch: {num_queries[j-1] - num_queries[j-2]}")
                        print(f"Number of clusters: {num_clusters[j-1]}")  # Add cluster count
                        print(f"Number of labels: {num_labels[j-1]}")
                        print(f"fill % of buffer: {fill_pct:.2f}%")

                if MAKE_EVO_GRAPHS:
                    evo_plotter.plot_clusters_colored_by_label(ared, X_skewed, y_w_rel, title="Cluster Visualization by Label")

                if SINGLETON_MERGE:
                    ared.SMALL_CLUSTER_THRESHOLD = SMALL_CLUSTER_THRESHOLD  # expose the constant
                    ared.small_cluster_merge()

            # end save and print -------------------------------------------------------------

            pt_dist, num_pts_searched = ared.process_point(data_stream.stream_new_data_point())
            pt_dists.append(pt_dist)
            num_pts_searched_list.append(num_pts_searched)

        if MAKE_EVO_GRAPHS:
            evo_plotter.add_snapshot(ared, X_skewed, y_w_rel, "c) with Forgetting")
            evo_plotter.plot()
            evo_plotter.plot_dataset(X_skewed, y_w_rel)

        print("ARED DONE")

        relevant_per_batch = np.diff(cumulative_relevants, prepend=0)
        num_correct_queries_per_batch = np.diff(num_correct_queries, prepend=0)
        num_queries_per_batch = np.diff(num_queries, prepend=0)

        PLOT_FLAG = True
        single_rel_recall_list, query_precision_list, rel_individual_recalls, query_rate = \
            calc_rel_recall_query_precision(sparsity_levels, conf_matrices, rel_classes, ared, num_correct_queries, \
                                     num_queries, PLOT_FLAG, GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS)

        # with np.printoptions(threshold=sys.maxsize):
        #     print(ared.conf_matrix)

        if MAKE_GRAPHS:
            try:
                # --- Plot #1: Number of clusters ---
                batch_num_pts = list(range(GRAPH_BATCH_SIZE, NUM_POINTS_TO_PROCESS + 1, GRAPH_BATCH_SIZE))
                plt.figure(figsize=(10, 5))
                plt.plot(batch_num_pts, num_clusters, 'b-o', label="Number of clusters")
                plt.grid(True)
                plt.xlabel("Processed points")
                plt.ylabel("Count")
                plt.legend()
                plt.title("Cluster Evolution")

                # --- Plot #2: Time per batch + Buffer fill % ---
                if len(times) > 1 and len(buffer_fill_percents) == len(np.diff(times)):
                    plt.figure(figsize=(10, 6))

                    batch_indices = list(range(1, len(times)))

                    time_per_batch = np.diff(times)

                    ax1 = plt.gca()
                    col = 'tab:red'
                    ax1.set_xlabel("Batch #")
                    ax1.set_ylabel("Time per batch (sec)", color=col)
                    ax1.plot(batch_indices, time_per_batch, 'o-', color=col, label="Time per batch")
                    ax1.tick_params(axis='y', labelcolor=col)
                    ax1.grid(True, alpha=0.3)

                    ax2 = ax1.twinx()
                    col = 'tab:green'
                    ax2.set_ylabel("Buffer fill %", color=col)
                    ax2.plot(batch_indices, buffer_fill_percents, 's--', color=col, label="Buffer fill %")
                    ax2.tick_params(axis='y', labelcolor=col)

                    plt.title("Performance per Batch")
                    ax1.legend(loc="upper left")
                    ax2.legend(loc="upper right")

            except Exception as e:
                print("Plot error:", e)


        current_time = time.time()
        time_elapsed = current_time - start_time
        print(f"Run took {time_elapsed:.2f} seconds")
        print("ARED COMPLETE")
        #
        # batch_times = np.diff(np.array(times))
        # batch_queries = np.diff(np.array(num_queries))
        # plt.figure()
        # plt.plot(batch_times/np.max(batch_times))
        # plt.plot(np.array(batch_queries)/max(batch_queries))
        # plt.plot(np.array(num_clusters)/max(num_clusters))
        # plt.plot(np.array(num_labels))
        # # plt.plot(np.array(precision)/1.0)
        # plt.legend(["time per batch", "num queries per batch", "num_clusters,num_labels"])
        # plt.figure()
        # #DEBUG ONLY------------------------------------------------
        # pt_dists = np.array(pt_dists)
        # num_pts_searched_list = np.array(num_pts_searched_list)
        # total_num_pts_searched = np.sum(num_pts_searched_list,axis=1)
        # plt.plot(pt_dists/max(pt_dists)*max(total_num_pts_searched),'o-') # don't care about scale of this
        # plt.plot(num_pts_searched_list,'o-') # do care about scale of this!
        # plt.legend(["closest pt dists", "num pts searched in l_buf"])
        # plt.figure()
        # plot_stacked_area(num_pts_searched_list[:,0], num_pts_searched_list[:,1], num_pts_searched_list[:,2], \
        #                   legend_labels=['# tail_pts', '# ball_trees_pts', '# head_pts'])
        plt.show()


