'''
DATA_SOURCE: Dataset to run ARED on
|- MNIST: MNIST dataset
|- EMNIST: EMNIST dataset
|- P_LOT
'''
DATA_SOURCE = "MNIST"

'''
N_REL_CLASSES: Specified number of relevant classes
|- This is the number of classes (starting from sparsest) that are to be considered relevant
|- MNIST settings:
|=== High relevance: 8 relevant classes ~25% of data as relevant
|=== Low relevance: 4 relevant classes ~1.4% of data as relevant`
'''
N_REL_CLASSES = 8

'''
KAPPAS: Paranoia Parameter
|- Array of Kappas to run ARED on
|- Run more than one for graphing purposes
'''
KAPPAS = [1] #0.5, , 1.4, 10

'''
K_COMP_CLUST: Number of clusters to compare to when looking for relevance
|- 1: Standard ARED
|- 2 or more: k ARED
@WARNING: must be 1 or greater
'''
K_COMP_CLUST = 2

'''
QS_VAR: Query Strategy Variants
|- 0: Diameter check
|- 1: Approx. Ave Single Linkage Average 
|- 2: Approx. Ave Single Linkage Average w/ o_pt_idxs
'''
QS_VAR = 0

'''
SM_VAR: Split Method Var 
|- 0: Voronoi Split
|- 1: 
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
DATA_WINDOW_SIZE = 1000

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

# Imports ===================================
from Circular_Buffer import *
from MNIST_Data_Processing import *
from Data_Stream import *
from Oracle import *
# from A_RED import *
from A_REDIN import *
from sklearn.datasets import fetch_openml
from Stats import *
import numpy as np
import struct
import time
import pickle
import cv2
import pandas as pd
import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if DATA_SOURCE == "MNIST":
        # Note: this assumes 10 classes - is totally "MNIST centric"
        stats = Stats()

        for kappa in KAPPAS:

            stats.init_for_kappa_loop(kappa)

            for seed in range(NUM_RUNS_TO_AVE):

                # Get data and skew and add relevance
                X_skewed, y_w_rel = MNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)

                # Initialize Oracle and ARED ===================================
                data_stream = Data_Stream(X_skewed, y_w_rel)
                oracle = Oracle(X_skewed, y_w_rel)

                ared = ARED(oracle, kappa, DATA_WINDOW_SIZE, K_COMP_CLUST, QS_VAR, REL_PROC_VAR, SM_VAR, VERBOSE_FLAGS)

                # Start overall timer
                start_time = time.time()
                last_batch_time = start_time

                # Stream and Process data =========================================
                ared.process_first_point(data_stream.stream_new_data_point())

                if NUM_POINTS_TO_PROCESS == -1:
                    points_to_process = data_stream.get_remaining_num_points()
                else:
                    points_to_process = NUM_POINTS_TO_PROCESS - 1

                points_queried = 1 # already processed first point
                current_time = time.time()
                for i in range(2, points_to_process + 1):

                    if i % 1000 == 0:
                        current_time = time.time()
                        time_elapsed = current_time - last_batch_time
                        last_batch_time = current_time
                        if 0 in VERBOSE_FLAGS:
                            print(f"Processing point {i}... (last 1000 points took {time_elapsed:.2f} seconds)")
                            print(f"Points queried: {len(ared.l_buf.data_array) - points_queried}, Query Rate: {(len(ared.l_buf.data_array) - points_queried) / 1000 * 100}%")
                            points_queried = len(ared.l_buf.data_array)

                    ared.process_point(data_stream.stream_new_data_point())

                current_time = time.time()
                time_elapsed = current_time - start_time
                print(f"Last points took {time_elapsed:.2f} seconds")
                print("ARED COMPLETE")

                num_queries = ared.num_queries
                num_pts_streamed = ared.num_pts_streamed
                num_total_points = NUM_POINTS_TO_PROCESS
                num_total_relevant_points = sum(y_w_rel[:num_pts_streamed,1]) # need to test this...
                # # Note: THIS NEEDS TO BE REWORKED - IT'S AN APPROXIMATION THAT IS ONLY REALLY TRUE IF ALL PTS ARE STREAMED
                # num_total_relevant_points = num_pts_streamed * sum(sparsity_levels[-N_REL_CLASSES:])

                num_relevant_points_found = 0
                for cluster in ared.subspace_partition.cluster_dict:
                    # print(cluster.cluster_id,'# l',len(cluster.l_pt_idxs),'#o',len(cluster.o_pt_idxs),'label',cluster.label,cluster.relevance,cluster.comp_distance)
                    if cluster.relevance:
                        num_relevant_points_found += len(cluster.l_pts)

                stats.precisions.append(num_relevant_points_found / num_queries)  # Relevant points found / total query
                stats.recalls.append(num_relevant_points_found / num_total_relevant_points)  # Relevant points found / total num rel points

                # Random Baseline
                # Note: see math explanation
                # Precision just equals ratio of num_relevant_pts to num_total_points
                stats.precision_baseline.append(num_total_relevant_points / num_total_points)
                # Recall just equals query rate
                query_rate = num_queries/ num_total_points
                stats.recall_baseline.append(query_rate)

                if 0 in VERBOSE_FLAGS:
                    total_elapsed_time = time.time() - start_time
                    print(f"Total streaming time: {total_elapsed_time:.2f} seconds")
                    print(f"kappa = {kappa}")
                    print(f"k = {K_COMP_CLUST}")
                    print(f"Number of points processed = {num_pts_streamed}")
                    print(f"number of classes discovered {len(ared.subspace_partition.set_of_known_labels)}")
                    print(f"classes discovered {ared.subspace_partition.set_of_known_labels}")
                    print(f"Number of queries: {num_queries}")
                    print(f"Relevant points found {num_relevant_points_found}")
                    print(f"Relevant recall percentage: {100*num_relevant_points_found/num_total_relevant_points:.2f}%")
                    print(f"Equivalent Random Relevant recall percentage: {(100*num_queries/num_pts_streamed):.2f}%")

                #only use the first seed for queries over time and query rate
                # if i == 0:
                #     stats.store_ared_query_information(ared)

            print(stats.precisions, stats.recalls, stats.precision_baseline, stats.recall_baseline)

            stats.averaged_precision_recalls[-1][1] = sum(stats.precisions) / len(stats.precisions)
            stats.averaged_precision_recalls[-1][2] = sum(stats.recalls) / len(stats.recalls)
            stats.averaged_precision_recalls[-1][3] = sum(stats.precision_baseline) / len(stats.precision_baseline)
            stats.averaged_precision_recalls[-1][4] = sum(stats.recall_baseline) / len(stats.recall_baseline)
            stats.store_ared_precision_recall(kappa, stats.averaged_precision_recalls[-1][1], stats.averaged_precision_recalls[-1][2], stats.averaged_precision_recalls[-1][3], stats.averaged_precision_recalls[-1][4])

        if MAKE_GRAPHS:
            stats.graph_all_queries_over_time("./queries_over_time_mnist.pdf")
            stats.graph_all_query_rates_over_time(100, "./query_rate_over_time_mnist.pdf")
            stats.plot_precision_recall_curve("./precision_recall_curve_mnist.pdf")

    # elif DATA_SOURCE == "EMNIST":
    #     print("Loading EMNIST dataset... ")
    #     DATA_DIR = "./gzip"
    #     # File paths
    #     TRAIN_IMAGE_FILE = 'emnist-byclass-train-images-idx3-ubyte'
    #     TRAIN_LABEL_FILE = 'emnist-byclass-train-labels-idx1-ubyte'
    #     TEST_IMAGE_FILE = 'emnist-byclass-test-images-idx3-ubyte'
    #     TEST_LABEL_FILE = 'emnist-byclass-test-labels-idx1-ubyte'
    #     MAPPING_FILE = 'emnist-byclass-mapping.txt'
    #
    #     # --- Load mapping file ---
    #     mapping = {}
    #     with open(os.path.join(DATA_DIR, MAPPING_FILE), 'r') as f:
    #         for line in f:
    #             label_idx, unicode_val = map(int, line.strip().split())
    #             mapping[label_idx] = chr(unicode_val)
    #     # --- Load train images ---
    #     with open(os.path.join(DATA_DIR, TRAIN_IMAGE_FILE), 'rb') as f:
    #         magic, num_images_train, rows, cols = struct.unpack('>IIII', f.read(16))
    #         train_img_data = np.frombuffer(f.read(), dtype=np.uint8)
    #         train_images = train_img_data.reshape((num_images_train, rows, cols))
    #     # --- Load train labels ---
    #     with open(os.path.join(DATA_DIR, TRAIN_LABEL_FILE), 'rb') as f:
    #         magic, num_labels_train = struct.unpack('>II', f.read(8))
    #         train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    #     # --- Load test images ---
    #     with open(os.path.join(DATA_DIR, TEST_IMAGE_FILE), 'rb') as f:
    #         magic, num_images_test, rows_test, cols_test = struct.unpack('>IIII', f.read(16))
    #         test_img_data = np.frombuffer(f.read(), dtype=np.uint8)
    #         test_images = test_img_data.reshape((num_images_test, rows_test, cols_test))
    #     # --- Load test labels ---
    #     with open(os.path.join(DATA_DIR, TEST_LABEL_FILE), 'rb') as f:
    #         magic, num_labels_test = struct.unpack('>II', f.read(8))
    #         test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    #
    #     # --- Combine train and test images and labels ---
    #     all_images = np.concatenate((train_images, test_images), axis=0)
    #     all_labels = np.concatenate((train_labels, test_labels), axis=0)
    #     n_events = len(all_labels)
    #
    #     # --- Fix orientation and flatten images ---
    #     X_full = [np.fliplr(img.T).flatten() / 255.0 for img in all_images]
    #     y_full = all_labels
    #
    #     # --- Identify least common labels ---
    #     label_counts = Counter(y_full)
    #     least_common_labels = [label for label, _ in label_counts.most_common()[-N_REL_CLASSES:]]
    #
    #     if 0 in VERBOSE_FLAGS:
    #         print(f"Running ARED on combined EMNIST train+test with {n_events} samples")
    #         print(f"Least common characters (relevant): {[mapping[l] for l in least_common_labels]}")
    #
    #     # --- Generate relevance array ---
    #     relevance_array = [label in least_common_labels for label in y_full]
    #     y_w_rel = list(zip(y_full, relevance_array))
    #     # --- Initialize and run ARED ---
    #     data_stream = Data_Stream(X_full, y_w_rel)
    #     oracle = Oracle(X_full, y_w_rel)
    #
    #     kappas = KAPPAS
    #     stats = Stats()
    #
    #     for kappa in kappas:
    #         data_stream.reset_stream_counter()
    #         ared = ARED(oracle, kappa, DATA_WINDOW_SIZE, K_COMP_CLUST, QS_VAR, REL_PROC_VAR, VERBOSE_FLAGS)
    #
    #         # Start overall timer
    #         start_time_total = time.time()
    #         last_batch_time = start_time_total
    #
    #         # Stream and Process data =========================================
    #         ared.process_first_point(data_stream.stream_new_data_point())
    #
    #         if NUM_POINTS_TO_PROCESS == -1:
    #             points_to_process = data_stream.get_remaining_num_points()
    #         else:
    #             points_to_process = NUM_POINTS_TO_PROCESS - 1
    #
    #         points_queried = 0
    #         for i in range(1, points_to_process + 1):
    #             if 0 in VERBOSE_FLAGS and not i % 1000:
    #                 current_time = time.time()
    #                 time_elapsed = current_time - last_batch_time
    #                 print(f"Processing point {i}... (last 1000 points took {time_elapsed:.2f} seconds)")
    #                 print(
    #                     f"Points queried: {len(ared.l_buf.data_array) - points_queried}, Query Rate: {(len(ared.l_buf.data_array) - points_queried) / 1000 * 100}%")
    #                 points_queried = len(ared.l_buf.data_array)
    #                 last_batch_time = current_time
    #
    #             ared.process_point(data_stream.stream_new_data_point())
    #
    #         time_elapsed = current_time - last_batch_time
    #         print(f"Last points took {time_elapsed:.2f} seconds")
    #         print("ARED COMPLETE")
    #
    #         print(points_queried)
    #
    #         if 0 in VERBOSE_FLAGS:
    #             total_elapsed_time = time.time() - start_time_total
    #             print(f"Total streaming time: {total_elapsed_time:.2f} seconds")
    #             num_queries = len(ared.l_buf.abs_idx_array)
    #             num_pts_streamed = ared.data_window.abs_idx_max + 1
    #             print(f"kappa = {kappa}")
    #             print(f"k = {K_COMP_CLUST}")
    #             print(f"Points processed = {num_pts_streamed}")
    #             print(f"Classes discovered: {[mapping[l] for l in ared.subspace_partition.set_of_known_labels]}")
    #             print(f"Number of queries: {num_queries}")
    #
    #             num_relevant_points_found = sum(
    #                 len(c.l_pt_idxs) + len(c.o_pt_idxs)
    #                 for c in ared.subspace_partition.cluster_dict
    #                 if c.relevance
    #             )
    #
    #             num_total_relevant_points = sum(label_counts[l] for l in least_common_labels)
    #             print(f"Relevant points found: {num_relevant_points_found}")
    #             print(f"Relevant recall: {100 * num_relevant_points_found / num_total_relevant_points:.2f}%")
    #             print(f"Random baseline recall: {100 * num_queries / num_pts_streamed:.2f}%")
    #
    #         stats.store_ared_query_information(ared)
    #
    #     stats.graph_all_queries_over_time("./queries_over_time.pdf")
    #     stats.graph_all_query_rates_over_time(100, "./query_rate_over_time.pdf")
    #
    #
    # elif DATA_SOURCE == "P_LOT":
    #     # === Load Dataset ===
    #     features_path = "./Parking_Lot_Data/features.pkl"
    #     labels_path = "./Parking_Lot_Data/labels.csv"
    #
    #     with open(features_path, 'rb') as f:
    #         features = pickle.load(f)  # shape (4410, 128, 128, 3)
    #
    #     labels_df = pd.read_csv(labels_path)
    #
    #     # Convert RGB to grayscale, normalize to [0, 1], and flatten
    #     X_full = np.array([
    #         cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32).flatten() / 255.0
    #         for img in features
    #     ])
    #
    #     y_full = labels_df["label"].tolist()
    #     assert len(X_full) == len(y_full), "Mismatch between features and labels"
    #
    #     print(X_full.shape)
    #
    #     # Combine X and y into a list of pairs
    #     combined = list(zip(X_full, y_full))
    #
    #     # Shuffle the combined list
    #     random.shuffle(combined)
    #
    #     # Unzip the shuffled list back into X and y
    #     X_full, y_full = zip(*combined)
    #
    #     # Convert back to NumPy arrays
    #     X_full = np.array(X_full)
    #     y_full = np.array(y_full)
    #
    #     # === Generate Relevance Labels ===
    #     non_relevant_labels = {"normal", "shadow"} #
    #     relevance_array = [label not in non_relevant_labels for label in y_full]
    #     y_w_rel = list(zip(y_full, relevance_array))
    #     num_rel_points = sum(relevance_array)
    #
    #     print(f"Total samples: {len(X_full)}")
    #     print(f"Relevant samples: {sum(relevance_array)}")
    #
    #     data_stream = Data_Stream(X_full, y_w_rel)
    #     oracle = Oracle(X_full, y_w_rel)
    #
    #     kappas = KAPPAS
    #     stats = Stats()
    #
    #     for kappa in kappas:
    #         data_stream.reset_stream_counter()
    #         ared = ARED(oracle, kappa, DATA_WINDOW_SIZE, K_COMP_CLUST, QS_VAR, REL_PROC_VAR, VERBOSE_FLAGS)
    #
    #         # Start overall timer
    #         start_time_total = time.time()
    #         last_batch_time = start_time_total
    #
    #         # Stream and Process data =========================================
    #         ared.process_first_point(data_stream.stream_new_data_point())
    #
    #         if NUM_POINTS_TO_PROCESS == -1:
    #             points_to_process = data_stream.get_remaining_num_points()
    #         else:
    #             points_to_process = NUM_POINTS_TO_PROCESS - 1
    #
    #         points_queried = 0
    #         for i in range(1, points_to_process + 1):
    #             if 0 in VERBOSE_FLAGS and not i % 1000:
    #                 current_time = time.time()
    #                 time_elapsed = current_time - last_batch_time
    #                 print(f"Processing point {i}... (last 1000 points took {time_elapsed:.2f} seconds)")
    #                 print(
    #                     f"Points queried: {len(ared.l_buf.data_array) - points_queried}, Query Rate: {(len(ared.l_buf.data_array) - points_queried) / 1000 * 100}%")
    #                 points_queried = len(ared.l_buf.data_array)
    #                 last_batch_time = current_time
    #
    #             ared.process_point(data_stream.stream_new_data_point())
    #
    #         time_elapsed = current_time - last_batch_time
    #         print(f"Last points took {time_elapsed:.2f} seconds")
    #         print("ARED COMPLETE")
    #
    #         print(points_queried)
    #
    #         if 0 in VERBOSE_FLAGS:
    #             total_elapsed_time = time.time() - start_time_total
    #             print(f"Total streaming time: {total_elapsed_time:.2f} seconds")
    #
    #             num_queries = len(ared.l_buf.abs_idx_array)
    #             num_pts_streamed = ared.data_window.abs_idx_max + 1
    #
    #             print(f"kappa = {kappa}")
    #             print(f"k = {K_COMP_CLUST}")
    #             print(f"Number of relevant points: {num_rel_points}")
    #             print(f"Number of points processed = {num_pts_streamed}")
    #             print(f"number of classes discovered {len(ared.subspace_partition.set_of_known_labels)}")
    #             print(f"classes discovered {ared.subspace_partition.set_of_known_labels}")
    #             print(f"Number of queries: {num_queries}")
    #
    #             num_relevant_points_found = 0
    #             for cluster in ared.subspace_partition.cluster_dict:
    #                 # print(cluster.cluster_id,'# l',len(cluster.l_pt_idxs),'#o',len(cluster.o_pt_idxs),'label',cluster.label,cluster.relevance,cluster.comp_distance)
    #                 if cluster.relevance:
    #                     num_relevant_points_found += len(cluster.l_pt_idxs) + len(cluster.o_pt_idxs)
    #
    #             print(f"Relevant points found {num_relevant_points_found}")
    #             print(
    #                 f"Relevant recall percentage: {100 * num_relevant_points_found / num_rel_points:.2f}%")
    #             print(f"Equivalent Random Relevant recall percentage: {(100 * num_queries / num_pts_streamed):.2f}%")
    #
    #         stats.store_ared_query_information(ared)
    #
    #     stats.graph_all_queries_over_time("./queries_over_time.pdf")
    #     stats.graph_all_query_rates_over_time(100, "./query_rate_over_time.pdf")

# Planning on moving the printing and stuff like that to it's own function since it is used by each ARED run and it will hopefully make things cleaner.
def print_ared_results(start_time, kappa, k_comp_cluster, num_pts_streamed, ared, num_queries, num_relevant_points_found, num_total_relevant_points):
    total_elapsed_time = time.time() - start_time
    print(f"Total streaming time: {total_elapsed_time:.2f} seconds")
    print(f"kappa = {kappa}")
    print(f"k = {k_comp_cluster}")
    print(f"Number of points processed = {num_pts_streamed}")
    print(f"number of classes discovered {len(ared.subspace_partition.set_of_known_labels)}")
    print(f"classes discovered {ared.subspace_partition.set_of_known_labels}")
    print(f"Number of queries: {num_queries}")
    print(f"Relevant points found {num_relevant_points_found}")
    print(f"Relevant recall percentage: {100 * num_relevant_points_found / num_total_relevant_points:.2f}%")
    print(f"Equivalent Random Relevant recall percentage: {(100 * num_queries / num_pts_streamed):.2f}%")