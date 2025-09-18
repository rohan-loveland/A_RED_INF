from Circular_Buffer import *
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
import threading
from collections import defaultdict
import heapq
from sklearn.mixture import GaussianMixture
from FiniteBuffer import FiniteBuffer

from main import QS_VAR


"""# Subspace Partition"""
# NOTE: ARED_IN doesn't use o-pts at the moment, and ALL o_pt support has been removed
# NOTE: all point data is referred to using absolute indexes including _all the points that have been streamed - _both
# l_pts and o_pts (the Finite Buffer has an internal conversion that keeps track of just l_pts, since o_pts aren't
# streamed into it)

class Subspace_Partition:
    def __init__(self,l_buf):
        #                                                                   (l_pt_idxs)          (o_pt_idxs)
        self.cluster_dict = {} # cluster is expected to be in the format of [label, relevance, [abs_idx_l_pt], [abs_idx_o_pt], diameter]
        self.set_of_known_labels = set()
        # cluster key is the cluster's key in cluster_dict - they start at 0 and just keep incrementing as new ones are needed
        # NOTE: they are _not re-used/recycled
        self.next_cluster_key_num = 0
        self.l_buf = l_buf

    def create_new_cluster(self, label, relevance, l_pt_idxs, o_pt_idxs, QS_VAR):
        self.set_of_known_labels.add(label)
        # can use len(cluster_dict) as cluster_id of new cluster because new cluster goes at end of list
        this_cluster_key_num = self.next_cluster_key_num
        self.cluster_dict[this_cluster_key_num] = Cluster(label, relevance, l_pt_idxs, self.l_buf, this_cluster_key_num, QS_VAR)
        self.next_cluster_key_num += 1
        return this_cluster_key_num

    def remove_l_pt_from_partition(self, pt_abs_idx, pt_cluster_key):
        self.cluster_dict[pt_cluster_key].remove_l_pt_from_cluster(pt_abs_idx)
        if len(self.cluster_dict[pt_cluster_key].l_pt_idxs) == 0:
            # zero'd out points in cluster so need to remove cluster
            del self.cluster_dict[pt_cluster_key]

"""# Cluster"""
class Cluster:
    def __init__(self, label, relevance, l_pt_idxs, o_pt_idxs, l_buf, cluster_key, QS_VAR = 0):
        self.label = label
        self.relevance = relevance
        self.l_pt_idxs = l_pt_idxs
        self.o_pt_idxs = o_pt_idxs
        self.comp_distance = 0  # QR_VAR=0: Diameter, QS_VAR=1: approx_nn_distance

        # cluster id is this cluster's position in Subspace_Partition.cluster_dict
        self.cluster_id = cluster_key
        if len(l_pt_idxs) > 1 and QS_VAR == 0:
            self.update_diameter(l_buf)

        elif len(l_pt_idxs) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist(l_buf)


    def add_l_pt(self, abs_idx, l_buf, QS_VAR = 0):
        self.l_pt_idxs.append(abs_idx)

        if QS_VAR == 0:
            self.update_diameter(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist(l_buf)
        # elif QS_VAR == 2:
        #     self.update_ave_nn_dist_w_o_pt_idxs(l_buf)

    def add_l_pt_no_comp_dist_update(self, abs_idx):
        self.l_pt_idxs.append(abs_idx)

    def remove_l_pt_from_cluster(self, pt_abs_idx):
        self.l_pt_idxs.remove(pt_abs_idx)

    def update_comp_distance(self, l_buf, QS_VAR = 0):
        if QS_VAR == 0:
            self.update_diameter(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist(l_buf)

    def merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
        if QS_VAR == 0:
            self.update_diameter(l_buf)
        if QS_VAR == 1:
            self.comp_distance = (
                                         self.comp_distance * num_points_from_merged_cluster
                                         + comp_distance_to_add * len(self.l_pt_idxs)
                                 ) / (len(self.l_pt_idxs) + num_points_from_merged_cluster)

    def update_diameter(self, l_buf):
        largest_distance = 0
        for i in range(len(self.l_pt_idxs)):
            for j in range(i):
                data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
                data_l_pt_j = l_buf.get_pt_data_abs(self.l_pt_idxs[j])
                distance = np.linalg.norm(data_l_pt_i - data_l_pt_j)
                if largest_distance < distance:
                    largest_distance = distance
        self.comp_distance = largest_distance

    def update_ave_nn_dist(self, l_buf):
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return

        # Retrieve data for all labeled points in the cluster
        data_points = np.array([l_buf.get_pt_data(abs_idx) for abs_idx in self.l_pt_idxs])

        # Compute the average nearest neighbor distance
        self.comp_distance = self.average_nearest_neighbor_distance(data_points)


    # for QS_VAR == 2
    # AI helper function
    def average_nearest_neighbor_distance(self, X):
        if len(X) < 2:
            raise ValueError("At least two points are required to compute nearest neighbor distances.")
        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(X)
        # Query for the two nearest neighbors (including itself)
        distances, _ = tree.query(X, k=2)

        # Extract the distance to the nearest neighbor (exclude self)
        nearest_distances = distances[:, 1]

        # Compute and return the average
        return np.mean(nearest_distances)

"""# AREDIN"""

class ARED:

    def __init__(self, oracle, kappa=1.0, l_buf_size=1000, k_closest_pts = 1, QS_VAR = 0, REL_PROC_VAR = 0, SM_VAR=0, VERBOSE_FLAGS = ()):
        self.kappa = kappa
        self.k_closest_pts = k_closest_pts
        self.l_buf = FiniteBuffer(l_buf_size)
        self.subspace_partition = Subspace_Partition(self.l_buf)
        self.oracle = oracle
        self.num_queries = 0
        self.anom_queries = 0 # queries arising from kappa comparison - NOT IMPLEMENTED YET
        self.rel_queries = 0 # queries arising from relevance assignment- NOT IMPLEMENTED YET
        self.num_pts_streamed = 0
        # Note: this is equivalent to abs_idx + 1
        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.REL_PROC_VAR = REL_PROC_VAR
        self.SM_VAR = SM_VAR
        self.verbose_flags = VERBOSE_FLAGS


    def process_first_point(self, data_point):
        # START QUERY
        self.num_pts_streamed += 1
        data_point_abs_idx = self.num_pts_streamed - 1

        label, relevance = self.query(data_point_abs_idx)
        # END QUERY

        # UPDATE CLUSTER Dictionary
        # Create new cluster
        cluster_key = self.subspace_partition.create_new_cluster(label, relevance, [data_point_abs_idx], [], self.QS_VAR)
        self.l_buf.insert_pt(data_point, cluster_key, label, relevance, data_point_abs_idx)
        # no maintenance required because this is the first point (so buffer's not full)
        # UPDATE CLUSTER LIST

        if 1 in self.verbose_flags:
            print("new cluster:", 0, [0])

    def merge_clusters(self, cluster_key_a, cluster_key_b):
        # DONE (by Nate Mediocrely)

        # failsafe to ensure that we don't try merging the same cluster.
        if cluster_key_a == cluster_key_b:
            return self.subspace_partition.cluster_dict[cluster_key_a]

        # get the smaller key (I am not sure that matters)
        smaller_cluster_key, larger_cluster_key = (
            (cluster_key_a, cluster_key_b)
            if cluster_key_a < cluster_key_b
            else (cluster_key_b, cluster_key_a)
        )

        if 5 in self.verbose_flags:
            print("Merging clusters:", smaller_cluster_key, larger_cluster_key)

        cluster_a = self.subspace_partition.cluster_dict[smaller_cluster_key]
        cluster_b = self.subspace_partition.cluster_dict[larger_cluster_key]

        [cluster_a.add_l_pt_no_comp_dist_update(l_pt) for l_pt in cluster_b.l_pts]
        # RL - or "cluster_a.l_pt_idxs += cluster_b_lpt_idxs"?

        l_buf_key_array = self.l_buf.cluster_key_circular_buffer.get_array()
        for i in range(l_buf_key_array):
            if l_buf_key_array[i] == cluster_key_b:
                l_buf_key_array[i] = cluster_key_a


        self.subspace_partition.cluster_dict.pop(larger_cluster_key)

        if QS_VAR == 0:
            #merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
            cluster_a.merge_comp_distances(self.l_buf, QS_VAR=self.QS_VAR)

        elif QS_VAR == 1:
            cluster_a.merge_comp_distances(self.l_buf, cluster_b.comp_distance, len(cluster_b.l_pts), self.QS_VAR)


        return cluster_a

    def determine_comparison_cluster(self, data_point):
        #DONE (by Nate Badly)
        comparison_point_info = None
        #                                 0            1                2     3      4     5    6
        # Get k closest points in l_buf [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
        k_closest_pts = self.l_buf.find_closest_pts(data_point, self.k_closest_pts)
        comparison_point_info = k_closest_pts[0]

        if len(k_closest_pts) > 1 and k_closest_pts[0][0] == k_closest_pts[1][0] and k_closest_pts[0][3] != k_closest_pts[1][3]:
            self.merge_clusters(k_closest_pts[0], k_closest_pts[1])

        #print(k_closest_pts)

        # Check for relevance in k the closest points
        k_closest_pts.reverse()
        for pt in k_closest_pts:
            if pt[5]: # if pt[5] is true
                comparison_point_info = pt

        # No relevant cluster in top-k, return closest overall
        return comparison_point_info # 0            1                 2     3      4     5          6
                                    # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]


    def anomalous(self, data_point, cluster_key, distance):
        cluster = self.subspace_partition.cluster_dict[cluster_key]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        # return (label, relevance) from oracle
        self.num_queries += 1
        return self.oracle.answer_query(abs_data_index)

    def update_structs_w_new_pt(self, abs_idx, data_point, cluster_key, label, relevance):
        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary

        # update l_buf to have the new point
        #                      0    1          2                 3
        # forgotten_pt_info = (key, relevance, internal_abs_idx, true_abs_idx)
        forgotten_pt_info = self.l_buf.insert_pt(data_point, cluster_key, label, relevance, abs_idx)

        if forgotten_pt_info:
            forgotten_pt_cluster_key = forgotten_pt_info[0]
            forgotten_pt_abs_idx = forgotten_pt_info[3]
            self.subspace_partition.remove_l_pt_from_partition(forgotten_pt_abs_idx, forgotten_pt_cluster_key)

    def add_l_pt_to_existing_cl(self, abs_idx, data_point, cluster_key):
        # Done (by Ro)
        # run when we add a new labeled data point to a known cluster
        # this adds to both l_buf and appropriate cluster in subspace partition
        # it also performs subspace partition maintenance if necessary

        if 1 in self.verbose_flags:
            print("add_l_pt:", abs_idx, cluster_key)

        this_cluster = self.subspace_partition.cluster_dict[cluster_key]

        # add point to cluster, so diameter gets updated properly
        this_cluster.add_l_pt_no_comp_dist_update(abs_idx)

        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary
        self.update_structs_w_new_pt(abs_idx, data_point, cluster_key,this_cluster.label, this_cluster.relevance)

        this_cluster.update_comp_distance(self.l_buf, self.QS_VAR)


    def split(self, data_point, data_point_idx, new_cluster_label, new_cluster_relevance, old_cluster_id):
        # Done (by Ro Not-Goodly)
        # NOTE: there are no o_pt_idxs in this implementation, so "splitting" consists only of creating a new cluster

        # make new cluster with 1 new l_pt
        this_cluster_key_num = self.subspace_partition.create_new_cluster(new_cluster_label, new_cluster_relevance, [data_point_idx], [], self.QS_VAR)

        # do maintenance by adding pt to l_buf, forgetting from subspace_partition if necessary
        self.update_structs_w_new_pt(data_point_idx, data_point, this_cluster_key_num, new_cluster_label, new_cluster_relevance)

    def process_point(self, data_point):
        self.num_pts_streamed += 1
        data_point_abs_idx = self.num_pts_streamed - 1

        # START DETERMINE COMPARISON CLUSTER
        comp_cluster_key, pt_abs_idx, distance, label, data, relevance, true_abs_idx = self.determine_comparison_cluster(data_point)
        comp_cluster_relevant = self.subspace_partition.cluster_dict[comp_cluster_key].relevance
        comp_cluster_label = self.subspace_partition.cluster_dict[comp_cluster_key].label

        is_anomalous = self.anomalous(data_point, comp_cluster_key, distance)


        if comp_cluster_relevant or is_anomalous:
            # Query!
            new_pt_label, new_pt_relevant = self.query(data_point_abs_idx)

            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, comp_cluster_key)
                # this adds pt to l_buf and updates appropriate cluster in subspace partition
            else:
                self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)
