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
    def __init__(self, label, relevance, l_pt_idxs, l_buf, cluster_key, QS_VAR = 0):
        self.label = label
        self.relevance = relevance
        self.l_pt_idxs = l_pt_idxs
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

    def merge_comp_distances(
            self,
            l_buf,
            other_l_pt_idxs=None,
            QS_VAR=0,
    ):
        """
        Merge cluster distances.

        Parameters
        ----------
        l_buf : buffer providing get_pt_data_abs
        other_l_pt_idxs : list of absolute indices of points
                          in the cluster being merged.
        QS_VAR : 0 -> diameter merge
                 1 -> average nearest-neighbor distance merge
        """
        if QS_VAR == 0:
            # Recompute diameter on the combined set
            self.update_diameter(l_buf)

        elif QS_VAR == 1:
            if other_l_pt_idxs is None:
                raise ValueError(
                    "For QS_VAR=1 you must pass other_l_pt_idxs to recompute average NN distance."
                )

            # Combine point indices
            combined_idxs = self.l_pt_idxs + list(other_l_pt_idxs)

            # Retrieve data like update_diameter
            data_points = np.array([l_buf.get_pt_data_abs(idx) for idx in combined_idxs])

            # Recompute average nearest-neighbor distance on the union
            self.comp_distance = self.average_nearest_neighbor_distance(data_points)

    # def update_diameter(self, l_buf):
    #     largest_distance = 0
    #     for i in range(len(self.l_pt_idxs)):
    #         for j in range(i):
    #             data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
    #             data_l_pt_j = l_buf.get_pt_data_abs(self.l_pt_idxs[j])
    #             distance = np.linalg.norm(data_l_pt_i - data_l_pt_j)
    #             if largest_distance < distance:
    #                 largest_distance = distance
    #     self.comp_distance = largest_distance

    def update_diameter(self, l_buf):
        latest_l_pt_idx = self.l_pt_idxs[-1]
        latest_pt_data = l_buf.get_pt_data_abs(latest_l_pt_idx)
        largest_distance = 0
        for i in range(len(self.l_pt_idxs)-1): # all except the latest...
                data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
                distance = np.linalg.norm(data_l_pt_i - latest_pt_data)
                if largest_distance < distance:
                    largest_distance = distance
        # if new distance is > existing diameter, update diameter, otherwise leave alone
        if largest_distance > self.comp_distance:
            self.comp_distance = largest_distance


    def update_ave_nn_dist(self, l_buf):
        """
        Compute the average nearest-neighbor distance of all labeled points in the cluster.
        Updated to retrieve data the same way QS_VAR = 0 does (using absolute indices).
        """
        # MAKE NOT O-N2 OR DIE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return

        # Retrieve data for all labeled points using absolute index accessor,
        # matching the approach in update_diameter.
        data_points = np.array([l_buf.get_pt_data_abs(abs_idx) for abs_idx in self.l_pt_idxs])

        # Compute and store the average nearest-neighbor distance
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
        self.l_buf = FiniteBuffer(l_buf_size, .8, 3)
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

        #                                 0            1                2     3      4     5    6
        # Get k closest points in l_buf [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]

        # failsafe to ensure that we don't try merging the same cluster # NOTE REDUNDANT SINCE we do this check before merging.
        if cluster_key_a == cluster_key_b:
            return self.subspace_partition.cluster_dict[cluster_key_a]
        # ^ CONSIDER REMOVING THIS

        if 5 in self.verbose_flags:
            print("Merging clusters:", cluster_key_a, cluster_key_b)

        cluster_a = self.subspace_partition.cluster_dict[cluster_key_a]
        cluster_b = self.subspace_partition.cluster_dict[cluster_key_b]

        cluster_a.l_pt_idxs += cluster_b.l_pt_idxs

        cb = self.l_buf.cluster_key_circular_buffer
        for i in range(len(self.l_buf.cluster_key_circular_buffer)):
            if cb.get(i) == cluster_key_b:
                cb.set_at(i, cluster_key_a)

        self.subspace_partition.cluster_dict.pop(cluster_key_b)

        if self.QS_VAR == 0:
            #merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
            cluster_a.merge_comp_distances(self.l_buf, QS_VAR=self.QS_VAR)

        elif self.QS_VAR == 1:
            cluster_a.merge_comp_distances(self.l_buf, cluster_b.l_pt_idxs, self.QS_VAR)


        return cluster_key_a

    def determine_comparison_cluster(self, data_point):
        #DONE (by Nate Okly)
        comparison_point_info = None
        relevant_point_info = None
        #                                 0            1                2     3      4     5    6
        # Get k closest points in l_buf [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
        # FOR NOW JUST DO ALL BRUTE FORCE TO DEBUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        k_closest_pts, num_pts_searched, all_pts_arr_brute = self.l_buf.find_closest_pts_all_brute(data_point, self.k_closest_pts)
        # LATER FIX 998 INSTEAD OF 1000 PROBLEM...
        # k_closest_pts, num_pts_searched = self.l_buf.find_closest_pts(data_point, self.k_closest_pts)
        # if self.num_pts_streamed >= 64982:
        #     k_closest_pts2, num_pts_searched, all_pts_arr_brute = self.l_buf.find_closest_pts_all_brute(data_point, self.k_closest_pts)
        #     dists2 = tuple([k[2] for k in k_closest_pts2])
        #     dists = tuple([k[2] for k in k_closest_pts])
        #     if dists2[0] != dists[0]:
        #         print(f"Different answers from ball tree and brute force search at point {self.num_pts_streamed}.")
        #         k_closest_pts2, num_pts_searched, all_pts_arr_brute = self.l_buf.find_closest_pts_all_brute(data_point, self.k_closest_pts)
        #         closest_data_pt = k_closest_pts2[0][4]
        #         k_closest_pts, num_pts_searched, all_pts_arr_btrees = self.l_buf.find_closest_pts_debug(data_point, self.k_closest_pts,closest_data_pt)
        #         a = all_pts_arr_brute
        #         b = all_pts_arr_btrees
        #         dtype = [('x', float), ('y', float)]
        #         a_struct = np.array([tuple(row) for row in a], dtype=dtype)
        #         b_struct = np.array([tuple(row) for row in b], dtype=dtype)
        #
        #         # Step 2: Find unique elements in a_struct not in b_struct
        #         mask = ~np.isin(a_struct, b_struct)
        #         result_struct = np.unique(a_struct[mask])
        #
        #         # Step 3: Convert back to original format (2D array)
        #         result = np.array([[row['x'], row['y']] for row in result_struct])
        #         print(f"num non-shared points: {result}")




        if len(k_closest_pts) > 1 and k_closest_pts[0][3] == k_closest_pts[1][3] and k_closest_pts[0][0] != k_closest_pts[1][0]:
        # i.e. if we have more than one point, and the closest points have the same label, and they're not in the same cluster...
            if k_closest_pts[0][0] < k_closest_pts[1][0]:
                keep_idx, merge_idx = 0, 1
            else:
                keep_idx, merge_idx = 1, 0

            keep_key = k_closest_pts[keep_idx][0]
            merge_key = k_closest_pts[merge_idx][0]

            self.merge_clusters(keep_key, merge_key)

            # update the merged point’s key so k_closest_pts is correct
            pt_info = list(k_closest_pts[merge_idx])
            pt_info[0] = keep_key
            k_closest_pts[merge_idx] = tuple(pt_info)

        comparison_point_info = k_closest_pts[0]

        # Check for relevance in k the closest points
        for pt in k_closest_pts:
            if pt[5]: # if pt[5] is true
                relevant_point_info = pt
                break

        # No relevant cluster in top-k, return closest overall
        return comparison_point_info, relevant_point_info, num_pts_searched # 0            1                 2     3      4     5          6
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
        #  0              1                2         3      4     5          6
        #  cluster_key,   pt_internal_idx, dist,     label, data, rel,       true_abs_idxz
        comp_cl_data, rel_cl_data, num_pts_searched = self.determine_comparison_cluster(data_point)
        comp_cluster_key, pt_internal_idx, distance, label, data, relevance, true_abs_idx = comp_cl_data
        comp_cluster_label = self.subspace_partition.cluster_dict[comp_cluster_key].label

        is_anomalous = self.anomalous(data_point, comp_cluster_key, distance)
        is_relevant = (rel_cl_data is not None)

        if is_relevant  or is_anomalous:

            # Query!
            new_pt_label, new_pt_relevant = self.query(data_point_abs_idx)

            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, comp_cluster_key)
                # this adds pt to l_buf and updates appropriate cluster in subspace partition
            else:
                # if self.num_pts_streamed > 80000:
                #     pass
                self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)

        # DEBUG ONLY -----------------------
        return distance, num_pts_searched
