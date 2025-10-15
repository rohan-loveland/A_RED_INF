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
            self.update_ave_nn_dist(l_buf)

    def update_diameter(self, l_buf):
        latest_l_pt_idx = self.l_pt_idxs[-1]
        latest_pt_data = l_buf.get_pt_data_abs(latest_l_pt_idx)
        largest_distance = 0
        for i in range(len(self.l_pt_idxs)-1): # all except the latest...
                data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
                distance = np.linalg.norm(data_l_pt_i - latest_pt_data)
                if  distance > largest_distance:
                    largest_distance = distance
        # if new distance is > existing diameter, update diameter, otherwise leave alone
        if largest_distance > self.comp_distance:
            self.comp_distance = largest_distance


    def update_ave_nn_dist(self, l_buf):
        """
        Compute the average nearest-neighbor distance of all labeled points in the cluster.
        Updated to retrieve data the same way QS_VAR = 0 does (using absolute indices).
        """
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return

        latest_l_pt_idx = self.l_pt_idxs[-1]
        latest_pt_data = l_buf.get_pt_data_abs(latest_l_pt_idx)
        data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[0])
        smallest_distance = np.linalg.norm(data_l_pt_i - latest_pt_data)
        num_pts_in_cluster = len(self.l_pt_idxs)
        for i in range(num_pts_in_cluster - 1):  # all except the latest...
            data_l_pt_i = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
            distance = np.linalg.norm(data_l_pt_i - latest_pt_data)
            if distance < smallest_distance:
                smallest_distance = distance

        # calculate new average
        self.comp_distance = (self.comp_distance*(num_pts_in_cluster - 1) + smallest_distance)/num_pts_in_cluster


"""# AREDIN"""

class ARED:

    def __init__(self, oracle, kappa, l_buf_size, K_COMP_PTS, QS_VAR, REL_PROC_VAR, SM_VAR, NGHBHOOD_MERGE, SINGLETON_MERGE,VERBOSE_FLAGS):
        self.kappa = kappa
        self.K_COMP_PTS = K_COMP_PTS
        self.l_buf = FiniteBuffer(l_buf_size, .8, 2)
        self.subspace_partition = Subspace_Partition(self.l_buf)
        self.oracle = oracle
        self.num_correct_queries = 0
        self.num_queries = 0
        self.anom_queries = 0 # queries arising from kappa comparison - NOT IMPLEMENTED YET
        self.rel_queries = 0 # queries arising from relevance assignment- NOT IMPLEMENTED YET
        self.num_pts_streamed = 0
        # Note: this is equivalent to abs_idx + 1
        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.REL_PROC_VAR = REL_PROC_VAR
        self.SM_VAR = SM_VAR
        self.NGHBHOOD_MERGE = NGHBHOOD_MERGE
        self.SINGLETON_MERGE = SINGLETON_MERGE
        self.verbose_flags = VERBOSE_FLAGS
        self.conf_matrix = np.zeros((oracle.num_classes, oracle.num_classes), dtype=int)



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


        # update confusion matrix
        int_label = self.oracle.int_str_label_bidict[label]
        self.conf_matrix[int_label,int_label] += 1

    def singleton_merge(self):
        """
        Periodically merge singleton clusters (len(l_pt_idxs) == 1) with nearest neighbor clusters
        that share the same label. Updates cluster keys in l_buf and subspace partition.
        """
        if not self.SINGLETON_MERGE or self.K_COMP_PTS < 2:
            if 5 in self.verbose_flags:
                print("Singleton merge skipped: SINGLETON_MERGE=False or K_COMP_PTS<2")
            return

        # Identify singleton clusters
        singleton_keys = [key for key, cluster in self.subspace_partition.cluster_dict.items()
                          if len(cluster.l_pt_idxs) == 1]

        if 5 in self.verbose_flags:
            print(f"Singleton merge: Found {len(singleton_keys)} singleton clusters: {singleton_keys}")

        for singleton_key in singleton_keys:
            if singleton_key not in self.subspace_partition.cluster_dict:
                continue  # Cluster may have been merged/removed already

            cluster = self.subspace_partition.cluster_dict[singleton_key]
            singleton_idx = cluster.l_pt_idxs[0]  # Singleton's only point (absolute index)
            singleton_label = cluster.label
            singleton_data = self.l_buf.get_pt_data_abs(singleton_idx)

            if singleton_data is None:
                if 5 in self.verbose_flags:
                    print(f"Singleton {singleton_key} point {singleton_idx} no longer in buffer, skipping")
                continue

            # Find K_COMP_PTS closest points
            # Format: [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
            k_closest_pts, _ = self.l_buf.find_closest_pts(singleton_data, self.K_COMP_PTS)

            # Look for a matching label in the closest points' clusters
            for pt_info in k_closest_pts:
                neighbor_cluster_key = pt_info[0]
                neighbor_label = pt_info[3]

                # Skip if same cluster or cluster no longer exists
                if neighbor_cluster_key == singleton_key or neighbor_cluster_key not in self.subspace_partition.cluster_dict:
                    continue

                # Merge if labels match
                if neighbor_label == singleton_label:
                    if 5 in self.verbose_flags:
                        print(f"Merging singleton cluster {singleton_key} into cluster {neighbor_cluster_key} "
                              f"(label: {singleton_label})")

                    # Merge clusters (updates l_buf cluster keys and comp_distance)
                    self.merge_clusters(neighbor_cluster_key, singleton_key)
                    break  # Stop after first merge to avoid merging same singleton multiple times

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
        comparison_point_info = None
        relevant_point_info = None
        #                                 0            1                2     3      4     5    6
        # Get k closest points in l_buf [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
        k_closest_pts, num_pts_searched = self.l_buf.find_closest_pts(data_point, self.K_COMP_PTS)

        if self.NGHBHOOD_MERGE and len(k_closest_pts) > 1 and k_closest_pts[0][3] == k_closest_pts[1][3] and k_closest_pts[0][0] != k_closest_pts[1][0]:
        # i.e. if we're doing neighborhood merging and we've already processed more than one point overall,
        # and the 2 closest points have the same label,
        # and they're not in the same cluster - merge the 2 closest clusters!
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
            if pt[5]: # if pt[5] - relevance label - is true
                relevant_point_info = pt
                break

        # No relevant cluster in top-k, return closest overall
        return comparison_point_info, relevant_point_info, num_pts_searched, k_closest_pts
        # for k_closest_pts indices...
        #    0            1                 2     3      4     5          6
        # [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]


    def anomalous(self, data_point, cluster_key, distance):
        cluster = self.subspace_partition.cluster_dict[cluster_key]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        (label, relevance) = self.oracle.answer_query(abs_data_index)
        # query is "correct" if we a) discovered a new class, or b) are querying a relevant class
        if label not in self.subspace_partition.set_of_known_labels:
            new_class_flag = True
        else:
            new_class_flag = False
        relevance_flag = relevance
        if new_class_flag or relevance_flag:
            self.num_correct_queries += 1
        self.num_queries += 1

        return (label, relevance)

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
        comp_cl_data, rel_cl_data, num_pts_searched, k_closest_pts = self.determine_comparison_cluster(data_point)
        comp_cluster_key, pt_internal_idx, distance, label, data, relevance, true_abs_idx = comp_cl_data
        comp_cluster_label = self.subspace_partition.cluster_dict[comp_cluster_key].label

        is_anomalous = self.anomalous(data_point, comp_cluster_key, distance)
        comp_cl_is_relevant = (rel_cl_data is not None)

        if comp_cl_is_relevant or is_anomalous:
            # Query!
            new_pt_label, new_pt_relevant = self.query(data_point_abs_idx)

            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, comp_cluster_key)
                # this adds pt to l_buf and updates appropriate cluster in subspace partition
            else: # new pt label different from comp. cluster
                if self.K_COMP_PTS > 1 and self.NGHBHOOD_MERGE and len(k_closest_pts) > 1:
                    #                     0            1                2     3      4     5    6
                    # k closest points [(cluster_key, pt_internal_idx, dist, label, data, rel, true_abs_idx)]
                    second_cluster_label = k_closest_pts[1][3]
                    if new_pt_label == second_cluster_label:
                        second_cluster_key = k_closest_pts[1][0]
                        self.add_l_pt_to_existing_cl(data_point_abs_idx, data_point, second_cluster_key)
                        # this adds pt to l_buf and updates appropriate cluster in subspace partition
                        # print("NEIGHBORHOOD MERGE!")
                    else:
                        self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)
                else:
                    self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_key)
            # update confusion matrix
            new_pt_label_int = self.oracle.int_str_label_bidict[new_pt_label]
            self.conf_matrix[new_pt_label_int,new_pt_label_int] += 1
        else:
            # update confusion matrix
            # this is o_pt case, so ARED doesn't actually know label, so we have to "peek" at it
            actual_new_pt_label = self.oracle.y[data_point_abs_idx][0]
            actual_new_pt_label_int = self.oracle.int_str_label_bidict[actual_new_pt_label]
            comp_cluster_label_int = self.oracle.int_str_label_bidict[comp_cluster_label]
            self.conf_matrix[actual_new_pt_label_int,comp_cluster_label_int] += 1

        # DEBUG ONLY -----------------------
        return distance, num_pts_searched
