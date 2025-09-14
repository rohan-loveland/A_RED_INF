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
# NOTE: ARED_IN doesn't use o-pts at the moment, but it might in the future, so I'm leaving support
# for them in
# NOTE: all point data is referred to using absolute indexes (as much as possible, outside of l_buf)

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
        self.cluster_dict[this_cluster_key_num] = Cluster(label, relevance, l_pt_idxs, o_pt_idxs, this_cluster_key_num, QS_VAR)
        self.next_cluster_key_num += 1
        return this_cluster_key_num

    def remove_l_pt_from_partition(self, pt_abs_idx, pt_cluster_key):
        self.cluster_dict[pt_cluster_key].remove_l_pt_from_cluster(pt_abs_idx)
        if len(self.cluster_dict[pt_cluster_key].l_pt_idxs) == 0:
            # zero'd out points in cluster so need to remove cluster
            del self.cluster_dict[pt_cluster_key]

"""# Cluster"""
class Cluster:
    def __init__(self, label, relevance, l_pts, o_pts, l_buf, cluster_key, QS_VAR = 0):
        self.label = label
        self.relevance = relevance
        self.l_pts = l_pts
        self.o_pts = o_pts
        self.comp_distance = 0  # QR_VAR=0: Diameter, QS_VAR=1: approx_nn_distance

        # cluster id is this cluster's position in Subspace_Partition.cluster_dict
        self.cluster_id = cluster_key
        if len(l_pts) > 1 and QS_VAR == 0:
            self.update_diameter(l_buf)

        elif len(l_pts) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist(l_buf)

        elif QS_VAR == 2:
            self.update_ave_nn_dist_w_o_pts(l_buf)

    def add_l_pt(self, abs_idx, l_buf, QS_VAR = 0):
        self.l_pts.append(abs_idx)

        if QS_VAR == 0:
            self.update_diameter(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist(l_buf)
        elif QS_VAR == 2:
            self.update_ave_nn_dist_w_o_pts(l_buf)

    def add_l_pt_no_comp_dist_update(self, abs_idx):
        self.l_pts.append(abs_idx)

    def remove_l_pt_from_cluster(self, pt_abs_idx):
        del self.l_pts[pt_abs_idx]

    def merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
        if QS_VAR == 0:
            self.update_diameter(l_buf)
        if QS_VAR == 1:
            self.comp_distance = (
                                         self.comp_distance * num_points_from_merged_cluster
                                         + comp_distance_to_add * len(self.l_pts)
                                 ) / (len(self.l_pts) + num_points_from_merged_cluster)

    def add_o_pt(self, abs_idx):
        self.o_pts.append(abs_idx)

    def update_diameter(self, l_buf):
        largest_distance = 0
        for i in range(len(self.l_pts)):
            for j in range(i):
                data_l_pt_i = l_buf.get_pt_data(self.l_pts[i])
                data_l_pt_j = l_buf.get_pt_data(self.l_pts[j])
                distance = np.linalg.norm(data_l_pt_i - data_l_pt_j)
                if largest_distance < distance:
                    largest_distance = distance
        self.comp_distance = largest_distance

    def update_ave_nn_dist(self, l_buf):
        if len(self.l_pts) < 2:
            self.comp_distance = 0.0
            return

        # Retrieve data for all labeled points in the cluster
        data_points = np.array([l_buf.get_pt_data(abs_idx) for abs_idx in self.l_pts])

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

    # NOT RE-IMPLEMENTED YET - STILL OLD CODE - CURRENT VERSION DOESN'T HAVE O-PTS
    # def update_ave_nn_dist_w_o_pts(self, l_buf):
    #     data_window = l_buf.data_window
    #     # make 2D array of all point data vectors
    #     newest_l_pt_ind = self.l_pts[-1]
    #     newest_l_pt_data = l_buf.get_pt_data(newest_l_pt_ind)
    #     l_and_o_pt_data = []
    #     for l_pt_ind in self.l_pts[:-1]:
    #         l_and_o_pt_data.append(l_buf.get_pt_data(l_pt_ind))
    #     for o_pt_ind in self.o_pts[:-1]:
    #         l_and_o_pt_data.append(data_window.get_data_point(o_pt_ind))
    #     l_and_o_pt_data.append(newest_l_pt_data)
    #     l_and_o_pt_data = np.array(l_and_o_pt_data)
    #     self.comp_distance = self.average_nearest_neighbor_distance(l_and_o_pt_data)

"""# AREDIN"""

class ARED:

    def __init__(self, oracle, kappa=1.0, l_buf_size=1000, k_closest_pts = 1, QS_VAR = 0, REL_PROC_VAR = 0, SM_VAR=0, VERBOSE_FLAGS = []):
        self.kappa = kappa
        self.k_closest_pts = k_closest_pts
        self.l_buf = FiniteBuffer(l_buf_size)
        self.subspace_partition = Subspace_Partition()
        self.oracle = oracle
        self.num_queries = 0 # NEW!  - NEED TO KEEP TRACK OF THESE IN HERE
        self.num_pts_streamed = 0 # NEW!  - NEED TO KEEP TRACK OF THESE IN HERE
        # Note: this is equivalent to abs_idx + 1
        # VARIATION CONTROL FLAGS
        self.QS_VAR = QS_VAR # {0: diameter, 1: Ave Single Link Dist in Cluster
        self.REL_PROC_VAR = REL_PROC_VAR
        self.SM_VAR = SM_VAR
        self.verbose_flags = VERBOSE_FLAGS


    def process_first_point(self, data_point):
        # START QUERY
        data_point_abs_idx = self.num_pts_streamed - 1
        self.num_pts_streamed = 1
        label, relevance = self.query(data_point_abs_idx)
        # END QUERY

        # UPDATE CLUSTER LIST
        # Create new cluster
        cluster_key = self.subspace_partition.create_new_cluster(label, relevance, [data_point_abs_idx], [], self.QS_VAR)
        self.l_buf.insert_pt(data_point, cluster_key, label, relevance)
        # UPDATE CLUSTER LIST

        if 1 in self.verbose_flags:
            print("new cluster:", 0, [0])

    def merge_clusters(self, cluster_a, cluster_b):
        if cluster_a.cluster_id == cluster_b.cluster_id:
            return cluster_a.cluster_id

        # find the cluster with the smaller id
        cluster_with_smaller_id = cluster_a
        cluster_with_larger_id = cluster_b
        if cluster_b.cluster_id < cluster_a.cluster_id:
            cluster_with_larger_id = cluster_a
            cluster_with_smaller_id = cluster_b

        if 5 in self.verbose_flags:
            print("Merging clusters:", cluster_with_smaller_id.cluster_id, cluster_with_larger_id.cluster_id)

        for abs_l_pt_idx_in_larger_id_cluster in cluster_with_larger_id.l_pts:
            #update data window
            if abs_l_pt_idx_in_larger_id_cluster > self.data_window.abs_idx_min:
                self.data_window.update_cluster_id_at(abs_l_pt_idx_in_larger_id_cluster, cluster_with_smaller_id.cluster_id)

            idx = self.l_buf.get_index_of_abs_idx(abs_l_pt_idx_in_larger_id_cluster)
            self.l_buf.cluster_id_array[idx] = cluster_with_smaller_id.cluster_id

            cluster_with_smaller_id.add_l_pt_no_comp_dist_update(abs_l_pt_idx_in_larger_id_cluster)

        for abs_o_pt_idx_in_larger_id_cluster in cluster_with_larger_id.o_pts:
            self.data_window.update_cluster_id_at(abs_o_pt_idx_in_larger_id_cluster, cluster_with_smaller_id.cluster_id)
            cluster_with_smaller_id.add_o_pt(abs_o_pt_idx_in_larger_id_cluster)

        if QS_VAR == 0:
            #merge_comp_distances(self, l_buf, comp_distance_to_add = 0, num_points_from_merged_cluster = 0, QS_VAR = 0):
            cluster_with_smaller_id.merge_comp_distances(self.l_buf)

        elif QS_VAR == 1:
            cluster_with_smaller_id.merge_comp_distances(self.l_buf, cluster_with_larger_id.comp_distance, len(cluster_with_larger_id.l_pts), self.QS_VAR)

        # replace old cluster with a cluster with an empty shell
        self.subspace_partition.cluster_dict[cluster_with_larger_id.cluster_id] = Cluster(None, False, [], [], self.l_buf, cluster_with_larger_id.cluster_id)

        return cluster_with_smaller_id.cluster_id

    def determine_comparison_cluster(self, data_point):
        comparison_point_info = None
        #                                 0            1           2     3      4     5
        # Get k closest points in l_buf [(cluster_key, pt_abs_idx, dist, label, data, rel)]
        k_closest_pts = self.l_buf.find_closest_pts(data_point, self.k_closest_pts)

        if len(k_closest_pts) > 1 and k_closest_pts[0][0] == k_closest_pts[1][0] and k_closest_pts[0][3] != k_closest_pts[1][3]:
            # MERGING SHOULD HAPPEN HERE
            pass

        #print(k_closest_pts)

        # Check for relevance in k the closest points
        for pt in k_closest_pts.reversed():
            if pt[5] == True: # redundant equality statement for confused programmers [making me add more compute cycles :c ]
                comparison_point_info = pt

        # No relevant cluster in top-k, return closest overall
        return comparison_point_info


    def anomalous(self, data_point, cluster_id, distance):
        cluster = self.subspace_partition.cluster_dict[cluster_id]

        # Point is anomalous if its distance is greater than the cluster's diameter
        return distance * self.kappa > cluster.comp_distance

    def query(self, abs_data_index):
        self.data_window.updated_labeled_window(abs_data_index)
        # return (label, relevance) from oracle
        return self.oracle.answer_query(abs_data_index)


    # ran when we add a new o_pt to a cluster
    def add_o_pt(self, abs_idx, cluster_id):

        if 1 in self.verbose_flags:
            print("add_o_pt:", abs_idx, cluster_id)

        cluster = self.subspace_partition.cluster_dict[cluster_id]
        cluster.add_o_pt(abs_idx)

        # update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(abs_idx, cluster_id)


    # ran when we add a new labeled data point to a known cluster
    def add_l_pt(self, abs_idx, data_point, cluster_id):

        if 1 in self.verbose_flags:
            print("add_l_pt:", abs_idx, cluster_id)

        # update cluster in subspace partition
        cluster = self.subspace_partition.cluster_dict[cluster_id]

        # get label and relevance
        label = cluster.label
        relevance = cluster.relevance

        # update data_window.assigned_cluster_id_window
        self.data_window.update_cluster_id_at(abs_idx, cluster_id)

        # update l_buf to have the new point
        self.l_buf.add_point(abs_idx, data_point, cluster_id, label, relevance)

        # add point to cluster, so diameter gets updated properly
        cluster.add_l_pt(abs_idx, self.l_buf, self.QS_VAR)


    def split(self, data_point, data_point_idx, new_cluster_label, new_cluster_relevance, old_cluster_id):

        if self.SM_VAR == 0: # VORONOI Splitting method
            new_cluster_id = len(self.subspace_partition.cluster_dict)
            self.l_buf.add_point(data_point_idx, data_point, new_cluster_id, new_cluster_label, new_cluster_relevance)
            self.data_window.update_cluster_id_at(data_point_idx, new_cluster_id)
            self.subspace_partition.create_new_cluster(new_cluster_label, new_cluster_relevance, [data_point_idx], [], self.l_buf, self.QS_VAR)

            if 1 in self.verbose_flags:
                print("new cluster:", new_cluster_id, [data_point_idx])

            # array to hold o_pt indexes during the split process
            new_cluster_o_pts_abs_inds = []
            old_cluster_o_pts_abs_inds = []

            # get o_pt indices
            o_pts_abs_inds_to_split = self.subspace_partition.cluster_dict[old_cluster_id].o_pts

            if (len(o_pts_abs_inds_to_split) == 0):
                return

            # get l_pt indices
            l_pt_inds = self.subspace_partition.cluster_dict[old_cluster_id].l_pts

            # o_pt_index is an abs_idx
            for o_pt_index in o_pts_abs_inds_to_split:
                o_pt = self.data_window.get_data_point(o_pt_index)

                # find the closest labeled point in the exisiting cluster
                distance_to_existing = min([
                    np.linalg.norm(o_pt - self.l_buf.get_data(l_pt_index))
                    for l_pt_index in l_pt_inds
                ])

                # get the distance to the labeled point in the new cluster
                distance_to_new = np.linalg.norm(o_pt - data_point)

                # put the o_pt in the closest cluster of the two
                if distance_to_existing < distance_to_new:
                    old_cluster_o_pts_abs_inds.append(o_pt_index)
                else:
                    #print(distance_to_new, distance_to_existing, o_pt_index)
                    new_cluster_o_pts_abs_inds.append(o_pt_index)

                    # update the data window so the assigned_label_id_window is correct for window maintenance later
                    self.data_window.update_cluster_id_at(o_pt_index, new_cluster_id)

            if 2 in self.verbose_flags:
                print("Split :")
                print("old_cluster_id w/ o_pt_idxs:", old_cluster_id, old_cluster_o_pts_abs_inds)
                print("new_cluster_id w/ o_pt_idxs:", new_cluster_id, new_cluster_o_pts_abs_inds)

            # put the o_pt_idxs in their correct cluster
            self.subspace_partition.cluster_dict[new_cluster_id].o_pts = new_cluster_o_pts_abs_inds # update o_pt_idxs new_cluster
            self.subspace_partition.cluster_dict[old_cluster_id].o_pts = old_cluster_o_pts_abs_inds # update o_pt_idxs old_cluster

        # if self.SM_VAR == 1: # 2 GMM splitting method
        #
        #     old_cluster_l_pts_idxes = self.subspace_partition.cluster_dict[old_cluster_id].l_pt_idxs
        #     old_cluster_l_pts = [self.l_buf.get_data(l_pt_idx) for l_pt_idx in old_cluster_l_pts_idxes]
        #     l_pt_idxs = old_cluster_l_pts.append(data_point)
        #
        #     # get o_pt indices
        #     o_pts_abs_inds_to_split = self.subspace_partition.cluster_dict[old_cluster_id].o_pt_idxs
        #     o_pt_idxs = [self.data_window.get_data_point(o_pt_idx) for o_pt_idx in o_pts_abs_inds_to_split]
        #
        #     all_pts = list(set(l_pt_idxs) | set(o_pt_idxs))
        #
        #     # Make 2 GMM using the labeled data points from each cluster
        #     gm = GaussianMixture(n_components=2).fit(all_pts)
        #
        #     # Predict where the unlabeled points belong based on the labeled data
        #     for o_pt_index in o_pt_idxs:
        #         o_pt =


    def relevance_processing(self, new_cluster_id):

        if self.REL_PROC_VAR == 0:
            pass
        elif self.REL_PROC_VAR == 1:
            current_rel_cluster = self.subspace_partition.cluster_dict[new_cluster_id]
            additional_cls_to_process = [current_rel_cluster]

            while len(additional_cls_to_process) > 0:
                current_rel_cluster = additional_cls_to_process.pop()
                if len(current_rel_cluster.o_pts) > 0:
                    # here we need to check whether the o_pt_idxs that have been assigned to this cluster are still part of the relevant class
                    current_cluster_label = current_rel_cluster.label
                    # currently this is a new cluster with only 1 l_pt
                    # so sort o_pt_idxs based on distance from that 1 pt, and query in that order, splitting when new label is encountered
                    o_pts_abs_inds = current_rel_cluster.o_pts
                    o_pts_data = []
                    dists = []
                    for i, o_pt_index in enumerate(o_pts_abs_inds):
                        o_pts_data.append(self.data_window.get_data_point(o_pt_index))
                        dists.append(np.linalg.norm(o_pts_data[i] - current_rel_cluster.l_pts[-1])) # gets dist from most recent l_pt in cluster

                    # sort everything by dists
                    sorted_triplets = sorted(zip(dists, o_pts_abs_inds, o_pts_data))

                    for dist, o_pt_abs_ind, o_pt_data in sorted_triplets:
                        new_pt_label, new_pt_relevance = self.query(o_pt_abs_ind)
                        current_rel_cluster.o_pts.remove(o_pt_abs_ind)
                        if new_pt_label == current_cluster_label:
                            # change o_pt to l_pt in current cluster
                            self.data_window.update_cluster_id_at(o_pt_abs_ind, current_rel_cluster.cluster_id)
                            self.data_window.updated_labeled_window(o_pt_abs_ind)
                            self.add_l_pt(o_pt_abs_ind,o_pt_data,current_rel_cluster.cluster_id)

                        else:
                            self.split(o_pt_data,o_pt_abs_ind,new_pt_label,new_pt_relevance, current_rel_cluster.cluster_id)
                            newest_cluster = self.subspace_partition.cluster_dict[-1]
                            self.data_window.update_cluster_id_at(o_pt_abs_ind, newest_cluster.cluster_id)
                            self.data_window.updated_labeled_window(o_pt_abs_ind)

                            if newest_cluster.relevance:
                                # add new cluster to additional_cls_to_process
                                additional_cls_to_process.append(newest_cluster)
                            # add split old (originally relevant) cluster to additional_cls_to_process
                            additional_cls_to_process.append(current_rel_cluster)

                            break


    # Removing forgotten o_pt_idxs from the subspace partition
    def subspace_partition_maintenance(self, forgotten_abs_idx, forgotten_point_cluster_id):

        cluster = self.subspace_partition.cluster_dict[forgotten_point_cluster_id]

        if 4 in self.verbose_flags:
            print(forgotten_abs_idx, forgotten_point_cluster_id)

        cluster.o_pts.remove(forgotten_abs_idx)

    def maintenance(self, data_point):
        if 3 in self.verbose_flags:
            print("labeled id array:", self.l_buf.cluster_id_array)
            print("labeled abs array:", self.l_buf.abs_idx_array)
            print("data window assigned id:", self.data_window.assigned_cluster_id_window.get_array())
        # THIS STUFF NEEDS TO GO IN A MAINTENANCE FUNCTION !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        is_forgotten_point_labeled = self.data_window.is_point_labeled_window.get(0) # the '0' here is the index of the oldest element in data window

        self.data_window.insert_data(data_point)
        data_point_abs_idx = self.data_window.abs_idx_max

        forgotten_abs_idx = self.data_window.abs_idx_min - 1
        forgotten_pt_cluster_id = self.data_window.last_removed_cluster_id

        # if forgotten_pt_cluster_id is NOT None (ie a point has been forgotten) do maintenance
        if forgotten_pt_cluster_id != None and not is_forgotten_point_labeled:
            self.subspace_partition_maintenance(forgotten_abs_idx, forgotten_pt_cluster_id)

    def process_point(self, data_point):
        # START MAINTENANCE
        self.maintenance(data_point)

        # START DETERMINE COMPARISON CLUSTER
        comp_cluster_id, distance = self.determine_comparison_cluster(data_point)
        comp_cluster_relevant = self.subspace_partition.cluster_dict[comp_cluster_id].relevance
        comp_cluster_label = self.subspace_partition.cluster_dict[comp_cluster_id].label

        is_anomalous = self.anomalous(data_point, comp_cluster_id, distance)

        data_point_abs_idx = self.num_pts_streamed

        # IGNORE O POINTS FOR NOW
        # if not comp_cluster_relevant and not is_anomalous:
        #     self.add_o_pt(data_point_abs_idx, comp_cluster_id)

        # else:

        if comp_cluster_relevant or is_anomalous:
            # Query!
            new_pt_label, new_pt_relevant = self.query(data_point_abs_idx)
            label_is_same = (new_pt_label == comp_cluster_label)
            if label_is_same:  # if not a new label
                self.add_l_pt(data_point_abs_idx, data_point, comp_cluster_id)
            else:
                self.split(data_point, data_point_abs_idx, new_pt_label, new_pt_relevant, comp_cluster_id)
                if new_pt_relevant:
                    self.relevance_processing(len(self.subspace_partition.cluster_dict) - 1)

        # POINT PROCESSED