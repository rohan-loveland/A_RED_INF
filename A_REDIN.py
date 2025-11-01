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
    def __init__(self, label, relevance, l_pt_idxs, l_buf, cluster_key, QS_VAR=0):
        self.label = label
        self.relevance = relevance
        self.l_pt_idxs = l_pt_idxs
        self.comp_distance = 0
        self.cluster_id = cluster_key
        self.last_updated = 0  # <<< FIX >>> For aging (optional future use)

        if len(l_pt_idxs) > 1 and QS_VAR == 0:
            self.update_diameter_all(l_buf)
        elif len(l_pt_idxs) > 1 and QS_VAR == 1:
            self.update_ave_nn_dist_all(l_buf)

    def touch(self):
        import time
        self.last_updated = time.time()  # <<< FIX >>> Track activity

    def add_l_pt(self, abs_idx, l_buf, QS_VAR=0):
        self.l_pt_idxs.append(abs_idx)
        self.touch()  # <<< FIX >>> Mark as active
        if QS_VAR == 0:
            self.update_diameter_single(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_single(l_buf)

    def add_l_pt_no_comp_dist_update(self, abs_idx):
        self.l_pt_idxs.append(abs_idx)
        self.touch()  # <<< FIX >>>

    def remove_l_pt_from_cluster(self, pt_abs_idx):
        self.l_pt_idxs.remove(pt_abs_idx)

    def update_comp_distance_single(self, l_buf, QS_VAR=0):
        if QS_VAR == 0:
            self.update_diameter_single(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_single(l_buf)

    def merge_comp_distances(self, l_buf, QS_VAR):
        if QS_VAR == 0:
            self.update_diameter_all(l_buf)
        elif QS_VAR == 1:
            self.update_ave_nn_dist_all(l_buf)

    def update_diameter_single(self, l_buf):
        latest_idx = self.l_pt_idxs[-1]
        latest_pt = l_buf.get_pt_data_abs(latest_idx)
        max_dist = 0
        for idx in self.l_pt_idxs[:-1]:
            dist = np.linalg.norm(l_buf.get_pt_data_abs(idx) - latest_pt)
            if dist > max_dist:
                max_dist = dist
        if max_dist > self.comp_distance:
            self.comp_distance = max_dist

    def update_diameter_all(self, l_buf):
        max_dist = 0
        for i in range(len(self.l_pt_idxs)):
            p1 = l_buf.get_pt_data_abs(self.l_pt_idxs[i])
            for j in range(i + 1, len(self.l_pt_idxs)):
                p2 = l_buf.get_pt_data_abs(self.l_pt_idxs[j])
                dist = np.linalg.norm(p1 - p2)
                if dist > max_dist:
                    max_dist = dist
        self.comp_distance = max_dist

    def update_ave_nn_dist_single(self, l_buf):
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return
        latest_idx = self.l_pt_idxs[-1]
        latest_pt = l_buf.get_pt_data_abs(latest_idx)
        min_dist = np.inf
        for idx in self.l_pt_idxs[:-1]:
            dist = np.linalg.norm(l_buf.get_pt_data_abs(idx) - latest_pt)
            if dist < min_dist:
                min_dist = dist
        n = len(self.l_pt_idxs)
        self.comp_distance = (self.comp_distance * (n - 1) + min_dist) / n

    def update_ave_nn_dist_all(self, l_buf):
        if len(self.l_pt_idxs) < 2:
            self.comp_distance = 0.0
            return
        data = np.array([l_buf.get_pt_data_abs(i) for i in self.l_pt_idxs])
        nn = NearestNeighbors(n_neighbors=2).fit(data)
        distances, _ = nn.kneighbors(data)
        self.comp_distance = np.mean(distances[:, 1])

    # AI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------
    # Inside class Cluster (unchanged except for the two helpers below)
    # ------------------------------------------------------------------
    def _diameter_of_union(self, l_buf, other_idxs):
        """Return the exact diameter of *self + other_idxs* without modifying self."""
        pts = [l_buf.get_pt_data_abs(i) for i in self.l_pt_idxs + other_idxs]
        max_d = 0.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d = np.linalg.norm(pts[a] - pts[b])
                if d > max_d: max_d = d
        return max_d

    def _avg_nn_of_union(self, l_buf, other_idxs):
        """Return the exact avg-NN distance of *self + other_idxs*."""
        pts = np.vstack([l_buf.get_pt_data_abs(i) for i in self.l_pt_idxs + other_idxs])
        if pts.shape[0] < 2:
            return 0.0
        nn = NearestNeighbors(n_neighbors=2).fit(pts)
        distances, _ = nn.kneighbors(pts)
        return np.mean(distances[:, 1])

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
                    # break  # Stop after first merge to avoid merging same singleton multiple times

    def merge_clusters(self, keep_key, merge_key):
        if keep_key == merge_key:
            return

        if 5 in self.verbose_flags:
            print("Merging clusters:", keep_key, merge_key)

        keep_cl = self.subspace_partition.cluster_dict[keep_key]
        merge_cl = self.subspace_partition.cluster_dict[merge_key]

        # 1. move points
        keep_cl.l_pt_idxs.extend(merge_cl.l_pt_idxs)

        # 2. fix circular-buffer cluster keys — ONLY VALID INDICES
        cb = self.l_buf.cluster_key_circular_buffer
        for i in range(cb.count):          # ← Fix 2
            if cb.get(i) == merge_key:
                cb.set_at(i, keep_key)

        # 3. delete the merged cluster
        del self.subspace_partition.cluster_dict[merge_key]

        # 4. UPDATE METRIC
        if self.QS_VAR == 0:  # DIAMETER
            max_cross = 0.0
            for k_idx in keep_cl.l_pt_idxs:
                k_pt = self.l_buf.get_pt_data_abs(k_idx)
                if k_pt is None: continue    # ← Fix 3
                for m_idx in merge_cl.l_pt_idxs:
                    m_pt = self.l_buf.get_pt_data_abs(m_idx)
                    if m_pt is None: continue
                    d = np.linalg.norm(k_pt - m_pt)
                    if d > max_cross: max_cross = d
            keep_cl.comp_distance = max(keep_cl.comp_distance,
                                        merge_cl.comp_distance,
                                        max_cross)
        else:  # AVG NN
            keep_cl.comp_distance = keep_cl._avg_nn_of_union(
                self.l_buf, merge_cl.l_pt_idxs)


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

        this_cluster.update_comp_distance_single(self.l_buf, self.QS_VAR)


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
