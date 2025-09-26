from Circular_Buffer import Circular_Buffer
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
import threading
import numpy as np
import bisect

# READ AND ACT UPON.
# NOTE by Ro: There are 3 types of indexes here, and we should improve the names:
# indexes into circular buffer, ranging from 0 to size of circular buffer - 1
# henceforth "cb_idxs"
# indexes based on total number of l_pts, ever.  These are the "internal abs" at the moment
# henceforth "abs_l_idxs"
# indexes based on total number of l AND o_pts, ever.  These are "true abs" at the moment
# henceforth "abs_l_o_idxs"
# the max internal abs index should correspond to the max circ buff index
# at the moment, ball tree indexes in calls are ball tree indexes, stuff stored in class is l_idxs
# henceforth "bt_idxs"

class BallTreeWithIndexes(BallTree):
    def __init__(self, X, min_index, max_index, leaf_size = 40, metric: str | DistanceMetric = "minkowski"):
        super().__init__(X, leaf_size=leaf_size, metric=metric)
        self.max_index = max_index
        self.min_index = min_index

class FiniteBuffer:
    def __init__(self, buffer_size: int, ball_tree_ratio: float = 0.8, num_ball_trees: int = 2):
        self.data_circular_buffer = Circular_Buffer(buffer_size)
        self.label_circular_buffer = Circular_Buffer(buffer_size)
        self.relevance_circular_buffer = Circular_Buffer(buffer_size)
        self.cluster_key_circular_buffer = Circular_Buffer(buffer_size)
        self.true_abs_idx_circular_buffer = Circular_Buffer(buffer_size)

        self.buffer_size = buffer_size

        self.num_ball_trees = num_ball_trees
        self.ball_tree_interval = int(self.buffer_size * ball_tree_ratio)
        self.non_overlap_interval = np.ceil(self.buffer_size * (1 - ball_tree_ratio) / self.num_ball_trees)
        self.ball_trees = []

        self.num_ball_trees_completed = 0 # DEBUGGGG

        self.max_internal_abs_idx = -1
        self.min_internal_abs_idx = 0

        # --- threading-related attributes ---
        self._tree_build_lock = threading.Lock()
        self._building_tree = False  # flag so we don’t start multiple builds

        self.balling = False    # stays false until we complete the first ball tree

    def insert_pt(self, X, cluster_key, label, relevance, true_abs_idx):

        forgotten_pt_info = None
        build_ball_tree = False

        # Check if buffer is fulls
        if self.data_circular_buffer.is_full():
            forgotten_pt_info = self._forget_pt()
            self.min_internal_abs_idx += 1

        self.data_circular_buffer.append(X)
        self.label_circular_buffer.append(label)
        self.relevance_circular_buffer.append(relevance)
        self.cluster_key_circular_buffer.append(cluster_key)
        self.true_abs_idx_circular_buffer.append(true_abs_idx)
        self.max_internal_abs_idx += 1

        if len(self.ball_trees) != 0 and self.ball_trees[0].min_index < self.min_internal_abs_idx:
            self.ball_trees.pop(0)

            # if we have a btree     and # ball trees < max # btree                 and max_abs_idx is greater than max index of newest btree + non overlap interval
        if len(self.ball_trees) != 0 and len(self.ball_trees) < self.num_ball_trees and self.max_internal_abs_idx >= self.ball_trees[-1].max_index + self.non_overlap_interval:
            build_ball_tree = True
             # If we have no btrees    and we have as many points as the btree interval
        elif len(self.ball_trees) == 0 and self.ball_tree_interval <= self.max_internal_abs_idx:
            build_ball_tree = True

        # if ball tree is invalid, forget it and start building new tree
        if build_ball_tree and not self._building_tree:
            self._building_tree = True
            threading.Thread(target=self._build_new_tree, daemon=True).start()

        return forgotten_pt_info # (key, relevance, internal_abs_index, true_abs_idx)

    '''
    returns the information about the oldest remembered streamed point
    '''
    def _forget_pt(self):
        forgotten_pt_cluster_key = self.cluster_key_circular_buffer.get(0)
        forgotten_pt_relevance = self.relevance_circular_buffer.get(0)
        forgotten_pt_index = self.min_internal_abs_idx
        forgotten_pt_true_abs_idx = self.true_abs_idx_circular_buffer.get(0)

        return forgotten_pt_cluster_key, forgotten_pt_index, forgotten_pt_relevance, forgotten_pt_true_abs_idx

    # temporary function to check ball tree
    def find_closest_pts_all_brute(self, X, k):

        all_pts = []

        closest_pts = []
        if self.data_circular_buffer.is_full():
            num_pts = 1000
        else:
            num_pts = self.data_circular_buffer.count

        for i in range(num_pts):
            all_pts.append(self.data_circular_buffer.get(i))
            dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
            distances = [d[2] for d in closest_pts]
            pos = bisect.bisect_left(distances, dist)

            if pos < k:
                closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                             i + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(i),
                                             self.data_circular_buffer.get(i),
                                             self.relevance_circular_buffer.get(i),
                                             self.true_abs_idx_circular_buffer.get(i)))
                if len(closest_pts) > k:
                    closest_pts.pop()
            # if pos == 1:
            #     print(i)
            num_pts_searched = (self.max_internal_abs_idx - self.min_internal_abs_idx + 1,0,0)

        all_pts_arr = np.array(all_pts)

        return closest_pts, num_pts_searched, all_pts_arr

    def find_closest_pts(self, X, k):
        #DEBUG ONLY
        all_pts = []

        closest_pts = []

        if len(self.ball_trees) != 0:
            self.balling = True # we've now completed at least 1 ball tree

            # brute force tail end, search ball trees, brute force head end
            min_l_idx_covered_by_btree = self.ball_trees[0].min_index
            max_l_idx_covered_by_btree = self.ball_trees[-1].max_index
            min_cb_index_covered_by_btree = min_l_idx_covered_by_btree - self.min_internal_abs_idx
            max_cb_index_covered_by_btree = max_l_idx_covered_by_btree - self.min_internal_abs_idx

            # brute force tail end
            for cb_i in range(0,min_cb_index_covered_by_btree):

                all_pts.append(self.data_circular_buffer.get(cb_i))

                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_i))
                # distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                distances = [d[2] for d in closest_pts]

                pos = bisect.bisect_left(distances, dist)
                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_i),
                                             cb_i + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(cb_i),
                                             self.data_circular_buffer.get(cb_i),
                                             self.relevance_circular_buffer.get(cb_i),
                                             self.true_abs_idx_circular_buffer.get(cb_i)))
                    if len(closest_pts) > k:
                        closest_pts.pop()

            # search ball trees
            for ball_tree in self.ball_trees:

                ball_tree_pts = list(ball_tree.get_arrays()[0])
                all_pts = all_pts + ball_tree_pts

                X = X.reshape((1,-1))
                dists, bt_idxs = ball_tree.query(X, k) # returned value is dist, index rel. to ball tree 0
                # for i in range(len(dist)): # len(dist) == 1 since it's a 2D array!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # renamed from 'dist' to 'dists' also
                for j in range(dists.shape[1]):
                    l_idx = bt_idxs[0][j] + ball_tree.min_index # now index rel. to l_pts
                    cb_idx = l_idx - self.min_internal_abs_idx
                    dist = dists[0][j]

                    # distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                    distances = [d[2] for d in closest_pts]
                    pos = bisect.bisect_left(distances, dist)

                    if pos < k:
                        closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_idx),
                                                 l_idx,
                                                 dist,
                                                 self.label_circular_buffer.get(cb_idx),
                                                 self.data_circular_buffer.get(cb_idx),
                                                 self.relevance_circular_buffer.get(cb_idx),
                                                 self.true_abs_idx_circular_buffer.get(cb_idx)))

                        if len(closest_pts) > k:
                            closest_pts.pop()

            # brute force head end
            num_head_pts = self.max_internal_abs_idx - max_l_idx_covered_by_btree

            # for i in range(self.max_internal_abs_idx - max_idx_covered_by_btree):
            #     c_buf_idx = i + max_idx_covered_by_btree - 1
            for cb_idx in range(max_cb_index_covered_by_btree + 1, max_cb_index_covered_by_btree + num_head_pts):

                all_pts.append(self.data_circular_buffer.get(cb_idx))

                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_idx))

                distances = [d[2] for d in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_idx),
                                             cb_idx + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(cb_idx),
                                             self.data_circular_buffer.get(cb_idx),
                                             self.relevance_circular_buffer.get(cb_idx),
                                             self.true_abs_idx_circular_buffer.get(cb_idx)))
                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_tail_pts = min_cb_index_covered_by_btree
            num_ball_trees_pts = self.ball_trees[-1].max_index - self.ball_trees[0].min_index + 1
            num_head_pts = num_head_pts
            num_pts_searched = (num_tail_pts, num_ball_trees_pts, num_head_pts)

            all_pts_arr = np.array(all_pts)
            num_unique_pts_searched = len(np.unique(all_pts_arr,axis = 0))
            # print(f"total num unique points: {len(np.unique(all_pts_arr,axis = 0))}")

        else: # brute force all points
            for i in range(self.max_internal_abs_idx - self.min_internal_abs_idx + 1):



                dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
                distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                                 i + self.min_internal_abs_idx,
                                                 dist,
                                                 self.label_circular_buffer.get(i),
                                                 self.data_circular_buffer.get(i),
                                                 self.relevance_circular_buffer.get(i),
                                                 self.true_abs_idx_circular_buffer.get(i)))

                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_pts_searched = (self.max_internal_abs_idx - self.min_internal_abs_idx + 1,0,0)

        return closest_pts, num_pts_searched

    def find_closest_pts_debug(self, X, k,closest_data_pt):
        #DEBUG ONLY
        all_pts = []
        closest_pts = []

        if len(self.ball_trees) != 0:
            self.balling = True # we've now completed at least 1 ball tree

            # brute force tail end, search ball trees, brute force head end
            min_l_idx_covered_by_btree = self.ball_trees[0].min_index
            max_l_idx_covered_by_btree = self.ball_trees[-1].max_index
            min_cb_index_covered_by_btree = min_l_idx_covered_by_btree - self.min_internal_abs_idx
            max_cb_index_covered_by_btree = max_l_idx_covered_by_btree - self.min_internal_abs_idx

            # brute force tail end
            for cb_i in range(0,min_cb_index_covered_by_btree):

                all_pts.append(self.data_circular_buffer.get(cb_i))

                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_i))
                # distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                distances = [d[2] for d in closest_pts]

                pos = bisect.bisect_left(distances, dist)
                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_i),
                                             cb_i + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(cb_i),
                                             self.data_circular_buffer.get(cb_i),
                                             self.relevance_circular_buffer.get(cb_i),
                                             self.true_abs_idx_circular_buffer.get(cb_i)))
                    if len(closest_pts) > k:
                        closest_pts.pop()

            # search ball trees
            for ball_tree in self.ball_trees:

                ball_tree_pts = list(ball_tree.get_arrays()[0])
                all_pts = all_pts + ball_tree_pts

                X = X.reshape((1,-1))
                dists, bt_idxs = ball_tree.query(X, k) # returned value is dist, index rel. to ball tree 0
                # for i in range(len(dist)): # len(dist) == 1 since it's a 2D array!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # renamed from 'dist' to 'dists' also
                for j in range(dists.shape[1]):
                    l_idx = bt_idxs[0][j] + ball_tree.min_index # now index rel. to l_pts
                    cb_idx = l_idx - self.min_internal_abs_idx
                    dist = dists[0][j]

                    # distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                    distances = [d[2] for d in closest_pts]
                    pos = bisect.bisect_left(distances, dist)

                    if pos < k:
                        closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_idx),
                                                 l_idx,
                                                 dist,
                                                 self.label_circular_buffer.get(cb_idx),
                                                 self.data_circular_buffer.get(cb_idx),
                                                 self.relevance_circular_buffer.get(cb_idx),
                                                 self.true_abs_idx_circular_buffer.get(cb_idx)))

                        if len(closest_pts) > k:
                            closest_pts.pop()

            # brute force head end
            num_head_pts = self.max_internal_abs_idx - max_l_idx_covered_by_btree

            # for i in range(self.max_internal_abs_idx - max_idx_covered_by_btree):
            #     c_buf_idx = i + max_idx_covered_by_btree - 1
            for cb_idx in range(max_cb_index_covered_by_btree + 1, max_cb_index_covered_by_btree + num_head_pts):

                all_pts.append(self.data_circular_buffer.get(cb_idx))

                dist = np.linalg.norm(X - self.data_circular_buffer.get(cb_idx))

                distances = [d[2] for d in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(cb_idx),
                                             cb_idx + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(cb_idx),
                                             self.data_circular_buffer.get(cb_idx),
                                             self.relevance_circular_buffer.get(cb_idx),
                                             self.true_abs_idx_circular_buffer.get(cb_idx)))
                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_tail_pts = min_cb_index_covered_by_btree
            num_ball_trees_pts = self.ball_trees[-1].max_index - self.ball_trees[0].min_index + 1
            num_head_pts = num_head_pts
            num_pts_searched = (num_tail_pts, num_ball_trees_pts, num_head_pts)

            all_pts_arr = np.array(all_pts)
            num_unique_pts_searched = len(np.unique(all_pts_arr,axis = 0))
            match_flag = False
            for n in range(len(all_pts_arr)):
                if np.array_equal(all_pts_arr[n, :], closest_data_pt):
                    match_flag = True
            if match_flag:
                print("there was a matching point in all_pts_arr")
            else:
                print(f"NO MATCHING POINT in all_pts_arr, # of ball trees = {len(self.ball_trees)}")
            print(f"total num unique points: {len(np.unique(all_pts_arr,axis = 0))}")

        else: # brute force all points

            for i in range(self.max_internal_abs_idx - self.min_internal_abs_idx + 1):

                all_pts.append(self.data_circular_buffer.get(i))

                dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
                distances = [d for _, __, d, ___, ____, _____, ______ in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if pos < k:
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                                 i + self.min_internal_abs_idx,
                                                 dist,
                                                 self.label_circular_buffer.get(i),
                                                 self.data_circular_buffer.get(i),
                                                 self.relevance_circular_buffer.get(i),
                                                 self.true_abs_idx_circular_buffer.get(i)))

                    if len(closest_pts) > k:
                        closest_pts.pop()

            num_pts_searched = (self.max_internal_abs_idx - self.min_internal_abs_idx + 1,0,0)

        all_pts_arr = np.array(all_pts)
        return closest_pts, num_pts_searched, all_pts_arr



    def get_pt_data(self, internal_abs_idx):
        if self.min_internal_abs_idx <= internal_abs_idx < self.max_internal_abs_idx:
            return self.data_circular_buffer.get(internal_abs_idx - self.min_internal_abs_idx)

        return None

    def get_pt_data_abs(self, true_abs_idx):
        internal_abs_idx = self.true_abs_idx_circular_buffer.get_array().index(true_abs_idx)
        return self.data_circular_buffer.get(internal_abs_idx)

    def _build_new_tree(self):
        """
        Take a snapshot of the latest window of points and build
        a BallTree covering absolute indices [abs_min, abs_max).
        Runs in its own thread.
        """
        try:
            # === 1. Atomic snapshot of current state ===
            with self._tree_build_lock:
                min_abs_snapshot = self.min_internal_abs_idx
                max_abs_snapshot = self.max_internal_abs_idx
                full_array = list(self.data_circular_buffer.get_array())

            # === 2. Compute absolute window bounds ===
            window_size = self.ball_tree_interval
            abs_min = max(min_abs_snapshot, max_abs_snapshot - window_size) # CEHCK HERE
            abs_max = max_abs_snapshot  # exclusive upper bound

            # === 3. Convert to relative slice positions ===
            rel_start = abs_min - min_abs_snapshot
            rel_end = abs_max - min_abs_snapshot

            # === 4. Extract the snapshot window ===
            data_snapshot = full_array[rel_start:rel_end]

            if not data_snapshot:
                return

            # === 5. Build the BallTree OFF the lock ===
            new_tree = BallTreeWithIndexes(data_snapshot, min_index=abs_min, max_index=abs_max, leaf_size=2)

            # === 6. Append finished tree under lock ===
            with self._tree_build_lock:
                self.ball_trees.append(new_tree)

        finally:
            self._building_tree = False
            self.num_ball_trees_completed += 1

            if self.num_ball_trees_completed < self.num_ball_trees:
                print("BUILD UP PERIOD: ", len(self.ball_trees), self.max_internal_abs_idx, self.true_abs_idx_circular_buffer.get(self.max_internal_abs_idx))

            if self.num_ball_trees_completed == self.num_ball_trees:
                print("STEADY STATE REACHED: ", len(self.ball_trees), self.max_internal_abs_idx, self.true_abs_idx_circular_buffer.get(self.max_internal_abs_idx))
