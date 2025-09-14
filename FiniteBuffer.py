from Circular_Buffer import Circular_Buffer
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
import threading
import numpy as np
import bisect

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
        self.non_overlap_interval = self.buffer_size * (1 - ball_tree_ratio) / self.num_ball_trees
        self.ball_trees = []

        self.max_internal_abs_idx = -1
        self.min_internal_abs_idx = 0

        # --- threading-related attributes ---
        self._tree_build_lock = threading.Lock()
        self._building_tree = False  # flag so we don’t start multiple builds

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

    def find_closest_pts(self, X, k):
        closest_pts = []

        if len(self.ball_trees) != 0:
            # brute force tail end, search ball trees, brute force head end
            min_idx_covered_by_btree = self.ball_trees[0].min_index
            max_idx_covered_by_btree = self.ball_trees[-1].max_index

            # brute force tail end
            for i in range (min_idx_covered_by_btree - self.min_internal_abs_idx):
                dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
                distances = [d for _, d, __, ___ in closest_pts]

                if len(closest_pts) < k:

                    pos = bisect.bisect_left(distances, dist)
                    # just insert at the right position
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                             self.min_internal_abs_idx + i,
                                             dist,
                                             self.label_circular_buffer.get(i),
                                             self.data_circular_buffer.get(i),
                                             self.relevance_circular_buffer.get(i),
                                             self.true_abs_idx_circular_buffer.get(i)))
                elif dist < closest_pts[-1]:

                    pos = bisect.bisect_left(distances, dist)
                    # insert and drop the farthest (last) element
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                             self.min_internal_abs_idx + i,
                                             dist,
                                             self.label_circular_buffer.get(i),
                                             self.data_circular_buffer.get(i),
                                             self.relevance_circular_buffer.get(i),
                                             self.true_abs_idx_circular_buffer.get(i)))
                    closest_pts.pop()

            # search ball trees
            for ball_tree in self.ball_trees:

                dist, idx = ball_tree.query(X) # returned value is dist, index
                idx = idx + ball_tree.min_index

                distances = [d for _, d, __, ___ in closest_pts]
                pos = bisect.bisect_left(distances, dist)
                if len(closest_pts) < k:

                    # just insert at the right position
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             ball_tree.min_index + idx,
                                             dist,
                                             self.label_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.data_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.relevance_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.true_abs_idx_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx)))

                elif dist < closest_pts[-1][1]:

                    # insert and drop the farthest (last) element
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             ball_tree.min_index + idx,
                                             dist,
                                             self.label_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.data_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.cluster_key_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.relevance_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx),
                                             self.true_abs_idx_circular_buffer.get(ball_tree.min_index + idx - self.min_internal_abs_idx)))
                    closest_pts.pop()


            # brute force head end
            for i in range(self.max_internal_abs_idx - max_idx_covered_by_btree):
                i = i + 1 # adjust index to not include data[max_inx_covered_by_btree] from bein used (it had it's chance in the ball tree), and to include data[max_abs_idx] *since max_abs_idx is a valid index*
                dist = np.linalg.norm(X - self.data_circular_buffer.get(i + max_idx_covered_by_btree))

                distances = [d for _, d, __, ___ in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if len(closest_pts) < k:

                    # just insert at the right position
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             max_idx_covered_by_btree + i,
                                             dist,
                                             self.label_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.data_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.relevance_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.true_abs_idx_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx)))

                elif dist < closest_pts[-1][1]:

                    # insert and drop the farthest (last) element
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             max_idx_covered_by_btree + i,
                                             dist,
                                             self.label_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.data_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.relevance_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx),
                                             self.true_abs_idx_circular_buffer.get(i + max_idx_covered_by_btree - self.min_internal_abs_idx)))
                    closest_pts.pop()

        else:
            # brute force all points
            for i in range(self.max_internal_abs_idx - self.min_internal_abs_idx + 1): #NOTE THIS SHOULD JUST BE from 0 to max_abs_idx (inclusive) since if min_abs_idx != 0 then we should have ball tree/s

                dist = np.linalg.norm(X - self.data_circular_buffer.get(i))
                distances = [d for _, d, __, ___ in closest_pts]
                pos = bisect.bisect_left(distances, dist)

                if len(closest_pts) < k:

                    # just insert at the right position
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                             i + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(i),
                                             self.data_circular_buffer.get(i),
                                             self.relevance_circular_buffer.get(i),
                                             self.true_abs_idx_circular_buffer.get(i)))

                elif dist < closest_pts[-1][1]:

                    # insert and drop the farthest (last) element
                    closest_pts.insert(pos, (self.cluster_key_circular_buffer.get(i),
                                             i + self.min_internal_abs_idx,
                                             dist,
                                             self.label_circular_buffer.get(i),
                                             self.data_circular_buffer.get(i),
                                             self.relevance_circular_buffer.get(i),
                                             self.true_abs_idx_circular_buffer.get(i)))
                    closest_pts.pop()

        # Return in format: list of (cluster_key, pt_abs_idx, dist, label, data, rel)
        return closest_pts

    def get_pt_data(self, internal_abs_idx):
        if self.min_internal_abs_idx <= internal_abs_idx < self.max_internal_abs_idx:
            return self.data_circular_buffer.get(internal_abs_idx - self.min_internal_abs_idx)

        return None

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
            abs_min = max(min_abs_snapshot, max_abs_snapshot - window_size)
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