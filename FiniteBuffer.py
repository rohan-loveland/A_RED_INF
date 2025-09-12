from Circular_Buffer import Circular_Buffer
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric
import threading
import heapq
import numpy as np

class BallTreeWithIndexes(BallTree):
    def __init__(self, X, min_index, max_index, leaf_size = 40, metric: str = "minkowski" | DistanceMetric):
        super().__init__(X, leaf_size, metric)
        self.max_index = max_index
        self.min_index = min_index

class FiniteBuffer:
    def __init__(self, buffer_size: int, ball_tree_ratio: float = 0.5, num_ball_trees: int = 2):
        self.data_circular_buffer = Circular_Buffer(buffer_size)
        self.label_circular_buffer = Circular_Buffer(buffer_size)
        self.cluster_id_circular_buffer = Circular_Buffer(buffer_size)
        self.relevance_circular_buffer = Circular_Buffer(buffer_size)

        self.buffer_size = buffer_size

        self.num_ball_trees = num_ball_trees
        self.ball_tree_ratio = ball_tree_ratio
        self.ball_trees = []

        self.max_abs_idx = 0
        self.min_abs_idx = 0

        self.dist = DistanceMetric.get_metric('minkowski')

        # --- threading-related attributes ---
        self._tree_build_lock = threading.Lock()
        self._building_tree = False  # flag so we don’t start multiple builds

    def insert_pt(self, X, label, cluster_id, relevance):
        forgotten_pt_info = None
        # Check if buffer is fulls
        if self.data_circular_buffer.is_full():
            forgotten_pt_info = self._forget_pt()

        self.data_circular_buffer.append(X)
        self.label_circular_buffer.append(label)
        self.cluster_id_circular_buffer.append(cluster_id)
        self.relevance_circular_buffer.append(relevance)

        if len(self.ball_trees) != 0 and self.ball_trees[0].min_index < self.min_abs_idx:
            self.ball_trees.pop(0)
            ball_tree_forgotten = True

        elif len(self.ball_trees) == 0:
            # determine whether we should build the first ball tree.

        # if ball tree is invalid, forget it and start building new tree
        if ball_tree_forgotten and self._building_tree == False:
            # start building new ball tree on separate thread
            self._building_tree = True
            self._build_new_tree()

        self.max_abs_idx += 1
        self.min_abs_idx += 1

        return forgotten_pt_info # (id, relevance, abs_index)

    '''
    returns the information about the oldest remembered streamed point
    '''
    def _forget_pt(self):
        forgotten_pt_cluster_id = self.cluster_id_circular_buffer.get(0)
        forgotten_pt_relevance = self.relevance_circular_buffer.get(0)
        forgotten_pt_index = self.min_abs_idx
        return (forgotten_pt_cluster_id, forgotten_pt_relevance, forgotten_pt_index)

    def find_closest_pts(self, X, k):
        closest_pt_in_clusters = {}

        if len(self.ball_trees) != 0:
            # brute force tail end, search ball trees, brute force head end
            min_idx_covered_by_btree = self.ball_trees[0].min_index
            max_idx_covered_by_btree = self.ball_trees[-1].max_index

            # brute force tail end
            for i in range (min_idx_covered_by_btree - self.min_abs_idx):
                cluster_id = self.cluster_id_circular_buffer.get(i)
                dist = self.dist(X, self.data_circular_buffer.get(i))

                if cluster_id not in closest_pt_in_clusters or dist < closest_pt_in_clusters[cluster_id][0]:
                    closest_pt_in_clusters[cluster_id] = (dist, i + self.min_abs_idx)

            # search ball trees
            for ball_tree in self.ball_trees:

                dist, idx = ball_tree.query(X) # returned value is dist, index
                cluster_id = self.cluster_id_circular_buffer.get(self.min_index + idx)

                if cluster_id not in closest_pt_in_clusters or dist < closest_pt_in_clusters[cluster_id][0]:
                    closest_pt_in_clusters[cluster_id] = (dist, i + ball_tree.min_index)


            # brute force head end
            for i in range(self.max_abs_idx - max_idx_covered_by_btree):
                cluster_id = self.cluster_id_circular_buffer.get(i)
                dist = self.dist(X, self.data_circular_buffer.get(i))

                if cluster_id not in closest_pt_in_clusters or dist < closest_pt_in_clusters[cluster_id][0]:
                    closest_pt_in_clusters[cluster_id] = (dist, i + self.ball_trees[-1].min_index)

        else:
            # brute force all points
            for i in range(self.max_abs_idx - self.min_abs_idx + 1):
                cluster_id = self.cluster_id_circular_buffer.get(i)
                dist = self.dist(X, self.data_circular_buffer.get(i))

                if cluster_id not in closest_pt_in_clusters or dist < closest_pt_in_clusters[cluster_id][0]:
                    closest_pt_in_clusters[cluster_id] = (dist, i + self.min_abs_idx)

        # Get k clusters with smallest distances
        closest_k = heapq.nsmallest(k, closest_pt_in_clusters.items(), key=lambda x: x[1][0])

        # Return in format: list of (cluster_id, min_dist, data_idx)
        return [(cluster_id, min_dist, data_idx) for cluster_id, (min_dist, data_idx) in closest_k]


    def get_pt_data(self, abs_idx):
        if abs_idx >= self.min_abs_idx and abs_idx < self.max_abs_idx:
            return self.data_circular_buffer.get(abs_idx-self.min_abs_idx)

        return None

    def _build_new_tree(self):
        """
                Build a new tree from current buffer snapshot.
                Runs in a background thread.
                """
        try:
            # snapshot data to avoid locking for long periods
            with self._tree_build_lock:
                window_size = int(self.buffer_size * self.ball_tree_ratio)
                min_idx = self.max_abs_idx - window_size
                max_idx = self.max_abs_idx
                data_snapshot = list(self.data_circular_buffer.get_array()[min_idx:max_idx])

            if data_snapshot:
                new_tree = BallTreeWithIndexes(data_snapshot, min_idx, max_idx, leaf_size=2)
                # insert new tree
                with self._tree_build_lock:
                    self.ball_trees.append(new_tree)

        finally:
            self._building_tree = False


