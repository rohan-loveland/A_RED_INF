from A_REDIN import ARED
from Circular_Buffer import Circular_Buffer
from DAGMM import DAGMM
import threading
import numpy as np


class ARED_w_DAGMM:
    def __init__(
        self,
        oracle,
        kappa,
        l_buf_size,
        K_COMP_PTS,
        QS_VAR,
        REL_PROC_VAR,
        SM_VAR,
        NGHBHOOD_MERGE,
        SINGLETON_MERGE,
        VERBOSE_FLAGS,
        dagmm_data_buffer_size,
        dagmm_latent_dim=2,
        dagmm_n_components=3,
        lambda_energy=0.1,
        lambda_cov=0.005,
        dagmm_epochs=100,
        dagmm_batch_size=1024,
        dagmm_lr=1e-4,
    ):
        # DAGMM buffers & models
        self.dagmm_data_buffer = Circular_Buffer(dagmm_data_buffer_size)
        self._dagmm_data_buffer_size = dagmm_data_buffer_size

        self.dagmm_stale = None          # Model currently used for inference
        self.dagmm_fresh = None          # Newly trained model waiting to be swapped
        self.dagmm_training_thread = None
        self.dagmm_lock = threading.Lock()

        # --- Generation tracking: which points has the current stale model seen? ---
        self.dagmm_generation = 0                    # Increments on every successful swap
        self.current_stale_generation = -1           # Generation of the currently active stale model
        self.buffer_generation = [0] * dagmm_data_buffer_size  # Per-slot generation ID

        # ARED (operates in latent space)
        self.ared = ARED(
            oracle, kappa, l_buf_size, K_COMP_PTS,
            QS_VAR, REL_PROC_VAR, SM_VAR,
            NGHBHOOD_MERGE, SINGLETON_MERGE, VERBOSE_FLAGS
        )

        # Training triggers
        self.first_point_processed = False
        self.min_points_to_train = dagmm_data_buffer_size // 2
        self.unseen_fraction_threshold = 0.5   # Retrain when ≥50% of buffer is "new"

        # DAGMM hyperparameters
        self.dagmm_params = {
            "latent_dim": dagmm_latent_dim,
            "n_components": dagmm_n_components,
            "lambda_energy": lambda_energy,
            "lambda_cov": lambda_cov,
            "epochs": dagmm_epochs,
            "batch_size": dagmm_batch_size,
            "lr": dagmm_lr,
        }

    # ------------------------------------------------------------------ #
    # Transform using current best model (thread-safe)
    # ------------------------------------------------------------------ #
    def _transform_with_current_model(self, x):
        with self.dagmm_lock:
            model = self.dagmm_stale or self.dagmm_fresh
        if model is None:
            return x  # No model yet → pass raw data
        return model.encode(np.array([x]))[0]

    # ------------------------------------------------------------------ #
    # Background training thread
    # ------------------------------------------------------------------ #
    def _train_dagmm_async(self):
        try:
            # Collect all current (non-None) points
            data = [pt for pt in self.dagmm_data_buffer.get_array() if pt is not None]
            if len(data) < self.min_points_to_train:
                return

            X = np.stack(data)

            new_model = DAGMM(
                input_dim=X.shape[1],
                latent_dim=self.dagmm_params["latent_dim"],
                n_components=self.dagmm_params["n_components"],
                lambda_energy=self.dagmm_params["lambda_energy"],
                lambda_cov=self.dagmm_params["lambda_cov"],
            )
            new_model.fit(
                X,
                epochs=self.dagmm_params["epochs"],
                batch_size=self.dagmm_params["batch_size"],
                lr=self.dagmm_params["lr"],
                verbose=False,
            )

            # Atomically set fresh model
            with self.dagmm_lock:
                self.dagmm_fresh = new_model

        finally:
            # Always clean up the thread reference
            with self.dagmm_lock:
                self.dagmm_training_thread = None

    # ------------------------------------------------------------------ #
    # Swap fresh → stale and reproject ARED's buffer into new latent space
    # ------------------------------------------------------------------ #
    def _swap_and_reproject(self):
        """
        Atomically swap in the new DAGMM model and reproject ALL points in ARED's FiniteBuffer
        to the new latent space using the stored raw high-dimensional points.
        This ensures 100% consistent latent space — no drift, no mixing of old/new representations.
        """
        with self.dagmm_lock:
            if self.dagmm_fresh is None:
                return False

            # 1. Atomically swap models
            new_model = self.dagmm_fresh
            self.dagmm_stale = new_model
            self.dagmm_fresh = None

            # 2. Update generation tracking
            self.current_stale_generation = self.dagmm_generation
            self.dagmm_generation += 1

        # ------------------------------------------------------------------
        # 3. NOW SAFE TO REPROJECT — outside lock, no race with incoming points
        # ------------------------------------------------------------------
        print(f"[DAGMM] Swapping to new model (gen {self.current_stale_generation}) — reprojecting buffer...")

        # Grab raw high-dim points from FiniteBuffer
        raw_points = []
        indices_to_update = []
        l_buf = self.ared.l_buf

        for i in range(l_buf.data_circular_buffer.count):
            raw_pt = l_buf.data_circular_buffer.get(i)  # ← raw high-dim point
            if raw_pt is not None:
                raw_points.append(raw_pt)
                indices_to_update.append(i)

        if not raw_points:
            print("[DAGMM] No raw points to reproject (buffer empty)")
            return True

        # Batch encode with new DAGMM
        raw_array = np.stack(raw_points)
        try:
            new_latents = new_model.transform(raw_array)  # (N, latent_dim)
        except Exception as e:
            print(f"[DAGMM] Transform failed during reprojection: {e}")
            return False

        # ------------------------------------------------------------------
        # 4. Critical: Update latent buffer + invalidate/rebuild search structures
        # ------------------------------------------------------------------
        with l_buf._tree_build_lock:
            # Overwrite current latent representations
            for idx, new_z in zip(indices_to_update, new_latents):
                l_buf.dagmm_data_circular_buffer.set_at(idx, new_z)

            # Invalidate all existing BallTrees — they point to old latent space
            old_tree_count = len(l_buf.ball_trees)
            l_buf.ball_trees.clear()
            l_buf.num_ball_trees_completed = 0
            l_buf.build_up_period = True  # Force fresh rebuild

        # Rebuild the first new tree immediately
        if not l_buf._building_tree:
            l_buf._building_tree = True
            threading.Thread(target=l_buf._build_new_tree, daemon=True).start()

        print(f"[DAGMM] Reprojection complete: {len(new_latents)} points → new latent dim {new_latents.shape[1]}")
        print(f"        Invalidated {old_tree_count} old BallTrees → rebuilding fresh ones")

        return True

    # ------------------------------------------------------------------ #
    # Main streaming entry point
    # ------------------------------------------------------------------ #
    def process_point(self, data_point):
        # 1. Insert raw point into circular buffer and tag with current generation
        insert_idx = self.dagmm_data_buffer.append(data_point)
        if insert_idx is not None:  # Circular_Buffer should return the index written
            self.buffer_generation[insert_idx] = self.dagmm_generation

        # 2. First-time training trigger
        if self.dagmm_stale is None and self.dagmm_data_buffer.count >= self.min_points_to_train:
            if self.dagmm_training_thread is None or not self.dagmm_training_thread.is_alive():
                self.dagmm_training_thread = threading.Thread(target=self._train_dagmm_async, daemon=True)
                self.dagmm_training_thread.start()

        # 3. Swap in a freshly trained model if available
        if self.dagmm_fresh is not None:
            self._swap_and_reproject()

        # 4. Trigger retraining when enough *new* (unseen by current model) data arrives
        if self.dagmm_stale is not None:
            unseen_count = 0
            valid_count = 0
            for i in range(len(self.dagmm_data_buffer)):
                pt = self.dagmm_data_buffer.get(i)
                if pt is not None:
                    valid_count += 1
                    if self.buffer_generation[i] > self.current_stale_generation:
                        unseen_count += 1

            if valid_count > 0 and unseen_count / valid_count >= self.unseen_fraction_threshold:
                if self.dagmm_training_thread is None or not self.dagmm_training_thread.is_alive():
                    self.dagmm_training_thread = threading.Thread(target=self._train_dagmm_async, daemon=True)
                    self.dagmm_training_thread.start()

        # 5. Transform current point using best available model
        latent_point = self._transform_with_current_model(data_point)

        # 6. Feed to ARED
        if not self.first_point_processed:
            dist, num_pts_searched = self.ared.process_first_point(latent_point, data_point)
            self.first_point_processed = True
        else:
            dist, num_pts_searched = self.ared.process_point(latent_point, data_point)

        return dist, num_pts_searched

    # ------------------------------------------------------------------ #
    # Optional: force immediate (blocking) training
    # ------------------------------------------------------------------ #
    def train_dagmm(self):
        """Force synchronous training on current buffer contents."""
        self._train_dagmm_async()
        # Wait for completion if thread was spawned
        if self.dagmm_training_thread is not None:
            self.dagmm_training_thread.join()
        if self.dagmm_fresh is not None:
            self._swap_and_reproject()