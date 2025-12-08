from A_REDIN import ARED
from Circular_Buffer import Circular_Buffer
from DAGMM import DAGMM
import numpy as np
import torch
import threading


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
        dagmm_data_buffer_size=10000,
        dagmm_latent_dim=2,
        dagmm_n_components=3,
        lambda_energy=0.1,
        lambda_cov=0.005,
        dagmm_epochs=100,
        dagmm_batch_size=1024,
        dagmm_lr=1e-4,
        retrain_every_n_points=5000,
    ):
        # Buffers
        self.dagmm_data_buffer = Circular_Buffer(dagmm_data_buffer_size)
        self.pending_raw_points = []           # Points waiting for first model (in order!)

        # Current DAGMM model
        self.dagmm_model = None

        # ARED instance
        self.ared = ARED(
            oracle, kappa, l_buf_size, K_COMP_PTS,
            QS_VAR, REL_PROC_VAR, SM_VAR,
            NGHBHOOD_MERGE, SINGLETON_MERGE, VERBOSE_FLAGS
        )

        # Control flags
        self.first_model_ready = False
        self.first_point_processed = False
        self.min_points_to_train = dagmm_data_buffer_size // 2
        self.retrain_every_n_points = retrain_every_n_points
        self.points_since_last_training = 0

        # Hyperparameters
        self.dagmm_params = {
            "latent_dim": dagmm_latent_dim,
            "n_components": dagmm_n_components,
            "lambda_energy": lambda_energy,
            "lambda_cov": lambda_cov,
            "epochs": dagmm_epochs,
            "batch_size": dagmm_batch_size,
            "lr": dagmm_lr,
        }

        self.buffer_turnover_counter = 0

        print(f"[ARED_w_DAGMM] Initialized. Buffering points until {self.min_points_to_train} collected for first DAGMM training.")

    # ------------------------------------------------------------------ #
    def _transform_with_current_model(self, x):
        x_np = np.asarray(x, dtype=np.float32)
        x_tensor = torch.from_numpy(x_np).to(self.dagmm_model.device).unsqueeze(0)
        with torch.no_grad():
            self.dagmm_model.eval()
            z = self.dagmm_model.encode(x_tensor)
        return z.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ #
    def _train_dagmm_blocking(self):
        print("\n[DAGMM] Starting first synchronous training...")

        data = [pt for pt in self.dagmm_data_buffer.get_array() if pt is not None]
        X = np.stack(data)

        print(f"[DAGMM] Training on {X.shape[0]} points (dim={X.shape[1]})")

        model = DAGMM(
            input_dim=X.shape[1],
            latent_dim=self.dagmm_params["latent_dim"],
            n_components=self.dagmm_params["n_components"],
            lambda_energy=self.dagmm_params["lambda_energy"],
            lambda_cov=self.dagmm_params["lambda_cov"],
        )

        model.fit(
            X,
            epochs=self.dagmm_params["epochs"],
            batch_size=self.dagmm_params["batch_size"],
            lr=self.dagmm_params["lr"],
            verbose=True,
        )

        self.dagmm_model = model
        self.points_since_last_training = 0
        self.first_model_ready = True

        print(f"[DAGMM] First model trained! Latent dim = {self.dagmm_params['latent_dim']}")
        print(f"[ARED] Now catching up on {len(self.pending_raw_points)} buffered points...\n")

        # ------------------------------------------------------------------
        # Replay all buffered points in original order
        # ------------------------------------------------------------------
        for raw_pt in self.pending_raw_points:
            latent_pt = self._transform_with_current_model(raw_pt)
            if not self.first_point_processed:
                self.ared.process_first_point(latent_pt, raw_pt)
                self.first_point_processed = True
            else:
                self.ared.process_point(latent_pt, raw_pt)

        print(f"[ARED] Catch-up complete! {len(self.pending_raw_points)} points processed.")
        self.pending_raw_points.clear()  # Free memory
        self._reproject_ared_buffer_to_new_model()

    def _reproject_ared_buffer_to_new_model(self):
        """
        After a new DAGMM model is trained, re-encode ALL points currently stored
        in ARED's latent buffer using the NEW model, and invalidate all BallTrees.
        This keeps the entire latent space consistent.
        """
        l_buf = self.ared.l_buf  # This is the FiniteBuffer instance inside ARED

        if l_buf.data_circular_buffer.count == 0:
            print("[DAGMM] ARED buffer empty — nothing to reproject.")
            return

        print(f"[DAGMM] Reprojecting {l_buf.data_circular_buffer.count} points in ARED buffer to new latent space...")

        # Step 1: Collect raw high-dim points + their indices
        raw_points = []
        indices = []
        for i in range(l_buf.data_circular_buffer.count):
            raw_pt = l_buf.data_circular_buffer.get(i)  # ← this stores the ORIGINAL high-dim point
            if raw_pt is not None:
                raw_points.append(raw_pt)
                indices.append(i)

        if not raw_points:
            return

        # Step 2: Batch encode with the NEW DAGMM model
        raw_array = np.stack(raw_points)
        new_latents = self.dagmm_model.transform(raw_array)  # shape: (N, latent_dim)

        # Step 3: Overwrite the latent vectors in the correct buffer
        # In the original A-REDIN code, latent vectors are stored in:
        #   l_buf.dagmm_data_circular_buffer  ← this is the one!
        latent_buffer = l_buf.dagmm_data_circular_buffer

        with l_buf._tree_build_lock:  # Critical: prevent race with tree builder
            for idx, new_z in zip(indices, new_latents):
                latent_buffer.set_at(idx, new_z.astype(np.float32))

            # Invalidate all existing BallTrees — they are now garbage
            old_tree_count = len(l_buf.ball_trees)
            l_buf.ball_trees.clear()
            l_buf.num_ball_trees_completed = 0
            l_buf.build_up_period = True  # Force full rebuild

        print(f"[DAGMM] Reprojection complete: {len(new_latents)} points updated.")
        print(f"        Invalidated {old_tree_count} old BallTrees → will rebuild in new latent space.")

        # Optional: kick off immediate rebuild of first tree
        if not getattr(l_buf, '_building_tree', False):
            l_buf._building_tree = True
            threading.Thread(target=l_buf._build_new_tree, daemon=True).start()

    # ------------------------------------------------------------------ #
    def process_point(self, data_point):
        # Always store in circular buffer for future training
        self.dagmm_data_buffer.append(data_point)

        # If first model not ready → just buffer and wait
        if not self.first_model_ready:
            if self.dagmm_data_buffer.count >= self.min_points_to_train:
                # First time we have enough → train now
                if self.dagmm_model is None:
                    self._train_dagmm_blocking()
            else:
                # Still collecting → store in pending list
                self.pending_raw_points.append(data_point)
                print(f"[BUFFER] Stored point {len(self.pending_raw_points) + self.dagmm_data_buffer.count - len(self.pending_raw_points)}/{self.min_points_to_train}", end="\r")
            return None, 0

        # ——— After first model is ready ———
        self.points_since_last_training += 1

        # Periodic retraining
        # ——— After first model is ready ———
        # ——— Normal streaming after first model is ready ———
        self.points_since_last_training += 1
        self.buffer_turnover_counter += 1

        # Retrain whenever the circular buffer has turned over by ≥50%
        if self.buffer_turnover_counter >= self.min_points_to_train:
            print(
                f"\n[DAGMM] 50% buffer turnover detected ({self.buffer_turnover_counter}/{self.min_points_to_train} new points) → retraining")
            self._train_dagmm_blocking()
            self.buffer_turnover_counter = 0  # fresh start

        # Normal processing...
        latent_point = self._transform_with_current_model(data_point)

        if not self.first_point_processed:
            dist, searched = self.ared.process_first_point(latent_point, data_point)
            self.first_point_processed = True
        else:
            dist, searched = self.ared.process_point(latent_point, data_point)

        return dist, searched

    # ------------------------------------------------------------------ #
    def force_retrain(self):
        print("[DAGMM] Manual retraining forced...")
        self._train_dagmm_blocking()