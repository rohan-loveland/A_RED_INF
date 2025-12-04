# train_parking_dagmm_best.py
# Fixed version – works with scikit-learn 1.2+ (max_iter instead of n_iter)

from DAGMM import DAGMM
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from umap import UMAP

warnings.filterwarnings('ignore')

# -------------------------- Config --------------------------
N_REL_CLASSES = 8
RANDOM_SEED = 42
MAX_EPOCHS = 300
PATIENCE = 150
BATCH_SIZE = 512
N_COMPONENTS = 12
LATENT_DIM = 64
PCA_COMPS = 1024 # DON'T USE

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------- 1. Load Data --------------------------
from main_helper_functions import parking_lot_setup_for_main

print(f"Loading Parking Lot data with N_REL_CLASSES = {N_REL_CLASSES}")
X_raw, y_w_rel, sparsity_levels, relevant_labels = parking_lot_setup_for_main(
    N_REL_CLASSES=N_REL_CLASSES,
    VERBOSE_FLAGS=[0],
    seed=RANDOM_SEED
)


# UNNECESSARY?
# # Flatten images
# X_flat = X_raw.reshape(len(X_raw), -1).astype(np.float32)
# print(f"Original shape: {X_flat.shape}")
X_flat = X_raw

# -------------------------- PCA (Smart & Safe) --------------------------
PCA_COMPS = 512
print(f"Applying PCA(n_components={PCA_COMPS}, whiten=False)...")

from sklearn.decomposition import PCA
pca = PCA(n_components=PCA_COMPS, random_state=RANDOM_SEED, svd_solver='full')
X_pca = pca.fit_transform(X_flat)

explained_variance = pca.explained_variance_ratio_.sum()
print(f"PCA retained {explained_variance*100:.2f}% of total variance with {PCA_COMPS} components")

if explained_variance < 0.90:
    print(f"   WARNING: Only {explained_variance*100:.1f}% variance retained! Consider increasing PCA_COMPS.")

# Scale each component to [-1, 1]
X_min = X_pca.min(axis=0)
X_max = X_pca.max(axis=0)
X_full = (X_pca - X_min) / (X_max - X_min + 1e-8)
X_full = X_full * 2.0 - 1.0

input_dim = X_full.shape[1]
print(f"After PCA + scaling: {X_full.shape[0]:,} samples × {input_dim} dims")
print(f"Explained variance ratio: {explained_variance:.4f}\n")

# ==================== CRITICAL: Extract labels HERE ====================
raw_labels = np.array([y[0] for y in y_w_rel])
relevance_flags = np.array([y[1] for y in y_w_rel])
# =========================================================================
# -------------------------- Train/Val Split --------------------------
val_ratio = 0.1
n_val = int(len(X_full) * val_ratio)
indices = np.random.RandomState(RANDOM_SEED).permutation(len(X_full))
train_idx, val_idx = indices[:-n_val], indices[-n_val:]

X_train = torch.from_numpy(X_full[train_idx])
X_val = torch.from_numpy(X_full[val_idx])

train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

# -------------------------- Model --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")
print(f"Hyperparams → latent_dim={LATENT_DIM}, GMM={N_COMPONENTS}, PCA={PCA_COMPS}")

model = DAGMM(
    input_dim=input_dim,
    latent_dim=LATENT_DIM,
    n_components=N_COMPONENTS,
    lambda_energy= 0.1,
    lambda_cov=0.005,
    enc_hidden=[2048, 1024, 512, 256, 128],
    est_hidden=[256, 128, 64],
    dropout=0.3,
    activation="tanh",
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

# -------------------------- Training Loop --------------------------
best_val_energy = float('inf')
patience_counter = 0

print("Starting training with OPTIMIZED DaGMM...\n")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    # Full-batch GMM update
    z_collect, gamma_collect = [], []
    with torch.no_grad():
        for (x,) in train_loader:
            x = x.to(device)
            _, _, z, gamma = model(x)
            z_collect.append(z.cpu())
            gamma_collect.append(gamma.cpu())
    z_all = torch.cat(z_collect).to(device)
    gamma_all = torch.cat(gamma_collect).to(device)
    model.compute_gmm_params(z_all, gamma_all)

    # Train nets
    epoch_loss = 0.0
    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        z_c, x_rec, z, gamma = model(x)
        loss_dict = model.loss_function(x, x_rec, z, gamma)
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        epoch_loss += loss_dict["loss"].item() * x.size(0)
    epoch_loss /= len(train_loader.dataset)

    # Validation energy
    model.eval()
    with torch.no_grad():
        val_z = []
        for (x,) in val_loader:
            x = x.to(device)
            _, _, z, _ = model(x)
            val_z.append(z.cpu())
        val_z = torch.cat(val_z).to(device)
        val_energy = model.compute_energy(val_z).mean().item()

    if epoch <= 10 or epoch % 30 == 0 or epoch == MAX_EPOCHS:
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Val Energy: {val_energy:.4f}")

    if val_energy < best_val_energy - 1e-4:
        best_val_energy = val_energy
        patience_counter = 0
        torch.save(model.state_dict(), f"results/dagmm_best_NREL{N_REL_CLASSES}.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load(f"results/dagmm_best_NREL{N_REL_CLASSES}.pth"))

# -------------------------- Final Inference --------------------------
print("\nFinal inference...")
model.eval()
with torch.no_grad():
    X_tensor = torch.from_numpy(X_full).to(device)
    z_c_all, _, z_all, _ = model(X_tensor)

    z_c_all = z_c_all.cpu().numpy()  # 64-dim compressed latent
    z_all = z_all.cpu().numpy()  # 66-dim full z (includes recon error features)

    # Compute anomaly scores
    anomaly_scores = model.predict_energy(X_full)
    if torch.is_tensor(anomaly_scores):
        anomaly_scores = anomaly_scores.cpu().numpy()

# === Compute top-100 recall (so rel_count is defined!) ===
top_idx = np.argsort(anomaly_scores)[-100:][::-1]
rel_count = relevance_flags[top_idx].sum()  # number of relevant/rare in top-100

# === Save everything for A/RED ===
os.makedirs("results", exist_ok=True)

# 1. Original compressed latent (you already tried)
with open(f"results/preprocessed_X_latent_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(z_c_all, f)

# 2. NEW & RECOMMENDED: full z with reconstruction error features
with open(f"results/preprocessed_X_full_z_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(z_all, f)

# Labels
with open(f"results/y_w_rel_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(y_w_rel, f)

# Bonus: save the actual DaGMM energy scores too
with open(f"results/dagmm_anomaly_scores_NREL{N_REL_CLASSES}.pkl", "wb") as f:
    pickle.dump(anomaly_scores, f)

print("Latent representations saved!")
print(f"   → z_c_all shape       : {z_c_all.shape}  (compressed latent)")
print(f"   → z_all shape         : {z_all.shape}    (full z = [z_c, euc, cos]) ← USE THIS IN A/RED!")
print(f"   → Top-100 recall      : {rel_count}/100 ({rel_count / 100:.1%}) relevant rare classes")
print(
    f"   → Anomaly scores saved: min={anomaly_scores.min():.3f}, max={anomaly_scores.max():.3f}, mean={anomaly_scores.mean():.3f}")

# -------------------------- Top 100 Anomalies --------------------------
print("\n" + "=" * 90)
print(f"TOP 100 MOST ANOMALOUS (N_REL_CLASSES={N_REL_CLASSES})")
print("=" * 90)
top_idx = np.argsort(anomaly_scores)[-100:][::-1]
rel_count = 0
for rank, idx in enumerate(top_idx, 1):
    label = raw_labels[idx]
    is_rel = "REL" if relevance_flags[idx] else "   "
    if relevance_flags[idx]:
        rel_count += 1
    print(f"{rank:3d}. [{is_rel}] Energy: {anomaly_scores[idx]:8.2f} → '{label}'")
print(f"\n→ {rel_count}/100 relevant rare classes in top 100! ({rel_count / 100:.1%})")

print("\nRunning UMAP (much better than t-SNE)...")
Z_2d = UMAP(n_neighbors=30, min_dist=0.0, n_components=2,
            random_state=RANDOM_SEED).fit_transform(z_c_all)

plt.figure(figsize=(16, 12))
# Subsample normals
normal_idx = np.where(~np.isin(raw_labels, relevant_labels))[0]
sub_idx = np.random.choice(normal_idx, min(2000, len(normal_idx)), replace=False)
plt.scatter(Z_2d[sub_idx, 0], Z_2d[sub_idx, 1], c='lightgray', s=30, alpha=0.7,
            label=f'normal (subsampled, n={len(sub_idx)})')

# All rare points — big and bold
palette = sns.color_palette("tab10", len(relevant_labels))
for i, cls in enumerate(relevant_labels):
    mask = raw_labels == cls
    if mask.sum() > 0:
        plt.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[palette[i]], s=120,
                    label=f"{cls} ({mask.sum()})", edgecolors='black', linewidth=1.2)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.title(f"DaGMM + UMAP — {rel_count}% rare class recall in top-100")
plt.tight_layout()
plt.savefig(f"results/dagmm_UMAP_NREL{N_REL_CLASSES}.png", dpi=300, bbox_inches='tight')
plt.show()