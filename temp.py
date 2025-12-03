from DAGMM import DAGMM   # ← This is now the correct implementation!
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

from sklearn.manifold import TSNE
import seaborn as sns
from collections import Counter
import random
import math
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

# -------------------------- Config --------------------------
N_REL_CLASSES = 8
RANDOM_SEED   = 42
MAX_EPOCHS    = 300
PATIENCE      = 120
BATCH_SIZE    = 256
N_COMPONENTS  = 4
VERBOSE_FLAGS = [0]

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------- 1. Load Data --------------------------
from main_helper_functions import parking_lot_setup_for_main

print(f"Loading Parking Lot data with N_REL_CLASSES = {N_REL_CLASSES}")
X_raw, y_w_rel, sparsity_levels, relevant_labels = parking_lot_setup_for_main(
    N_REL_CLASSES=N_REL_CLASSES,
    VERBOSE_FLAGS=VERBOSE_FLAGS,
    seed=RANDOM_SEED
)

# Normalize to [-1, 1] (DaGMM expects this range)
X_full = X_raw.astype(np.float32)
X_full = X_full * 2.0 - 1.0
input_dim = X_full.shape[1]

raw_labels = [y[0] for y in y_w_rel]
class_labels = np.array(raw_labels)
relevance_flags = np.array([y[1] for y in y_w_rel])

print(f"Dataset: {X_full.shape[0]:,} samples × {input_dim} features")
print(f"Selected {N_REL_CLASSES} rarest classes as RELEVANT:")
for lbl in relevant_labels:
    cnt = np.sum(class_labels == lbl)
    print(f"   • '{lbl}': {cnt} samples")
print()

# -------------------------- 2. Train/Val Split --------------------------
val_ratio = 0.10
n_val = int(len(X_full) * val_ratio)
indices = np.random.RandomState(RANDOM_SEED).permutation(len(X_full))
train_idx, val_idx = indices[:-n_val], indices[-n_val:]

X_train = torch.from_numpy(X_full[train_idx])
X_val   = torch.from_numpy(X_full[val_idx])

train_dataset = TensorDataset(X_train)
val_dataset   = TensorDataset(X_val)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# -------------------------- 3. Model + Training Setup --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nTraining on {device} | GMM Components: {N_COMPONENTS}")

model = DAGMM(
    input_dim=input_dim,
    latent_dim=16,
    n_components=N_COMPONENTS,
    lambda_energy=0.1,
    lambda_cov=0.005,
    enc_hidden=[128, 64, 32],
    est_hidden=[64, 32],
    dropout=0.5,
    activation="tanh",
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

# -------------------------- 4. Training Loop (with proper GMM updates) --------------------------
best_val_energy = float('inf')
patience_counter = 0
train_losses = []

print("Starting training with REAL DaGMM...")
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    # === Full-batch GMM parameter estimation (as in paper) ===
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

    for (x,) in train_loader:
        x = x.to(device)
        optimizer.zero_grad()

        z_c, x_rec, z, gamma = model(x)
        loss_dict = model.loss_function(x, x_rec, z, gamma)

        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        epoch_loss += loss_dict["loss"].item() * x.size(0)

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # === Validation energy ===
    model.eval()
    with torch.no_grad():
        val_z_collect = []
        for (x,) in val_loader:
            x = x.to(device)
            _, _, z, _ = model(x)
            val_z_collect.append(z.cpu())
        val_z = torch.cat(val_z_collect).to(device)
        val_energy = model.compute_energy(val_z).mean().item()

    if epoch <= 10 or epoch % 20 == 0 or epoch == MAX_EPOCHS:
        print(f"Epoch {epoch:3d} | Train Loss: {epoch_loss:.4f} | Val Energy: {val_energy:.4f}")

    # Early stopping on validation energy
    if val_energy < best_val_energy - 1e-4:
        best_val_energy = val_energy
        patience_counter = 0
        torch.save(model.state_dict(), f"results/dagmm_real_4comp_NREL{N_REL_CLASSES}_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(torch.load(f"results/dagmm_real_4comp_NREL{N_REL_CLASSES}_best.pth"))

# -------------------------- 5. Final Inference --------------------------
print("\nFinal inference on full dataset...")
model.eval()
with torch.no_grad():
    X_tensor = torch.from_numpy(X_full).to(device)
    z_c_all, _, z_all, _ = model(X_tensor)
    z_c_all = z_c_all.cpu().numpy()
    anomaly_scores = model.predict_energy(X_full)  # Uses real GMM energy

# -------------------------- 5.1 Save for A/RED --------------------------
os.makedirs("results", exist_ok=True)
latent_path = f"results/preprocessed_X_latent_NREL{N_REL_CLASSES}.pkl"
yrel_path   = f"results/y_w_rel_NREL{N_REL_CLASSES}.pkl"

with open(latent_path, "wb") as f:
    pickle.dump(z_c_all, f)        # Save compressed latent z_c (not full z with euc/cos)
with open(yrel_path, "wb") as f:
    pickle.dump(y_w_rel, f)

print("\n" + "="*80)
print("PREPROCESSED DATA SAVED FOR A/RED (REAL DAGMM)")
print(f"   • Latent z_c : {latent_path}")
print(f"   • Labels     : {yrel_path}")
print("="*80)

# -------------------------- 6. Top 100 Anomalies --------------------------
print("\nTOP 100 MOST ANOMALOUS SAMPLES (Real DaGMM Energy)")
print("-"*90)
top_idx = np.argsort(anomaly_scores)[-100:][::-1]
for rank, idx in enumerate(top_idx, 1):
    label = class_labels[idx]
    is_rel = "REL" if relevance_flags[idx] else "   "
    print(f"{rank:3d}. [{is_rel}] Energy: {anomaly_scores[idx]:8.2f} → '{label}'")

# -------------------------- 7. t-SNE Visualization --------------------------
print("\nRunning t-SNE on compressed latent space...")
tsne = TSNE(n_components=2, perplexity=80, learning_rate='auto', init='pca',
            random_state=RANDOM_SEED, n_jobs=-1, metric='euclidean', max_iter=4000)
Z_2d = tsne.fit_transform(z_c_all)

unique_classes = sorted(set(class_labels))
palette = sns.color_palette("tab20", len(unique_classes))
class_to_color = {cls: palette[i] for i, cls in enumerate(unique_classes)}

plt.figure(figsize=(14, 10))
for cls in unique_classes:
    mask = class_labels == cls
    color = 'red' if cls in relevant_labels else class_to_color[cls]
    alpha = 1.0 if cls in relevant_labels else 0.6
    plt.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=[color], s=50, label=f"{cls} ({mask.sum()})",
                alpha=alpha, edgecolors='black', linewidth=0.6 if cls in relevant_labels else 0)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Class (red = relevant)")
plt.title(f'Real DaGMM Latent Space — N_REL_CLASSES={N_REL_CLASSES}', fontsize=18)
plt.tight_layout()
plt.savefig(f'results/real_dagmm_tsne_NREL{N_REL_CLASSES}.png', dpi=300, bbox_inches='tight')
plt.show(block=True)

print("\n" + "="*85)
print("ALL DONE WITH THE REAL DAGMM!")
print("You now have state-of-the-art unsupervised anomaly scores.")
print("="*85)