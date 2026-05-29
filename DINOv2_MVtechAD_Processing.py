#!/usr/bin/env python3
# DINOv2_MVtechAD_Processing.py
#
# Pipeline: DINOv2 feature extraction (768-D)
# Simplified version - only DINOv2 embeddings (no autoencoder).

import os
import numpy as np
import pickle
from collections import Counter
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

MVTECHAD_OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MODEL_NAME     = "facebook/dinov2-base"
DINO_IMG_SIZE  = (224, 224)
BATCH_SIZE_VIT = 32


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helper
# ──────────────────────────────────────────────────────────────────────────────

class _MVtechImageDataset(Dataset):
    def __init__(self, img_paths, processor):
        self.img_paths = img_paths
        self.processor = processor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(
            DINO_IMG_SIZE, Image.LANCZOS
        )
        inputs = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}


# ──────────────────────────────────────────────────────────────────────────────
# DINOv2 feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_dino_features(img_paths, device, verbose=False):
    """Extract z-score normalized DINOv2 CLS embeddings."""
    if verbose:
        print(f"Loading DINOv2 model: {MODEL_NAME}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model_vit = AutoModel.from_pretrained(MODEL_NAME)
    model_vit.eval()
    model_vit.to(device)

    dataset = _MVtechImageDataset(img_paths, processor)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_VIT,
        shuffle=False,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )

    features_list = []
    total = len(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            pixel_values = batch["pixel_values"].to(device)
            outputs = model_vit(pixel_values)
            cls_tokens = outputs.last_hidden_state[:, 0, :]   # (B, 768)
            features_list.append(cls_tokens.cpu().numpy())
            if verbose and (i % 10 == 0 or i == total):
                print(f"  DINOv2 batch {i}/{total}")

    X = np.vstack(features_list).astype(np.float32)

    # z-score normalization per dimension
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    return X


# ──────────────────────────────────────────────────────────────────────────────
# Image path + label collector
# ──────────────────────────────────────────────────────────────────────────────

def _collect_mvtechad_category(mvtechad_root, object_name, include_train=True):
    """Walk one MVtechAD object category and return image paths + labels."""
    obj_dir = os.path.join(mvtechad_root, object_name)
    if not os.path.isdir(obj_dir):
        raise FileNotFoundError(
            f"MVtechAD object directory not found: {obj_dir}\n"
            f"Expected structure: {mvtechad_root}/<object>/test/<label>/"
        )

    img_paths = []
    y_w_rel = []

    def _ingest(folder, label_str, is_anomaly):
        if not os.path.isdir(folder):
            return
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue
            full = os.path.join(folder, fname)
            img_paths.append(full)
            y_w_rel.append((label_str, is_anomaly))

    # train/good -> normal
    if include_train:
        _ingest(os.path.join(obj_dir, "train", "good"),
                f"normal_{object_name}", is_anomaly=False)

    # test/<label> folders
    test_dir = os.path.join(obj_dir, "test")
    if os.path.isdir(test_dir):
        for label_folder in sorted(os.listdir(test_dir)):
            label_path = os.path.join(test_dir, label_folder)
            if not os.path.isdir(label_path):
                continue
            if label_folder.lower() == "good":
                _ingest(label_path, f"normal_{object_name}", is_anomaly=False)
            else:
                _ingest(label_path, f"{label_folder}_{object_name}", is_anomaly=True)

    return img_paths, y_w_rel


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def mvtechad_dino_ae_setup_for_main(
    N_REL_CLASSES,
    VERBOSE_FLAGS,
    seed,
    mvtechad_root: str = None,                    # ← Set to None for auto-detection
    object_categories=None,
    include_train=True,
):
    """
    Load MVtechAD dataset using DINOv2 (768-D) embeddings with smart path detection.
    """
    verbose = 0 in VERBOSE_FLAGS

    # ── Smart Dataset Path Detection ───────────────────────────────────────
    if mvtechad_root is None:
        nate_path = "Datasets/MVtechAD"
        rohan_path = os.path.expanduser("~/Dropbox/Research/Datasets/mvtec")

        if os.path.isdir(nate_path):
            mvtechad_root = nate_path
            if verbose:
                print(f"✅ Using Nate's path: {mvtechad_root}")
        elif os.path.isdir(rohan_path):
            mvtechad_root = rohan_path
            if verbose:
                print(f"✅ Using Rohan's path: {mvtechad_root}")
        else:
            raise FileNotFoundError(
                "MVtec AD dataset not found in any default location.\n"
                f"Checked:\n"
                f"  • Nate:  {os.path.abspath(nate_path)}\n"
                f"  • Rohan: {rohan_path}\n\n"
                "Please pass the correct path manually:\n"
                "mvtechad_dino_ae_setup_for_main(mvtechad_root='/your/path')"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    cache_feat = os.path.join(mvtechad_root, "mvtechad_dino_features.pkl")
    cache_y    = os.path.join(mvtechad_root, "mvtechad_dino_y.pkl")

    # Load from cache if available
    if os.path.exists(cache_feat) and os.path.exists(cache_y):
        if verbose:
            print(f"Loading cached MVtechAD DINOv2 data from:\n  {cache_feat}")
        with open(cache_feat, "rb") as f:
            X = pickle.load(f)
        with open(cache_y, "rb") as f:
            y_w_rel = pickle.load(f)
    else:
        if object_categories is None:
            object_categories = MVTECHAD_OBJECT_CATEGORIES

        # Collect images and labels
        all_img_paths = []
        all_y_w_rel = []
        for obj in object_categories:
            if verbose:
                print(f"Collecting MVtechAD category: {obj} ...")
            paths, y_cat = _collect_mvtechad_category(
                mvtechad_root, obj, include_train=include_train
            )
            all_img_paths.extend(paths)
            all_y_w_rel.extend(y_cat)

        if verbose:
            print(f"\nTotal images collected: {len(all_img_paths)}")

        # Extract DINOv2 features
        if verbose:
            print("Extracting DINOv2 features ...")
        X = _extract_dino_features(all_img_paths, device, verbose=verbose)

        y_w_rel = all_y_w_rel

        # Save cache
        if verbose:
            print(f"Saving cache to:\n  {cache_feat}\n  {cache_y}")
        with open(cache_feat, "wb") as f:
            pickle.dump(X, f)
        with open(cache_y, "wb") as f:
            pickle.dump(y_w_rel, f)

    # Compute sparsity levels
    labels_only = [lbl for lbl, _ in y_w_rel]
    label_counts = Counter(labels_only)
    total = len(labels_only)
    sparsity_levels = [(lbl, cnt / total) for lbl, cnt in label_counts.most_common()]

    # Relevant labels (anomalous ones)
    label_relevance = {}
    for lbl, is_rel in y_w_rel:
        label_relevance[lbl] = label_relevance.get(lbl, False) or is_rel
    relevant_labels = sorted(lbl for lbl, rel in label_relevance.items() if rel)

    if verbose:
        print(f"\nMVtechAD DINOv2 dataset ready")
        print(f"  X shape       : {X.shape}")
        print(f"  Total samples : {total}")
        print(f"  Total classes : {len(label_counts)}")
        n_anom = sum(1 for _, r in y_w_rel if r)
        print(f"  Anomalous     : {n_anom} ({n_anom/total*100:.1f}%)")

    return X, y_w_rel, sparsity_levels, relevant_labels


# ──────────────────────────────────────────────────────────────────────────────
# Stand-alone demo
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y_w_rel, sparsity_levels, relevant_labels = mvtechad_dino_ae_setup_for_main(
        N_REL_CLASSES=None,
        VERBOSE_FLAGS=[0],
        seed=42,
    )

    total = len(y_w_rel)
    n_anomaly = sum(1 for _, r in y_w_rel if r)

    print("\n-- Summary --------------------------------------------------")
    print(f"X shape        : {X.shape}")
    print(f"Total samples  : {total}")
    print(f"Anomaly rate   : {n_anomaly/total*100:.2f}%")
    print()
    for lbl, prop in sparsity_levels[:10]:
        print(f"  {lbl:<45} {prop*100:.2f}%")