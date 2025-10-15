import numpy as np
import pickle
import pandas as pd
import cv2
from collections import Counter


def parking_lot_setup_for_main(VERBOSE_FLAGS, seed):
    # === Load Dataset ===
    features_path = "./Parking_Lot_Data/features.pkl"
    labels_path = "./Parking_Lot_Data/labels.csv"

    with open(features_path, 'rb') as f:
        features = pickle.load(f)  # shape (4410, 128, 128, 3)

    labels_df = pd.read_csv(labels_path)

    # Convert RGB to grayscale, normalize to [0, 1], and flatten
    X = np.array([
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32).flatten() / 255.0
        for img in features
    ])

    y = labels_df["label"].tolist()
    assert len(X) == len(y), "Mismatch between features and labels"

    n_events = len(y)

    # Get label counts
    label_counts = Counter(y)

    # Define relevant labels
    relevant_labels = ["sticks", "cigarette", "costco card", "library card", "paperclip", "zone"]

    if 0 in VERBOSE_FLAGS:
        print(f"Running on parking lot dataset with {n_events} events")
        print(f"All found labels: {list(label_counts.keys())}")
        print(f"Relevant labels: {relevant_labels}")

    # Generate relevance info
    relevance_array = [label in set(relevant_labels) for label in y]
    y_w_rel = list(zip(y, relevance_array))

    # Sparsity levels: number of points in each class from highest to lowest
    sparsity_levels = label_counts.most_common()
    total_count = 0
    for _, l_count in sparsity_levels:
        total_count += l_count
    sparsity_levels = [(label, l_count/total_count) for label, l_count in sparsity_levels]
    return X, y_w_rel, sparsity_levels, relevant_labels