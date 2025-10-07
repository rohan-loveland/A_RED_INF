import numpy as np
import pickle
import pandas as pd
import cv2
from collections import Counter


def parking_lot_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, seed):
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

    # Least common labels (for relevance)
    least_common_labels = [label for label, _ in label_counts.most_common()[-N_REL_CLASSES:]]

    if 0 in VERBOSE_FLAGS:
        print(f"Running on parking lot dataset with {n_events} events")
        print(f"Least common labels: {least_common_labels} (marked as relevant)")

    # Generate relevance info
    relevance_array = [label in set(least_common_labels) for label in y]
    y_w_rel = list(zip(y, relevance_array))

    # Sparsity levels: number of points in each class from highest to lowest
    sparsity_levels = label_counts.most_common()

    return X, y_w_rel, sparsity_levels