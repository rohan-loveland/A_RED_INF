import numpy as np
import pickle
import os
from collections import Counter


def load_emnist(save_path="emnist.pkl"):
    """
    Loads the EMNIST dataset from a pickle file.

    Args:
        save_path: Path to the EMNIST pickle file.

    Returns:
        X: EMNIST images, shape (n_samples, 784).
        y: EMNIST labels, shape (n_samples,).
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"EMNIST file not found at {save_path}. "
                                "Please generate it first using the EMNIST loading script.")

    # Load the EMNIST dataset
    print(f"Loading EMNIST from {save_path}...")
    with open(save_path, "rb") as file:
        data = pickle.load(file)

    # Extract images and labels
    X = data['images']  # Shape: (814255, 784)
    y = data['labels']  # Shape: (814255,)

    print(f"EMNIST loaded: {X.shape[0]} samples")
    return X, y


def generate_is_relevant(label_list, relevant_set):
    """
    Generates a boolean array indicating whether each label is in the relevant set.

    Args:
        label_list: List of labels.
        relevant_set: Set of relevant labels.

    Returns:
        List of booleans indicating relevance.
    """
    return [label in relevant_set for label in label_list]


def EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS, save_path="emnist.pkl"):
    """
    Sets up the EMNIST dataset for main processing, loading the full dataset and
    marking the least common classes as relevant.

    Args:
        N_REL_CLASSES: Number of least common classes to mark as relevant.
        VERBOSE_FLAGS: List of flags for controlling verbosity.
        save_path: Path to the EMNIST pickle file.

    Returns:
        X: EMNIST images, shape (n_samples, 784).
        y_w_rel: List of (label, relevance) tuples for the dataset.
    """
    # Load the full EMNIST dataset
    X, y = load_emnist(save_path)
    n_events = len(y)

    # Identify the N_REL_CLASSES least common classes
    class_counts = Counter(y)
    least_common_classes = [cls for cls, _ in class_counts.most_common()[-N_REL_CLASSES:]]
    lc_class_freqs = [f for _, f in class_counts.most_common()[-N_REL_CLASSES:]]
    num_relevant_points = sum(lc_class_freqs)
    num_points_total = X.shape[0]
    print(f"Number of classes: {len(set(y))}, number of relevant classes: {N_REL_CLASSES}")
    print(f"% of relevant class points = {100*num_relevant_points/num_points_total:.2f}")

    if 0 in VERBOSE_FLAGS:
        print(f"Running ARED on EMNIST dataset with {n_events} events")
        print(f"Least common classes: {least_common_classes} (marked as relevant)")

    # Generate relevance info
    relevance_array = generate_is_relevant(y, set(least_common_classes))
    y_w_rel = list(zip(y, relevance_array))
    # y_w_rel = np.column_stack((y, np.array(relevance_array),))

    return X, y_w_rel


if __name__ == "__main__":
    # Example usage
    N_REL_CLASSES = 2
    VERBOSE_FLAGS = [0]
    X, y_w_rel = EMNIST_setup_for_main(N_REL_CLASSES, VERBOSE_FLAGS)
    print(f"EMNIST dataset shape: {X.shape}")
    print(f"Sample labels with relevance: {y_w_rel[:5]}")