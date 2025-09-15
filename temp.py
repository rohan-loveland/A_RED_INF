import numpy as np
import pickle
import os
import struct


def load_and_replicate_emnist(data_dir="./gzip", save_path="emnist_replicated_10x.pkl"):
    """
    Loads EMNIST dataset (byclass split), duplicates all samples 10 times,
    and saves to a pickle file.

    Args:
        data_dir: Directory containing EMNIST .idx files and mapping.txt.
        save_path: Path to save the replicated EMNIST dataset pickle file.

    Returns:
        None (saves replicated dataset to pickle file).
    """
    # File paths
    TRAIN_IMAGE_FILE = 'emnist-byclass-train-images-idx3-ubyte'
    TRAIN_LABEL_FILE = 'emnist-byclass-train-labels-idx1-ubyte'
    TEST_IMAGE_FILE = 'emnist-byclass-test-images-idx3-ubyte'
    TEST_LABEL_FILE = 'emnist-byclass-test-labels-idx1-ubyte'
    MAPPING_FILE = 'emnist-byclass-mapping.txt'

    if os.path.exists(save_path):
        print(f"Replicated EMNIST already exists at {save_path}. Skipping...")
        return

    print("Loading EMNIST dataset from .idx files...")

    # Load mapping file (for reference, not strictly needed for pickle)
    mapping = {}
    with open(os.path.join(data_dir, MAPPING_FILE), 'r') as f:
        for line in f:
            label_idx, unicode_val = map(int, line.strip().split())
            mapping[label_idx] = chr(unicode_val)

    # Load train images
    with open(os.path.join(data_dir, TRAIN_IMAGE_FILE), 'rb') as f:
        magic, num_images_train, rows, cols = struct.unpack('>IIII', f.read(16))
        train_img_data = np.frombuffer(f.read(), dtype=np.uint8)
        train_images = train_img_data.reshape((num_images_train, rows, cols))

    # Load train labels
    with open(os.path.join(data_dir, TRAIN_LABEL_FILE), 'rb') as f:
        magic, num_labels_train = struct.unpack('>II', f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Load test images
    with open(os.path.join(data_dir, TEST_IMAGE_FILE), 'rb') as f:
        magic, num_images_test, rows_test, cols_test = struct.unpack('>IIII', f.read(16))
        test_img_data = np.frombuffer(f.read(), dtype=np.uint8)
        test_images = test_img_data.reshape((num_images_test, rows_test, cols_test))

    # Load test labels
    with open(os.path.join(data_dir, TEST_LABEL_FILE), 'rb') as f:
        magic, num_labels_test = struct.unpack('>II', f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Combine train and test
    all_images = np.concatenate((train_images, test_images), axis=0)  # Shape: (814255, 28, 28)
    all_labels = np.concatenate((train_labels, test_labels), axis=0)  # Shape: (814255,)

    print(f"Original EMNIST loaded: {all_images.shape[0]} samples")

    # Replicate 10 times
    all_images_replicated = np.tile(all_images, (10, 1, 1))  # Shape: (8142550, 28, 28)
    all_labels_replicated = np.tile(all_labels, (10,))  # Shape: (8142550,)

    # Fix orientation, normalize, and flatten images
    X_replicated = np.array(
        [np.fliplr(img.T).flatten() / 255.0 for img in all_images_replicated])  # Shape: (8142550, 784)
    y_replicated = all_labels_replicated.astype(str)  # Convert to strings

    # Save replicated dataset
    emnist_replicated = {
        'images': X_replicated,
        'labels': y_replicated
    }
    with open(save_path, 'wb') as f:
        pickle.dump(emnist_replicated, f)
    print(f"Replicated EMNIST (10x) saved to {save_path}")
    print(f"Replicated dataset shape: images {X_replicated.shape}, labels {y_replicated.shape}")


if __name__ == "__main__":
    load_and_replicate_emnist()