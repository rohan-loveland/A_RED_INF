#!/usr/bin/env python3
# resize_parking_lot_features.py
# Resize flattened 128x128 RGB parking lot images to 224x224 and save as a new pickle.

import os
import pickle
import cv2
import numpy as np


def resize_parking_lot_features(
    input_path: str = "./Parking_Lot_Data/features.pkl",
    output_path: str = "./Parking_Lot_Data/resized_features_224.pkl",
) -> None:
    """
    Load the original flattened 128x128 RGB parking lot features from pickle,
    reshape to (n, 128, 128, 3), resize each image to 224x224 using cubic interpolation,
    and save the resized features as a new pickle file with shape (n, 224, 224, 3).

    Parameters
    ----------
    input_path : str
        Path to the original features.pkl file (expected shape: (n, 128*128*3)).
    output_path : str
        Path to save the resized features.pkl file.

    Notes
    -----
    - Assumes original features are flattened RGB: (n_samples, 128*128*3).
    - Uses cv2.INTER_CUBIC for high-quality upscaling.
    - Preserves original dtype (typically uint8).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Loading original features...")
    with open(input_path, "rb") as f:
        features = pickle.load(f)

    print(f"Original shape: {features.shape}")

    # Reshape flattened features to (n, 128, 128, 3)
    if features.shape[1] != 128 * 128 * 3:
        raise ValueError(
            f"Unexpected flattened dim: {features.shape[1]}, "
            f"expected {128 * 128 * 3}"
        )

    n_samples = features.shape[0]
    features = features.reshape(n_samples, 128, 128, 3)
    print(f"Reshaped to: {features.shape}")

    resized_features = np.zeros((n_samples, 224, 224, 3), dtype=features.dtype)

    print("Resizing images...")
    for i in range(n_samples):
        img = features[i]  # (128, 128, 3)
        resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        resized_features[i] = resized_img

        if (i + 1) % 500 == 0 or (i + 1) == n_samples:
            print(f"Processed {i + 1}/{n_samples} images...")

    print(f"Resized shape: {resized_features.shape}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(resized_features, f)

    print(f"Resized features saved to: {output_path}")


if __name__ == "__main__":
    resize_parking_lot_features()