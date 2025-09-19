from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def plot_clusters_colored_by_label(ared, X_skewed, y_w_rel, title="Cluster Visualization by Label"):
    """
    Visualizes points in ARED's subspace partition clusters, color-coded by their labels.

    Parameters:
    - ared: ARED instance with subspace_partition.cluster_dict containing clusters.
    - X_skewed: numpy array of data points (n_samples, n_features).
    - y_w_rel: numpy array of labels (e.g., relevance or class labels).
    - title: Plot title (default: "Cluster Visualization by Label").
    """
    # Extract points and their cluster assignments
    cluster_dict = ared.subspace_partition.cluster_dict
    if not cluster_dict:
        print("No clusters to visualize.")
        return

    # Collect points, their labels, and cluster IDs
    points = []
    labels = []
    cluster_ids = []

    for cluster_id, cluster in cluster_dict.items():
        # Assume cluster stores point indices (adjust based on actual cluster object)
        # Replace with actual attribute accessing point indices/data in your cluster object
        if hasattr(cluster, 'point_indices'):
            indices = cluster.point_indices
        else:
            # Fallback: assume cluster stores data directly (modify as needed)
            print(f"Warning: cluster.point_indices not found for cluster {cluster_id}. Skipping.")
            continue

        for idx in indices:
            if idx < len(X_skewed):
                points.append(X_skewed[idx])
                labels.append(y_w_rel[idx])
                cluster_ids.append(cluster_id)

    if not points:
        print("No points to visualize.")
        return

    points = np.array(points)
    labels = np.array(labels)
    cluster_ids = np.array(cluster_ids)

    # Dimensionality reduction with t-SNE to 2D
    if points.shape[1] > 2:
        print("Applying t-SNE to reduce dimensions to 2D...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(points) - 1))
        points_2d = tsne.fit_transform(points)
    else:
        points_2d = points

    # Get unique labels for color mapping
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    # Use a colormap with enough distinct colors
    colormap = plt.cm.get_cmap('tab10' if num_labels <= 10 else 'tab20')(np.linspace(0, 1, num_labels))

    # Plot points, color-coded by label
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            points_2d[mask, 0], points_2d[mask, 1],
            c=[colormap[i]], label=f'Label {label}',
            alpha=0.6, s=50
        )

    # Add cluster boundaries or annotations (optional, based on Voronoi split)
    # For simplicity, we'll just annotate cluster IDs at the centroid of each cluster
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        if np.sum(mask) > 0:
            centroid = np.mean(points_2d[mask], axis=0)
            plt.text(centroid[0], centroid[1], f'C{cid}', fontsize=12, weight='bold',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()