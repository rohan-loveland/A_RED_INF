import numpy as np
from sklearn.decomposition import PCA  # Explicitly import PCA
from sklearn.manifold import TSNE  # Import t-SNE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or 'wxAgg' depending on your system

# Visualize data using t-SNE (subsample for large datasets)
def tSNE_3D_vis(X_skewed,y_w_rel):
    # Visualize data using t-SNE (subsample for large datasets)
    subsample_size = 10000  # Adjust this based on memory and computation time constraints
    if len(X_skewed) > subsample_size:
        indices = np.random.choice(len(X_skewed), subsample_size, replace=False)
        X_subsampled = X_skewed[indices]
        y_w_rel_subsampled = [y_w_rel[i] for i in indices]
    else:
        X_subsampled = X_skewed
        y_w_rel_subsampled = y_w_rel

    tsne = TSNE(n_components=3, random_state=42, perplexity=30, max_iter=1000)  # 3D t-SNE
    X_3d = tsne.fit_transform(X_subsampled)

    # Extract class labels and relevance
    class_labels = [y[0] for y in y_w_rel_subsampled]  # Use class number for coloring
    # Convert class labels to numeric values if they are strings
    class_labels = np.array(class_labels, dtype=float) if isinstance(class_labels[0], str) else np.array(class_labels)
    relevance = [y[1] for y in y_w_rel_subsampled]  # Extract relevance for size and alpha

    # Create 3D scatter plot with varying marker size and transparency
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Split into relevant and irrelevant points
    relevant_indices = [i for i, r in enumerate(relevance) if r]
    irrelevant_indices = [i for i, r in enumerate(relevance) if not r]

    # Plot irrelevant points with smaller size and lower opacity
    if irrelevant_indices:
        ax.scatter([X_3d[i, 0] for i in irrelevant_indices],
                   [X_3d[i, 1] for i in irrelevant_indices],
                   [X_3d[i, 2] for i in irrelevant_indices],
                   c=[class_labels[i] for i in irrelevant_indices],
                   cmap='jet',
                   s=20,  # Base size for irrelevant
                   alpha=0.2)  # Mostly transparent

    # Plot relevant points with larger size and higher opacity
    if relevant_indices:
        ax.scatter([X_3d[i, 0] for i in relevant_indices],
                   [X_3d[i, 1] for i in relevant_indices],
                   [X_3d[i, 2] for i in relevant_indices],
                   c=[class_labels[i] for i in relevant_indices],
                   cmap='jet',
                   s=50,  # Base size + 5 for relevant
                   alpha=1)  # Opaque

    plt.colorbar(ax.collections[0], label='Class Number')  # Use the first collection for colorbar
    ax.set_title(f't-SNE 3D Visualization of EMNIST Data (Subsample Size: {subsample_size})')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def plot_stacked_area(list1, list2, list3,legend_labels=['List 1', 'List 2', 'List 3']):
    # Convert lists to numpy arrays for easier manipulation
    y1 = np.array(list1)
    y2 = np.array(list2)
    y3 = np.array(list3)
    x = range(len(list1))

    # Create cumulative arrays for stacking
    y2_cum = y1 + y2
    y3_cum = y1 + y2 + y3

    # Create the stacked area plot

    plt.stackplot(x, y1, y2, y3, labels=legend_labels, colors=['#FF9999', '#66B2FF', '#99FF99'])

    # Customize the plot
    plt.title('Cumulative Stacked Area Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    list1 = [1, 2, 3, 4, 5]
    list2 = [2, 3, 2, 3, 4]
    list3 = [3, 4, 5, 4, 3]
    plot_stacked_area(list1, list2, list3)
