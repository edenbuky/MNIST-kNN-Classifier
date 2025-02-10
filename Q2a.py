import sklearn
import numpy as np

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", as_frame=False)
data = mnist["data"]
labels = mnist["target"]

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

from scipy.spatial import distance


def k_nearest_neighbor(train_data, train_labels, query_image, k):
    # Calculate distances between the query image and all training images
    dists = distance.cdist(train_data, [query_image], metric='euclidean').flatten()

    # Find the indices of the k nearest neighbors
    nearest_neighbors_indices = np.argsort(dists)[:k]

    # Fetch the labels of the k nearest neighbors
    nearest_labels = train_labels[nearest_neighbors_indices]

    # Count the occurrence of each label among the nearest neighbors
    label_counts = np.bincount(nearest_labels)

    # Determine the most frequent label (ties are broken by selecting the smallest label)
    predicted_label = np.argmax(label_counts)

    return predicted_label
