import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score




def get_kmeans_f1(positions: np.ndarray, labels: np.ndarray):

    unique_labels = np.unique(labels)

    kmeans = KMeans(n_clusters=unique_labels.size, ).fit(positions)
    real_means = {}
    for label in unique_labels:
        mean = np.mean(positions[labels == label], axis=0)
        real_means[label] = mean

    # find the kmeans centres best corresponding to the real centres and compute F1 score for each class
    map = []
    f1_scores = []
    for label, mean in real_means.items():
        dist = np.linalg.norm(kmeans.cluster_centers_ - mean[None, :], axis=1)
        closest = np.argmin(dist)
        map.append((label, np.argmin(dist)))

        real_binary_labels = np.where(labels == label, 1, 0)
        kmeans_binary_labels = np.where(kmeans.labels_ == closest, 1, 0)
        f1 = f1_score(real_binary_labels, kmeans_binary_labels)
        f1_scores.append(f1)


    return sum(f1_scores)/unique_labels.size