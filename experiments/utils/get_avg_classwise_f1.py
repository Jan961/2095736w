import numpy as np
from sklearn.metrics import f1_score



def get_avg_classwise_f1(real_labels: np.ndarray, predicted_labels: np.ndarray):
    unique_labels = np.unique(real_labels)

    f1_scores = []
    for label in unique_labels:
        real_binary_labels = np.where(real_labels == label, 1, 0)
        predicted_binary_labels = np.where(predicted_labels == label, 1, 0)

        f1 = f1_score(real_binary_labels, predicted_binary_labels)
        f1_scores.append(f1)

    return sum(f1_scores)/unique_labels.size