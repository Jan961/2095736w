import numpy as np
from experiments.utils.SimpleComparison import SimpleComparison
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.visualise_layouts_and_metrics.plot import show_layout,show_generation_metrics
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.create_low_d_layout.LayoutBase import LowDLayoutBase
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from pathlib import Path
from definitions import PROJECT_ROOT


def get_avg_classwise_f1(real_labels: np.ndarray, predicted_labels: np.ndarray):
    unique_labels = np.unique(real_labels)

    f1_scores = []
    for label in unique_labels:
        real_binary_labels = np.where(real_labels == label, 1, 0)
        predicted_binary_labels = np.where(predicted_labels == label, 1, 0)

        f1 = f1_score(real_binary_labels, predicted_binary_labels)
        f1_scores.append(f1)

    return sum(f1_scores)/unique_labels.size