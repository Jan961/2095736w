import numpy as np

from experiments.utils.SimpleComparison import SimpleComparison
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.visualise_layouts_and_metrics.plot import show_layout,show_generation_metrics
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from experiments.utils.get_avg_classwise_f1 import get_avg_classwise_f1
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from pathlib import Path
from definitions import PROJECT_ROOT
import matplotlib.pyplot as plt


def get_f1_for_best_k_with_knn(lower_bound: int, upper_bound: int,
                               cross_validation_folds: int,
                               data:np.ndarray,
                               labels: np.ndarray,
                               show_averages: bool = False):

    kf = KFold(n_splits=cross_validation_folds)

    averages = np.zeros((upper_bound- lower_bound, 2))

    for index, j in enumerate(range(lower_bound, upper_bound)):

        cross_val_avg = 0
        for i, (train_index, test_index) in enumerate(kf.split(data)):
            # print(f"Fold {i}")
            neigh = KNeighborsClassifier(n_neighbors=j)
            train = data[train_index]
            test = data[test_index]
            neigh.fit(train, labels[train_index])

            predicted_labels = neigh.predict(test)
            # print(f" predicted {predicted_labels}")
            true_labels = labels[test_index]
            # print(f" true :{true_labels}")
            f1 = get_avg_classwise_f1(true_labels, predicted_labels)
            cross_val_avg += f1
            # print(f1)

        cross_val_avg /= 10

        averages[index,0] = j
        averages[index, 1] = cross_val_avg

    if show_averages:
        fig, ax = plt.subplots()
        ax.plot(averages[:,0], averages[:,1])
        plt.ylabel("F1 score")
        plt.xlabel("k")
        plt.show()



    index_best = np.argmax(averages[:,1])
    return averages[index_best][0], averages[index_best][1] #return best k, and the corresponding f1