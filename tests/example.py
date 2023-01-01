from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.visualise_low_d_layout.plot import show_layout, show_generation_metrics
from sklearn.decomposition import PCA
import numpy as np


metric_collection = {'stress': 100, 'average speed': 100}

dataset = DataFetcher().fetch_data('mnist')
Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
# Xld *= 10/np.std(Xld)


algo96 = Chalmers96(dataset=dataset, initial_layout=Xld, alpha=0.7, distance_fn=euclidean)


layout = LowDLayoutCreation().create_layout(algo96, dataset, metric_collection=metric_collection, no_iters=500)


print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
print(f"velocity: {layout.collected_metrics['average speed'][1]} \n")
print(f" stress: {layout.collected_metrics['stress'][1]} \n")
show_layout(layout, use_labels=True)
show_generation_metrics(layout, average_speed=True)
