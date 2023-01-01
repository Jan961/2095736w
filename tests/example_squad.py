from sklearn.decomposition import PCA
import numpy as np

from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_low_d_layout.plot import show_layout, show_generation_metrics
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists

metric_collection = {'stress': 100}
dataset = DataFetcher().fetch_data('mnist')

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
# Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=dataset, initial_layout=Xld, distance_fn= relative_rbf_dists)
layout = LowDLayoutCreation().create_layout(squad, dataset, metric_collection=metric_collection, no_iters=1000)

show_layout(layout, use_labels=True)
show_generation_metrics(layout)
