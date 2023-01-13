from sklearn.decomposition import PCA
import numpy as np

from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists

metric_collection = {'stress': 20}
dataset = DataFetcher().fetch_data('coil20')

# Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
# Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=dataset, distance_fn= relative_rbf_dists)
layout = LowDLayoutCreation().create_layout(squad, dataset, optional_metric_collection=metric_collection, no_iters=200)

show_layouts(layout, use_labels=True)
show_generation_metrics(layout)
