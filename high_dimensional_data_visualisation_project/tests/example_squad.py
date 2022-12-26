from sklearn.decomposition import PCA
import numpy as np

from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_low_d_layout.show_layout import show_layout
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists


data, labels = DataFetcher().fetch_data('coil20')

Xld = PCA(n_components=2, whiten=True, copy=True).fit_transform(data).astype(np.float64)
Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=data, initial_layout=Xld, distance_fn= relative_rbf_dists)
layout = LowDLayoutCreation().create_layout(squad, data, labels, no_iters=100)

show_layout(layout, dataset=data, use_labels=True)