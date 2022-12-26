from sklearn.decomposition import PCA
import numpy as np

from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_low_d_layout.show_layout import show_layout


def get_RNAseq_3k():
    XY = np.load('RNAseq_N3k.npy')
    return XY[:, :-1].astype(np.float64), XY[:, -1]

data, labels = DataFetcher().fetch_data('poker', size=500)

Xld = PCA(n_components=2, whiten=True, copy=True).fit_transform(data).astype(np.float64)
Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=data, initial_layout=Xld, distance_fn=poker_distance)
layout = LowDLayoutCreation().create_layout(squad, data, labels, no_iters=500)

show_layout(layout, dataset=data, use_labels=True)