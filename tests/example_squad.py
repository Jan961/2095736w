from sklearn.decomposition import PCA
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists
import numpy as np


all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']


metric_collection = {'Average quartet stress': 200, 'Stress': 200}
dataset = DataFetcher().fetch_data('coil20')

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=dataset, initial_layout=Xld, nesterovs_momentum=False)
layout = LowDLayoutCreation().create_layout(squad, optional_metric_collection=None, no_iters=200)
print(layout.collected_metrics)
show_layouts(layout, use_labels=True, color_map='rainbow', title="new vectorisation")
show_generation_metrics(layout, quartet_stress=True)
