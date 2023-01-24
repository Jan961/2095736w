from sklearn.decomposition import PCA
import numpy as np

from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists

metric_collection = {'Average quartet stress': 50, 'Stress': 200}
dataset = DataFetcher().fetch_data('shuttle')

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

squad = SQuaD(dataset=dataset, initial_layout=Xld)
layout = LowDLayoutCreation().create_layout(squad, optional_metric_collection=metric_collection, no_iters=1000)
print(layout.collected_metrics)
show_layouts(layout, use_labels=True, color_map='rainbow')
show_generation_metrics(layout, quartet_stress=True)
