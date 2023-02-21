from sklearn.decomposition import PCA
from hdimvis.algorithms.stochastic_ntet_algo.SNaD import SNaD
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import manhattan
import numpy as np
import matplotlib.pyplot as plt


all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']


metric_collection = {'Average quartet stress': 200, 'Stress': 200}
dataset = DataFetcher.fetch_data('rna N3k')

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

squad = SNaD(dataset=dataset, initial_layout=None, nesterovs_momentum=False, ntet_size=12, distance_fn=manhattan)
layout = LowDLayoutCreation().create_layout(squad, optional_metric_collection=None, no_iters=1000)
print(layout.collected_metrics)
show_layouts(layout, use_labels=True, color_map='rainbow', title=" 2k iter ntet 2")
show_generation_metrics(layout, quartet_stress=True)

fig, axis = plt.subplots()
axis.scatter(Xld[:,0], Xld[:,1], c=dataset.labels, cmap='rainbow')

