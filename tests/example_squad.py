from sklearn.decomposition import PCA
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import manhattan
import numpy as np
import matplotlib.pyplot as plt
from experiments.cube.Cube import Cube


all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']

cube = Cube(num_points=100, side=30, angle=0.4)
dataset_cube= cube.get_sample_dataset(3000)


metric_collection = { "Average n-tet stress": 1, 'Stress': 100}
dataset = DataFetcher.fetch_data('rna N3k')

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)
random_initial =  10*np.random.randn(dataset.data.shape[0], 2)


squad = SNeD(dataset=dataset, initial_layout=Xld, use_nesterovs_momentum=False, ntet_size=4, momentum=0.1)
layout = LayoutCreation.create_layout(squad, no_iters=200,optional_metric_collection=metric_collection, use_decay=False)
show_layout(layout, use_labels=True, color_map='rainbow', title=f"SNeD - {layout.iteration_number}")
show_generation_metrics(layout, quartet_stress=True, title=f"iters {layout.iteration_number} ", iters_from = 100)

# cube.plot_2d(layout=layout, title="no decay ")
print(layout.collected_metrics)
# fig, axis = plt.subplots()
# axis.scatter(Xld[:,0], Xld[:,1], c=dataset.labels, cmap='rainbow')

