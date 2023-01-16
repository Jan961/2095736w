from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from sklearn.decomposition import PCA
import numpy as np
import tracemalloc

all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']


metric_collection = {'stress': 20, 'average speed': 20}

dataset = DataFetcher().fetch_data('rna N3k')
# Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
# Xld *= 10/np.std(Xld)

# tracemalloc.start()
algo96 = Chalmers96(dataset=dataset, alpha=0.7,  distance_fn=euclidean, use_knnd=False, sample_set_size=4, neighbour_set_size=0)


layout = LowDLayoutCreation().create_layout(algo96, dataset, optional_metric_collection=None, no_iters=200)
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()


print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
print(f"velocity: {layout.collected_metrics['average speed'][1]} \n")
print(f" stress: {layout.collected_metrics['stress'][1]} \n")
print("total time: {}")
show_layouts(layout, use_labels=True)
show_generation_metrics(layout, average_speed=True)

