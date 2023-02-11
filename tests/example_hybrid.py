from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layouts, show_generation_metrics
from sklearn.decomposition import PCA
import numpy as np
import tracemalloc

all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']


metric_collection = {'Stress': 50}

dataset = DataFetcher().fetch_data('rna N3k')
Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

zero_initial = np.zeros((dataset.data.shape[0], 2))


tracemalloc.start()
hybrid = Hybrid(dataset=dataset, initial_layout=zero_initial, alpha=0.7,  distance_fn=euclidean,
                    use_knnd=False, sample_set_size=10, neighbour_set_size=20)


layout = LowDLayoutCreation().create_layout(hybrid, optional_metric_collection=None)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()


# print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
# print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
# print(f"velocity: {layout.collected_metrics['average speed'][1]} \n")
# print(f" stress: {layout.collected_metrics['stress'][1]} \n")
# print("total time: {}")
show_layouts(layout, use_labels=True, color_map='rainbow', title="Hybrid")
show_generation_metrics(layout, title="Hybrid ")
