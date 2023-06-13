from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics
from sklearn.decomposition import PCA
from experiments.cube.Cube import Cube
import numpy as np
import tracemalloc

all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']


metric_collection = {'Stress': 50}
cube = Cube(num_points=100, side=30, angle=0.4)
dataset_cube= cube.get_sample_dataset(3000)

dataset = DataFetcher.fetch_data('rna N3k')
Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset_cube.data).astype(np.float64)
Xld *= 10/np.std(Xld)

zero_initial = np.zeros((dataset.data.shape[0], 2))


tracemalloc.start()
hybrid = Hybrid(dataset=dataset_cube, initial_layout=Xld, alpha=0.6,  distance_fn=euclidean,
                    use_knnd=False, sample_set_size=5, neighbour_set_size=10, use_random_sample=False,
                use_correct_interpolation_error=True)


layout = LayoutCreation().create_layout(hybrid, optional_metric_collection=None)
print(tracemalloc.get_traced_memory())
tracemalloc.stop()


# print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
# print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
# print(f"velocity: {layout.collected_metrics['average speed'][1]} \n")
# print(f" stress: {layout.collected_metrics['stress'][1]} \n")
# print("total time: {}")
show_layout(layout, use_labels=True, color_map='rainbow', title="Hybrid")
show_generation_metrics(layout, title="Hybrid ")
cube.plot_2d(layout)
