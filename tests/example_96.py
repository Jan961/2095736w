from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import manhattan,euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics
from sklearn.decomposition import PCA
import numpy as np
from experiments.cube.Cube import Cube








all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                                                                                                'shuttle',
                     'flow cytometry']

cube = Cube(num_points=100, side=30, angle=0.4)
cube_dataset= cube.get_sample_dataset(3000)


metric_collection = {'Average speed': 1}

dataset = DataFetcher.fetch_data('coil20')
Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(cube_dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

zero_initial = np.zeros((dataset.data.shape[0], 2))

algo96 = Chalmers96(dataset=cube_dataset, initial_layout=Xld,  distance_fn=euclidean,
                    damping_constant=0, spring_constant=1, integrate_sum=True, sc_scaling_factor=0,
                    use_knnd=False, sample_set_size=6, neighbour_set_size=4)


layout = LayoutCreation().create_layout(algo96, optional_metric_collection=metric_collection, no_iters=50)

# print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
# print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
# print(f"velocity: {layout.collected_metrics['Average speed'][1]} \n")
# print(f" stress: {layout.collected_metrics['Stress'][1]} \n")
# print("total time: {}")
# show_layouts(layout, use_labels=True, color_map='rainbow', title=f"damp: {algo96.damping_constant}, sk: {algo96.spring_constant},\
#  n: {algo96.neighbour_set_size}, s:{algo96.sample_set_size}")
show_generation_metrics(layout, average_speed=True, stress=True, title=f"damp: {algo96.damping_constant}, sk: {algo96.spring_constant},\
 n: {algo96.neighbour_set_size}, s:{algo96.sample_set_size}")

cube.plot_2d(layout)