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



metric_collection = {'Average speed': 1, "Stress": 5}

dataset = DataFetcher.fetch_data('rna N3k')
Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(cube_dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

# show_layout(positions=Xld, labels=dataset.labels, title="PCA")

zero_initial = np.zeros((dataset.data.shape[0], 2))
random_initial =  10*np.random.randn(dataset.data.shape[0], 2)

algo96 = Chalmers96(dataset=dataset,  distance_fn=euclidean, initial_layout=None,
                    damping_constant=0, spring_constant=0.5,
                    use_knnd=False, sample_set_size=10, neighbour_set_size=5)


layout = LayoutCreation.create_layout(algo96, optional_metric_collection=None, num_iters=100)
#





# print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
# print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
# print(f"velocity: {layout.collected_metrics['Average speed'][1]} \n")
# print(f" stress: {layout.collected_metrics['Stress'][1]} \n")
# print("total time: {}")
# show_layout(layout, use_labels=True, color_map='rainbow', title=f"damp: {algo96.damping_constant}, sk: {algo96.spring_constant},\
# #  n: {algo96.neighbour_set_size}, s:{algo96.sample_set_size}, iters: {layout.iteration_number}")
show_generation_metrics(layout, average_speed=True, stress=True, title=f"Chalmers' 96 - Cube dataset ")

# cube.plot_2d(layout, title="Chalmers' 96")

show_layout(layout, use_labels=True, color_map='rainbow', title='Chalmers\' 96, Mouse cortex scRNA')