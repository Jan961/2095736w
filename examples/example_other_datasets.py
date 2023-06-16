from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import manhattan,euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics
from sklearn.decomposition import PCA
import numpy as np
from experiments.cube.Cube import Cube
from sklearn.decomposition import PCA
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics


dataset = DataFetcher.fetch_data('rna N3k')
metric_collection = {'Average speed': 1, "Stress": 20}

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

show_layout(positions=Xld, labels=dataset.labels, title="PCA")

zero_initial = np.zeros((dataset.data.shape[0], 2))

algo96 = Chalmers96(dataset=dataset,  distance_fn=manhattan, initial_layout=zero_initial,
                    damping_constant=0, spring_constant=0.5,
                    use_knnd=False, sample_set_size=10, neighbour_set_size=5)


layout = LayoutCreation.create_layout(algo96, optional_metric_collection=None, num_iters=100, )

show_layout(layout, use_labels=True, color_map='rainbow')
show_generation_metrics(layout=layout, average_speed=True, iters_from=20)

metric_collection_squad = { "Average n-tet stress": 1, "Stress": 300}


squad = SNeD(dataset=dataset, initial_layout=Xld, use_nesterovs_momentum=True, ntet_size=4, is_test=True, use_rbf_adjustment=False)
layout = LayoutCreation.create_layout(squad, num_iters=1000, optional_metric_collection=metric_collection_squad, use_decay=False)

show_layout(layout, use_labels=True, title=f"SNeD")
show_generation_metrics(layout, quartet_stress=True, title=f"SNeD" )