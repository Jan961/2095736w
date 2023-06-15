from hdimvis.data_fetchers.DataFetcher import DataFetcher
import numpy as np
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from sklearn.decomposition import PCA
from hdimvis.visualise_layouts_and_metrics.plot import show_generation_metrics
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean,manhattan
from experiments.globe_and_tsne_umap_compare.plot import show_original_globe, show_globe_embedding


dataset = DataFetcher.fetch_data('globe', swiss_roll =False, size=5000)



show_original_globe(dataset)


embedding_PCA = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
embedding_PCA *= 10/np.std(embedding_PCA)
show_globe_embedding(dataset=dataset, points_2d=embedding_PCA)


metric_collection_sned = {"Average n-tet stress": 1, "Stress": 200}
sned = SNeD(dataset=dataset, initial_layout=embedding_PCA)
layout_sned = LayoutCreation.create_layout(sned, num_iters=1000, optional_metric_collection=metric_collection_sned)
show_globe_embedding(dataset=dataset, points_2d=layout_sned.get_final_positions())
show_generation_metrics(layout_sned, quartet_stress=True, title="SNeD generation metrics")


metric_collection_96 = {"Average speed": 1, "Stress": 40}
algo96 = Chalmers96(dataset=dataset, initial_layout=embedding_PCA, distance_fn=euclidean,
                    damping_constant=0, spring_constant=0.07,
                    use_knnd=False, sample_set_size=10, neighbour_set_size=5)

layout_96 = LayoutCreation().create_layout(algo96, optional_metric_collection=metric_collection_96, num_iters=100)
show_globe_embedding(dataset=dataset, points_2d=layout_96.get_final_positions())
show_generation_metrics(layout=layout_96, average_speed=True)