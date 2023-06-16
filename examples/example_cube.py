import numpy as np
import umap
from experiments.cube.Cube import Cube
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from sklearn.decomposition import PCA
from  sklearn.manifold import SpectralEmbedding, MDS
from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid

from sklearn.manifold import TSNE

# **Cube Parameters**
# side : int, optional
#     length of the side of the cube
#
# angle : float, optional
#     angle between the bottom side of the cube and the x-y plane,
#     the angle is multiplied by pi so that angle=1/3 will
#     mean that the angle pi/3 will be used when generation the cube
#
# num_points: int, optional
#     number of points per side
#
# distance_axes: int, optional
#     distance from the axes, e.g. distance 1 will mean that cube will be translated
#     by the vector (1,1,1) from the position at the origin and then rotated by the *angle*


cube = Cube(num_points=100, side=30, angle=0.4)
dataset= cube.get_sample_dataset(3000)
cube.plot_3d()


cube.plot_3d(title="Cube dataset", axes_labels_off=True, size_inches=10)



embedding_PCA = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
embedding_PCA *= 10/np.std(embedding_PCA)
cube.plot_2d(layout_points=embedding_PCA, hd_points=dataset.data, title=f"PCA")


embedding_MDS = MDS(n_components=2).fit_transform(dataset.data).astype(np.float64)
cube.plot_2d(layout_points=embedding_MDS, hd_points=dataset.data, title=f"scikit-learn's MDS")


embedding_umap = umap.UMAP(n_neighbors=500).fit_transform(dataset.data)
cube.plot_2d(layout_points=embedding_umap, hd_points=dataset.data, title=f"UMAP, n_neighbours = 500",)




algo96 = Chalmers96(dataset=dataset)
layout_96 = LayoutCreation.create_layout(algo96, num_iters=100)
cube.plot_2d(layout=layout_96)


embedding_tsne = TSNE(n_components=2, perplexity=500).fit_transform(dataset.data)
cube.plot_2d(layout_points=embedding_tsne, hd_points=dataset.data, title="tSNE embedding")


sned = SNeD(dataset=dataset, ntet_size=6)
layout_sned = LayoutCreation().create_layout(sned, num_iters=1000)
cube.plot_2d(layout=layout_sned)


hybrid = Hybrid(dataset=dataset)
hybrid_layout = LayoutCreation().create_layout(hybrid)
cube.plot_2d(layout=hybrid_layout)

laplacian_eigenmaps_embedding = SpectralEmbedding(n_components=2, n_neighbors=500).fit_transform(dataset.data)
cube.plot_2d(layout_points=laplacian_eigenmaps_embedding, hd_points=dataset.data, title=f"Laplacian eigenmaps",
             save_to=r"C:\Users\Owner\Desktop\2095736w\experiments\cube\out" )