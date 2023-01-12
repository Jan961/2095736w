import numpy as np
import matplotlib.pyplot as plt
from generate_placement_data import get_data

from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
import math

all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist',
                     'shuttle','flow cytometry']

parents, parent_hd_distances, sample_ld_pos, r, dims, data, sample, sample_indx = get_data('coil20')

point_idx = np.random.randint(0, r)
the_point_hd = data[point_idx]
the_parent_indx = parents[point_idx]
print(f"dims: {dims}")

print(f"the point: {the_point_hd}")

radius = parent_hd_distances[point_idx]
the_parent = sample_ld_pos[the_parent_indx]


extra_space = 0
x = np.arange( the_parent[0] - radius - extra_space, the_parent[0] + radius + extra_space)
y = np.arange( the_parent[1] - radius - extra_space, the_parent[1] + radius + extra_space)
grid_size = x.size
print(f"x: {x}")
print(f"y: {y}")
xv, yv = np.meshgrid(x, y)

hd_dist= np.linalg.norm(sample- the_point_hd, axis=1)

ld_dist = np.zeros((grid_size,grid_size, sample.shape[0]))

for i in range(sample_ld_pos.shape[0]):
    xy =  np.dstack((xv - sample_ld_pos[i, 0], yv - sample_ld_pos[i, 1]))
    distance_one_p = np.linalg.norm(xy, axis=2)
    ld_dist[... ,i] = distance_one_p

diffs = (ld_dist - hd_dist[None, None,:])**2
error = np.sum((diffs**2), axis=2)

print(f"hd distances : {hd_dist}")

print(f"parent: {the_parent}")

plt.contour(x, y, error, levels= 50)
plt.plot(the_parent[0], the_parent[1], 'ro')
plt.axis('scaled')
plt.colorbar()
plt.show()