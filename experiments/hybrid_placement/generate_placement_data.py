import numpy as np
import matplotlib.pyplot as plt

from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.data_fetchers.Dataset import Dataset
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.distance_measures.poker_distance import poker_distance
import math

all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist',
                     'shuttle','flow cytometry']

def get_data(dataset_name):
    dataset = DataFetcher().fetch_data(dataset_name)

    output_dict = dict()
    N = dataset.data.shape[0]
    dims = dataset.data.shape[1]
    sample_indx = np.random.randint(0, N, size=math.floor(math.sqrt(N)))
    sample = Dataset(dataset.data[sample_indx], None, "sample")

    algo96 = Chalmers96(dataset=sample, alpha=0.7, distance_fn=euclidean, use_knnd=False)
    layout = LowDLayoutCreation().create_layout(algo96, optional_metric_collection=None, no_iters=50)

    diffs = np.broadcast_to(dataset.data[:,:,None], shape=(dataset.data.shape[0], dataset.data.shape[1],
                                                 sample.data.shape[0])) - sample.data.T[None,:,:]

    parents = np.argmin(np.linalg.norm(diffs, axis=1), axis=1)

    parent_hd_distances = np.min(np.linalg.norm(diffs, axis=1), axis=1)
    sample_ld_pos = layout.get_final_positions()


    return parents, parent_hd_distances, sample_ld_pos, N, dims, dataset.data, sample.data, sample_indx, layout



