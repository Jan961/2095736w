from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import manhattan,euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.visualise_layouts_and_metrics.plot import show_layout, show_generation_metrics
from sklearn.decomposition import PCA
from time import perf_counter
import numpy as np
from pathlib import Path
from definitions import PROJECT_ROOT
import pickle
#%%
rna = DataFetcher.fetch_data('rna N3k')
coli20 =  DataFetcher.fetch_data('coil20')
globe = DataFetcher.fetch_data('globe_and_tsne_umap_compare', size=100)
fmnist = DataFetcher.fetch_data('fashion mnist', size=100)
mnist = DataFetcher.fetch_data('mnist', size=100)

datasets = [rna, coli20,globe,fmnist, mnist]
dataset_names = ['rna N3k', 'coli20', 'globe_and_tsne_umap_compare', 'fashion mnist', 'mnist']
#%%
n_neigh = 1
n_sample = 1
increment_neigh = False
num_repeats = 0
results = {dataset_name : (np.zeros((20,20,num_repeats)),[[[0 for k in range(3)] for j in range(20)] for i in range(20)])
           for dataset_name in dataset_names }

for i, dataset in enumerate(datasets):
    print("outer loop")
    for j in range(num_repeats):
        algo96 = Chalmers96(dataset=dataset, distance_fn=euclidean,
                        damping_constant=0, spring_constant=0.5,
                        use_knnd=False, sample_set_size=n_neigh, neighbour_set_size=n_sample)
        start = perf_counter()
        layout = LayoutCreation.create_layout(algo96, num_iters=1)
        results[dataset_names[i]][0][n_neigh, n_sample, j] = perf_counter() - start
        results[dataset_names[i]][1][n_neigh, n_sample, j] = layout

    if increment_neigh:
        n_neigh += 1
    else:
        n_sample += 1
    increment_neigh = not increment_neigh


output_dir= (Path(PROJECT_ROOT).joinpath(
    Path(f"experiments/varing_V_and_S/out/"))).resolve().absolute()

path_to_pickle = (Path(output_dir).joinpath(Path(f"results.pickle"))).resolve()
with open(path_to_pickle, 'wb') as pickle_out:
    pickle.dump(results, pickle_out)