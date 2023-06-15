import numpy as np
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from sklearn.decomposition import PCA
from time import perf_counter
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean
from pathlib import Path
from definitions import PROJECT_ROOT
import pickle





dataset= DataFetcher.fetch_data("metro", size=20000)
results = [[],[]]
num_repeats = 2

Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
Xld *= 10/np.std(Xld)

for i in range(num_repeats):

    no_knnd_start = perf_counter()
    algo96 = Chalmers96(dataset=dataset, distance_fn=euclidean,
             spring_constant=0.05, initial_layout=Xld,
            use_knnd=False)
    layout = LayoutCreation.create_layout(algo96, num_iters=100)
    results[0].append(perf_counter() - no_knnd_start)



    knnd_start = perf_counter()
    algo96 = Chalmers96(dataset=dataset, distance_fn=euclidean,
             spring_constant=0.05, initial_layout=Xld,
            use_knnd=True)
    layout_knnd = LayoutCreation.create_layout(algo96, num_iters=100)
    results[1].append(perf_counter() - knnd_start)


print(results)
output_dir= (Path(PROJECT_ROOT).joinpath(
    Path(f"experiments/knnd/out/"))).resolve().absolute()

path_to_pickle = (Path(output_dir).joinpath(Path(f"results_large_dataset.pickle"))).resolve()
with open(path_to_pickle, 'wb') as pickle_out:
    pickle.dump(results, pickle_out)