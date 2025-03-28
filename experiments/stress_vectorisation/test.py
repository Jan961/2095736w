from hdimvis.metrics.stress.stress import vectorised_stress,unvectorised_stress
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean,manhattan
import numpy as np
import tracemalloc
from time import perf_counter
import definitions
import os
import matplotlib.pylab as plt
import re
from pathlib import Path
import pickle


large_dataset = DataFetcher.fetch_data('mnist', size=4000)

unable_to_alloc_sizes = []

size = 4000

while True:
    try:
        sample_indices = np.random.randint(0, 4000, size)
        sample = large_dataset.data[sample_indices]
        ld_positions = 20 * np.random.rand(sample.shape[0], 2)
        stress_ts = vectorised_stress(sample, ld_positions, euclidean)
        print(f"Computed successfully size {size}")
        break

    except Exception as exception:
        except_str = str(exception)
        print(except_str)
        mem_size = re.search(r'\d+\.\d GiB', except_str).group()
        unable_to_alloc_sizes.append((size, mem_size))
        print((size, mem_size))

    size -=2

output_dir = os.path.realpath(os.path.join(definitions.PROJECT_ROOT, "experiments/stress_vectorisation/out/"))

path_to_pickle = (Path(output_dir).joinpath(Path(f"exceptions.pickle"))).resolve()
with open(path_to_pickle, 'wb') as pickle_out:
    pickle.dump(unable_to_alloc_sizes, pickle_out)

# size = 2615
# sample_indices = np.random.randint(0, 4000, size)
# sample = large_dataset.data[sample_indices]
# ld_positions = 20 * np.random.rand(sample.shape[0], 2)
# stress_ts = vectorised_stress(sample, ld_positions, euclidean)