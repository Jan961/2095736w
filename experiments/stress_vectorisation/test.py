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


large_dataset = DataFetcher.fetch_data('mnist', size='max')

unable_to_alloc_sizes = []

size = 4000

while True:
    try:
        sample_indices = np.random.randint(0, 69999, size)
        sample = large_dataset.data[sample_indices]
        ld_positions = 20 * np.random.rand(sample.shape[0], 2)
        stress_ts = vectorised_stress(sample, ld_positions, euclidean)
        print(f"Computed successfully size {size}")
        break

    except np.core._exceptions._ArrayMemoryError as exception:
        except_str = str(exception)
        print(except_str)
        mem_size = re.search(r'\d+\.\d GiB', except_str).group()
        unable_to_alloc_sizes.append((size, mem_size))
        print((size, mem_size))

    size -=2


