from typing import Dict, List

import numpy as np

from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
from hdimvis.algorithms.spring_force_algos.SpringForceBase import SpringForceBase
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.data_fetchers.Dataset import Dataset
from experiments.basic_comparison.ComparisonBase import ExperimentBase
from time import perf_counter
import tracemalloc

# this is all very messy code

class BasicComparison(ExperimentBase):

    def __init__(self, *algorithms: BaseAlgorithm, **kwargs):

        super().__init__(**kwargs)
        assert len(algorithms) >= 2
        self.algorithms = algorithms
        self.results = { dataset:{} for dataset in self.dataset_names }
        self.layouts = { dataset:{} for dataset in self.dataset_names }

    def run(self):
        for dataset_name in self.dataset_names:

            dataset = DataFetcher().fetch_data(dataset_name)

            # basci metrics which will be added to self.result
            bm = np.zeros((self.num_repeats, 4))
            # 4 columns for memory; time; final normal stress; and (optional) final special squad stress

            for i, algorithm in enumerate(self.algorithms):

                # since we want to reuse the same alog with different datasets
                # given the earlier implementation now many initialisations have to be done manually:
                self._complete_algorithm_initialisation(algorithm, dataset)

                # initialise datastructures used for collecting measurements:
                self.layouts[dataset_name][algorithm.name] = []

                if self.metric_collection is not None:
                    filtered_metric_collection = {metric: freq for metric, freq in self.metric_collection.items()
                                                  if metric in algorithm.available_metrics}
                else:
                    filtered_metric_collection = None

                for j in range(self.num_repeats):
                    basic_metrics, layout = self.one_experiment(dataset, algorithm,
                                                                                     filtered_metric_collection)
                    bm[i][j][0] = basic_metrics.get('peak memory', np.NAN)
                    bm[i][j][1] = basic_metrics.get('time', np.NAN)
                    bm[i][j][2] = basic_metrics.get('final stress', np.NAN)

                    self.layouts[dataset_name][algorithm.name].append(layout)

            self.results[dataset_name].append(bm)
            self.results[dataset_name].append(ogm)


    def one_experiment(self, dataset: Dataset, algorithm: BaseAlgorithm, filtered_metric_collection: Dict[str,int]):

        basic_metrics = dict()

        if self.record_memory:
            tracemalloc.start()
        else:
            t1 = perf_counter()

        layout = LowDLayoutCreation().create_layout(algorithm, dataset,
                                                    optional_metric_collection=filtered_metric_collection,
                                                    no_iters=self.iterations)

        if self.record_memory:
            basic_metrics['current memory'], basic_metrics['peak memory'] = tracemalloc.get_tracemalloc_memory()
            tracemalloc.stop()
        else:
            basic_metrics['time'] = perf_counter() - t1

        basic_metrics['final stress'] = layout.get_final_stress()

        return basic_metrics, layout

    def _initialise_dict_for_optional_metrics(self, algorithm_index: int, filtered_metric_collection: Dict,
                                              algorithms_optional_generation_metrics: List[Dict]):
        for metric, freq in filtered_metric_collection.items():
            if self.iterations % freq == 0:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, int(self.iterations / freq) +1))
            else:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, (self.iterations // freq) + 2))

    def _complete_algorithm_initialisation(self, algorithm: BaseAlgorithm, dataset: Dataset ):
        algorithm.dataset = dataset.data
        if isinstance(algorithm, SpringForceBase):
            algorithm.nodes = algorithm.build_nodes()
        if isinstance(algorithm, SQuaD):
            algorithm.N, M = dataset.data.shape
            algorithm.perms = np.arange(algorithm.N)
            algorithm.batch_indices = np.arange((algorithm.N - algorithm.N % 4)).reshape((-1, 4))
            algorithm.grad_acc = np.ones((algorithm.N, 2))
            algorithm.low_d_positions = algorithm.initial_layout
