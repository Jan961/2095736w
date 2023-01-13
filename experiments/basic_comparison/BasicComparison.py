from typing import Dict, List

import numpy as np

from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.data_fetchers.Dataset import Dataset
from ..ExperimentBase import ExperimentBase
from time import perf_counter
import tracemalloc


class BasicComparison(ExperimentBase):

    def __init__(self, experiment_name: str, *algorithms: BaseAlgorithm, dataset_names: List[str] = None,
                 metric_collection: Dict[str,int] = None, num_repeats: int = 3, record_memory=False,
                 iterations: int = 100):
        super().__init__(experiment_name, dataset_names, metric_collection, num_repeats, record_memory, iterations)
        assert len(algorithms) >= 2
        self.algorithms = algorithms

    def run(self):
        for dataset_name in self.dataset_names:

            self.h()
            self.h()
            self.pr(f'Dataset: {dataset_name} \n')
            dataset = DataFetcher().fetch_data(dataset_name)
            self.pr(f"{self.num_repeats} repeats of every algorithm run")
            algorithms_basic_metrics = [np.zeros((self.num_repeats, 4)) for algo in self.algorithms]
            # 4 columns for memory; time; final normal stress; and (optional) final special squad stress

            algorithms_optional_generation_metrics = []

            for i, algorithm in enumerate(self.algorithms):
                if self.metric_collection is not None:
                    algorithms_optional_generation_metrics.append(dict())
                    filtered_metric_collection = {metric: freq for metric, freq in self.metric_collection.items()
                                                  if metric in algorithm.available_metrics}

                    self.initialise_dict_for_optional_metrics( i, filtered_metric_collection,
                                                              algorithms_optional_generation_metrics )

                else:
                    filtered_metric_collection = None

                for j in range(self.num_repeats):
                    basic_metrics, optional_generation_metrics = self.one_experiment(dataset, algorithm,
                                                                                     filtered_metric_collection)

                    algorithms_basic_metrics[i][j][0] = basic_metrics.get('peak memory',np.NAN)
                    algorithms_basic_metrics[i][j][1] = basic_metrics.get('time', np.NAN)
                    algorithms_basic_metrics[i][j][2] = basic_metrics.get('final stress', np.NAN)

                    if len(optional_generation_metrics) > 0:
                        for metric in algorithms_optional_generation_metrics[i]:
                            algorithms_optional_generation_metrics[i][metric][j] = \
                                np.array(optional_generation_metrics[metric])


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
        optional_generation_metrics = layout.collected_metrics

        return basic_metrics, optional_generation_metrics


    def initialise_dict_for_optional_metrics(self, algorithm_index: int, filtered_metric_collection: Dict,
                                             algorithms_optional_generation_metrics: List[Dict]):
        for metric, freq in filtered_metric_collection.items():
            if self.iterations % freq == 0:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, self.iterations / freq))
            else:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, (self.iterations // freq) + 1))
