from typing import Dict, List

import numpy as np

from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
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
        self.results = { dataset:[] for dataset in self.dataset_names }
        self.layouts = { dataset:{} for dataset in self.dataset_names }

    def run(self):
        for dataset_name in self.dataset_names:

            # self.h()
            # self.h()
            # self.pr(f'Dataset: {dataset_name} \n')
            dataset = DataFetcher().fetch_data(dataset_name)
            # self.pr(f"{self.num_repeats} repeats of every algorithm run")
            # basci metrics which will be added to self.result
            bm = [np.zeros((self.num_repeats, 4)) for algo in self.algorithms]
            # 4 columns for memory; time; final normal stress; and (optional) final special squad stress

            # optional generation metrics which will be added to self.result
            ogm = []

            for i, algorithm in enumerate(self.algorithms):
                self.layouts[dataset_name][algorithm.name] = []

                if self.metric_collection is not None:
                    ogm.append(dict())
                    filtered_metric_collection = {metric: freq for metric, freq in self.metric_collection.items()
                                                  if metric in algorithm.available_metrics}

                    self.initialise_dict_for_optional_metrics(i, filtered_metric_collection,
                                                              ogm)

                else:
                    filtered_metric_collection = None

                for j in range(self.num_repeats):
                    basic_metrics, generation_metrics, layout = self.one_experiment(dataset, algorithm,
                                                                                     filtered_metric_collection)

                    bm[i][j][0] = basic_metrics.get('peak memory', np.NAN)
                    bm[i][j][1] = basic_metrics.get('time', np.NAN)
                    bm[i][j][2] = basic_metrics.get('final stress', np.NAN)

                    if len(generation_metrics) > 0:
                        for metric in ogm[i]:
                            ogm[i][metric][j] = \
                                np.array(generation_metrics[metric][1])


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
        optional_generation_metrics = layout.collected_metrics

        return basic_metrics, optional_generation_metrics, layout

    def initialise_dict_for_optional_metrics(self, algorithm_index: int, filtered_metric_collection: Dict,
                                             algorithms_optional_generation_metrics: List[Dict]):
        for metric, freq in filtered_metric_collection.items():
            if self.iterations % freq == 0:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, int(self.iterations / freq) +1))
            else:
                algorithms_optional_generation_metrics[algorithm_index][metric] = \
                    np.zeros((self.num_repeats, (self.iterations // freq) + 2))
