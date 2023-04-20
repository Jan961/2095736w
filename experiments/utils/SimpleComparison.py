from typing import Dict

import numpy as np

from hdimvis.create_low_d_layout.LayoutCreation import LayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
from hdimvis.algorithms.spring_force_algos.SpringForceBase import SpringForceBase
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.data_fetchers.Dataset import Dataset
from experiments.utils.ComparisonBase import ComparisonBase
from time import perf_counter
import tracemalloc

# this is all very messy code

class SimpleComparison(ComparisonBase):


    def __init__(self, algorithms: dict, **kwargs):

        # the algorithms to be analysed must be passed as a dict of algorithm: name pairs
        # - the names will allow the algorithms to be distinguished in case both are of the same type
        # and to access the results easily and intuitively
        for algorithm, name in algorithms.items():
            assert isinstance(algorithm, BaseAlgorithm) and isinstance(name, str)

        super().__init__(**kwargs)

        self.clean_algorithms = algorithms
        self.algorithms = algorithms
        self.results = { dataset:{} for dataset in self.dataset_names }
        self.layouts = { dataset:{} for dataset in self.dataset_names }

    def run(self):
        for dataset_name in self.dataset_names:

            dataset = DataFetcher().fetch_data(dataset_name)

            for algorithm, name in self.algorithms.items():

                # basic metrics which will be added to self.result
                bm = {'time': [], 'final stress': [], 'peak memory': [], 'baseline memory': []}
                if isinstance(algorithm, SNeD):
                    bm['final squad stress'] = []

                # since we want to reuse the same algo with different datasets
                # given the earlier implementation now many initialisations have to be done manually:
                self._algorithm_initialisation(algorithm, name, dataset)

                # initialise data structures used for collecting measurements:
                self.layouts[dataset_name][algorithm.get_name(only_additional=True)] = []
                self.results[dataset_name][algorithm.get_name(only_additional=True)] = None

                if self.metric_collection_during_layout_creation is not None:
                    filtered_metric_collection = {metric: freq for metric, freq
                                                  in self.metric_collection_during_layout_creation.items()
                                                  if metric in algorithm.available_metrics}
                else:
                    filtered_metric_collection = None

                for j in range(self.num_repeats):
                    basic_metrics, layout = self.one_experiment( algorithm, filtered_metric_collection)
                    bm['peak memory'].append(basic_metrics.get('peak memory', np.NAN))
                    bm['baseline memory'].append(basic_metrics.get('baseline memory', np.NAN))
                    bm['time'].append(basic_metrics.get('time', np.NAN))
                    bm['final stress'].append(basic_metrics.get('final stress', np.NAN))
                    if isinstance(algorithm, SNeD):
                        bm['final squad stress'].append(basic_metrics.get('final squad stress', np.NAN))

                    self.layouts[dataset_name][algorithm.get_name(only_additional=True)].append(layout)

                self.results[dataset_name][algorithm.get_name(only_additional=True)] = bm
            self.algorithms = self.clean_algorithms     # reset the initialisations

    def one_experiment(self, algorithm: BaseAlgorithm, filtered_metric_collection: Dict[str,int]):

        # separate runs for memory measurement and runtime measurements as tracemalloc adds some runtime overhead

        basic_metrics = dict()
        if self.measure_memory_use:
            tracemalloc.start()

            # include time spent creating k-nn index/nodes in the measurements
            if isinstance(algorithm, SpringForceBase):
                self._initialise_nodes_or_knn_index(algorithm)

            layout = LayoutCreation().create_layout(algorithm,
                                                    optional_metric_collection=filtered_metric_collection,
                                                    no_iters=self.iterations)
            basic_metrics['baseline memory'], basic_metrics['peak memory'] = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        if self.measure_time:
            t1 = perf_counter()

            # include time spent building k-nn index/nodes in the measurements
            if isinstance(algorithm, SpringForceBase):
                self._initialise_nodes_or_knn_index(algorithm)

            layout = LayoutCreation().create_layout(algorithm,
                                                    optional_metric_collection=filtered_metric_collection,
                                                    no_iters=self.iterations)
            basic_metrics['time'] = perf_counter() - t1

        basic_metrics['final stress'] = layout.get_final_stress()
        if isinstance(algorithm, SNeD):
            basic_metrics['final squad stress'] = algorithm.get_average_quartet_stress()

        return basic_metrics, layout



    # since we want to reuse the same algo with different datasets and rerun the layout creation multiple times
    # given the earlier implementation now many initialisations have to be done manually
    # at the same we want to include the creation of the k-nn index in the measurements - see
    # :
    def _algorithm_initialisation(self, algorithm: BaseAlgorithm, name:str, dataset: Dataset):
        algorithm.dataset = dataset
        algorithm.data = dataset.data
        algorithm.initial_layout = algorithm.initialise_layout()
        algorithm.additional_name = name
        if isinstance(algorithm, SpringForceBase):
            algorithm.nodes = algorithm.build_nodes()
            if algorithm.use_knnd:
                algorithm.knnd_index = algorithm.create_knnd_index()

        if isinstance(algorithm, SNeD):
            algorithm.N, M = dataset.data.shape
            algorithm.perms = np.arange(algorithm.N)
            algorithm.batch_indices = np.arange((algorithm.N - algorithm.N % 4)).reshape((-1, 4))
            algorithm.grad_acc = np.ones((algorithm.N, 2))
            algorithm.low_d_positions = algorithm.initial_layout


    def _initialise_nodes_or_knn_index(self, algorithm: SpringForceBase):
        algorithm.nodes = algorithm.build_nodes()
        if algorithm.use_knnd:
            algorithm.knnd_index = algorithm.create_knnd_index()

