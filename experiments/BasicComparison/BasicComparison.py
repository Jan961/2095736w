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

    def __init__(self, experiment_name: str, *algorithms: BaseAlgorithm):
        super().__init__(experiment_name)
        assert len(algorithms) >= 2

    def run(self):
        for dataset_name in self.dataset_names:

            self.h()
            self.h()
            self.pr(f'Dataset: {dataset_name} \n')
            dataset = DataFetcher().fetch_data(dataset_name)
            final_metrics

            for i in range(self.num_repeats):
                pass

    def one_experiment(self, dataset: Dataset, algorithm: BaseAlgorithm):

        basic_metrics = dict()
        optional_generation_metrics = dict()

        distance_fn = poker_distance if dataset.name == 'poker' and isinstance(algorithm, Chalmers96) else euclidean

        tracemalloc.start()
        t1 = perf_counter()
        layout = LowDLayoutCreation().create_layout(algorithm, dataset, optional_metric_collection=self.metric_collection,
                                                    no_iters=200)
        time = perf_counter() - t1
        curr, peak = tracemalloc.get_tracemalloc_memory()

        stress =


        basic_metrics['final stress'] = layout.get_final_stress()

        if isinstance(algorithm, Chalmers96):



        return layout, time, curr, peak, stress
