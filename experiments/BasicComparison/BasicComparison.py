from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.data_fetchers.Dataset import Dataset
from ..Experiment import Experiment
from time import perf_counter
import tracemalloc


class BasicComparison(Experiment):

    def __init__(self, experiment_name: str, *algorithms: BaseAlgorithm ):
        super().__init__(experiment_name)
        assert len(algorithms) >=2
        self.final_eval_metrics = {algo.get_name(): {'time':[],
                                                     'peak memory':[],
                                                     'final stress':[]} for algo in algorithms}

    def run(self):
        for dataset_name in self.dataset_names:

            self.h()
            self.h()
            self.pr(f'Dataset: {dataset_name} \n')
            dataset = DataFetcher().fetch_data(dataset_name)

            for i in range(self.repeat):
                pass





    def one_experiment(self, dataset: Dataset, algorithm: BaseAlgorithm ):

        distance_fn = poker_distance if dataset.name == 'poker' and isinstance(algorithm, Chalmers96 ) else euclidean

        tracemalloc.start()
        t1 = perf_counter()
        layout = LowDLayoutCreation().create_layout(algorithm, dataset, metric_collection = self.metric_collection,
                                                                                   no_iters = 200)
        time = perf_counter() - t1
        curr, peak = tracemalloc.get_tracemalloc_memory()
        stress = layout.collected_metrics['stress'][1][-1]
        tracemalloc.stop()

        return layout, time, curr, peak, stress





