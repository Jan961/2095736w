from typing import Dict, Tuple, List

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from abc import abstractmethod
import pickle
import numpy as np
from ..data_fetchers.Dataset import Dataset

class LowDLayoutBase:

    def __init__(self, algorithm, dataset: Dataset, optional_metric_collection: dict[str: int]):
        self.algorithm = algorithm
        self.final_positions = None
        self.optional_metric_collection = optional_metric_collection # a dict specifying what metrics to collect
        # the below is a dict where each value - a tuple- contains a list of iteration numbers at pos 0
        # and a list of collected valuesof a metric measured at the corresponding iteration number
        self.collected_metrics = {metric : ([],[]) for metric in self.algorithm.available_metrics }
        self.data = dataset.data
        self.labels = dataset.labels
        self.iteration_number = 0 # current number of iterations performed

    # method to create the layout - it repeatedly calls "collect_metrics" as it runs
    # and increments self.iteration_number after each call to the one_iteration method of self.algorithm
    @abstractmethod
    def run(self):
        pass

    def get_final_positions(self):
        return self.final_positions

    def get_final_stress(self):
        if self.optional_metric_collection is None:
            return self.collected_metrics['stress'][1].append(self.algorithm.get_stress())
        else:
            return self.collected_metrics['stress'][1][-1]

    # collect various metrics as specified by self.metric_collection dict
    # the "final" parameter is used to allow for collection of the final measurements
    # regardless of the collection interval specified by self.metric-collection
    def collect_metrics(self, final = False):
        if 'stress' in self.optional_metric_collection:
            if final or self._check_collection_interval('stress'):
                self.collected_metrics['stress'][0].append(self.iteration_number)
                self.collected_metrics['stress'][1].append(self.algorithm.get_stress())

        if 'average speed' in self.optional_metric_collection:
            if final or self._check_collection_interval('average speed'):
                self.collected_metrics['average speed'][0].append(self.iteration_number)
                self.collected_metrics['average speed'][1].append(self.algorithm.get_average_speed())

        if 'average quartet stress' in self.optional_metric_collection:
            if final or self._check_collection_interval('average quartet stress'):
                self.collected_metrics['average quartet stress'][0].append(self.iteration_number)
                self.collected_metrics['average quartet stress'][1].append(self.algorithm.get_average_speed())


    # this is separated from metric collection to allow the calculation of metrics not to influence memory
    # usage, speed etc measurements

    def save(self):
        try:
            with open("data.pickle", "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

    def _check_collection_interval(self, metric: str):
        if self.iteration_number % self.optional_metric_collection[metric] == 0:
            return True
        else:
            return False






