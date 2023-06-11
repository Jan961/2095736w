from typing import Dict, Tuple, List

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from abc import abstractmethod
import pickle
import numpy as np
from ..data_fetchers.Dataset import Dataset

class LowDLayoutBase:

    def __init__(self, algorithm, optional_metric_collection: dict[str: int]= None, num_iters: int = None):
        self.algorithm = algorithm
        self.final_positions = None
        self.num_iters = num_iters  #number of iterations to be performed
        self.optional_metric_collection = optional_metric_collection # a dict specifying what metrics to collect
        # the below is a dict where each value - a tuple- contains a list of iteration numbers at pos 0
        # and a list of collected values of a metric measured at the corresponding iteration number
        self.collected_metrics = {metric : ([],[]) for metric in self.algorithm.available_metrics }
        self.data = algorithm.data
        self.labels = algorithm.dataset.labels
        self.iteration_number = 0 # current number of iterations performed

        if self.num_iters is not None:
            assert self.num_iters >= 0, "Number of iterations must be greater than 0"




    # method to create the layout - it repeatedly calls "collect_metrics" as it runs
    # and increments self.iteration_number after each call to the one_iteration method of self.algorithm
    @abstractmethod
    def run(self):
        pass

    def get_final_positions(self):
        return self.final_positions
    def get_data(self):
        return self.data

    def get_final_stress(self):
        norm = 'euclidian'
        if self.optional_metric_collection and 'norm' in self.optional_metric_collection:
            norm = self.optional_metric_collection['norm']

        if not self.collected_metrics["Stress"][1]:
            self.collected_metrics['Stress'][0].append(self.iteration_number)
            self.collected_metrics['Stress'][1].append(self.algorithm.get_stress(norm=norm))
            return self.collected_metrics['Stress'][1][-1]
        else:
            return self.collected_metrics['Stress'][1][-1]

    # collect various metrics as specified by self.metric_collection dict
    # the "final" parameter is used to allow for collection of the final measurements
    # regardless of the collection interval specified by self.metric-collection
    def collect_metrics(self, final = False):

        norm = 'euclidian'
        if 'norm' in self.optional_metric_collection:
            norm= self.optional_metric_collection['norm']

        if 'Stress' in self.optional_metric_collection:
            if final or self._check_collection_interval('Stress'):
                self.collected_metrics['Stress'][0].append(self.iteration_number)
                self.collected_metrics['Stress'][1].append(self.algorithm.get_stress(norm=norm))

        if 'Average speed' in self.optional_metric_collection:
            if final or self._check_collection_interval('Average speed'):
                self.collected_metrics['Average speed'][0].append(self.iteration_number)
                self.collected_metrics['Average speed'][1].append(self.algorithm.get_average_speed())

        if 'Average n-tet stress' in self.optional_metric_collection:
            if final or self._check_collection_interval('Average n-tet stress'):
                self.collected_metrics['Average n-tet stress'][0].append(self.iteration_number)
                self.collected_metrics['Average n-tet stress'][1].append(self.algorithm.get_average_quartet_stress())

        if 'Average grad' in self.optional_metric_collection:
            if final or self._check_collection_interval('Average n-tet stress'):
                self.collected_metrics['Average grad'][0].append(self.iteration_number)
                self.collected_metrics['Average grad'][1].append(self.algorithm.get_avg_grad())


    # this is separated from metric collection to allow the calculation of metrics not to influence memory
    # usage, speed etc measurements

    def save(self):
        try:
            with open("data.pickle", "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object:", ex)

    def _check_collection_interval(self, metric: str):
        if self.iteration_number % self.optional_metric_collection[metric] == 0:
            return True
        else:
            return False






