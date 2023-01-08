from typing import Dict, Tuple, List

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from abc import abstractmethod
import pickle
import numpy as np
from ..data_fetchers.Dataset import Dataset

class LowDLayoutBase:

    def __init__(self, algorithm, dataset: Dataset, metric_collection: dict[str: int] ):
        self.algorithm = algorithm
        self.final_positions = None
        self.metric_collection = metric_collection # a dict specifying what metrics to collect
        # the below is a dict where each value - a tuple- contains a list of iteration numbers at pos 0
        # and a list of collected values of a metric measured at the corresponding iteration number
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

    # collect various metrics as specified by self.metric_collection dict
    def collect_metrics(self):
        if 'stress' in self.metric_collection:
            if self.iteration_number % self.metric_collection['stress'] == 0 :
                self.collected_metrics['stress'][0].append(self.iteration_number)
                self.collected_metrics['stress'][1].append(self.algorithm.get_stress())

        if 'average speed' in self.metric_collection:
            if self.iteration_number % self.metric_collection['average speed'] == 0:
                self.collected_metrics['average speed'][0].append(self.iteration_number)
                self.collected_metrics['average speed'][1].append(self.algorithm.get_average_speed())





    def save(self):
        try:
            with open("data.pickle", "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

