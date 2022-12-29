from typing import Dict, Tuple, List

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from abc import abstractmethod
import pickle
import numpy as np

class LowDLayoutBase:

    def __init__(self, algorithm, data:np.ndarray, labels: np.ndarray, metric_collection: dict[str: int] = None ):
        self.algorithm = algorithm
        self.final_positions = None
        self.generation_metrics: Dict[str: np.ndarray]  # values of various metrics (e.g. stress) collected during
        self.data = data                                # the generation of the layout
        self.labels = labels
        self.metric_collection = metric_collection


    # method to create the layout - it repeatedly calls "collect_metrics" as it runs
    @abstractmethod
    def run(self):
        pass

    def get_final_positions(self):
        return self.final_positions

    def collect_metrics(self):

        pass

    def save(self):
        try:
            with open("data.pickle", "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print("Error during pickling object (Possibly unsupported):", ex)

