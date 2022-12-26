from typing import Dict, Tuple, List

from ..algorithms import BaseAlgorithm
from abc import abstractmethod
import pickle
import numpy as np

class LowDLayoutBase:

    def __init__(self, algorithm: BaseAlgorithm, data:np.ndarray, labels: np.ndarray):
        self.algorithm = algorithm
        self.final_positions: np.ndarray = None
        self.metrics: Dict[str: np.ndarray] = None
        self.data = data
        self.labels = labels


    # method to create the layout - it repeatedly calls "collect_metrics" as it runs
    @abstractmethod
    def run(self, metric_collection: List[Tuple] =None):
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

