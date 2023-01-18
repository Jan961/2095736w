from abc import abstractmethod
from typing import Callable, List
from ..data_fetchers.Dataset import Dataset

import numpy as np
from ..distance_measures.euclidian_and_manhattan import euclidean

class BaseAlgorithm:

    name: str
    available_metrics: List[str]

    def __init__(self, dataset: Dataset = None, initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 **kwargs):
        self.dataset = dataset.data if dataset is not None else None
        self.initial_layout = initial_layout if initial_layout is not None else self._initialise_layout()
        self.distance_fn = distance_fn


    @abstractmethod
    def get_name(self):
        pass
    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_stress(self, **kwargs) -> float:
        pass
    # @abstractmethod
    # def get_available_metrics(self) -> List:
    #     pass
    # @abstractmethod
    # def get_time_per_iter(self) -> int:
    #     pass

    def get_vectorised_stress(self):
        print("  euclidian stress")
        data = self.dataset
        hd_dist = np.sqrt(((data[:,:,None] - data[:,:,None].T)**2).sum(axis=1))
        ld_dist = np.sqrt(((self.get_positions()[:,:,None] - self.get_positions()[:,:,None].T)**2).sum(axis=1))
        numerator = np.sum((hd_dist - ld_dist)**2)/4
        denominator = np.sum(ld_dist**2)/4
        if denominator == 0:
            return np.inf
        else:
            return numerator/denominator

    def _initialise_layout(self):
        if self.dataset is not None:
            return np.zeros((self.dataset.shape[0], 2))
        else:
            return None
