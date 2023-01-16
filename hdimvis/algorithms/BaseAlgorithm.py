from abc import abstractmethod
from typing import Callable
from ..data_fetchers.Dataset import Dataset

import numpy as np
from ..distance_measures.euclidian_and_manhattan import euclidean

class BaseAlgorithm:
    name: str
    def __init__(self, dataset: Dataset, initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean):
        self.dataset = dataset.data
        self.initial_layout = initial_layout if initial_layout is not None \
            else np.zeros((self.dataset.shape[0], 2))
        self.distance_fn = distance_fn
        self.available_metrics = ['stress']


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
    # def get_time_per_iter(self) -> int:
    #     pass
    @abstractmethod
    def get_memory(self) ->int:
        pass

    def get_vectorised_stress(self):
        print("calculating euclidian stress")
        data = self.dataset
        hd_dist = np.sqrt(((data[:,:,None] - data[:,:,None].T)**2).sum(axis=1)/2)
        ld_dist = np.sqrt(((self.get_positions()[:,:,None] - self.get_positions()[:,:,None].T)**2).sum(axis=1)/2)
        numerator = np.sum((hd_dist - ld_dist)**2)
        denominator = np.sum(ld_dist**2)/2
        if denominator == 0:
            return np.inf
        else:
            return numerator/denominator


