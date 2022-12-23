from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from ..distance_measures.euclidian_and_manhattan import euclidean

class BaseAlgorithm(ABC):
    def __init__(self, dataset: np.ndarray, initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean):
        self.dataset = dataset
        self.initial_layout = initial_layout if initial_layout is not None \
            else np.zeros((self.dataset.shape[0], 2))
        self.distance_fn = distance_fn

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_metrics(self, **kwargs) -> dict:
        pass
    # @abstractmethod
    # def get_time_per_iter(self) -> int:
    #     pass
    @abstractmethod
    def get_memory(self) ->int:
        pass


