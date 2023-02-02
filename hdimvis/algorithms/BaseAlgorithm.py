from abc import abstractmethod
from typing import Callable, List
from ..data_fetchers.Dataset import Dataset

import numpy as np
from ..distance_measures.euclidian_and_manhattan import euclidean

class BaseAlgorithm:

    name: str
    available_metrics: List[str]

    def __init__(self, dataset: Dataset | None, additional_name: str =None,
                 initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean,
                 **kwargs):
        self.dataset = dataset
        self.data = dataset.data if dataset is not None else None
        self.initial_layout = initial_layout if initial_layout is not None else self.initialise_layout()
        self.distance_fn = distance_fn
        self.additional_name = additional_name

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_unvectorised_euclidian_stress(self):
        pass

    def get_stress(self) -> float:
        try:
            stress = self.get_vectorised_euclidian_stress()
        except np.core._exceptions._ArrayMemoryError:
            stress = self.get_unvectorised_euclidian_stress()
        finally:
            return stress


    # @abstractmethod
    # def get_available_metrics(self) -> List:
    #     pass
    # @abstractmethod
    # def get_time_per_iter(self) -> int:
    #     pass

    def get_name(self, only_additional=False):
        # only additional name as a key in result dictionaries - to make it easier to access them
        if self.additional_name is None:
            return self.name
        elif only_additional:
            return self.additional_name
        else:
            return self.name + ' - ' + self.additional_name

    def get_vectorised_euclidian_stress(self):
        print("vectorised euclidian stress")
        data = self.data
        hd_dist = np.sqrt(((data[:,:,None] - data[:,:,None].T)**2).sum(axis=1))
        ld_dist = np.sqrt(((self.get_positions()[:,:,None] - self.get_positions()[:,:,None].T)**2).sum(axis=1))
        numerator = np.sum((hd_dist - ld_dist)**2)/4
        denominator = np.sum(ld_dist**2)/4
        if denominator == 0:
            return np.inf
        else:
            return numerator/denominator

    def initialise_layout(self):
        if self.dataset is not None:
            print("The algorithm will use a random initialization for the low D embedding/layout")
            return 20*np.random.rand(self.data.shape[0], 2)
        else:
            return None
