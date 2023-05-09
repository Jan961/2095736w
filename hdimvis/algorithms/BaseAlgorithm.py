from abc import abstractmethod
from typing import List, Callable
from ..data_fetchers.Dataset import Dataset
from ..metrics.stress.stress import vectorised_stress, unvectorised_stress
import numpy as np
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean, manhattan
import copy

class BaseAlgorithm:

    name: str
    available_metrics: List[str]

    def __init__(self, dataset: Dataset | None, additional_name: str =None,
                 initial_layout: np.ndarray = None,
                 distance_fn = euclidean,
                 **kwargs):
        self.dataset = copy.deepcopy(dataset) if dataset is not None else None
        self.data = self.dataset.data if self.dataset is not None else None
        self.initial_layout = initial_layout.copy() if initial_layout is not None else self.initialise_layout()
        self.distance_fn = distance_fn
        self.additional_name = additional_name

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass

    @abstractmethod
    def one_iteration(self, *args, **kwargs):
        pass

    def get_stress(self, norm: str = "euclidian") -> float:

        if norm == "euclidian":
            distance_fn = euclidean
        elif norm == "manhattan":
            distance_fn = manhattan

        try:
            stress = self.get_vectorised_stress(distance_fn)
            print(f"\n Computing vectorised {distance_fn.__name__} stress \n")
            return stress

        except np.core._exceptions._ArrayMemoryError:
            print("Not enough memory to allocate for a numpy array for stress calculation. \n"
                  "Stress will be calculated with a Python loop")
            print(f"\n Computing vectorised {distance_fn.__name__} stress \n")
            stress = self.get_unvectorised_stress(distance_fn)
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

    def get_vectorised_stress(self, distance_function: Callable):
        return vectorised_stress(self.data, self.get_positions(), distance_function)


    def get_unvectorised_stress(self, distance_function: Callable):
        return unvectorised_stress(self.data, self.get_positions(), distance_function)

    def initialise_layout(self):
        if self.dataset is not None:
            print("#" * 20)
            print("The algorithm will use a random initialization for the low D embedding/layout")
            return 20*np.random.rand(self.data.shape[0], 2)
        else:
            return None
