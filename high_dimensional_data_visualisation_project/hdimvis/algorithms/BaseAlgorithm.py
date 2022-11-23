from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):

    def __init__(self):
        self.name: str = None

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        pass
    @abstractmethod
    def get_evaluation_metrics(self, **kwargs) -> dict:
        pass
    @abstractmethod
    def get_time_per_iter(self) -> int:
        pass
    @abstractmethod
    def get_memory(self) ->int:
        pass


