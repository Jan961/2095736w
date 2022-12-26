from abc import ABC, abstractmethod
import numpy as np


class LowLevelDataFetcherBase(ABC):

    #returns data and labels as a tuple
    @abstractmethod
    def fetch_dataset(self, **kwargs) -> (np.ndarray, np.ndarray) :
        pass


