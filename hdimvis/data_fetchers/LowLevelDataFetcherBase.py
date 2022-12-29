from abc import ABC, abstractmethod
import numpy as np


class LowLevelDataFetcherBase:

    #returns data and labels as a tuple
    @abstractmethod
    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):
        pass


