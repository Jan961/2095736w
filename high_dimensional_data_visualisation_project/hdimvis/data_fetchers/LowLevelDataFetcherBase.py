from abc import ABC, abstractmethod


import numpy as np


class LowLevelDataFetcherBase(ABC):

    @abstractmethod
    def fetch_dataset(self) -> np.ndarray:
        pass

    def fetch_labels(self) -> np.ndarray:
        return None
