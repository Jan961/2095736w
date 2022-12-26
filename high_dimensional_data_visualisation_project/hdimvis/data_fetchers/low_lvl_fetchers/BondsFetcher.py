from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from ..config import DATA_ROOT
import os


class BondsFetcher(LowLevelDataFetcherBase):

    def fetch_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):
        np.genfromtxt('data/winequality-red.csv', delimiter=";", skip_header=1)
