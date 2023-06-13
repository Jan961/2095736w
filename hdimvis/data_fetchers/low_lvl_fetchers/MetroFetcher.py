import os
import numpy as np
from ..config import DATA_ROOT
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase


class MetroFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size=100000, **kwargs) -> (np.ndarray, np.ndarray):
        rng = np.random.default_rng()
        data = np.loadtxt( os.path.join(DATA_ROOT,"MetroPT3(AirCompressor).csv"), delimiter=',',
                           skiprows=1, usecols=(i for i in range(3,16)))
        sample_indx = rng.choice(1516947, size=size, replace=False)
        labels = None
        sample = data[sample_indx]

        print(sample)
        return sample,labels