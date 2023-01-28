import numpy as np
import os
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
from ..config import DATA_ROOT

# (58000, 9)
class ShuttleFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):

        shuttle_train = np.loadtxt(os.path.join(DATA_ROOT,'shuttle.trn'))
        shuttle_test = np.loadtxt(os.path.join(DATA_ROOT,'shuttle.tst'))
        shuttle = np.vstack([shuttle_train, shuttle_test])
        data = shuttle[:, :-1]
        labels = shuttle[:, -1]

        return data, labels

