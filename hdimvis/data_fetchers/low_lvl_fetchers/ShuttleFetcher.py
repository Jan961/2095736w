import numpy as np
import os
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
from ..config import DATA_ROOT

# (58000, 9)
class ShuttleFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size=15000, **kwargs) -> (np.ndarray, np.ndarray):

        shuttle_train = np.loadtxt(os.path.join(DATA_ROOT,'shuttle.trn'))
        shuttle_test = np.loadtxt(os.path.join(DATA_ROOT,'shuttle.tst'))
        shuttle = np.vstack([shuttle_train, shuttle_test])
        indices = np.random.choice(57999, size=size, replace=False)
        data = shuttle[:, :-1][indices]
        labels = shuttle[:, -1][indices]

        return data, labels

