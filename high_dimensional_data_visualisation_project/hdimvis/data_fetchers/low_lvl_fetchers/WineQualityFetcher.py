from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
import scipy
from ..config import DATA_ROOT
import os


class WineQualityFetcher(LowLevelDataFetcherBase):


    def load_dataset(self, normalise_with_zscore =True):
        XY = np.genfromtxt( os.path.join(DATA_ROOT, 'winequality-red.csv'), delimiter=";", skip_header=1)
        Y = XY[:, -1]
        X = XY[:, :-1]

        if normalise_with_zscore:
            X = scipy.stats.zscore(X, axis=0)
        return X, Y