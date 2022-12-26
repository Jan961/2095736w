from ..config import DATA_ROOT
import scipy
import numpy as np
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
import os


class AirfoilNoiseFetcher(LowLevelDataFetcherBase):

    def fetch_dataset(self, normalise_with_z_score = True):
        label_idx = -1
        XY = np.genfromtxt( os.path.join(DATA_ROOT,'airfoil_noise.csv'), delimiter=";", skip_header=1)
        Y = XY[:, label_idx]
        X = np.delete(XY, label_idx, axis=1)

        if normalise_with_z_score:
            X = scipy.stats.zscore(X, axis=0)

        return X, Y