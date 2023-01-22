import scipy
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
from ..config import DATA_ROOT
import os

#(1440, 1024)


class COIL20Fetcher(LowLevelDataFetcherBase):

    def load_dataset(self):
        from scipy.io import loadmat
        mat = loadmat(os.path.join(DATA_ROOT,"COIL20.mat"))
        X, Y = mat['X']+1e-8, mat['Y']
        Y = (Y.astype(int) - 1).reshape((-1,))
        return (X, Y)