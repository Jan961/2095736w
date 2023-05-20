import scipy
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
from ..config import DATA_ROOT
import os
import numpy as np



#(1440, 1024)
# stress up
class COIL20Fetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size: int = 1440):
        from scipy.io import loadmat
        mat = loadmat(os.path.join(DATA_ROOT,"COIL20.mat"))
        X, Y = mat['X']+1e-10, mat['Y']
        Y = (Y.astype(int) - 1).reshape((-1,))
        indices= np.random.randint(0, X.shape[0], size=size)
        return (X[indices], Y[indices])