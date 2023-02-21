from sklearn.datasets import fetch_openml

import numpy as np
from ..config import DATA_ROOT
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase


# (70000, 784)
#stress works normally

class MnistFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size=5000) -> (np.ndarray, np.ndarray):

        mnist = fetch_openml("mnist_784", as_frame=False, cache=True)
        if size == 'max':
            size = mnist.data.shape[0]
        data = mnist.data[:size]
        labels = mnist.target.astype(np.int8)[:size]
        return data, labels


