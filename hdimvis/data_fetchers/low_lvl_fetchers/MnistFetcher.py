from sklearn.datasets import fetch_openml

import numpy as np
from ..config import DATA_ROOT
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase



class MnistFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, size=1000) -> (np.ndarray, np.ndarray):
        mnist = fetch_openml("mnist_784", as_frame=False, cache=True)
        data = mnist.data[:size]
        labels = mnist.target.astype(np.int8)[:size]
        return data, labels

