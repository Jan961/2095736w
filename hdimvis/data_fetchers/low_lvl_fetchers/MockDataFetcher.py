from ..Dataset import Dataset
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np

class MockDataFetcher(LowLevelDataFetcherBase):

    def load_dataset(self):
        mock_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
        
        return mock_data, np.array([0,1,2,3])