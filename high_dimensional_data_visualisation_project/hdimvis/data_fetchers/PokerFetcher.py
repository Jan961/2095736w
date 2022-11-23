from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from hdimvis.data_fetchers.definitions import DATA_ROOT
import os
from typing import Any


class PokerFetcher(LowLevelDataFetcherBase):


    def fetch_dataset(self, size) -> Any:
        return self.load_file(f'/poker{size}.csv', np.int16)

    def _load_file(self, dtype):
        with open(os.path.join(DATA_ROOT, f'poker/'), encoding='utf8') as data_file:
            return np.loadtxt(
                data_file,
                skiprows=1,
                delimiter=',',
                dtype=dtype,
                comments='#'
            )






