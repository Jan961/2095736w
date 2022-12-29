from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from ..config import DATA_ROOT
import os
from typing import Any


class PokerFetcher(LowLevelDataFetcherBase):


    def load_dataset(self, size: int =1000) -> np:
        return self._load_file(f'{size}.csv', np.int16), None

    def _load_file(self, name, dtype):
        with open(os.path.join(DATA_ROOT, f'poker\poker{name}'), encoding='utf8') as data_file:
            return np.loadtxt(
                data_file,
                skiprows=1,
                delimiter=',',
                dtype=dtype,
                comments='#'
            )






