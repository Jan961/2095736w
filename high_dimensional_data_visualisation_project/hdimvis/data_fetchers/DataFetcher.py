import numpy as np


from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
from PokerFetcher import PokerFetcher


class DataFetcher:

    def create_low_lvl_data_fetcher(self, dataset_name: str) -> LowLevelDataFetcherBase:
        if dataset_name == 'poker':
            return PokerFetcher()


    def fetch_data(self, dataset_name) -> np.ndarray:
        low_lvl_data_fetcher = self.create_low_lvl_data_fetcher(dataset_name)
        data: np.ndarray = low_lvl_data_fetcher.fetch_dataset()
        return data
