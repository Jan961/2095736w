import numpy as np


from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
from .PokerFetcher import PokerFetcher


def create_low_lvl_data_fetcher(dataset_name: str='poker', **kwargs) -> LowLevelDataFetcherBase:
    if dataset_name == 'poker':
        return PokerFetcher()


class DataFetcher:

    def fetch_data(self, dataset_name='poker', **kwargs):
        low_lvl_data_fetcher = create_low_lvl_data_fetcher(dataset_name, **kwargs)
        data = low_lvl_data_fetcher.fetch_dataset()
        labels = low_lvl_data_fetcher.fetch_labels()
        return data, labels
