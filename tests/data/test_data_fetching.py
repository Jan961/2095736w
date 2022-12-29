import hdimvis.data_fetchers.low_lvl_fetchers
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.PokerFetcher import PokerFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.BondsFetcher import BondsFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.COIL20Fetcher import COIL20Fetcher
from hdimvis.data_fetchers.low_lvl_fetchers.AirfoilNoiseFetcher import AirfoilNoiseFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.RNA_N3kFetcher import RNA_N3kFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.WineQualityFetcher import WineQualityFetcher
import numpy as np




datasets= ['poker', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality']
fetchers = [PokerFetcher, BondsFetcher, COIL20Fetcher, RNA_N3kFetcher, AirfoilNoiseFetcher, WineQualityFetcher]


# noinspection PyTypeHints
def test_data_fetched_correctly():
    for i, dataset in enumerate(datasets):
        fetcher = DataFetcher()
        data, labels = fetcher.fetch_data(dataset)
        assert isinstance(fetcher.low_lvl_data_fetcher, fetchers[i])
        assert isinstance(data, np.ndarray)
        assert isinstance(labels, np.ndarray) or labels is None



