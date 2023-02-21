import hdimvis.data_fetchers.low_lvl_fetchers
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.PokerFetcher import PokerFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.BondsFetcher import BondsFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.COIL20Fetcher import COIL20Fetcher
from hdimvis.data_fetchers.low_lvl_fetchers.AirfoilNoiseFetcher import AirfoilNoiseFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.RNA_N3kFetcher import RNA_N3kFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.WineQualityFetcher import WineQualityFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.FmnistFetcher import FmnistFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.ShuttleFetcher import ShuttleFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.FlowCytometryFetcher import FlowCytometryFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.MnistFetcher import MnistFetcher
from sklearn.datasets import fetch_openml
import numpy as np

dataset_names = ['poker', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist', 'shuttle',
            'flow cytometry', 'mnist']
fetchers = [PokerFetcher, BondsFetcher, COIL20Fetcher, RNA_N3kFetcher, AirfoilNoiseFetcher, WineQualityFetcher,
            FmnistFetcher, ShuttleFetcher, FlowCytometryFetcher, MnistFetcher]


# noinspection PyTypeHints
def test_data_fetched_correctly():
    for i, dataset_name in enumerate(dataset_names):
        fetcher = DataFetcher()
        dataset = fetcher.fetch_data(dataset_name)
        assert isinstance(dataset.data, np.ndarray)
        assert isinstance(dataset.labels, np.ndarray) or dataset.labels is None
        assert dataset.name == dataset_name


def test_fmnist_fetching():
    f = FmnistFetcher()
    x, y = f.load_dataset(size=100)
    assert y.min() >= 0 and y.max() <= 9

def test_mnist_fetching():
    m = MnistFetcher()
    x, y = m.load_dataset(size=100)
    assert y.min() >= 0 and y.max() <= 9
