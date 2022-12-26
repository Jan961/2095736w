from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
from hdimvis.data_fetchers.low_lvl_fetchers.PokerFetcher import PokerFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.BondsFetcher import BondsFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.COIL20Fetcher import COIL20Fetcher
from hdimvis.data_fetchers.low_lvl_fetchers.AirfoilNoiseFetcher import AirfoilNoiseFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.RNA_N3kFetcher import RNA_N3kFetcher
from hdimvis.data_fetchers.low_lvl_fetchers.WineQualityFetcher import WineQualityFetcher



class DataFetcher:

    def fetch_data(self, dataset_name='poker', **kwargs):

        if dataset_name == 'poker':
            low_lvl_data_fetcher = PokerFetcher()
        elif dataset_name == 'bonds':
            low_lvl_data_fetcher = BondsFetcher()
        elif dataset_name == 'coil20':
            low_lvl_data_fetcher = BondsFetcher()
        elif dataset_name == 'rna N3k':
            low_lvl_data_fetcher = RNA_N3kFetcher()
        elif dataset_name == 'airfoil':
            low_lvl_data_fetcher = AirfoilNoiseFetcher()
        elif dataset_name == 'wine quality':
            low_lvl_data_fetcher = WineQualityFetcher


        data, labels = low_lvl_data_fetcher.fetch_dataset()

        return data, labels
