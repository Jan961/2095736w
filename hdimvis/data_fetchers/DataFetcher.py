from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
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
from hdimvis.data_fetchers.low_lvl_fetchers.MockDataFetcher import MockDataFetcher
from .Dataset import Dataset



class DataFetcher:

    all_datasets_list = ['poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                         'shuttle', 'flow cytometry']


    @staticmethod
    def fetch_data( dataset_name='rna N3k', for_bhtsne: bool = False, **kwargs):

        low_lvl_data_fetcher : LowLevelDataFetcherBase | None = None
        dataset_name = dataset_name.strip()
        print("#"*20)
        print(f"Fetching the \"{dataset_name}\" dataset")

        if dataset_name == 'poker':
            low_lvl_data_fetcher = PokerFetcher()
        elif dataset_name == 'bonds':
            low_lvl_data_fetcher = BondsFetcher()
        elif dataset_name == 'coil20':
            low_lvl_data_fetcher = COIL20Fetcher()
        elif dataset_name == 'rna N3k':
            low_lvl_data_fetcher = RNA_N3kFetcher()
        elif dataset_name == 'airfoil':
            low_lvl_data_fetcher = AirfoilNoiseFetcher()
        elif dataset_name == 'wine quality':
            low_lvl_data_fetcher = WineQualityFetcher()
        elif dataset_name == 'fashion mnist':
            low_lvl_data_fetcher = FmnistFetcher()
        elif dataset_name == 'shuttle':
            low_lvl_data_fetcher = ShuttleFetcher()
        elif dataset_name == 'flow cytometry':
            low_lvl_data_fetcher = FlowCytometryFetcher()
        elif dataset_name == 'mnist':
            low_lvl_data_fetcher = MnistFetcher()
        elif dataset_name == 'mock data':
            low_lvl_data_fetcher = MockDataFetcher()
        else:
            print("Dataset name not recognised")

        data, labels = low_lvl_data_fetcher.load_dataset( **kwargs)
        print("#" * 20)
        print("Dataset loaded")
        print(f"Dataset shape: {data.shape[0], data.shape[1]}")
        print("#" * 20)
        return Dataset(data, labels, dataset_name)

