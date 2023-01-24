import os
import fcsparser
from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from hdimvis.data_fetchers.config import DATA_ROOT
#code from https://github.com/lmcinnes/umap_paper_notebooks/blob/master/UMAP%20Flow%20cytometry%20embedding.ipynb


#(10000, 17)
# normal stress increasing

class FlowCytometryFetcher(LowLevelDataFetcherBase):

    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):
        fcs_data = fcsparser.parse(os.path.join(DATA_ROOT,'pbmc_luca.fcs'))
        raw_data = fcs_data[1]

        # this is the pre-processing used by Leland McInnes
        # he uses the last column to colour the data
        selected_columns = [col for col in raw_data.columns if col.endswith("-A")] + ['Time']
        prepped_data = np.arcsinh(raw_data[selected_columns].values / 150.0).astype(np.float32, order='C')[:10000]

        return prepped_data, None


