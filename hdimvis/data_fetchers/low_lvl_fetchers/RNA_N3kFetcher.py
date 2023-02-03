import numpy as np
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
from ..config import DATA_ROOT
import os

#(3000, 50)
#stress works normally

class RNA_N3kFetcher(LowLevelDataFetcherBase):

    def load_dataset(self):
        XY = np.load( os.path.join(DATA_ROOT,'RNAseq_N3k.npy'))
        return XY[:, :-1], XY[:, -1]