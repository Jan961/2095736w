from ..Dataset import Dataset
from ..config import DATA_ROOT
from ..LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from pathlib import Path
import csv
from collections import defaultdict
from itertools import count


# (801, 20531)

class CancerRNAFetcher(LowLevelDataFetcherBase):

    def load_dataset(self):

        data_and_idx = np.genfromtxt( Path(DATA_ROOT).joinpath('TCGA-PANCAN-HiSeq-801x20531/data.csv'), delimiter=",",
                              skip_header=1)
        data = data_and_idx[:,1:]
        labels = []
        mapper = defaultdict(count().__next__)
        with open(Path(DATA_ROOT).joinpath('TCGA-PANCAN-HiSeq-801x20531/labels.csv')) as file:
            reader = csv.reader(file, delimiter=',')
            line_count = 0
            for row in reader:
                if line_count == 0:
                    line_count +=1
                else:
                    labels.append(mapper[row[1]])

        return data, np.array(labels)

