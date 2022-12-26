from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import numpy as np
from hdimvis.data_fetchers.config import DATA_ROOT
import os
import pandas as pd


class BondsFetcher(LowLevelDataFetcherBase):
    def __init__(self):
        self.indices = [1,1,1,1]
        self.dicts = [{},{},{},{}] #dicts and an indices to convert currency iso strings into integers from 1


    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):
        output_array = np.zeros((1000,9))
        with open(os.path.join(DATA_ROOT, 'd1000.csv'), 'r') as f:
            i = 0
            j = 0  #counter to skip the header
            while True:
                input_line = f.readline()

                if not input_line:
                    break
                if j > 1:
                    arr = self._convert_line_to_arr([s.strip() for s in input_line.split(',')])
                    output_array[i] = arr
                    i += 1
                else:
                    j += 1
        self._reset()
        return output_array, None


    def _convert_line_to_arr(self,input_list):
        arr = np.zeros(9)

        idx_to_convert =[2,3,7,8] #indices of the entries in the imput list that need to be converted from
                                # string names to some numerica value

        idx_simple_convert = [0,1,4,5,6] #indices that just need a simple float to str conversion

        for i, idx in enumerate(idx_to_convert):
            if not self.dicts[i].get(input_list[idx]):
                self.dicts[i][input_list[idx]] = self.indices[i]
                self.indices[i] += 1

            arr[idx] =  self.dicts[i].get(input_list[idx])

        for idx in idx_simple_convert:
            arr[idx] = float(input_list[idx])
        return arr

    def _reset(self):
        self.indices = [1,1,1,1]
        self.dicts = [{},{},{},{}]

