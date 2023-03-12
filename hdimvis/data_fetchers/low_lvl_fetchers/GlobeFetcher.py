from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from ..config import DATA_ROOT
from pathlib import Path

# code to create the continent png images used by this fetcher with cartopy and the idea of a using 3d points
# sampled from a world map - from https://github.com/NikolayOskolkov/tSNE_vs_UMAP_GlobalStructure
# 3d globe (rather than a 2d map rolled into a "swiss roll" as in the code above) and its implementation mine



class GlobeFetcher(LowLevelDataFetcherBase):

    continents = ['Africa.png', 'Europe.png', 'Asia.png', 'Australia.png', 'SouthAmerica.png', 'NorthAmerica.png']


    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):

        rng = np.random.RandomState(123)
        plt.figure(figsize=(20, 8))
        matplotlib.rcParams.update({'font.size': 22})

        labels = [1, 2, 3, 4, 5, 6]
        colours = ['yellow', 'limegreen', 'r', 'b','fuchsia', 'orange' ]
        sizes = []

        for i, continent_image in enumerate(self.continents):
            image_arr = self._get_image_array(continent_image)

            continent_arr_bw = np.flip(np.where(image_arr[:, :, 0] < 1, 1, 0), axis=0)
            land_points = np.argwhere(continent_arr_bw)
            plt.scatter(land_points[:, 1], land_points[:, 0], s=0.5, c=colours[i])

            if continent_image == 'Asia.png':
                zero_meridian = (np.nonzero(continent_arr_bw)[1].max() + np.nonzero(continent_arr_bw)[1].min()) / 2
                equator = continent_arr_bw.shape[0] / 2

        plt.axvline(x=zero_meridian, c='black')
        plt.axhline(y=equator, c='black')
        plt.show()

        return land_points[1], land_points[0]

    def _get_image_array(self, img_name: str):
        image_path = Path(DATA_ROOT).joinpath('globe_3d', img_name)
        return imread(image_path)


