from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from ..config import DATA_ROOT
from pathlib import Path


class GlobeFetcher(LowLevelDataFetcherBase):

    continents = ['Africa.png', 'Europe.png', 'Asia.png', 'Australia.png', 'SouthAmerica.png', 'NorthAmerica.png']


    def load_dataset(self, **kwargs) -> (np.ndarray, np.ndarray):

        rng = np.random.RandomState(123)
        plt.figure(figsize=(20, 8))
        matplotlib.rcParams.update({'font.size': 22})

        labels = [1, 2, 3, 4, 5, 6]
        colours = []
        sizes = []

        for i, continent_image in enumerate(self.continents):
            image_arr = self._get_image_array(continent_image)

            continent_arr_bw = np.flip(np.where(image_arr[:, :, 0] < 1, 1, 0), axis=0)
            land_points = np.argwhere(continent_arr_bw)
            plt.scatter(land_points[:, 1], land_points[:, 0], s=0.5)

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






# rng = np.random.RandomState(123)
# plt.figure(figsize=(20, 8))
# matplotlib.rcParams.update({'font.size': 22})



# for i, continent_image in enumerate(continents):
#     continent_arr = imread(continent_image) / 255
#     continent_arr_bw = np.flip(np.where(continent_arr[:, :, 0] < 1, 1, 0), axis=0)
#     land_points = np.argwhere(continent_arr_bw)
#     plt.scatter(land_points[:, 1], land_points[:, 0], s=0.5)
#
#     if continent_image == 'Eurasia.png':
#         zero_meridian = (np.nonzero(continent_arr_bw)[1].max() + np.nonzero(continent_arr_bw)[1].min()) / 2
#         equator = continent_arr_bw.shape[0] / 2
#
# plt.axvline(x=zero_meridian, c='black')
# plt.axhline(y=equator, c='black')
# plt.show()

# world_map_rgb_squashed = np.round(np.sum(world_map[:,:,0:3], axis=2)/3, decimals=1)
# world_map_gray = np.flip(np.where(world_map_rgb_squashed<1, 1, 0), axis=0)
# plt.imshow(world_map_rgb_squashed)

# plt.imshow(np.where(world_map_rgb_squashed==0.3, 0, 1))

# # 0.2 africa

# # 0.3 america and outline of africa

# print(np.unique(world_map_rgb_squashed))

# zero_meridian = (np.nonzero(world_map_gray)[1].max() + np.nonzero(world_map_gray)[1].min())/2
# print(f"zero: {zero_meridian}")


# land_points = np.argwhere(world_map_gray)

# plt.figure(figsize = (20, 8))
# plt.scatter(land_points[:,1], land_points[:,0], s=1)
# plt.axvline(x=zero_meridian, c='r')
# plt.show()
