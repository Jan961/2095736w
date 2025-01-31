from typing import Tuple

from hdimvis.data_fetchers.LowLevelDataFetcherBase import LowLevelDataFetcherBase
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from ..config import DATA_ROOT
from pathlib import Path

# code to create the continent png images used by this fetcher, with cartopy, and the idea of a using 3d points
# sampled from a world map - from https://github.com/NikolayOskolkov/tSNE_vs_UMAP_GlobalStructure
# 3d globe_and_tsne_umap_compare (rather than a 2d map rolled into a "swiss roll" as in the code above) and its implementation mine
colours = ['yellow', 'limegreen', 'r', 'b', 'fuchsia', 'orange']


class GlobeFetcher(LowLevelDataFetcherBase):

    continents = ['Africa.png', 'Europe.png', 'Asia.png', 'Australia.png', 'SouthAmerica.png', 'NorthAmerica.png']
    labels = [1, 2, 3, 4, 5, 6]

    def load_dataset(self, size: int = 10000,
                     swiss_roll: bool = False, revolutions: float = 2, tightness: float = 1,  # swiss roll parameters
                     **kwargs)->(np.ndarray, np.ndarray):

        # col 0 is latitude and col 0 longitude !!!!
        sampled_2d_points  = self._get_sample_2d_points(size)
        # print(f"sampled: {sampled_2d_points}")

        if swiss_roll:
            max_x = sampled_2d_points[:,1].max()
            min_x = sampled_2d_points[:,1].min()
            range = max_x - min_x
            angles = ((sampled_2d_points[:,1] - min_x)/range) * 2*np.pi * revolutions
            z = sampled_2d_points[:,0]
            y = tightness * angles * np.cos(angles)
            x = tightness * angles * np.sin(angles)

        else:
            pitch = sampled_2d_points[:, 0]/1  # radius = 1 in this case because of scaling in _get_sample_2d_points
            yaw = sampled_2d_points[:, 1]/1

            # print("radius")
            # print(radius)
            # print("scaled_points[:, 1")
            # print(sampled_2d_points[:, 1].max())

            z = np.sin(pitch)
            x = np.cos(pitch)*np.sin(yaw)
            y = np.cos(pitch) *np.cos(yaw)

        points_3d = np.vstack((x,y,z)).T
        labels = sampled_2d_points[:,2]

        return points_3d, labels


    def _get_image_array(self, img_name: str):
        image_path = Path(DATA_ROOT).joinpath('globe_3d', img_name)
        return imread(image_path)

    def _get_sample_2d_points(self, size: int):

        sizes = []
        all_two_d_points = []
        for i, continent_image in enumerate(self.continents):
            image_arr = self._get_image_array(continent_image)

            continent_arr_bw = np.flip(np.where(image_arr[:, :, 0] < 1, 1, 0), axis=0)
            land_points = np.argwhere(continent_arr_bw)
            # fig = plt.figure()
            # ax = fig.add_subplot()
            # ax.scatter(land_points[:,1], land_points[:,0])

            if i == 0:
                zero_meridian = continent_arr_bw.shape[1] / 2
                equator = continent_arr_bw.shape[0] / 2
                circumference = continent_arr_bw.shape[1]
                half_meridian_length = continent_arr_bw.shape[0]


            sizes.append(land_points.shape[0])
            all_two_d_points.append(land_points)

        sampled_2d_points = []
        for i, points in enumerate(all_two_d_points):
            sample_num = int(np.ceil( (sizes[i] / sum(sizes)) * size))
            sample_indices = np.random.randint(0, points.shape[0], sample_num)
            sampled_points = points[sample_indices]
            labels = np.ones(sample_num) * self.labels[i]
            # print(f" concat{np.concatenate((sampled_points, labels[:, None]), axis=1)}")
            sampled_2d_points.append(np.concatenate((sampled_points, labels[:, None]), axis=1))

        points_2d_one_arr = np.vstack(sampled_2d_points)


        # scale x (longitude) to range [-pi, pi] and y (latitude) to range [-pi/2, pi/2]
        points_2d_one_arr[:, 1] = np.pi * (points_2d_one_arr[:, 1] - zero_meridian)/ (circumference/2)
        points_2d_one_arr[:, 0] = np.pi * (points_2d_one_arr[:, 0] - equator) / (2*(half_meridian_length/2))

        # plot 2d map
        # fig = plt.figure()
        # ax = fig.add_subplot()
        # for i in self.labels:
        #     p = points_2d_one_arr[points_2d_one_arr[:,2] == self.labels[i-1] ]
        #     print(f" p{p}")
        #     ax.scatter(p[:, 1], p[:, 0], s=0.5, c=colours[i - 1])
        #
        # plt.axvline(x=0, c='black')
        # plt.axhline(y=0, c='black')

        return points_2d_one_arr  # col 0 is latitude and col 0 longitude !!!!

