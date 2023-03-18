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
# 3d globe (rather than a 2d map rolled into a "swiss roll" as in the code above) and its implementation mine
colours = ['yellow', 'limegreen', 'r', 'b', 'fuchsia', 'orange']


class GlobeFetcher(LowLevelDataFetcherBase):

    continents = ['Africa.png', 'Europe.png', 'Asia.png', 'Australia.png', 'SouthAmerica.png', 'NorthAmerica.png']
    labels = [1, 2, 3, 4, 5, 6]

    def load_dataset(self, size: int = 10000, **kwargs) -> (np.ndarray, np.ndarray):

        sampled_2d_points, params = self._get_sample_2d_points(size)
        print(f"sampled: {sampled_2d_points}")
        scaled_points = self._scale_2d_points(sampled_2d_points, params)

        radius = params[0]
        pitch =0.01*scaled_points[:, 1]/radius
        yaw = 0.01*scaled_points[:, 0]/radius

        print("radius")
        print(radius)

        print("scaled_points[:, 1")
        print(scaled_points[:, 1].max())

        z = np.sin(pitch)
        x = np.cos(pitch)*np.sin(yaw)
        y = np.cos(pitch) *np.cos(yaw)
        # x = np.sin(yaw) * np.cos(pitch)
        # y =  np.sin(yaw) * np.sin(pitch)
        # z =  np.cos(yaw)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        points_3d = np.vstack((x,y,z)).T
        labels = sampled_2d_points[:,2]
        print("x")
        print(scaled_points )

        colours = ['yellow', 'limegreen', 'r', 'b', 'fuchsia', 'orange']
        for i in self.labels:
            p =  points_3d[labels == i]
            print(f" p{p}")
            ax.scatter(p[:, 0], p[:, 1], p[:,2], s=0.5, c=colours[i-1])

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


            if continent_image == 'Asia.png':
                radius = (np.nonzero(continent_arr_bw)[1].max() - np.nonzero(continent_arr_bw)[1].min())/(2*np.pi)
                zero_meridian = (np.nonzero(continent_arr_bw)[1].max() + np.nonzero(continent_arr_bw)[1].min()) / 2
                equator = continent_arr_bw.shape[0] / 2


            sizes.append(land_points.shape[0])
            all_two_d_points.append(land_points)

        sampled_2d_points = []
        for i, points in enumerate(all_two_d_points):
            sample_num = int(np.floor( (sizes[i] / sum(sizes)) * size))
            sample_indices = np.random.randint(0, points.shape[0], sample_num)
            sampled_points = points[sample_indices]
            labels = np.ones((sample_num)) * self.labels[i]
            print(f" concat{np.concatenate((sampled_points, labels[:, None]), axis=1)}")
            sampled_2d_points.append(np.concatenate((sampled_points, labels[:, None]), axis=1))

        points_2d_one_arr = np.vstack(sampled_2d_points)
        print(f" one aray{points_2d_one_arr}")
        # min_x = np.min(points_2d_one_arr[:,1])
        # min_y = np.min(points_2d_one_arr[:, 0])
        # points_2d_one_arr[:,1]  -= min_x
        # points_2d_one_arr[:, 0] -= min_y
        radius1 =  radius/np.max(points_2d_one_arr[:,1])

        return points_2d_one_arr, (radius1, zero_meridian, equator)

    def _scale_2d_points(self, points: np.ndarray, params: Tuple):

        radius, zero_meridian, equator = params
        translated_y = points[:, 0].copy() - equator
        scaled_y = np.pi * translated_y /(2 * np.max(translated_y) )

        scaled_x_1 = 2* np.pi * points[:, 1].copy() /(np.max(translated_y) )
        scaled_x_2 = np.where(points[:, 0] >= zero_meridian, scaled_x_1 -zero_meridian, scaled_x_1 + zero_meridian)

        points[:, 0] = scaled_x_2
        points[:,1] =  translated_y

        print(points)

        return points
