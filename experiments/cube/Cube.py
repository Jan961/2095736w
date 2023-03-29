import numpy as np
import matplotlib.pyplot as plt
from hdimvis.data_fetchers.Dataset import Dataset
from hdimvis.create_low_d_layout.LayoutBase import LowDLayoutBase
import pathlib
from pathlib import Path


class Cube:
    def __init__(self, side: int = 10, angle: float = 1/6, num_points: int = 20, distance_axes: int = 2):
        self.side = side # side length
        self.angle = angle * np.pi # the acute angle of the front side to the x-y plane in radians
        self.num_points = num_points # num points per side length
        self.distance_axes = distance_axes # perpendicular distance of the bottom front edge from x-y and x-z
        # planes as well as the distance from the right side of the cube from the y-z plane

        self.colours = []


        #front
        front_y_len = np.cos(self.angle) * self.side
        front_z_len = np.sin(self.angle) * self.side
        x, y = np.meshgrid(np.linspace(self.distance_axes, self.distance_axes+ self.side, self.num_points),
                           np.linspace(self.distance_axes, self.distance_axes+ front_y_len, self.num_points))
        z = np.tile(np.linspace(self.distance_axes, self.distance_axes+ front_z_len, self.num_points)[:,None],
                    (1,self.num_points))
        label = np.zeros_like(z) # a label which will allow us to use the same colouring on the 2D embedding plot
        front_stacked = np.dstack((x,y,z, label))
        self.front = front_stacked.reshape(-1,4)

        #top
        top_y_len = np.cos(np.pi/2 - self.angle) * self.side
        top_z_len = np.sin(np.pi/2 - self.angle) * self.side
        x_top, y_top = np.meshgrid(np.linspace(self.distance_axes, self.distance_axes + self.side, self.num_points),
                           np.linspace(self.distance_axes+ front_y_len, self.distance_axes+ front_y_len+ top_y_len,
                                       self.num_points))
        z_top = np.tile(np.linspace(self.distance_axes+ front_z_len, self.distance_axes+ front_z_len - top_z_len ,
                                    self.num_points)[:,None], (1,self.num_points))
        label_top = np.ones_like(z_top) # a label which will allow us to use the same colouring on the 2D embedding plot
        top_stacked = np.dstack((x_top,y_top,z_top,label_top))[1:,:,:] #remove first row to eliminate duplicate points
        self.top = top_stacked.reshape(-1,4)

        #bottom
        top_copy = self.top.copy()
        top_copy[:, 1] = top_copy[:, 1] - np.cos(self.angle) * self.side
        top_copy[:,2] = top_copy[:,2] - np.sin(self.angle) * self.side
        top_copy[:,3] = 2
        self.bottom = top_copy

        #right
        y_right, z_right = np.meshgrid(np.linspace(0, self.side, self.num_points),
                                       np.linspace(0, self.side, self.num_points))
        y_right = y_right[1:-1,1:] #remove first rows and first column to eliminate duplicate points
        z_right = z_right[1:-1,1:] #remove first rows and first column to eliminate duplicate points
        yz = np.dstack((y_right,z_right)).reshape(-1,2).T
        angle2 = -(np.pi/2 - self.angle) # rotating clockwise hence the minus
        rotation_matrix = np.array([[np.cos(angle2), -np.sin(angle2)],
                                    [np.sin(angle2), np.cos(angle2)]])
        rotated_yz = rotation_matrix @ yz
        translated_yz = rotated_yz + np.ones_like(rotated_yz) * self.distance_axes
        x_right = (self.distance_axes + self.side) * np.ones(translated_yz.shape[1])
        label_right = 3*np.ones(translated_yz.shape[1])
        self.right = np.concatenate((x_right[:,None],translated_yz.T, label_right[:,None]), axis=1)

        #left
        copy_right = self.right.copy()
        copy_right[:,0] = copy_right[:,0] - self.side
        copy_right[:,3] = 4
        self.left = copy_right

    def get_colour_mapping(self, side_index: int, points: np.ndarray):

        # side indices are:
        # 0 - front
        # 1 - top
        # 2 - bottom
        # 3 - right
        # 4 - left

        if side_index == 0:
            return points[:, 2] / np.sin(self.angle)
        elif side_index == 1 or side_index == 2:
            return points[:, 2] / np.sin(np.pi/2 - self.angle)
        elif side_index == 3 or side_index == 4:
            return points[:, 2]


    def plot_3d(self, title: str = None, axes_labels_off: bool = False, size_inches: int = None ):
        fig = plt.figure()
        fig.suptitle(title)

        if size_inches:
            fig.set_size_inches(size_inches, size_inches)

        ax = fig.add_subplot(projection='3d')

        #front
        color = self.get_colour_mapping(0, self.front)
        ax.scatter(self.front[:, 0], self.front[:, 1], self.front[:, 2], c=color, cmap='viridis')

        #top
        color = self.get_colour_mapping(1, self.top)
        ax.scatter(self.top[:, 0], self.top[:, 1], self.top[:, 2], c=color, cmap='plasma_r')

        #bottom
        color = self.get_colour_mapping(2,self.bottom)
        ax.scatter(self.bottom[:, 0], self.bottom[:, 1], self.bottom[:, 2], c=color, cmap='spring_r')

        #right
        color = self.get_colour_mapping(3, self.right)
        ax.scatter(self.right[:, 0], self.right[:, 1], self.right[:, 2], c=color, cmap='cividis')

        #left
        color = self.get_colour_mapping(3, self.left)

        if axes_labels_off:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

            for line in ax.xaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.yaxis.get_ticklines():
                line.set_visible(False)
            for line in ax.zaxis.get_ticklines():
                line.set_visible(False)

        ax.scatter(self.left[:, 0], self.left[:, 1], self.left[:, 2], c=color, cmap='winter_r')

        ax.set_aspect('equal', adjustable='box')



    def get_sample_dataset(self, size: int):
        all_points = self.get_all_points()
        assert size <= all_points.shape[0]
        sample_indices = np.random.randint(0, all_points.shape[0], size)

        return Dataset(all_points[sample_indices], None, f"3d cube of {self.side}^3 points - {size} points sampled")

    def get_all_points(self):
        return np.vstack((self.front, self.top, self.bottom, self.right, self.left))

    def plot_2d(self, layout: LowDLayoutBase = None, layout_points: np.ndarray = None,
                hd_points : np.ndarray =None, opacity: float = 1, title: str = None,
                save_to: Path = None):

        assert layout is not None or layout_points is not None, "Must provide a  layout object or 2D points"

        if layout_points is not None:
            assert hd_points is not None, "Must provide both LD and HD points as numpy arrays" \
                                          " if not using a 2D layout object"
            data = hd_points
            pos = layout_points
        else:
            data = layout.get_data()
            pos = layout.get_final_positions()

        fig, ax = plt.subplots()
        cmaps = ['viridis', 'plasma_r', 'spring_r', 'cividis', 'winter_r']

        for i in range(5):
            cmap = cmaps[i]
            indices = np.squeeze(np.argwhere(data[:,3] == i))
            color = self.get_colour_mapping(i, data[indices])
            points_to_plot = pos[indices]
            ax.scatter(points_to_plot[:,0], points_to_plot[:,1], c=color, cmap=cmap, alpha= opacity)
        plt.title(title)
        plt.axis('off')

        if save_to:
            plt.savefig((Path(save_to).joinpath(Path(f"{title}.png"))).resolve())

        plt.show()









