import numpy as np
import matplotlib.pyplot as plt


class Cube:
    def __init__(self, side: int = 10, angle: float = 1/6, num_points: int = 20, distance_axes: int = 2):
        self.side = side # side length
        self.angle = angle * np.pi # the acute angle of the front side to the x-y plane in radians
        self.num_points = num_points # num points per side length
        self.distance_axes = distance_axes # perpendicular distance of the bottom front edge
        # and right/left side of the cube from the x-y and y-z planes respectively

        #front
        front_y_len = np.cos(self.angle) * self.side
        front_z_len = np.sin(self.angle) * self.side
        x, y = np.meshgrid(np.linspace(self.distance_axes, self.distance_axes+ self.side, self.num_points),
                           np.linspace(self.distance_axes, self.distance_axes+ front_y_len, self.num_points))
        z = np.tile(np.linspace(self.distance_axes, self.distance_axes+ front_z_len, self.num_points)[:,None],
                    (1,self.num_points))
        label = np.zeros_like(z) # a label which will allow us to us the same colouring on the 2D embedding plot
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
        label_top = np.ones_like(z_top) # a label which will allow us to us the same colouring on the 2D embedding plot
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
        angle2 = -(np.pi/2 - self.angle)
        rotation_matrix = np.array([[np.cos(angle2), -np.sin(angle2)],
                                    [np.sin(angle2), np.cos(angle2)]])
        rotated_yz = rotation_matrix @ yz
        #rotated_yz + np.ones_like(rotated_yz) * self.distance_axes
        translated_yz = rotated_yz + np.ones_like(rotated_yz) * self.distance_axes
        x_right = (self.distance_axes + self.side) * np.ones(translated_yz.shape[1])
        label_right = 3*np.ones(translated_yz.shape[1])
        self.right = np.concatenate((x_right[:,None],translated_yz.T, label_right[:,None]), axis=1)

        #left
        copy_right = self.right.copy()
        copy_right[:,0] = copy_right[:,0] - self.side
        copy_right[:,3] = 4
        self.left = copy_right




    def plot_yz(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(self.right[:, 1], self.right[:, 2])
    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        #front
        color = self.front[:, 2] / np.sin(self.angle)
        ax.scatter(self.front[:, 0], self.front[:, 1], self.front[:, 2], c=color, cmap='viridis')

        #top
        color = self.top[:, 2] / np.sin(np.pi/2 - self.angle)
        ax.scatter(self.top[:, 0], self.top[:, 1], self.top[:, 2], c=color, cmap='plasma')

        #bottom
        color = self.bottom[:, 2] / np.sin(np.pi / 2 - self.angle)
        ax.scatter(self.bottom[:, 0], self.bottom[:, 1], self.bottom[:, 2], c=color, cmap='plasma')

        #right
        color = self.right[:, 2]
        ax.scatter(self.right[:, 0], self.right[:, 1], self.right[:, 2], c=color, cmap='plasma')

        #left
        color = self.left[:, 2]
        ax.scatter(self.left[:, 0], self.left[:, 1], self.left[:, 2], c=color, cmap='plasma')

        ax.set_aspect('equal', adjustable='box')




