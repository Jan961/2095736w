import numpy as np
import matplotlib.pyplot as plt


class Cube:
    def __init__(self, side: int = 10, angle: float = 1/6, num_points: int = 20, distance_axes: int = 2):
        self.side = side # side length
        self.angle = angle * np.pi # the acute angle of the front side to the x-y plane in radians
        self.num_points = num_points # num points per side length
        self.distance_axes = distance_axes # perpendicular distance of the cube from each of x,y and z axes

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
        x_bottom, y_bottom = np.meshgrid(np.linspace(self.distance_axes, self.distance_axes + self.side, self.num_points),
                           np.linspace(self.distance_axes, self.distance_axes + top_y_len,
                                       self.num_points))
        z_bottom = np.tile(np.linspace(self.distance_axes, self.distance_axes - top_z_len ,
                                    self.num_points)[:,None], (1,self.num_points))
        label_bottom = 2*np.ones_like(z_top) # a label which will allow us to us the same colouring on the 2D embedding plot
        bottom_stacked = np.dstack((x_bottom,y_bottom,z_bottom,label_bottom))[1:,:,:] #remove first row to eliminate duplicate points
        self.bottom = bottom_stacked.reshape(-1,4)



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


        ax.set_aspect('equal', adjustable='box')




