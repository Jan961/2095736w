import numpy as np

side = 10
class Cube:
    def __init__(self, side: int, angle: int, num_points: int):
        self.side = side # side length
        self.angle = angle * np.pi # the acute angle of the front side to the x-y plane in radians
        self.num_points = num_points # num points per side length

        #front
        y_len = np.cos(self.angle) * self.side
        x, y = np.meshgrid





