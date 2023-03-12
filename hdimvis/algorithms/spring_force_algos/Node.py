import numpy as np


# code adapted from 2019 Project by Iain Cattermole

class Node:
    __slots__ = ['datapoint', 'x', 'y', 'ux', 'uy', 'old_ux', 'old_uy']

    def __init__(self, datapoint: np.ndarray,
                 ux: float=0.0, uy: float=0.0) -> None:
        self.datapoint = datapoint[:-2]
        self.x = datapoint[-2]
        self.y = datapoint[-1]
        self.ux = ux
        self.uy = uy
        self.old_ux = ux
        self.old_uy = uy

    def increment_position_update(self, ux: float, uy: float) -> None:
        self.ux += ux
        self.uy += uy

    def apply_position_update(self) -> None:
        # integration step
        self.x += self.ux
        self.y += self.uy
        self.clear_position_update()

    def clear_position_update(self) -> None:
        self.old_ux = self.ux
        self.old_uy = self.uy
        self.ux = 0.0
        self.uy = 0.0

    def __str__(self) -> str:
        return f'Node<{self.datapoint}>({self.x} + {self.ux}, {self.y} + {self.uy})'

    def __repr__(self) -> str:
        return f'Node<{self.datapoint}>'