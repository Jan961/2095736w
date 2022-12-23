import numpy as np


class Node:
    __slots__ = ['datapoint', 'x', 'y', 'vx', 'vy']

    def __init__(self, datapoint: np.ndarray,
                 vx: float=0.0, vy: float=0.0) -> None:
        self.datapoint = datapoint[:-2]
        self.x = datapoint[-2]
        self.y = datapoint[-1]
        self.vx = vx
        self.vy = vy

    def increment_velocity(self, vx: float, vy: float) -> None:
        self.vx += vx
        self.vy += vy

    def apply_velocity(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.clear_velocity()

    def clear_velocity(self) -> None:
        self.vx = 0.0
        self.vy = 0.0

    def __str__(self) -> str:
        return f'Node<{self.datapoint}>({self.x} + {self.vx}, {self.y} + {self.vy})'

    def __repr__(self) -> str:
        return f'Node<{self.datapoint}>'