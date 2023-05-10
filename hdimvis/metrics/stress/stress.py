from itertools import combinations
from typing import Callable
import math

import numpy as np


def vectorised_stress(data: np.ndarray, ld_positions: np.ndarray, distance_function: Callable):

    hd_dist = distance_function(data[:,:,None] - data[:,:,None].T, 1)
    ld_dist = distance_function(ld_positions[:,:,None] - ld_positions[:,:,None].T, 1)
    numerator = np.sum((hd_dist - ld_dist)**2)
    denominator = np.sum(ld_dist**2)
    if denominator == 0:
        return np.inf
    else:
        return np.sqrt(numerator/denominator)


def unvectorised_stress(data: np.ndarray, ld_positions: np.ndarray, distance_function: Callable):
    numerator: float = 0.0
    denominator: float = 0.0

    for source, target in combinations(zip(data.tolist(), ld_positions.tolist()), 2):
        high_d_distance = distance_function(np.array(source[0]) - np.array(target[0]))
        low_d_distance = distance_function(np.array(source[1])-  np.array(target[1]))
        numerator += (high_d_distance - low_d_distance) ** 2
        denominator += low_d_distance ** 2
    if denominator == 0:
        return math.inf
    return math.sqrt(numerator / denominator)