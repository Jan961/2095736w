import numpy as np
import math

def wiggle_index_vector_pair(v1 : np.ndarray, v2 : np.ndarray, ld_positions: np.ndarray = None, cutoff: float = None):

    assert v1.ndim == 1 and v2.ndim == 1, "Vectors have to be 1-dimensional np arrays"

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    cos = np.dot(v1, v2)/(l1*l2)

    if ld_positions is not None:
        cutoff = calculate_cutoff(ld_positions)
    elif cutoff is not None:
        cutoff = cutoff
    else:
        cutoff = 0.5

    transformed_l1 = vector_magnitude_map_function(l1, cutoff)
    transformed_l2 = vector_magnitude_map_function(l2, cutoff)
    transformed_cos = cosine_magnitude_map_function(cos)

    return transformed_l1 * transformed_l2 * transformed_cos


def vector_magnitude_map_function(input: float, cutoff: float):

    assert cutoff > 0, "Cutoff has to be greater than 0"

    if input < cutoff:
        transformed = (input)**6
    else:
        transformed = 5*(input - cutoff) + (cutoff)**6

    return transformed


def cosine_magnitude_map_function(input: float):

    return abs((input - 1)**2)


def calculate_cutoff( ld_positions: np.ndarray):

    x_range = np.max(ld_positions[:,0]) - np.min(ld_positions[:,0])
    y_range = np.max(ld_positions[:, 1]) - np.min(ld_positions[:, 1])

    avg = (x_range + y_range)/2

    return 0.02 * avg




