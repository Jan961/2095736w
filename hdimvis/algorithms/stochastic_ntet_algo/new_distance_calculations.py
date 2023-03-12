from typing import Callable

import numpy as np
import numba



def compute_quartet_dhd(exaggerate_dist: bool, HD_points: np.ndarray, distance_fn :Callable):

    Dhd_full_matrix = distance_fn(HD_points[:, :, np.newaxis] - HD_points[:, :, np.newaxis].T, 1)

    if exaggerate_dist:     # during exaggeration: don't take the square root of the distances
        Dhd_full_matrix = Dhd_full_matrix**2

    Dhd_full_matrix += 1e-12  # for some datasets 0 distance is also apparently an issue for hd dist
                                                         #  - hence the small number added
    np.fill_diagonal(Dhd_full_matrix, 0)
    Dhd_quartet = np.triu(Dhd_full_matrix) # zero duplicate and unneeded entries
    return Dhd_quartet


def compute_quartet_dld(LD_points: np.ndarray):

    Dld_full_matrix = np.sqrt(np.sum(
        (LD_points[:, :, None] - LD_points[:, :, None].T) ** 2, axis=1))
    Dld_full_matrix += 1e-12 # add a small number just in case - to avoid zero division
    zeroed_diag_ld = Dld_full_matrix.copy()
    np.fill_diagonal(zeroed_diag_ld, 0)
    Dld_quartet = np.triu(zeroed_diag_ld)

    return Dld_full_matrix, Dld_quartet
