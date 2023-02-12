import numpy as np


def relative_rbf_dists(Dhd_quartet: np.ndarray):
    if Dhd_quartet.ndim > 1:
        Dhd_quartet =  Dhd_quartet[np.nonzero(Dhd_quartet)]

    rel_dists = np.exp((Dhd_quartet-np.min(Dhd_quartet)) / (-2*np.std(Dhd_quartet)))
    rel_dists = 1 - rel_dists
    rel_dists /= np.sum(rel_dists)
    return rel_dists
