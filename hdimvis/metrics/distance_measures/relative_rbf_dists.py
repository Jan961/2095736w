import numpy as np

# this is just an adjustment - DO NOT use the "distance_function" parameter for algorithms
# set the "use_rbf_adjustment" to true for Sned if required instead
def relative_rbf_dists(Dhd_quartet: np.ndarray):

    Dhd_quartet_non_zero =  Dhd_quartet[np.nonzero(Dhd_quartet)]

    rel_dists = np.exp((Dhd_quartet-np.min(Dhd_quartet)) / (-2*np.std(Dhd_quartet_non_zero)))
    rel_dists = 1 - rel_dists
    rel_dists /= np.sum(rel_dists)
    return rel_dists
