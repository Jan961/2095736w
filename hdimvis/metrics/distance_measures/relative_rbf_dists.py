import numpy as np

# this is just an adjustment - DO NOT use the "distance_function" parameter for algorithms
# set the "use_rbf_adjustment" to true for Sned if required instead
def relative_rbf_dists(Dhd_quartet: np.ndarray, ntet_size: int):

    # use np.longdouble because float64 overflows in exp a few lines below
    # and float128 is not available in this project setup for some reason
    Dhd_quartet_triu_entries =  np.longdouble(Dhd_quartet.copy())
    Dhd_quartet_triu_entries[np.tril_indices(ntet_size)] = np.nan
    Dhd_quartet_triu_entries = Dhd_quartet_triu_entries[np.nonzero(np.invert(np.isnan(Dhd_quartet_triu_entries)))]
    # Dhd_quartet_triu_entries = flattened entries of the upper triangular matrix

    std = np.std(Dhd_quartet_triu_entries) + 1e-13 # small number just in case - to avoid zero division
    rel_dists = np.exp((np.longdouble(Dhd_quartet)-np.min(Dhd_quartet_triu_entries)) / (-2*std))
    rel_dists = 1 - rel_dists

    np.fill_diagonal(rel_dists, 0)
    rel_dists_zeroed = np.triu(rel_dists)  # zero duplicate and unneeded entries
    # this must be done to return arr in the same format as the original Dhd_quartet passed into the function

    rel_dists_zeroed /= np.sum(rel_dists_zeroed)
    return rel_dists_zeroed
