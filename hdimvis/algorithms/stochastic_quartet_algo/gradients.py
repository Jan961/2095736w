import numpy as np
from .original_calculations import compute_quartet_grads_original




def compute_quartet_grads(points : np.ndarray, Dhd : np.ndarray, Dld : np.ndarray,
                          Dld_distances_full_matrix: np.ndarray , test: bool):

    sum_dld = np.sum(Dld)
    Dld_relative = Dld/sum_dld     # make the distances in Dld relative
    diffs = Dld_relative - Dhd
    first_brackets = (2/sum_dld) * diffs # first bracket of the grad formula from the Squad paper

    # accumulate gradients from ech part of the sum here
    gradients = np.zeros_like(points)


    # iterate over the upper triangle of entries of the diffs matrix
    # computing parts of the sum for each distance
    row= 0
    col = 1
    helper1 =  np.zeros(points.shape[0]) # helper matrix
    helper2 =  np.zeros(points.shape[0]) # helper matrix

    while row < diffs.shape[0] - 1:

        helper1[[row, col]] = 1 # performs a similar role to the identity matrix in the gradient formula in the paper
                                # we set BOTH the element indexed by "row" and "col" indices to 1; the rest are zero
        points_copy = points.copy()
        # computing grads separately for x (i: 0) and y (i: 1)
        for i in range(2):
            temp_points = points_copy[:,i]

            helper2[row] = temp_points[col] # helper2 allows for subtraction of relevant LD values
            helper2[col] = temp_points[row] # - see the grad formula in the paper

            first_term = (helper1*(temp_points - helper2))/Dld[row,col]
            second_term = Dld_relative[row,col] * \
                          np.sum((temp_points[:, None] - temp_points[:, None].T)/Dld_distances_full_matrix, axis=1)

            second_brackets =  first_term - second_term

            gradients[:, i] += (first_brackets[row, col] * second_brackets)

        # reset the helpers
        helper1.fill(0)
        helper2.fill(0)

        if col == diffs.shape[1] - 1: # moving to the next row of the upper triangular matrix with the diagonal ignored
            row += 1
            col = row + 1
        else:
            col += 1

    if test:
        Dhd_1dim = Dhd[np.nonzero(Dhd)] #convert to the format used by the OG grad computation
        Dld_1dim = Dld[np.nonzero(Dld)]
        assert np.allclose(gradients.ravel(), compute_quartet_grads_original(points, Dhd_1dim, Dld_1dim))
        print ("Gradient equality assertion passed")

    return gradients





