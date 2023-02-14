import numpy as np

#code adapted and modified from https://github.com/PierreLambert3/SQuaD-MDS




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





# quartet gradients for a 2D projection, Dhd contains the top-right triangle of the HD distances
# the points are named a,b,c and d internally to keep track of who is who
# points shape: (4, 2)
def compute_quartet_grads_original(points, Dhd, Dld):


    xa, ya = points[0]
    xb, yb = points[1]
    xc, yc = points[2]
    xd, yd = points[3]

    # LD distances, add a small number just in case
    d_ab, d_ac, d_ad, d_bc, d_bd, d_cd = Dld[0], Dld[1], Dld[2], Dld[3], Dld[4], Dld[5]
    # HD distances
    pab, pac, pad, pbc, pbd, pcd = Dhd[0], Dhd[1], Dhd[2], Dhd[3], Dhd[4], Dhd[5]

    # for each element of the sum: use the same gradient function and just permute the points given in input
    gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD = ABCD_grad(
        xa, ya, xb, yb, xc, yc, xd, yd, \
        d_ab, d_ac, d_ad, d_bc, d_bd, d_cd, \
        pab)

    gxA2, gyA2, gxC2, gyC2, gxB2, gyB2, gxD2, gyD2 = ABCD_grad(
        xa, ya, xc, yc, xb, yb, xd, yd, \
        d_ac, d_ab, d_ad, d_bc, d_cd, d_bd, \
        pac)

    gxA3, gyA3, gxD3, gyD3, gxC3, gyC3, gxB3, gyB3 = ABCD_grad(
        xa, ya, xd, yd, xc, yc, xb, yb, \
        d_ad, d_ac, d_ab, d_cd, d_bd, d_bc, \
        pad)

    gxB4, gyB4, gxC4, gyC4, gxA4, gyA4, gxD4, gyD4 = ABCD_grad(
        xb, yb, xc, yc, xa, ya, xd, yd, \
        d_bc, d_ab, d_bd, d_ac, d_cd, d_ad, \
        pbc)

    gxB5, gyB5, gxD5, gyD5, gxA5, gyA5, gxC5, gyC5 = ABCD_grad(
        xb, yb, xd, yd, xa, ya, xc, yc, \
        d_bd, d_ab, d_bc, d_ad, d_cd, d_ac, \
        pbd)

    gxC6, gyC6, gxD6, gyD6, gxA6, gyA6, gxB6, gyB6 = ABCD_grad(
        xc, yc, xd, yd, xa, ya, xb, yb, \
        d_cd, d_ac, d_bc, d_ad, d_bd, d_ab, \
        pcd)

    gxA = gxA + gxA2 + gxA3 + gxA4 + gxA5 + gxA6
    gyA = gyA + gyA2 + gyA3 + gyA4 + gyA5 + gyA6

    gxB = gxB + gxB2 + gxB3 + gxB4 + gxB5 + gxB6
    gyB = gyB + gyB2 + gyB3 + gyB4 + gyB5 + gyB6

    gxC = gxC + gxC2 + gxC3 + gxC4 + gxC5 + gxC6
    gyC = gyC + gyC2 + gyC3 + gyC4 + gyC5 + gyC6

    gxD = gxD + gxD2 + gxD3 + gxD4 + gxD5 + gxD6
    gyD = gyD + gyD2 + gyD3 + gyD4 + gyD5 + gyD6

    return np.array([gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD])

def ABCD_grad(xa, ya, xb, yb, xc, yc, xd, yd, dab, dac, dad, dbc, dbd, dcd, pab):

    sum_dist = dab + dac + dad + dbc + dbd + dcd

    # relative ab distance
    dr_ab = (dab / sum_dist)

    # pab is the relative HD distance  between ab

    # the order of the terms in the difference: pab - dr_ab is flipped wrt the formula in the paper hence the order of
    # terms in the 2nd pair of brackets also flipped
    gxA = 2 * ((pab - dr_ab) / sum_dist) * ( (dab / sum_dist) * ((xa - xb) / dab + (xa - xc) / dac + (xa - xd) / dad) - (xa - xb) / dab)

    gyA = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((ya - yb) / dab + (ya - yc) / dac + (ya - yd) / dad) - (ya - yb) / dab)

    gxB = 2 * ((pab - dr_ab) / sum_dist) * ( (dab / sum_dist) * ((xb - xa) / dab + (xb - xc) / dbc + (xb - xd) / dbd) - (xb - xa) / dab)

    gyB = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((yb - ya) / dab + (yb - yc) / dbc + (yb - yd) / dbd) - (yb - ya) / dab)

    gxC = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((xc - xa) / dac + (xc - xb) / dbc + (xc - xd) / dcd))

    gyC = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((yc - ya) / dac + (yc - yb) / dbc + (yc - yd) / dcd))

    gxD = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((xd - xa) / dad + (xd - xb) / dbd + (xd - xc) / dcd))

    gyD = 2 * ((pab - dr_ab) / sum_dist) * ((dab / sum_dist) * ((yd - ya) / dad + (yd - yb) / dbd + (yd - yc) / dcd))

    return gxA, gyA, gxB, gyB, gxC, gyC, gxD, gyD