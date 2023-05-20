import numpy as np
import numba
#code copied from https://github.com/PierreLambert3/SQuaD-MDS

@numba.jit(nopython=True)
def original_dhd_calculation(exaggerat_dist: bool, HD_points: np.ndarray ):

    Dhd_quartet_og = np.zeros((6,))


    Dhd_quartet_og[0] = np.sum((HD_points[0] - HD_points[1]) ** 2)
    Dhd_quartet_og[1] = np.sum((HD_points[0] - HD_points[2]) ** 2)
    Dhd_quartet_og[2] = np.sum((HD_points[0] - HD_points[3]) ** 2)
    Dhd_quartet_og[3] = np.sum((HD_points[1] - HD_points[2]) ** 2)
    Dhd_quartet_og[4] = np.sum((HD_points[1] - HD_points[3]) ** 2)
    Dhd_quartet_og[5] = np.sum((HD_points[2] - HD_points[3]) ** 2)

    if not exaggerat_dist:
        Dhd_quartet_og = np.sqrt(Dhd_quartet_og)

    return Dhd_quartet_og


@numba.jit(nopython=True)
def original_dld_calculation(LD_points: np.ndarray):

    Dld_quartet_og = np.zeros((6,))

    xa, ya = LD_points[0]
    xb, yb = LD_points[1]
    xc, yc = LD_points[2]
    xd, yd = LD_points[3]

    Dld_quartet_og[0] = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) + 1e-12
    Dld_quartet_og[1] = np.sqrt((xa - xc) ** 2 + (ya - yc) ** 2) + 1e-12
    Dld_quartet_og[2] = np.sqrt((xa - xd) ** 2 + (ya - yd) ** 2) + 1e-12
    Dld_quartet_og[3] = np.sqrt((xb - xc) ** 2 + (yb - yc) ** 2) + 1e-12
    Dld_quartet_og[4] = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2) + 1e-12
    Dld_quartet_og[5] = np.sqrt((xc - xd) ** 2 + (yc - yd) ** 2) + 1e-12

    return Dld_quartet_og


# quartet gradients for a 2D projection, Dhd contains the top-right triangle of the HD distances
# the points are named a,b,c and d internally to keep track of who is who
# points shape: (4, 2)

@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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

