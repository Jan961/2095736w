import numpy as np
from numpy import sqrt
from .gradients import compute_quartet_grads
from ..BaseAlgorithm import BaseAlgorithm








# does the whole gradient descent process for the basic algorithm
# target dimension = 2



def fast_distance_scaling_update(N, X_LD, LR, perms, batches_idxes, grad_acc,  Xhd, squared_D, Dhd_quartet, relative_rbf_distance):
    grad_acc.fill(0.)

    for batch_idx in batches_idxes:
        quartet     = perms[batch_idx]
        LD_points   = X_LD[quartet]

        # compute quartet's HD distances
        if squared_D: # during exaggeration: dont take the square root of the distances
            Dhd_quartet[0] = np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2)
            Dhd_quartet[1] = np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2)
            Dhd_quartet[2] = np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2)
            Dhd_quartet[3] = np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2)
            Dhd_quartet[4] = np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2)
            Dhd_quartet[5] = np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2)
        else:
            Dhd_quartet[0] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[1]])**2))
            Dhd_quartet[1] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[2]])**2))
            Dhd_quartet[2] = sqrt(np.sum((Xhd[quartet[0]] - Xhd[quartet[3]])**2))
            Dhd_quartet[3] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[2]])**2))
            Dhd_quartet[4] = sqrt(np.sum((Xhd[quartet[1]] - Xhd[quartet[3]])**2))
            Dhd_quartet[5] = sqrt(np.sum((Xhd[quartet[2]] - Xhd[quartet[3]])**2))

        if relative_rbf_distance:
            quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet))
        else:
            Dhd_quartet  /= np.sum(Dhd_quartet)
            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)

        grad_acc[quartet[0], 0] += quartet_grads[0]
        grad_acc[quartet[0], 1] += quartet_grads[1]
        grad_acc[quartet[1], 0] += quartet_grads[2]
        grad_acc[quartet[1], 1] += quartet_grads[3]
        grad_acc[quartet[2], 0] += quartet_grads[4]
        grad_acc[quartet[2], 1] += quartet_grads[5]
        grad_acc[quartet[3], 0] += quartet_grads[6]
        grad_acc[quartet[3], 1] += quartet_grads[7]

    X_LD -= LR*grad_acc


def relative_rbf_dists(Dhd_quartet):
    rel_dists = np.exp((Dhd_quartet-np.min(Dhd_quartet)) / (-2*np.std(Dhd_quartet)))
    rel_dists = 1 - rel_dists
    rel_dists /= np.sum(rel_dists)
    return rel_dists
