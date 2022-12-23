import numpy as np
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, algorithm: BaseAlgorithm, data: np.ndarray, labels: np.ndarray,
                 metric: str = 'relative rbf distance', n_iter: int = 10, LR: int = 550, exaggerate_D: bool = False,
                 stop_exaggeration: float = 0.6):
        super().__init__(algorithm, data, labels)


SQuaD_MDS(hparams, np.random.uniform(size = 30*3).reshape((30,3)).astype(np.float64), 20*np.random.uniform(size = 30*2).reshape((30,2)).astype(np.float64))




def SQuaD_MDS(hparams, Xhd, Xld):
    N, M = Xhd.shape

    relative_rbf_distance = hparams["metric"] == "relative rbf distance" # transform the distances nonlinearly with 1 - exp(- (Dhd - min(Dhd))/(2*std(Dhd)) ) as described in the paper
    n_iter                = hparams["n iter"]
    LR                    = hparams["LR"]
    decay = np.exp(np.log(1e-3) / n_iter) # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    squared_D  = False
    stop_D_exa = 0
    if hparams["exaggerate D"]: # exaggeration of HD distances by taking them squared
        stop_D_exa = int(n_iter*hparams["stop exaggeration"]) # iteration when we stop the exaggeration
        squared_D  = True

    perms         = np.arange(N)
    batches_idxes = np.arange((N-N%4)).reshape((-1, 4)) # will point towards the indices for each random batch
    grad_acc      = np.ones((N, 2))
    Dhd_quartet   = np.zeros((6,))
    for i in range(n_iter):
        LR *= decay
        if i == stop_D_exa:
            squared_D = False

        np.random.shuffle(perms)
        fast_distance_scaling_update(N, Xld, LR, perms, batches_idxes, grad_acc, Xhd, squared_D, Dhd_quartet, relative_rbf_distance)