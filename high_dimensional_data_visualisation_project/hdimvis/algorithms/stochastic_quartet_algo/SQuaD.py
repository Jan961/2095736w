from typing import Callable

import numpy as np
from numpy import sqrt
from .gradients import compute_quartet_grads
from ..BaseAlgorithm import BaseAlgorithm
from ...distance_measures.euclidian_and_manhattan import euclidean
from ...distance_measures.relative_rbf_dists import relative_rbf_dists


class SQuaD(BaseAlgorithm):
    def __init__(self, dataset: np.ndarray, initial_layout: np.ndarray,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = relative_rbf_dists):  #on other dist measure implemented yet for this algo
        super().__init__(dataset, initial_layout, distance_fn)

        self. N, M = self.dataset.shape
        self.perms = np.arange(self.N)
        self.batch_indices = np.arange((self.N - self.N % 4)).reshape((-1, 4))  # will point towards the indices for each random batch
        self.grad_acc = np.ones((self.N, 2))
        self.low_d_positions = initial_layout




 # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    def get_positions(self):
        return self.low_d_positions

    def get_memory(self) ->int:
        pass
    def get_metrics(self, **kwargs) -> dict:
        pass

    def one_iteration(self, exaggerate_dist: bool, LR:float):

        np.random.shuffle(self.perms)

        self.grad_acc.fill(0.)
        Dhd_quartet = np.zeros((6,))

        for batch_idx in self.batch_indices:
            quartet = self.perms[batch_idx]
            LD_points = self.low_d_positions[quartet]

            # compute quartet's HD distances
            if exaggerate_dist:  # during exaggeration: dont take the square root of the distances
                Dhd_quartet[0] = np.sum((self.dataset[quartet[0]] - self.dataset[quartet[1]]) ** 2)
                Dhd_quartet[1] = np.sum((self.dataset[quartet[0]] - self.dataset[quartet[2]]) ** 2)
                Dhd_quartet[2] = np.sum((self.dataset[quartet[0]] - self.dataset[quartet[3]]) ** 2)
                Dhd_quartet[3] = np.sum((self.dataset[quartet[1]] - self.dataset[quartet[2]]) ** 2)
                Dhd_quartet[4] = np.sum((self.dataset[quartet[1]] - self.dataset[quartet[3]]) ** 2)
                Dhd_quartet[5] = np.sum((self.dataset[quartet[2]] - self.dataset[quartet[3]]) ** 2)
            else:
                Dhd_quartet[0] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[1]]) ** 2))
                Dhd_quartet[1] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[2]]) ** 2))
                Dhd_quartet[2] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[3]]) ** 2))
                Dhd_quartet[3] = sqrt(np.sum((self.dataset[quartet[1]] - self.dataset[quartet[2]]) ** 2))
                Dhd_quartet[4] = sqrt(np.sum((self.dataset[quartet[1]] - self.dataset[quartet[3]]) ** 2))
                Dhd_quartet[5] = sqrt(np.sum((self.dataset[quartet[2]] - self.dataset[quartet[3]]) ** 2))

            if self.distance_fn == relative_rbf_dists:
                quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet))
            else:
                Dhd_quartet /= np.sum(Dhd_quartet)
                quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet)

            self.grad_acc[quartet[0], 0] += quartet_grads[0]
            self.grad_acc[quartet[0], 1] += quartet_grads[1]
            self.grad_acc[quartet[1], 0] += quartet_grads[2]
            self.grad_acc[quartet[1], 1] += quartet_grads[3]
            self.grad_acc[quartet[2], 0] += quartet_grads[4]
            self.grad_acc[quartet[2], 1] += quartet_grads[5]
            self.grad_acc[quartet[3], 0] += quartet_grads[6]
            self.grad_acc[quartet[3], 1] += quartet_grads[7]

        self.low_d_positions -= LR * self.grad_acc





