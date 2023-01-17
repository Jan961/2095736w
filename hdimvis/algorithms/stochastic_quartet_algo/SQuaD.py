from itertools import combinations
from typing import Callable
from ...data_fetchers.Dataset import Dataset
import numpy as np
from numpy import sqrt
from .gradients import compute_quartet_grads
from ..BaseAlgorithm import BaseAlgorithm
from ...distance_measures.euclidian_and_manhattan import euclidean
from ...distance_measures.relative_rbf_dists import relative_rbf_dists


class SQuaD(BaseAlgorithm):
    def __init__(self, dataset: Dataset, initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean):  #on other dist measure implemented yet for this algo
        super().__init__(dataset, initial_layout, distance_fn) #the base class extracts data from the Dataset object

        self. N, M = self.dataset.shape
        self.perms = np.arange(self.N)
        self.batch_indices = np.arange((self.N - self.N % 4)).reshape((-1, 4))  # will point towards the indices for each random batch
        self.grad_acc = np.ones((self.N, 2))
        self.low_d_positions = self.initial_layout
        self.available_metrics = ['stress','average quartet stress']
        self.name = 'Stochastic Quartet Descent MDS'
        self.last_average_quartet_stress_measurement = 0



 # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    def get_name(self):
        return self.name

    def get_positions(self):
        return self.low_d_positions

    def get_stress(self) -> float:
        return self.get_vectorised_stress()

    def get_average_quartet_stress(self):
        return self.last_average_quartet_stress_measurement


    def one_iteration(self, exaggerate_dist: bool = False, LR:float = 550.0, calculate_average_stress: bool = False):

        np.random.shuffle(self.perms)

        self.grad_acc.fill(0.)
        Dhd_quartet = np.zeros((6,))
        Dld_quartet = np.zeros((6,))

        quartet_stress = 0

        for batch_idx in self.batch_indices:
            quartet = self.perms[batch_idx]
            LD_points = self.low_d_positions[quartet]
            xa, ya = LD_points[0]
            xb, yb = LD_points[1]
            xc, yc = LD_points[2]
            xd, yd = LD_points[3]


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

            # LD distances, add a small number just in case
            Dld_quartet[0] = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) + 1e-12
            Dld_quartet[1] = np.sqrt((xa - xc) ** 2 + (ya - yc) ** 2) + 1e-12
            Dld_quartet[2] = np.sqrt((xa - xd) ** 2 + (ya - yd) ** 2) + 1e-12
            Dld_quartet[3] = np.sqrt((xb - xc) ** 2 + (yb - yc) ** 2) + 1e-12
            Dld_quartet[4] = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2) + 1e-12
            Dld_quartet[5] = np.sqrt((xc - xd) ** 2 + (yc - yd) ** 2) + 1e-12

            if self.distance_fn == relative_rbf_dists:
                quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet), Dld_quartet)
            else:
                Dhd_quartet /= np.sum(Dhd_quartet)
                quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet, Dld_quartet)

            self.grad_acc[quartet[0], 0] += quartet_grads[0]
            self.grad_acc[quartet[0], 1] += quartet_grads[1]
            self.grad_acc[quartet[1], 0] += quartet_grads[2]
            self.grad_acc[quartet[1], 1] += quartet_grads[3]
            self.grad_acc[quartet[2], 0] += quartet_grads[4]
            self.grad_acc[quartet[2], 1] += quartet_grads[5]
            self.grad_acc[quartet[3], 0] += quartet_grads[6]
            self.grad_acc[quartet[3], 1] += quartet_grads[7]

            if calculate_average_stress:
                quartet_stress += np.sum((Dhd_quartet/np.sum(Dhd_quartet) - Dld_quartet/np.sum(Dld_quartet))**2)

        if calculate_average_stress:
            self.last_average_quartet_stress_measurement = quartet_stress/self.batch_indices.shape[0]


        self.low_d_positions -= LR * self.grad_acc





