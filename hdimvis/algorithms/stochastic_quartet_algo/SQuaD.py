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

    available_metrics = ['Stress', 'Average quartet stress']
    name = 'Stochastic Quartet Descent MDS'


 # numpy.core._exceptions._ArrayMemoryError -  stress
    def __init__(self, ntet_size: int = 4, vectorised: bool = True, **kwargs):
        super().__init__( **kwargs)

        # the optional "None" values are used to allow automatic data collection from many datasets
        self. N, M = self.data.shape if self.data is not None else (None, None)
        self.ntet_size = ntet_size
        self.perms = np.arange(self.N) if self.N is not None else None
        # will point towards the indices for each random batch
        self.batch_indices = np.arange((self.N - self.N % self.ntet_size)).reshape((-1, self.ntet_size)) if self.N is not None else None
        self.grad_acc = np.ones((self.N, 2)) if self.N is not None else None
        self.low_d_positions = self.initial_layout
        self.last_average_quartet_stress_measurement = 0
        self.vectorised = vectorised




 # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    def get_name(self):
        return self.name

    def get_positions(self):
        return self.low_d_positions

    def get_stress(self) -> float:
        return self.get_vectorised_euclidian_stress()

    def get_average_quartet_stress(self):
        return self.last_average_quartet_stress_measurement

    def one_iteration(self, exaggerate_dist: bool = False, LR:float = 550.0, calculate_average_stress: bool = False):

        assert self.data is not None

        np.random.shuffle(self.perms)

        self.grad_acc.fill(0.)
        # Dhd_quartet = np.zeros((6,))
        # Dld_quartet = np.zeros((6,))

        quartet_stress = 0

        for batch_idx in self.batch_indices:
            quartet = self.perms[batch_idx]
            LD_points = self.low_d_positions[quartet]
            # xa, ya = LD_points[0]
            # xb, yb = LD_points[1]
            # xc, yc = LD_points[2]
            # xd, yd = LD_points[3]

            # compute quartet's HD distances
            if exaggerate_dist:  # during exaggeration: don't take the square root of the distances
                Dhd_distances_matrix = np.sum(
                    (self.data[quartet][:, :, None] - self.data[quartet][:, :, None].T) ** 2, axis=1)
                Dhd_quartet = Dhd_distances_matrix[np.nonzero(np.triu(Dhd_distances_matrix))]
                # Dhd_quartet[0] = np.sum((self.data[quartet[0]] - self.data[quartet[1]]) ** 2)
                # Dhd_quartet[1] = np.sum((self.data[quartet[0]] - self.data[quartet[2]]) ** 2)
                # Dhd_quartet[2] = np.sum((self.data[quartet[0]] - self.data[quartet[3]]) ** 2)
                # Dhd_quartet[3] = np.sum((self.data[quartet[1]] - self.data[quartet[2]]) ** 2)
                # Dhd_quartet[4] = np.sum((self.data[quartet[1]] - self.data[quartet[3]]) ** 2)
                # Dhd_quartet[5] = np.sum((self.data[quartet[2]] - self.data[quartet[3]]) ** 2)
            else:
                Dhd_distances_matrix = np.sqrt(np.sum(
                    (self.data[quartet][:, :, None] - self.data[quartet][:, :, None].T) ** 2, axis=1))
                Dhd_quartet = Dhd_distances_matrix[np.nonzero(np.triu(Dhd_distances_matrix))]

                # Dhd_quartet[0] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[1]]) ** 2))
                # Dhd_quartet[1] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[2]]) ** 2))
                # Dhd_quartet[2] = sqrt(np.sum((self.dataset[quartet[0]] - self.dataset[quartet[3]]) ** 2))
                # Dhd_quartet[3] = sqrt(np.sum((self.dataset[quartet[1]] - self.dataset[quartet[2]]) ** 2))
                # Dhd_quartet[4] = sqrt(np.sum((self.dataset[quartet[1]] - self.dataset[quartet[3]]) ** 2))
                # Dhd_quartet[5] = sqrt(np.sum((self.dataset[quartet[2]] - self.dataset[quartet[3]]) ** 2))

            # print(f"HD quartet: {np.allclose(Dhd_quartet, Dhd_quartet_alt)}")
            # LD distances, add a small number just in case
            Dld_distances_matrix = np.sqrt(np.sum(
                (LD_points[:, :, None] - LD_points[:, :, None].T) ** 2, axis=1))
            Dld_quartet = Dld_distances_matrix[np.nonzero(np.triu(Dld_distances_matrix))] + 1e-12

            # Dld_quartet[0] = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) + 1e-12
            # Dld_quartet[1] = np.sqrt((xa - xc) ** 2 + (ya - yc) ** 2) + 1e-12
            # Dld_quartet[2] = np.sqrt((xa - xd) ** 2 + (ya - yd) ** 2) + 1e-12
            # Dld_quartet[3] = np.sqrt((xb - xc) ** 2 + (yb - yc) ** 2) + 1e-12
            # Dld_quartet[4] = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2) + 1e-12
            # Dld_quartet[5] = np.sqrt((xc - xd) ** 2 + (yc - yd) ** 2) + 1e-12


            # print(f"LD quartet: {np.allclose(Dld_quartet_alt, Dld_quartet)}")
            if self.distance_fn == relative_rbf_dists:
                quartet_grads = compute_quartet_grads(LD_points, relative_rbf_dists(Dhd_quartet), Dld_quartet)
            else:
                Dhd_quartet /= np.sum(Dhd_quartet)
                quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet, Dld_quartet)

            quartet_grads = quartet_grads.reshape((self.ntet_size,2))
            self.grad_acc[quartet] += quartet_grads
            # self.grad_acc[quartet[0], 0] += quartet_grads[0]
            # self.grad_acc[quartet[0], 1] += quartet_grads[1]
            # self.grad_acc[quartet[1], 0] += quartet_grads[2]
            # self.grad_acc[quartet[1], 1] += quartet_grads[3]
            # self.grad_acc[quartet[2], 0] += quartet_grads[4]
            # self.grad_acc[quartet[2], 1] += quartet_grads[5]
            # self.grad_acc[quartet[3], 0] += quartet_grads[6]
            # self.grad_acc[quartet[3], 1] += quartet_grads[7]

            if calculate_average_stress:
                quartet_stress += np.sum((Dhd_quartet/np.sum(Dhd_quartet) - Dld_quartet/np.sum(Dld_quartet))**2)

        if calculate_average_stress:
            self.last_average_quartet_stress_measurement = quartet_stress/self.batch_indices.shape[0]


        self.low_d_positions -= LR * self.grad_acc





