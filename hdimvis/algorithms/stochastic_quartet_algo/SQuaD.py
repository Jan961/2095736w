from itertools import combinations
from typing import Callable
from ...data_fetchers.Dataset import Dataset
import numpy as np
import math
from .gradients import compute_quartet_grads
from ..BaseAlgorithm import BaseAlgorithm
from ...distance_measures.euclidian_and_manhattan import euclidean
from ...distance_measures.relative_rbf_dists import relative_rbf_dists
from numpy import sqrt

#code adapted and modified from https://github.com/PierreLambert3/SQuaD-MDS
# the original implementation commented out for legibility, retained for testing,

class SQuaD(BaseAlgorithm):

    available_metrics = ['Stress', 'Average quartet stress']
    name = 'Stochastic Quartet Descent MDS'

    def __init__(self, dataset: Dataset | None, ntet_size: int = 4, nesterovs_momentum: bool = False,
                 momentum: float = 0.6, test: bool = False, **kwargs):
        super().__init__(dataset, **kwargs)

        # the optional "None" values are used to allow automatic data collection from many datasets in "Basic Comparison"
        self. N, M = self.data.shape if self.data is not None else (None, None)
        self.ntet_size = ntet_size # n-tet for: duet, trio (triplet), quartet, quintet, sextet etc.
        self.perms = np.arange(self.N) if self.N is not None else None
        # will point towards the indices for each random batch
        self.batch_indices = np.arange((self.N - self.N % self.ntet_size)).reshape((-1, self.ntet_size)) \
            if self.N is not None else None
        self.grad_acc = np.ones((self.N, 2)) if self.N is not None else None
        self.low_d_positions = self.initial_layout
        self.last_average_quartet_stress_measurement = 0
        self.test = test
        self.nesterovs_momentum = nesterovs_momentum
        self.momentum = momentum
        if self.nesterovs_momentum:
            self.nesterovs_v = np.zeros((self.N, 2)) if self.N is not None else None
            print("\n Nesterov's momentum will be used by the algorithm \n")


        if self.test: # for testing n-tet size must be set to 4
            assert self.ntet_size == 4


 # aim for an end LR of 1e-3 , if not initialised with a std of 10 (not recommended), then this value should be changed as well

    def get_positions(self) -> np.ndarray:
        return self.low_d_positions

    def get_unvectorised_euclidian_stress(self) -> float:

        numerator: float = 0.0
        denominator: float = 0.0

        for source, target in combinations(zip(self.data.tolist(), self.get_positions().tolist() ), 2):
            high_d_distance = euclidean(np.array(source[0]), np.array(target[0]))
            low_d_distance = math.sqrt((target[1][0] - source[1][0]) ** 2 + (target[1][1] - source[1][1]) ** 2)
            numerator += (high_d_distance - low_d_distance) ** 2
            denominator += low_d_distance ** 2
        if denominator == 0:
            return math.inf
        return numerator / denominator

    def get_average_quartet_stress(self):
        return self.last_average_quartet_stress_measurement

    def one_iteration(self, exaggerate_dist: bool = False, LR:float = 550.0, calculate_average_stress: bool = False):

        assert self.data is not None

        np.random.shuffle(self.perms)

        self.grad_acc.fill(0.)
        # Dhd_quartet_og = np.zeros((6,))
        # Dld_quartet_og = np.zeros((6,))

        quartet_stress = 0

        for batch_idx in self.batch_indices:
            quartet = self.perms[batch_idx]
            if self.nesterovs_momentum:
                LD_points = self.low_d_positions[quartet] + self.momentum * self.nesterovs_v[quartet]
            else:
                LD_points = self.low_d_positions[quartet]

            # xa, ya = LD_points[0]
            # xb, yb = LD_points[1]
            # xc, yc = LD_points[2]
            # xd, yd = LD_points[3]

            # compute quartet's HD distances
            if exaggerate_dist:  # during exaggeration: don't take the square root of the distances
                Dhd_distances_full_matrix = np.sum(
                    (self.data[quartet][:, :, None] - self.data[quartet][:, :, None].T) ** 2, axis=1)
                Dhd_distances_full_matrix += 1e-12             #for some datasets 0 distance is also apparently an issue
                                                                # for hd dist - hence the small number
                zeroed_diag_hd = Dhd_distances_full_matrix.copy()
                np.fill_diagonal(zeroed_diag_hd, 0)
                Dhd_quartet = np.triu(zeroed_diag_hd)


                # if self.test:
                #     Dhd_quartet = Dhd_distances_full_matrix[np.nonzero(np.triu(Dhd_distances_full_matrix))]
                #     Dhd_quartet_og[0] = np.sum((self.data[quartet[0]] - self.data[quartet[1]]) ** 2)
                #     Dhd_quartet_og[1] = np.sum((self.data[quartet[0]] - self.data[quartet[2]]) ** 2)
                #     Dhd_quartet_og[2] = np.sum((self.data[quartet[0]] - self.data[quartet[3]]) ** 2)
                #     Dhd_quartet_og[3] = np.sum((self.data[quartet[1]] - self.data[quartet[2]]) ** 2)
                #     Dhd_quartet_og[4] = np.sum((self.data[quartet[1]] - self.data[quartet[3]]) ** 2)
                #     Dhd_quartet_og[5] = np.sum((self.data[quartet[2]] - self.data[quartet[3]]) ** 2)
            else:
                Dhd_distances_full_matrix = np.sqrt(np.sum(
                    (self.data[quartet][:, :, None] - self.data[quartet][:, :, None].T) ** 2, axis=1))
                Dhd_distances_full_matrix += 1e-12             #for some datasets 0 distance is also apparently an issue
                                                                # for hd dist - hence the small number
                zeroed_diag_hd = Dhd_distances_full_matrix.copy()
                np.fill_diagonal(zeroed_diag_hd, 0)
                Dhd_quartet = np.triu(zeroed_diag_hd)

                # if self.test:
                #     Dhd_quartet = Dhd_distances_full_matrix[np.nonzero(np.triu(Dhd_distances_full_matrix))]
                #     Dhd_quartet_og[0] = sqrt(np.sum((self.data[quartet[0]] - self.data[quartet[1]]) ** 2))
                #     Dhd_quartet_og[1] = sqrt(np.sum((self.data[quartet[0]] - self.data[quartet[2]]) ** 2))
                #     Dhd_quartet_og[2] = sqrt(np.sum((self.data[quartet[0]] - self.data[quartet[3]]) ** 2))
                #     Dhd_quartet_og[3] = sqrt(np.sum((self.data[quartet[1]] - self.data[quartet[2]]) ** 2))
                #     Dhd_quartet_og[4] = sqrt(np.sum((self.data[quartet[1]] - self.data[quartet[3]]) ** 2))
                #     Dhd_quartet_og[5] = sqrt(np.sum((self.data[quartet[2]] - self.data[quartet[3]]) ** 2))



            # if self.test:
            #     assert np.allclose(Dhd_quartet, Dhd_quartet_og)
            #     print("HD distances equality assertion passed")


            # LD distances, add a small number just in case
            Dld_distances_full_matrix = np.sqrt(np.sum(
                (LD_points[:, :, None] - LD_points[:, :, None].T) ** 2, axis=1))
            Dld_distances_full_matrix += 1e-12
            zeroed_diag_ld = Dld_distances_full_matrix.copy()
            np.fill_diagonal(zeroed_diag_ld, 0)
            Dld_quartet = np.triu(zeroed_diag_ld)

            # if self.test:
            #     Dld_quartet = Dld_distances_full_matrix[np.nonzero(np.triu(Dld_distances_full_matrix))]
            #     Dld_quartet_og[0] = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2) + 1e-12
            #     Dld_quartet_og[1] = np.sqrt((xa - xc) ** 2 + (ya - yc) ** 2) + 1e-12
            #     Dld_quartet_og[2] = np.sqrt((xa - xd) ** 2 + (ya - yd) ** 2) + 1e-12
            #     Dld_quartet_og[3] = np.sqrt((xb - xc) ** 2 + (yb - yc) ** 2) + 1e-12
            #     Dld_quartet_og[4] = np.sqrt((xb - xd) ** 2 + (yb - yd) ** 2) + 1e-12
            #     Dld_quartet_og[5] = np.sqrt((xc - xd) ** 2 + (yc - yd) ** 2) + 1e-12
            #     assert np.allclose(Dld_quartet_og, Dld_quartet)
            #     print("LD distances equality assertion passed")



            # after the below couple of lines the Dhd_quartet contains the relative distances
            # the distances in Dld_quartet are NOT relative and are passed as such to the compute_quartet_grads() funct
            if self.distance_fn == relative_rbf_dists:
                Dhd_quartet = relative_rbf_dists(Dhd_quartet)

            else:
                Dhd_quartet /= np.sum(Dhd_quartet)

            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet,
                                                  Dld_quartet, Dld_distances_full_matrix, self.test)

            if self.nesterovs_momentum:
                self.nesterovs_v[quartet] = self.nesterovs_v[quartet]* self.momentum - LR*quartet_grads
            else:
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

        if self.nesterovs_momentum:
            self.low_d_positions += self.nesterovs_v
        else:
            self.low_d_positions -= LR * self.grad_acc





