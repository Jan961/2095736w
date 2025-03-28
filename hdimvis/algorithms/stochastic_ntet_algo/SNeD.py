from itertools import combinations
from ...data_fetchers.Dataset import Dataset
import numpy as np
import math
from .new_gradient_calculations import compute_quartet_grads
from ..BaseAlgorithm import BaseAlgorithm
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.metrics.distance_measures.relative_rbf_dists import relative_rbf_dists
from .new_distance_calculations import compute_quartet_dhd, compute_quartet_dld
from .all_original_calculations import original_dhd_calculation, original_dld_calculation, compute_quartet_grads_original



#code adapted and modified from https://github.com/PierreLambert3/SQuaD-MDS
# the original implementation commented out for legibility, but retained for testing,

class SNeD(BaseAlgorithm):

    available_metrics = ['Stress', 'Average n-tet stress', "Average grad"]
    name = 'Stochastic N-tet Descent MDS'

    def __init__(self, dataset: Dataset | None, ntet_size: int = 4, use_nesterovs_momentum: bool = False,
                 momentum: float = 0.6, is_test: bool = False, use_rbf_adjustment: bool = False, **kwargs):
        super().__init__(dataset, **kwargs)

        assert ntet_size > 2, "Only n-tet sizes of 3 or greater are available"

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
        self.is_test = is_test
        self.use_rbf_adjustment = use_rbf_adjustment
        self.use_nesterovs_momentum = use_nesterovs_momentum
        self.momentum = momentum
        self.avg_iter_grad = 0
        if self.use_nesterovs_momentum:
            self.nesterovs_v = np.zeros((self.N, 2)) if self.N is not None else None

        assert self.distance_fn == euclidean, \
            "For now, Squad only supports euclidian distance with optional rbf adjustment"
        if self.is_test:
            assert self.ntet_size == 4, "For comparing the original grad calculation with new vectorised ones," \
                                        "n-tet size must be set to 4"


    def get_positions(self) -> np.ndarray:
        return self.low_d_positions

    def one_iteration(self, exaggerate_dist: bool = False, LR:float = 550.0, calculate_average_stress: bool = False):

        assert self.data is not None

        np.random.shuffle(self.perms)
        self.grad_acc.fill(0.)
        quartet_stress = 0

        for batch_idx in self.batch_indices:
            quartet = self.perms[batch_idx]
            if self.use_nesterovs_momentum:
                # with Nesterov's momentum we use a projection of LD points positions instead of real positions
                LD_points = self.low_d_positions[quartet] + self.momentum * self.nesterovs_v[quartet]
            else:
                LD_points = self.low_d_positions[quartet]

            HD_points = self.data[quartet]

            # HD distances between quartet points
            Dhd_quartet = compute_quartet_dhd(exaggerate_dist, HD_points, self.distance_fn)

            # LD distances between quartet points and, including a full n_tet x n_tet sized matrix for grad computation
            Dld_full_matrix, Dld_quartet = compute_quartet_dld(LD_points)

            if self.is_test:
                self._test_dist_calc( Dhd_quartet, Dld_quartet, HD_points, LD_points, exaggerate_dist)

            # after the below couple of lines the Dhd_quartet contains the relative distances
            # the distances in Dld_quartet are NOT relative and are passed as such to the compute_quartet_grads() funct
            if self.use_rbf_adjustment:
                Dhd_quartet = relative_rbf_dists(Dhd_quartet)

            else:
                Dhd_quartet /= np.sum(Dhd_quartet)

            quartet_grads = compute_quartet_grads(LD_points, Dhd_quartet,
                                                  Dld_quartet, Dld_full_matrix)

            if self.is_test:
                self._test_grad_calc(LD_points, Dhd_quartet, Dld_quartet, quartet_grads)

            if self.use_nesterovs_momentum:
                self.nesterovs_v[quartet] = self.nesterovs_v[quartet]* self.momentum - LR*quartet_grads
            else:
                self.grad_acc[quartet] += quartet_grads


            quartet_stress += np.sum((Dhd_quartet - (Dld_quartet/np.sum(Dld_quartet)))**2, dtype=np.longdouble)

        if calculate_average_stress:
            self.last_average_quartet_stress_measurement = quartet_stress/self.batch_indices.shape[0]

        self.avg_iter_grad = np.sum(np.linalg.norm(self.grad_acc, axis=1))/self.grad_acc.shape[0]

        if self.use_nesterovs_momentum:
            self.low_d_positions += self.nesterovs_v
        else:
            self.low_d_positions -= LR * self.grad_acc



    def get_average_quartet_stress(self):
        return self.last_average_quartet_stress_measurement
    def get_avg_grad(self):
        return self.avg_iter_grad

    def _test_dist_calc(self, Dhd_quartet, Dld_quartet, HD_points, LD_points, exaggerate_dist):
        Dhd_quartet_alt = Dhd_quartet[np.nonzero(Dhd_quartet)]  # convert to the form used by the og code
        assert np.allclose(Dhd_quartet_alt, original_dhd_calculation(exaggerate_dist, HD_points))
        print("HD distance equality assertion passed")

        # worth keeping in mind that the conversion in the first line above and similar conversions
        # below are not perfect since if we are very, very unluckily some relevant distances
        #  in the Dhd_quartet matrix might be zero
        # but it should be perfectly fine for testing - for perfect conversion see the one in
        # metrics.distance_measures.rbf_distance

        Dld_quartet_alt = Dld_quartet[np.nonzero(Dld_quartet)]  # convert to the form used by the og code
        assert np.allclose(Dld_quartet_alt, original_dld_calculation(LD_points))
        print("LD distance equality assertion passed")

    def _test_grad_calc(self, LD_points, Dhd_quartet, Dld_quartet, quartet_grads):
        Dhd_1dim = Dhd_quartet[np.nonzero(Dhd_quartet)]  # convert to the format used by the OG grad computation
        Dld_1dim = Dld_quartet[np.nonzero(Dld_quartet)]
        print(Dhd_1dim)
        print(Dld_1dim)
        print(f"new {quartet_grads.ravel()}")
        print(f"old {compute_quartet_grads_original(LD_points, Dhd_1dim, Dld_1dim)}")
        assert np.allclose(quartet_grads.ravel(), compute_quartet_grads_original(LD_points, Dhd_1dim, Dld_1dim))
        print("Gradient equality assertion passed")




