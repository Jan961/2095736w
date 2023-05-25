from typing import List, Tuple
from progress.bar import IncrementalBar
import numpy as np
from .LayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_ntet_algo.SNeD import SNeD
from ..data_fetchers.Dataset import Dataset

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, num_iters: int = 50, *basic_layout_creation_parameters,
                 exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, LR: float = 550.0,
                 terminate_at: float = None, # threshold percentage difference in the 2 running means
                 # of the Average n-tet stress (of window size 20, separated by 20)
                 #  below which layout generation is terminated
                 ):
        super().__init__(num_iters, *basic_layout_creation_parameters)

        assert isinstance(self.algorithm, SNeD)
        assert num_iters is not None, "For this algorithm the number of iterations must be specified"
        self.exaggerate_D = exaggerate_D
        self.stop_exaggeration = stop_exaggeration
        self.decay = decay
        self.LR = LR
        self.decay = decay if decay is not None else np.exp(np.log(1e-3) / self.num_iters)
        self.terminate_at = terminate_at

        if self.terminate_at is not None:
            assert self.optional_metric_collection is not None
            assert self.optional_metric_collection.get("Average n-tet stress") is not None, \
                "Average n-tet stress must be calculated periodically to use it as a terminating condition"
        if self.algorithm.use_nesterovs_momentum:
            print("\n Nesterov's momentum will be used by the algorithm \n")

        assert self.num_iters is not None or self.terminate_at is not None, "Must provide a termination condition"


    def run(self):

        bar = None
        if not self.terminate_at:
            bar = IncrementalBar("Creating layout", max=self.num_iters)

        if self.exaggerate_D:  # exaggeration of HD distances by squaring
            stop_d_exa = int(self.num_iters * self.stop_exaggeration)  # iteration when we stop the exaggeration
        else:
            stop_d_exa = 0

        terminated = False
        while not terminated:

            calculate_ntet_stress = False

            # the below conditional allows us to avoid calculating avg n-tet stress (which is best
            # measured alongside other calculations during one iteration run)  on every iteration
            if self.num_iters and self.optional_metric_collection is not None:
                if self.iteration_number == self.num_iters - 1:
                    calculate_ntet_stress = True

                if self.optional_metric_collection.get('Average n-tet stress') and \
                        self._check_collection_interval('Average n-tet stress') :
                    calculate_ntet_stress = True

            if self.iteration_number == stop_d_exa:
                self.LR *= self.decay
                self.exaggerate_D = False

            self.algorithm.one_iteration(self.exaggerate_D, self.LR, calculate_ntet_stress)
            if self.optional_metric_collection is not None:
                self.collect_metrics()
            if bar:
                bar.next()
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()

        if self.num_iters is not None and self.iteration_number >= self.num_iters:
            terminated = True

        average_speed = self.algorithm.get_average_speed()
        if self.target_node_speed is not None \
                and self.target_node_speed > 0 \
                and self.target_node_speed >= average_speed:
            terminated = True


        if self.optional_metric_collection is not None:
            self.collect_metrics(final=True)
        bar.finish()



