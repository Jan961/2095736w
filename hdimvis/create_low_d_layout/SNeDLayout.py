from typing import List, Tuple
from progress.bar import IncrementalBar
import numpy as np
from .LayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_ntet_algo.SNeD import SNeD
from ..data_fetchers.Dataset import Dataset
import math

class SNeDLayout(LowDLayoutBase):
    def __init__(self, num_iters: int = 50, *basic_layout_creation_parameters,
                 exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, use_decay: bool = False,
                 LR: float = 550.0,
                 terminate_at: float = None, # threshold average n-tet stress value
                 #  below which layout generation is terminated
                 ):
        super().__init__(num_iters, *basic_layout_creation_parameters)

        assert isinstance(self.algorithm, SNeD)
        assert num_iters is not None, "For this algorithm the number of iterations must be specified"
        self.exaggerate_D = exaggerate_D
        self.stop_exaggeration = stop_exaggeration
        self.use_decay = use_decay
        self.LR = LR
        self.decay = decay if decay is not None else np.exp(np.log(1e-3) / self.num_iters) if self.num_iters is not None \
                    else np.exp(np.log(1e-3) / 200)
        self.terminate_at = terminate_at


        if self.terminate_at is not None:
            assert self.optional_metric_collection is not None
            assert self.optional_metric_collection["Average n-tet stress"] == 1, \
                "Average n-tet stress must be calculated on every iteration to use it as a terminating condition," \
                "please use the \"optional_metric_collection\" parameter dict to set \"Average n-tet stress\" to one"
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

            if self.use_decay:
                self.LR *= self.decay
            elif self.iteration_number == 0:
                self.LR *= self.decay

            calculate_ntet_stress = False
            if self.terminate_at:
                print(f"Iteration number: {self.iteration_number}, "
                      f"Average n-tet stress {self.algorithm.get_average_quartet_stress()}" )

            # the below conditional allows us to avoid calculating avg n-tet stress (which is best
            # measured alongside other calculations during one iteration run)  on every iteration
            if self.optional_metric_collection is not None:
                if self.num_iters and self.iteration_number == self.num_iters - 1:
                    calculate_ntet_stress = True

                if self.optional_metric_collection.get('Average n-tet stress') and \
                        self._check_collection_interval('Average n-tet stress') :
                    calculate_ntet_stress = True

            if self.iteration_number == stop_d_exa:
                self.exaggerate_D = False

            self.algorithm.one_iteration(self.exaggerate_D, self.LR, calculate_ntet_stress)
            if self.optional_metric_collection is not None:
                self.collect_metrics()

            if bar:
                bar.next()
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()

            # termination conditions
            if self.num_iters is not None and self.iteration_number >= self.num_iters:
                terminated = True

            if self.terminate_at and self.iteration_number > 100:
                latest_running_mean = sum(self.collected_metrics['Average n-tet stress'][1][-100:])/100
                print(f"Average n-tet stress running mean: {latest_running_mean}")
                if math.isclose(self.terminate_at, latest_running_mean) or self.terminate_at > latest_running_mean:
                    terminated = True

            # elif self.iteration_number > 3000:
            #     terminated = True

            # if terminated get the final measurements and wrap up the display
            if terminated and self.optional_metric_collection is not None:
                self.collect_metrics(final=True)

                if bar:
                    bar.finish()





