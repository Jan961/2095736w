from typing import List, Tuple
from progress.bar import IncrementalBar
import numpy as np
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_ntet_algo.SNeD import SNeD
from ..data_fetchers.Dataset import Dataset

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, no_iters: int = 50, *basic_layout_creation_parameters,
                 exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, LR: float = 550.0
                 ):
        super().__init__(no_iters, *basic_layout_creation_parameters)

        assert isinstance(self.algorithm, SNeD)
        assert no_iters is not None, "For this algorithm the number of iterations must be specified"
        self.exaggerate_D = exaggerate_D
        self.stop_exaggeration = stop_exaggeration
        self.decay = decay
        self.LR = LR

        if self.algorithm.use_nesterovs_momentum:
            print("\n Nesterov's momentum will be used by the algorithm \n")


    def run(self):

        bar = IncrementalBar("Creating layout", max=self.no_iters)
        decay = self.decay if self.decay is not None else np.exp(np.log(1e-3) / self.no_iters)
        if self.exaggerate_D:  # exaggeration of HD distances by squaring
            stop_d_exa = int(self.no_iters * self.stop_exaggeration)  # iteration when we stop the exaggeration
        else:
            stop_d_exa = 0

        for i in range(self.no_iters):
            calculate_quartet_stress = False

            # the below conditional allows us to avoid calculating avg quartet stress (which is best
            # measured alongside other calculations during one iteration run)  on every iteration
            if self.optional_metric_collection is not None:
                if i == self.no_iters - 1:
                    calculate_quartet_stress = True

                if self.optional_metric_collection.get('Average quartet stress') and \
                        self._check_collection_interval('Average quartet stress') :
                    calculate_quartet_stress = True

            if i == stop_d_exa:
                self.LR *= decay
                self.exaggerate_D = False

            self.algorithm.one_iteration(self.exaggerate_D, self.LR, calculate_quartet_stress)
            if self.optional_metric_collection is not None:
                self.collect_metrics()

            bar.next()
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()

        if self.optional_metric_collection is not None:
            self.collect_metrics(final=True)
        bar.finish()



