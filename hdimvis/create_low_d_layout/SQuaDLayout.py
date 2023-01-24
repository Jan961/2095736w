from typing import List, Tuple
from progress.bar import IncrementalBar
import numpy as np
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from ..data_fetchers.Dataset import Dataset

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, no_iters: int = 50, *basic_layout_creation_parameters):
        super().__init__(no_iters, *basic_layout_creation_parameters)

        assert isinstance(self.algorithm, SQuaD)

    def run(self, exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, LR: float = 550.0):

        calculate_quartet_stress = False
        bar = IncrementalBar("Creating layout", max=self.no_iters)
        decay = decay if decay is not None else np.exp(np.log(1e-3) / self.no_iters)
        if exaggerate_D:  # exaggeration of HD distances by taking them squared
            stop_d_exa = int(self.no_iters * stop_exaggeration)  # iteration when we stop the exaggeration
        else:
            stop_d_exa = 0

        for i in range(self.no_iters):
            if self.optional_metric_collection is not None:
                if self.iteration_number == self.no_iters:
                    calculate_quartet_stress = True

                if self.optional_metric_collection.get('Average quartet stress') and \
                        self.iteration_number % self.optional_metric_collection['Average quartet stress'] == 0 :
                    calculate_quartet_stress = True

            if i == stop_d_exa:
                LR *= decay
                exaggerate_D = False

            self.algorithm.one_iteration(exaggerate_D, LR, calculate_quartet_stress)
            if self.optional_metric_collection is not None:
                self.collect_metrics()
            bar.next()
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()

        if self.optional_metric_collection is not None:
            self.collect_metrics(final=True)
        bar.finish()

