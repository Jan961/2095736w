from typing import List, Tuple

import numpy as np
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, algorithm: SQuaD, data: np.ndarray, labels: np.ndarray,):
        super().__init__(algorithm, data, labels)

        assert isinstance(self.algorithm, SQuaD)



    def run(self, metric_collection: List[Tuple] =None, no_iters: int = 10,
            exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, LR: float = 550.0):

        decay = decay if decay is not None else np.exp(np.log(1e-3) / no_iters)
        if exaggerate_D:  # exaggeration of HD distances by taking them squared
            stop_d_exa = int(no_iters * stop_exaggeration)  # iteration when we stop the exaggeration

        stop_d_exa = 0
        for i in range(no_iters):
            if i == stop_d_exa:
                LR *= decay
                exaggerate_D = False

            self.algorithm.one_iteration(exaggerate_D, LR)
            self.final_positions = self.algorithm.get_positions()

