from typing import List, Tuple
from progress.bar import IncrementalBar
import numpy as np
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms import BaseAlgorithm
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from ..data_fetchers.Dataset import Dataset

class SQuaDLayout(LowDLayoutBase):
    def __init__(self, algorithm: SQuaD, dataset:Dataset, optional_metric_collection: dict[str: int] = None):
        super().__init__(algorithm, dataset, optional_metric_collection)

        assert isinstance(self.algorithm, SQuaD)

    def run(self, metric_collection: List[Tuple] = None, no_iters: int = 10,
            exaggerate_D: bool = False, stop_exaggeration: float = 0.6,
                 decay: float = None, LR: float = 550.0):

        bar = IncrementalBar("Creating layout", max=no_iters)
        decay = decay if decay is not None else np.exp(np.log(1e-3) / no_iters)
        if exaggerate_D:  # exaggeration of HD distances by taking them squared
            stop_d_exa = int(no_iters * stop_exaggeration)  # iteration when we stop the exaggeration
        else:
            stop_d_exa = 0

        for i in range(no_iters):
            if self.optional_metric_collection is not None:
                self.collect_metrics()

            if i == stop_d_exa:
                LR *= decay
                exaggerate_D = False

            self.algorithm.one_iteration(exaggerate_D, LR)
            self.iteration_number += 1
            bar.next()
            self.final_positions = self.algorithm.get_positions()

        if self.optional_metric_collection is not None:
            self.collect_metrics(final=True)
        bar.finish()