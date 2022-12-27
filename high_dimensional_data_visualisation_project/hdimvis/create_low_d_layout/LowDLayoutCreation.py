from typing import List, Tuple

import numpy as np

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from .Chalmers96Layout import Chalmers96Layout
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from .SQuaDLayout import SQuaDLayout


class LowDLayoutCreation:


    def create_layout(self, algorithm: BaseAlgorithm, data: np.ndarray, labels: np.ndarray,
                      metric_collection: dict[str: int] = None, **kwargs):

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(algorithm, data, labels)

        elif isinstance(algorithm, SQuaD):
            layout = SQuaDLayout(algorithm, data, labels)

        # print all the info here
        layout.run(metric_collection, **kwargs)
        return layout