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

        parameters = [algorithm, data, labels, metric_collection]

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(*parameters)

        elif isinstance(algorithm, SQuaD):
            layout = SQuaDLayout(*parameters)

        if metric_collection is None:
            print("No metrics will be collected during the generation of the layout. \n"
                  "To change this use the \'metric collection\' parameter \n " )
        else:
            for metric, freq in metric_collection.items():
                assert freq > 0, f"Frequency of metric collection has to be > 0, got: {freq} "
                assert metric in algorithm.available_metrics, f"{metric} not available for this algorithm"

        if np.all(np.where(algorithm.initial_layout==0, 1,0)):
            print("Warning: the initial 2D positions are all set to 0. \n"
                  " You might want to use the \"initial layout\" parameter \n"
                  "to specify a different initialisation \n")
        # print all the info here

        layout.run(**kwargs)
        return layout
