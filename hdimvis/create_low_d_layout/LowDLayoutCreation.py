from typing import List, Tuple

import numpy as np

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from ..algorithms.spring_force_algos.SpringForceBase import SpringForceBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from .Chalmers96Layout import Chalmers96Layout
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from .SQuaDLayout import SQuaDLayout
from ..data_fetchers.Dataset import Dataset


class LowDLayoutCreation:

    def create_layout(self, algorithm: BaseAlgorithm, no_iters: int | None, optional_metric_collection: dict[str: int] = None,
                       **additional_parameters):

        basic_layout_creation_parameters = [algorithm, optional_metric_collection, no_iters]

        print("#" * 20)
        print(f"A 2D layout of the \"{algorithm.dataset.name}\" dataset will be created \n"
              f"using the \"{algorithm.get_name()}\" algorithm")
        print("#" * 20)

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(*basic_layout_creation_parameters)

        elif isinstance(algorithm, SQuaD):
            layout = SQuaDLayout(*basic_layout_creation_parameters)

        if optional_metric_collection is None:
            print("#" * 20)
            print("No metrics will be collected during layout creation. \n"
                  "To change this use the \'metric collection\' parameter of the layout " )
            print("#" * 20)

        else:
            for metric, freq in optional_metric_collection.items():
                assert freq > 0, f"Frequency of metric collection has to be > 0, got: {freq} "
                assert metric in algorithm.available_metrics, f"{metric} not available for this algorithm"
                print(f"\"{metric.capitalize()}\" will be measured every {freq} iterations")
                print("#" * 20)

        if np.all(np.where(algorithm.initial_layout==0, 1,0)):
            print("#"*20)
            print("Warning: the initial 2D positions are all set to 0. \n"
                  "You might want to use the \"initial layout\" parameter of the algorithm\n"
                  "to specify a different initialisation")
            print("#" * 20)
        # print all the info here

        #the nodes are created here so that this operation is easier to include in memory and time measurements
        if isinstance(layout.algorithm, SpringForceBase):
            layout.algorithm.build_nodes()

        layout.run(**additional_parameters)
        return layout
