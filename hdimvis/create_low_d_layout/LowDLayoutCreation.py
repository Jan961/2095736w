from typing import List, Tuple

import numpy as np

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from .Chalmers96Layout import Chalmers96Layout
from ..algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from .SQuaDLayout import SQuaDLayout
from ..data_fetchers.Dataset import Dataset


class LowDLayoutCreation:

    def create_layout(self, algorithm: BaseAlgorithm, dataset: Dataset,
                      metric_collection: dict[str: int] = None, **kwargs):

        parameters = [algorithm, dataset, metric_collection]

        print("#" * 20)
        print(f"Creating a 2D layout of the \"{dataset.name}\" dataset \n"
              f"using the {algorithm.name} algorithm")
        print("#" * 20)

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(*parameters)

        elif isinstance(algorithm, SQuaD):
            layout = SQuaDLayout(*parameters)


        if metric_collection is None:
            print("#" * 20)
            print("No metrics will be collected during layout creation. \n"
                  "To change this use the \'metric collection\' parameter of the layout " )
            print("#" * 20)

        else:
            for metric, freq in metric_collection.items():
                assert freq > 0, f"Frequency of metric collection has to be > 0, got: {freq} "
                assert metric in algorithm.available_metrics, f"{metric} not available for this algorithm"
                print(f"Collecting \"{metric}\" every {freq} iterations")
                print("#" * 20)

        if np.all(np.where(algorithm.initial_layout==0, 1,0)):
            print("#"*20)
            print("Warning: the initial 2D positions are all set to 0. \n"
                  "You might want to use the \"initial layout\" parameter of the algorithm\n"
                  "to specify a different initialisation")
            print("#" * 20)
        # print all the info here

        layout.run(**kwargs)
        return layout
