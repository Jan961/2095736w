from typing import List, Tuple

import numpy as np

from ..algorithms.BaseAlgorithm import BaseAlgorithm
from ..algorithms.spring_force_algos.SpringForceBase import SpringForceBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from .Chalmers96Layout import Chalmers96Layout
from ..algorithms.stochastic_ntet_algo.SNeD import SNeD
from .SQuaDLayout import SQuaDLayout
from ..algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from .HybridLayout import HybridLayout
from ..data_fetchers.Dataset import Dataset


class LowDLayoutCreation:

    def create_layout(self, algorithm: BaseAlgorithm, no_iters: int = None ,
                      optional_metric_collection: dict[str: int] = None,
                       **additional_parameters):

        basic_layout_creation_parameters = [algorithm, optional_metric_collection, no_iters]

        print("#" * 20)
        print(f"A 2D layout of the \"{algorithm.dataset.name}\" dataset will be created \n"
              f"using the \"{algorithm.get_name()}\" algorithm")
        print("#" * 20)
        print(f"The HD distance measure used is: {algorithm.distance_fn.__name__}")
        print("#" * 20)

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(*basic_layout_creation_parameters, **additional_parameters)

        elif isinstance(algorithm, Hybrid):
            layout = HybridLayout(*basic_layout_creation_parameters, **additional_parameters)
            if no_iters is not None:
                print(f"The passed value of iteration numer \"no_iters\": {no_iters} will be ignored"
                      "and the total number of iterations calculated as the sum of parameters "
                      "\"sample_layout_iterations\" and \"refine_layout_iterations\""
                      "please use those to set the number iterations for this algorithm")
                print("#" * 20)

        elif isinstance(algorithm, SNeD):
            layout = SQuaDLayout(*basic_layout_creation_parameters, **additional_parameters)
            print(f" \"N-tet\" size: {algorithm.ntet_size}")
            if algorithm.is_test:
                print("Testing mode (i.e. comparing original grad calculations with new ones) is enabled")
            print("#" * 20)

        else:
            print("Error: unsupported algorithm type")

        if optional_metric_collection is None:
            print("#" * 20)
            print("No metrics will be collected during layout creation. \n"
                  "To change this use the \'metric collection\' parameter of the layout " )
            print("#" * 20)

        else:
            for metric, interval in optional_metric_collection.items():
                if metric != 'norm':
                    assert interval > 0, f"Interval of metric collection has to be > 0, got: {interval} "
                    assert metric in algorithm.available_metrics, f"{metric} not available for this algorithm"
                    if interval != 1:
                        print(f"\"{metric.capitalize()}\" will be measured every {interval} iterations")
                    else:
                        print(f"\"{metric.capitalize()}\" will be measured on every iteration")
                    print("#" * 20)

        norm = "euclidian"
        if optional_metric_collection and "norm" in optional_metric_collection:
            norm = optional_metric_collection["norm"]
        print(f"All stress calculations will be performed using the {norm} norm")
        print("#" * 20)

        if np.all(np.where(algorithm.initial_layout==0, 1,0)):
            print("#"*20)
            print("Warning: the initial 2D positions are all set to 0. \n"
                  "You might want to use the \"initial layout\" parameter of the algorithm\n"
                  "to specify a different initialisation")
            print("#" * 20)

        #the nodes are created here so that this operation is easier to include in memory and time measurements
        if isinstance(layout.algorithm, SpringForceBase):
            layout.algorithm.build_nodes()
            print(f"Spring constant is set to  {layout.algorithm.spring_constant} ")
            print(f"Damping constant is set to {layout.algorithm.damping_constant} ")
            print(f"Spring constant scaling factor is set to {layout.algorithm.sc_scaling_factor} ")
            print("#" * 20)

        layout.run()
        return layout
