from typing import List, Tuple
from ..data_fetchers.Dataset import Dataset
from progress.bar import IncrementalBar
from .LayoutBase import LowDLayoutBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
import numpy as np
import matplotlib.pyplot as plt


class Chalmers96Layout(LowDLayoutBase):
    def __init__(self, num_iters: int = 100, *basic_layout_creation_parameters,
                 target_node_speed: float = None, alpha: float=1):
        super().__init__(num_iters, *basic_layout_creation_parameters)
        assert isinstance(self.algorithm, Chalmers96)

        self.target_node_speed = target_node_speed
        self.alpha =alpha

        assert self.num_iters is not None or self.target_node_speed is not None, "Must provide a termination condition"

    def run(self ) -> None:
        """
        Method to perform the main spring layout calculation, move the nodes iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times.
        """

        bar = None
        if self.target_node_speed is None:
            bar = IncrementalBar("Creating layout", max=self.num_iters)

        terminated = False
        while not terminated:

            if self.optional_metric_collection is not None:
                self.collect_metrics()
            self.algorithm.one_iteration(self.alpha)
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()
            if bar:
                bar.next()



            if self.num_iters is not None and self.iteration_number >= self.num_iters:
                terminated = True


            if self.target_node_speed is not None:
                average_speed = self.algorithm.get_average_speed()
                print(f"Iteration number: {self.iteration_number}")
                print(f"Average force/speed/position update: {average_speed}")

                if self.target_node_speed >0 and self.target_node_speed >= average_speed:
                    terminated =True

            if terminated:
                if self.optional_metric_collection is not None:
                    self.collect_metrics(final=True)
                if bar:
                    bar.finish()


            # stress_og = self.algorithm.get_stress()
            # stress_new = self.algorithm.get_euclidian_stress()
            # print(f"stress og: {stress_og}")
            # print(f"stres new: {stress_new}")







