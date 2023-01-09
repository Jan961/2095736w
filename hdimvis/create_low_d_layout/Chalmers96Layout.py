from typing import List, Tuple
from ..data_fetchers.Dataset import Dataset
from progress.bar import IncrementalBar
from .LowDLayoutBase import LowDLayoutBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
import numpy as np
import matplotlib.pyplot as plt

class Chalmers96Layout(LowDLayoutBase):

    def __init__(self, algorithm: Chalmers96, dataset: Dataset, optional_metric_collection: dict[str: int] = None):
        super().__init__(algorithm, dataset, optional_metric_collection)
        assert isinstance(self.algorithm, Chalmers96)


    def run(self, metric_collection: List[Tuple] =None, no_iters: int = 50, target_node_speed: float = 0.0, ) -> None:
        """
        Method to perform the main spring layout calculation, move the nodes iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times. Subsequent calls to create will continue from the previous number of
        iterations.
        """
        bar = None

        if no_iters is not None:
            assert no_iters >= 0
            if target_node_speed == 0:
                bar = IncrementalBar("Creating layout", max=no_iters)

        assert target_node_speed >= 0
        assert no_iters is not None or target_node_speed > 0

        while True:
            if no_iters is not None and self.iteration_number >= no_iters:
                if self.optional_metric_collection is not None:
                    self.collect_metrics(final=True)
                bar.finish()
                return

            average_speed = self.algorithm.get_average_speed()
            if target_node_speed >0 and target_node_speed >= average_speed:
                if self.optional_metric_collection is not None:
                    self.collect_metrics(final=True)
                return

            if self.optional_metric_collection is not None:
                self.collect_metrics()

            self.algorithm.one_iteration()
            self.iteration_number += 1
            self.final_positions = self.algorithm.get_positions()
            if bar:
                bar.next()
            # stress_og = self.algorithm.get_stress()
            # stress_new = self.algorithm.get_euclidian_stress()
            # print(f"stress og: {stress_og}")
            # print(f"stres new: {stress_new}")







