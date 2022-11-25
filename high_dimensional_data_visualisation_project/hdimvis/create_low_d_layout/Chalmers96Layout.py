from .LowDLayoutBase import LowDLayoutBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
import numpy as np
import matplotlib.pyplot as plt

class Chalmers96Layout(LowDLayoutBase):

    def __init__(self, algorithm: Chalmers96):
        assert isinstance(algorithm, Chalmers96)
        self.algorithm = algorithm
        self.final_positions: np.ndarray = np.zeros(1)



    def get_final_positions(self):
        return self.final_positions

    def run(self, return_after: int = 50, target_node_speed: float = 0.0, ) -> None:
        """
        Method to perform the main spring layout calculation, move the nodes iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times. Subsequent calls to create will continue from the previous number of
        iterations.
        """


        if return_after is not None:
            assert return_after >= 0
        assert target_node_speed >= 0
        assert return_after is not None or target_node_speed > 0

        while True:
            # Return calculated positions for datapoints
            if return_after is not None and self.algorithm.get_iteration_no() >= return_after:
                return
            average_speed = self.algorithm.get_evaluation_metrics('average speed')
            if target_node_speed >0 and target_node_speed >= average_speed:
                return

            self.algorithm.one_iteration()
            self.final_positions = self.algorithm.get_positions()


