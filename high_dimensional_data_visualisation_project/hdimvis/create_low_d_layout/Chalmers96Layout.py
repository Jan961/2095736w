import LowDLayoutBase
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
import numpy as np

class Chalmers96Layout(LowDLayoutBase):

    def create(self, algorithm: Chalmers96, return_after: int = None,
               target_node_speed: float = 0.0,) -> np.ndarray:
        """
        Method to perform the main spring layout calculation, move the nodes iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times. Subsequent calls to create will continue from the previous number of
        iterations.
        """

        # assert iterations >= 0

        if return_after is not None and \
                algorithm.get_evaluation_metrics('average_speed') > target_node_speed:
            for i in range(return_after):
                algorithm.one_iteration()

        else:
            while algorithm.get_average_speed() > target_node_speed and \
                    algorithm.get_iteration_no() < iterations:
                algorithm.one_iteration()
                self._i += 1
        # Return calculated positions for datapoints
        return self.get_positions()

