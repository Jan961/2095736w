from ..algorithms.BaseAlgorithm import BaseAlgorithm
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
from .Chalmers96Layout import Chalmers96Layout


class LowDLayoutCreation:


    def create_layout(self, algorithm: BaseAlgorithm, **kwargs):

        if isinstance(algorithm, Chalmers96):
            layout = Chalmers96Layout(algorithm)
            layout.run(**kwargs)
            return layout
