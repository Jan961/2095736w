from .LowDLayoutBase import LowDLayoutBase
from ..algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from ..algorithms.spring_force_algos.hybrid_algo.HybridStage import HybridStage
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96



class HybridLayout(LowDLayoutBase):

    def __init__(self, sample_layout_iterations: int = 75,
                 interpolation_adjustment_iterations: int = 5,
                 refine_layout_iterations: int = 5,
                 *basic_layout_creation_parameters):

        super().__init__(*basic_layout_creation_parameters)
        assert isinstance(self.algorithm, Hybrid)

        self.no_iters = sample_layout_iterations + refine_layout_iterations + 1

        self.sample_layout_iterations:    int = sample_layout_iterations
        self.interpolation_adjustment_iterations: int = interpolation_adjustment_iterations
        self.refine_layout_iterations:    int = refine_layout_iterations


