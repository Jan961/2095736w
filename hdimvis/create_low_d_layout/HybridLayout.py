from progress.bar import IncrementalBar

from .LowDLayoutBase import LowDLayoutBase
from ..algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from ..algorithms.spring_force_algos.hybrid_algo.HybridStage import HybridStage
from ..algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96



class HybridLayout(LowDLayoutBase):

    def __init__(self, *basic_layout_creation_parameters, sample_layout_iterations: int = 75,
                interpolation_adjustment_iterations: int = 5,
                refine_layout_iterations: int = 5,
                 alpha: float = 1):

        super().__init__(*basic_layout_creation_parameters)
        assert isinstance(self.algorithm, Hybrid)

        if self.no_iters is not None:
            print(f"The passed value of iteration numer \"no_iters\": {self.no_iters} will be ignored"
                  "and the total number of iterations calculated as a sum of parameters "
                  "\"sample_layout_iterations\" and \"refine_layout_iterations\""
                  "please use those to set the number iterations for this algorithm" )

        self.alpha = alpha
        self.no_iters = sample_layout_iterations + refine_layout_iterations + 1
        self.stage_iteration_numbers:    tuple[int, int, int] = (sample_layout_iterations, 1, refine_layout_iterations)
        self.interpolation_adjustment_iterations: int = interpolation_adjustment_iterations # interpolation is an
        # atypical stage - the previous implementation means that it is performed as just one call to
        # self.algorithm.one_iteration (hence the 1 in the tuple above) function but that function then performs
        # a refinement of self.interpolation_adjustment_iterations for each interpolated datapoint in turn



    def run(self) -> None:
        """
        Method to perform the main spring layout calculation, move the nodes iterations
        number of times unless return_after is given.
        If return_after is specified then the nodes will be moved the value of return_after
        times.

        Best not to collect generation metrics for this layout as for most runtime most points are in
        their initial positions
        """
        assert self.algorithm.stage == HybridStage.PLACE_SAMPLE, "The Hybrid Algorithm object is not in its initial state" \
                                                                 "please create a new hybrid algorithm to create a layout"

        bar = IncrementalBar("Creating layout", max=self.no_iters)

        for stage, num_iters in zip([enum for enum in HybridStage],self.stage_iteration_numbers):
            self.algorithm.set_stage(stage)
            print(f"\n Stage: {stage.name} \n")

            for i in range(num_iters):
                if self.optional_metric_collection is not None:
                    self.collect_metrics()

                self.algorithm.one_iteration(self.alpha, interpolation_adjustment_iterations =
                                            self.interpolation_adjustment_iterations)
                self.iteration_number += 1
                self.final_positions = self.algorithm.get_positions()
                bar.next()

        if self.optional_metric_collection is not None:
            self.collect_metrics(final=True)
        bar.finish()


