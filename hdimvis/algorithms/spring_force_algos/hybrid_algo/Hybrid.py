from ..Node import Node
from ..SpringForceBase import SpringForceBase
from ..chalmers96_algo.Chalmers96 import Chalmers96
from ..utils import point_on_circle, random_sample_set
from typing import Callable, Tuple, List
from .HybridStage import HybridStage
import numpy as np
import math

# code adapted from 2019 Project by Iain Cattermole



class Hybrid(SpringForceBase):
    """
    An implementation of Chalmers, Morrison, and Ross' 2002 hybrid_algo layout.
    It performs the '96 neighbour sampling algorithm, on a sample set, sqrt(n) samples of the data.
    Other data points are then interpolated into the model. Finally, another spring simulation is
    performed to clean up the layout.
    """

    name = 'Hybrid'
    available_metrics = ['Stress', 'Average speed']

    def __init__(self, preset_sample : np.ndarray = None,
                 interpolation_adjustment_sample_size: int = 15,
                 use_correct_interpolation_error: bool = True,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.use_correct_interpolation_error = use_correct_interpolation_error
        self.preset_sample = preset_sample  # node indices - we can manually set the sample used for interpolation here
        self.initial_sample_size:             int = preset_sample.size if preset_sample is not None \
                                                 else round(math.sqrt(len(self.nodes)))

        self.interpolation_adjustment_sample_size:       int = interpolation_adjustment_sample_size
        self.sample_indices:        List[int] = self.create_sample_indices() # use preset_sample or select random
        self.remainder_indices:     List[int] = list(set(range(len(self.nodes))) - set(self.sample_indices))
        self.sample:               List[Node] = [self.nodes[i] for i in self.sample_indices]
        self.remainder:            List[Node] = [self.nodes[i] for i in self.remainder_indices]
        self.data_size_factor:          float = 1 / self.interpolation_adjustment_sample_size

        self.sample_layout_stage_algorithm = Chalmers96(nodes=self.sample, dataset=None, distance_fn=self.distance_fn,
                                                        use_knnd=self.use_knnd, sample_set_size=self.sample_set_size,
                                                        neighbour_set_size=self.neighbour_set_size)

        self.refine_stage_algorithm = Chalmers96(nodes=self.nodes, dataset=None, distance_fn=self.distance_fn,
                                                 use_knnd=self.use_knnd, sample_set_size=self.sample_set_size,
                                                 neighbour_set_size=self.neighbour_set_size)

        self.stage:               HybridStage = HybridStage.PLACE_SAMPLE


    def create_sample_indices(self):
        if self.preset_sample is not None:
            return self.preset_sample.tolist()
        else:
            return random_sample_set(self.initial_sample_size,
                                     len(self.nodes))

    def set_stage(self, stage: HybridStage):
        self.stage = stage

    def one_iteration(self, alpha: float = 1, interpolation_adjustment_iterations:int = 5) -> None:

        if self.stage == HybridStage.PLACE_SAMPLE:
            self._sample_stage_one_iteration()
        elif self.stage == HybridStage.INTERPOLATE:
            self._interpolate(interpolation_adjustment_iterations=interpolation_adjustment_iterations ,alpha=alpha)
        elif self.stage == HybridStage.REFINE:
            self._refine_stage_one_iteration(alpha=alpha)


    def _sample_stage_one_iteration(self):
        """
        Running the Chalmers' 96 algorithm on a sample of the whole dataset
         """
        self.sample_layout_stage_algorithm.one_iteration()
        self.sample = self.sample_layout_stage_algorithm.nodes

    def _interpolate(self, alpha: float, interpolation_adjustment_iterations :int) -> None:
        """
        Place the remainder (i.e. whole dataset\sample set) nodes close to similar nodes from the
        sample set
        """
        number_remainders = len(self.remainder_indices)
        for i in range(number_remainders):
            self._place_near_parent(self.remainder_indices[i], alpha, interpolation_adjustment_iterations)

    def _refine_stage_one_iteration(self, alpha: float) -> None:
        """
        Perform neighbour sampling algorithm on whole layout to refine it
        """
        self.refine_stage_algorithm.one_iteration()
        self.nodes = self.refine_stage_algorithm.nodes


    def _place_near_parent(self, index: int, alpha: float, interpolation_adjustment_iterations: int) -> None:
        """
        Determine the most similar parent node and place source node
        nearby by comparing distances to other nodes
        """
        source = self.nodes[index]
        parent_index, distances = self._find_parent(source)
        radius = distances[parent_index]
        parent = self.sample[parent_index]

        sample_distances_error = self._create_error_fn(parent_index, distances)

        lower_angle, upper_angle = self._find_circle_quadrant(sample_distances_error)
        best_angle = self._binary_search_angle(lower_angle, upper_angle, sample_distances_error)
        source.x, source.y = point_on_circle(parent.x, parent.y, best_angle, radius)
        self._force_layout_child(index, distances, alpha, interpolation_adjustment_iterations)

    def _create_error_fn(self, parent_index: int,  # parent index in the sample, not in the list of all nodes
                         distances: List[float],   # these are hd distances
                         ) -> Callable[[int, bool], float]:
        """
        Create function specific to current node to calculate
        error in distances at an angle on the circle centred at the parent
        at the calculated radius - i.e. radius equal to hd distance between the node and its parent
        """

        radius = distances[parent_index]
        parent = self.sample[parent_index]

        def sample_distances_error(angle: int, correct_error_calc: bool = None) -> float:

            if correct_error_calc is None:
                correct_error_calc = self.use_correct_interpolation_error

            point = point_on_circle(parent.x, parent.y, angle, radius)

            if correct_error_calc:
                point_arr = np.array(point)
                print(f"point arr {point_arr}")
                sample_arr = np.array([[node.x, node.y] for node in self.sample] )
                print(f"sample arr {sample_arr}")
                distances_2d = np.linalg.norm(sample_arr - point_arr, axis=1)
                print(f"ld dist algo: {distances_2d}")
                distances_hd = np.array(distances)
                print(f"hd dist algo: {distances_hd}")
                return np.sum((distances_hd - distances_2d)**2)

            else:
                return abs(sum(distances)- self._sample_distances_sum(*point))

        return sample_distances_error

    def _find_parent(self, source: Node) -> Tuple[int, List[float]]:
        """
        Find the cloest parent node to the source, also return the
        list of calculated distances to speed up later calculation
        Return the index of the parent in the sample not global
        """
        distances: List[float] = [self.hd_distance(source, target) for target in self.sample]
        parent_index: int = np.argmin(distances)
        return parent_index, distances

    def _find_circle_quadrant(self, error_fn: Callable[[int], float]) -> Tuple[int, int]:
        """
        Find the quadrant with the minimum error function at the edges
        """
        angles = [0, 90, 180, 270]
        distance_errors = [error_fn(angle) for angle in angles]
        print(f"distance errors {distance_errors}")

            # determine angle with lowest error and choose neighbour quadrant angle with lowest error
        best_angle_id = np.argmin(distance_errors)
        neighbour_angle_ids = (best_angle_id - 1) % 4, (best_angle_id + 1) % 4

        if distance_errors[neighbour_angle_ids[0]] < distance_errors[neighbour_angle_ids[1]]:
            closest_neighbour_id = neighbour_angle_ids[0]
        else:
            closest_neighbour_id = neighbour_angle_ids[1]

        if {angles[best_angle_id], angles[closest_neighbour_id]} == {0, 270}:
            lower_bound_angle, upper_bound_angle = 270, 360
        else:
            lower_bound_angle, upper_bound_angle = sorted((angles[best_angle_id], angles[closest_neighbour_id]))

        return lower_bound_angle, upper_bound_angle



    def _binary_search_angle(self, lower_angle: int, upper_angle: int,
                             error_fn: Callable[[int], float]) -> int:
        """
        Recursively binary search to find the best angle between upper and lower
        bound that minimises the error function
        """
        if upper_angle - lower_angle <= 4:
            return lower_angle  # found a good angle
        angle_range = upper_angle - lower_angle
        lower_error = error_fn(lower_angle + angle_range // 4)
        upper_error = error_fn(upper_angle - angle_range // 4)
        if lower_error < upper_error:
            return self._binary_search_angle(lower_angle, upper_angle - angle_range // 2, error_fn)
        elif upper_error < lower_error:
            return self._binary_search_angle(lower_angle + angle_range // 2, upper_angle, error_fn)
        # both sides are equal so search in the middle
        return self._binary_search_angle(
            lower_angle + angle_range // 4, upper_angle - angle_range // 4, error_fn
        )

    def _force_layout_child(self, index: int, distances: List[float], alpha: float,
                            interpolation_adjustment_iterations: int) -> None:
        """
        Apply forces from random subsets of the original sample to
        the child node to refine its position
        """
        source = self.nodes[index]
        for i in range(interpolation_adjustment_iterations):
            sample_set = random_sample_set(self.interpolation_adjustment_sample_size, self.initial_sample_size, {index})
            for j in sample_set:
                target = self.sample[j]
                f_x, f_y = self._calculate_force(source, target)
                source.increment_position_update(f_x, f_y)
            source.apply_position_update()

    def _sample_distances_sum(self, x: float, y: float):
        """
        Sum the total distance from all sample nodes to a point (x, y)
        """
        return sum([math.hypot(target.x - x, target.y - y) for target in self.sample])
