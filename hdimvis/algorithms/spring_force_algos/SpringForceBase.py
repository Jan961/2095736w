from hdimvis.algorithms.BaseAlgorithm import BaseAlgorithm
from hdimvis.algorithms.spring_force_algos.Node import Node
from hdimvis.algorithms.spring_force_algos.utils import jiggle, get_size, mean
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from itertools import combinations
from typing import Callable, Tuple, List, Dict, FrozenSet
from abc import abstractmethod
import numpy as np
import math



class SpringForceBase(BaseAlgorithm):
    def __init__(self, dataset: np.ndarray, initial_layout: np.ndarray = None,
                 distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean, nodes: List[Node] = None,
                 enable_cache: bool = True, alpha: float = 1) -> None:

        super().__init__(dataset, initial_layout, distance_fn)
        assert dataset is not None or nodes is not None, "must provide either dataset or nodes"

        self.nodes: List[Node] = nodes if nodes is not None else self.build_nodes(self.dataset, self.initial_layout)
        self.data_size_factor: float = 1
        self._average_speeds: List[float] = list()
        self.enable_cache: bool = enable_cache
        self.alpha = alpha
        if enable_cache:
            self.distances: Dict[FrozenSet[Node], float] = dict()
        else:
            # Change the distance function
            self.distance = self.distance_no_cache


    @abstractmethod
    def one_iteration(self, alpha: float) -> None:
        """
        Perform one iteration of the spring layout
        """
        pass


    def get_iteration_no(self):

        return self.iteration_no


    def build_nodes(self, dataset: np.ndarray, initial_layout: np.ndarray) -> List[Node]:
        """
        Construct a Node for each datapoint
        """
        #contactenate the datapoints with the initial positions for low-d mappings
        #for the apply_along_axis fn
        conc = np.concatenate((dataset, initial_layout), axis=1)

        return list(np.apply_along_axis(Node, axis=1, arr= conc))

    def get_metrics(self, *args) -> float:
        assert 'average speed' in args or 'stress' in args

        if 'average speed' in args:
            return self.get_average_speed()
        if 'stress' in  args:
            return self.get_stress()


    def get_positions(self) -> np.ndarray:
        return np.array([(n.x, n.y) for n in self.nodes])

    def set_positions(self, positions: np.ndarray) -> None:
        for pos, node in zip(positions, self.nodes):
            node.x, node.y = pos

    def get_stress(self) -> float:
        distance_diff: float = 0.0
        actual_distance: float = 0.0
        for source, target in combinations(self.nodes, 2):
            high_d_distance = self.distance(source, target, cache=False)
            low_d_distance = math.sqrt((target.x - source.x) ** 2 + (target.y - source.y) ** 2)
            distance_diff += (high_d_distance - low_d_distance) ** 2
            actual_distance += low_d_distance ** 2
        if actual_distance == 0:
            return math.inf
        return math.sqrt(distance_diff / actual_distance)

    def get_memory(self) -> int:
        return get_size(self)

    def get_average_speed(self) -> float:
        """ Return the 5-running mean of the average node speeds """
        return mean(self._average_speeds[-5:]) if len(self._average_speeds) > 0 else np.inf


    def distance_no_cache(self, source: Node, target: Node, cache: bool = False) -> float:
        """ Distance function to use when self.disable_cache = True """
        return self.distance_fn(source.datapoint, target.datapoint)

    def distance(self, source: Node, target: Node, cache: bool = False) -> float:
        """
        Returns the high dimensional distance between two nodes at source and target
        index using self.distance_fn
        """
        pair = frozenset({source, target})
        if pair in self.distances:
            return self.distances[pair]
        distance = self.distance_fn(source.datapoint, target.datapoint)
        if cache:
            self.distances[pair] = distance
        return distance


    def _force(self, current_distance, real_distance, alpha: float = 1) -> float:
        return (current_distance - real_distance) * alpha * self.data_size_factor / current_distance

    def _calculate_velocity(self, source: Node, target: Node, alpha: float = 1,
                            cache_distance: bool = False) -> Tuple[float, float]:
        """
        Calculate the spring force to apply between two nodes i and j
        """
        x, y = self._current_distance(source, target)
        dist = math.hypot(x, y)
        real_dist = self.distance(source, target, cache=cache_distance)
        force = self._force(dist, real_dist, alpha)
        return (x * force, y * force)

    def _current_distance(self, source: Node, target: Node) -> Tuple[float, float]:
        """
        Calculate the current 2d layout distance between two nodes.
        Apply a small non zero random value to remove values of zero
        """
        x = target.x - source.x
        y = target.y - source.y
        # x and y must be non zero
        x = x if x else jiggle()
        y = y if y else jiggle()
        return x, y

    def _set_velocity(self, source: Node, target: Node, alpha: float,
                      cache_distance: bool = False) -> None:
        """
        Calculate the force between two nodes and update the
        velocity of both nodes
        """
        vx, vy = self._calculate_velocity(source, target, alpha=alpha,
                                          cache_distance=cache_distance)
        source.increment_velocity(vx, vy)
        target.increment_velocity(-vx, -vy)

    def _apply_velocities(self) -> None:
        """
        Apply the current velocity of each node to its position
        and reset velocity
        """
        total: float = 0.0
        for node in self.nodes:
            total += math.hypot(node.vx, node.vy)
            node.apply_velocity()
        total /= len(self.nodes)
        self._average_speeds.append(total)

