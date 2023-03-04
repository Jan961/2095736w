import numpy as np

from hdimvis.algorithms.spring_force_algos.SpringForceBase import SpringForceBase
from ..utils import random_sample_set
from ....data_fetchers.Dataset import Dataset
from typing import List, Dict


# code adapted and modified from 2019 Project by Iain Cattermole

class Chalmers96(SpringForceBase):
    """
    An implementation of Chalmers' 1996 Neighbour and Sampling algorithm.
    Using random sampling to find the closest neighbours from the data set.

    """
    name = 'Chalmers\' 1996'

    def __init__(self, dataset: Dataset | None, **kwargs):

        # the base class extracts data from the Dataset object
        super().__init__( dataset, **kwargs)

        self.neighbours: Dict[int, List[int]] = dict() # dictionary used when no knnd index is supplied
        self.data_size_factor: float = 0.5 / (self.neighbour_set_size + self.sample_set_size)

    def one_iteration(self, alpha: float=1) -> None:
        """
        Perform one iteration of the spring layout
        """
        assert self.data is not None or self.nodes is not None, "must provide dataset or nodes"

        n = len(self.nodes)

        for i in range(n):

            if self.sample_set_size:
                sample_set = self._get_sample_set(i)
                for global_index in sample_set: # j tracks the position in self.sample_set_updates
                    self._set_position_update( source=self.nodes[i], target=self.nodes[global_index])

            if self.neighbour_set_size:
                neighbour_set = self._get_neighbours(i)
                for global_index in neighbour_set: # j tracks the position in self.neighbour_set_updates
                    self._set_position_update( source=self.nodes[i], target=self.nodes[global_index],
                                              cache_distance=True)

            if not self.use_knnd and self.neighbour_set_size and self.sample_set_size:
                self._update_neighbours(i, samples=sample_set)




        self._apply_position_update()

    def _get_neighbours(self, index: int) -> List[int]:
        """
        Get the list of neighbour indices for a given node index sorted by distance.
        If no neighbours exist yet then they are randomly sampled.
        """
        if self.use_knnd:
            return self.knnd_index.neighbor_graph[0][index][1:].tolist()

        if index not in self.neighbours:
            random_sample = random_sample_set(self.neighbour_set_size, len(self.nodes), {index})
            random_sample.sort(
                key=lambda j: self.distance(self.nodes[index], self.nodes[j])
            )
            self.neighbours[index] = random_sample
        return self.neighbours[index]

    def _get_sample_set(self, i: int) -> List[int]:
        """
        Get a valid sample set for a node index by randomly sampling, excluding
        current node and neighbours of the node.
        """
        exclude = {i}.union(set(self._get_neighbours(i)))
        return list(random_sample_set(self.sample_set_size, len(self.nodes), exclude))

    def _update_neighbours(self, i: int, samples: List[int]) -> None:
        """
        Update the neighbour set for a given index from a sample set.
        Sample nodes are added to the neighbour set in sorted order if
        they are closer than the furthest current neighbour.
        """
        source = self.nodes[i]
        neighbours = self._get_neighbours(i)
        furthest_neighbour = self.distance(source, self.nodes[neighbours[-1]])
        for s in samples:
            sample_distance = self.distance(source, self.nodes[s])
            if sample_distance < furthest_neighbour:
                n = self.neighbour_set_size - 2
                neighbour_distance = self.distance(source, self.nodes[neighbours[n]])
                while sample_distance < neighbour_distance:
                    n -= 1
                    if n < 0:
                        break
                    neighbour_distance = self.distance(source, self.nodes[neighbours[n]])
                neighbours.insert(n + 1, s)
                distance_key = frozenset({source, self.nodes[neighbours[-1]]})
                if self.enable_cache and distance_key in self.distances:
                    del self.distances[distance_key]  # Remove distance from cache to save memory
                del neighbours[-1]

