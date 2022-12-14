from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
import matplotlib.pyplot as plt
import numpy as np

dataset = DataFetcher().fetch_data('poker', size=500)

def test_get_neighbours_returns_correct_size_set():
    algorithm = Chalmers96(dataset=dataset, neighbour_set_size=5)

    neighbour_set = algorithm._get_neighbours(0)
    assert len(neighbour_set) == 5


def test_get_neighbours_saves_neighbours_after_first_call():
    algorithm = Chalmers96(dataset)
    assert 0 not in algorithm.neighbours
    algorithm._get_neighbours(0)
    assert 0 in algorithm.neighbours


def test_get_neighbours_returns_sorted_list_of_neighbours():
    algorithm = Chalmers96(dataset)
    neighbour_set = algorithm._get_neighbours(0)

    source = algorithm.nodes[0]
    previous = -1
    for i in neighbour_set:
        distance = algorithm.distance(source, algorithm.nodes[i])
        assert distance >= previous
        previous = distance


def test_update_neighbours_doesnt_change_neighbour_set_size():
    algorithm = Chalmers96(dataset,
                                  neighbour_set_size=3,
                                  sample_set_size=3)
    for i in range(10):
        sample_set = algorithm._get_sample_set(0)
        algorithm._update_neighbours(0, sample_set)
        neighbour_set = algorithm._get_neighbours(0)
        assert len(neighbour_set) == 3


def test_update_neighbours_keeps_sorted_neighbour_set():
    algorithm = Chalmers96(dataset,
                                  neighbour_set_size=5,
                                  sample_set_size=1)
    source = algorithm.nodes[0]

    for i in range(10):
        sample_set = algorithm._get_sample_set(0)
        algorithm._update_neighbours(0, sample_set)
        neighbour_set = algorithm._get_neighbours(0)

        # Check neighbour set is sorted
        current_distance = -1
        for j in neighbour_set:
            target = algorithm.nodes[j]
            distance = algorithm.distance(source, target)
            assert distance >= current_distance
            current_distance = distance


def test_get_sample_set_excludes_neighbours_and_current_node():
    algorithm = Chalmers96(dataset,
                                  neighbour_set_size=3,
                                  sample_set_size=3)
    for i in range(10):
        sample_set = set(algorithm._get_sample_set(0))
        neighbour_set = set(algorithm._get_neighbours(0))
        excluded = {0}.union(neighbour_set)
        empty_set = set()
        assert sample_set.intersection(excluded) == empty_set


def test_calling_distance_caches_result():
    algorithm = Chalmers96(dataset=dataset, enable_cache=True)

    source = algorithm.nodes[0]
    target = algorithm.nodes[1]
    key = frozenset({source, target})

    algorithm.distance(source, target, cache=False)
    assert key not in algorithm.distances

    distance = algorithm.distance(source, target, cache=True)
    assert key in algorithm.distances
    assert algorithm.distances[key] == distance


def test_calling_distance_doesnt_cache_result_when_cache_disabled():
    algorithm = Chalmers96(dataset=dataset, enable_cache=False)

    source = algorithm.nodes[0]
    target = algorithm.nodes[1]
    algorithm.distance(source, target, cache=True)
    assert not hasattr(algorithm, 'distances')


def test_spring_layout_return_after_performs_correct_number_of_iterations():
    algorithm = Chalmers96(dataset=dataset)

    for i in range(2, 10, 2):
        [algorithm.one_iteration() for i in range(2)]
        assert algorithm.get_iteration_no() == i



def test_spring_force_builds_nodes_correctly():
    spring_force = Chalmers96(dataset=dataset)
    assert np.array_equal(spring_force.nodes[0].datapoint, dataset[0, :])
    assert np.array_equal(spring_force.nodes[1].datapoint, dataset[1, :])


def test_get_stress_returns_expected_stress_with_perfect_layout():
    # Data that should result in 0 stress layout
    # with euclidean distance
    mock_dataset = np.array([
        [0, 0],
        [0, 1],
    ])

    algorithm = Chalmers96(dataset=mock_dataset)
    [algorithm.one_iteration() for i in range(100)]

    expected_stress = 0
    np.assert_almost_equal(algorithm.get_stress(), expected_stress)


def test_get_stress_returns_expected_stress_with_bad_layout():
    mock_dataset = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    # Distance function that does not allow for a zero
    # stress layout with given data
    def bad_distance(a, b):
        return 1

    algorithm = SpringForce(dataset=mock_dataset, distance_fn=bad_distance, iterations=100)
    algorithm.spring_layout()

    # precalculated nearly optimal stress for this case
    expected_stress = 0.1715728
    assert_almost_equal(algorithm.get_stress(), expected_stress)
