from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from sklearn.decomposition import PCA
import numpy as np

# code adapted and slightly extended from 2019 Project by Iain Cattermole

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



def test_spring_force_builds_nodes_correctly():
    spring_force = Chalmers96(dataset=dataset)
    assert np.array_equal(spring_force.nodes[0].datapoint, dataset.data[0, :])
    assert np.array_equal(spring_force.nodes[1].datapoint, dataset.data[1, :])


def test_knnd_index_created_and_neighbours_retrieved_correctly():
    algo = Chalmers96(dataset, neighbour_set_size=2, use_knnd=True, knnd_parameters={'metric': 'manhattan','n_trees': 100})
    assert len(algo._get_neighbours(3)) == 2
    assert algo._get_neighbours(4) == algo.knnd_index.neighbor_graph[0][4][1:].tolist()


def test_initial_layout():
    Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
    algorithm = Chalmers96(dataset=dataset, initial_layout=Xld, neighbour_set_size=5)
    for i in range(3,7):
        assert algorithm.nodes[i].x == Xld[i][0]
        assert algorithm.nodes[i].y == Xld[i][1]