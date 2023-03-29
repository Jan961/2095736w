import numpy as np
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.relative_rbf_dists import relative_rbf_dists
from hdimvis.data_fetchers.Dataset import Dataset


dataset = DataFetcher().fetch_data('coil20', size=50)
mock_data= np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
mock_dataset = Dataset(mock_data, None, "mock data")

mock_data_2 = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
initial_positions_2 = 20*np.random.rand(4,2)
mock_dataset_2 = Dataset(mock_data_2, np.array([0,1,2,3]), 'mock data 2')

def test_one_iteration_correctly_performed():
    algo = SNeD(dataset=mock_dataset, initial_layout=initial_positions)
    assert np.allclose(initial_positions, algo.get_positions())
    algo.one_iteration()
    assert np.allclose(initial_positions, algo.get_positions())
    algo.one_iteration()
    assert np.allclose(initial_positions, algo.get_positions()) #tests if the correspondence between low D and high D
                                                                # is maintained and the points are not shuffled
    assert np.allclose(initial_positions, algo.data)


def test_vectorised_calculations_produce_the_same_results_as_original():
    algo = SNeD(dataset=dataset, is_test=True) # this uses in-build testing functionality in the algorithm
    # implementation, as factoring this out would be very cumbersome
    for i in range(20):
        algo.one_iteration()


def test_nesterovs_momentum_v_increases_as_expected():
    algo = SNeD(dataset=mock_dataset_2, initial_layout=initial_positions_2, use_nesterovs_momentum=True, momentum=0.9)
    assert not np.any(algo.nesterovs_v)  # check if all initial Nesterov's momenutm "changes or v are 0
    previous_v = algo.nesterovs_v

    for i in range(5):
        algo.one_iteration()
        assert np.sum(np.abs(previous_v) < np.sum(np.abs(algo.nesterovs_v)))
        previous_v = algo.nesterovs_v

def test_rbf_distance_return_correct_array_format():
    Dhd_quartet = np.tile(np.array([[1.,3.,5.,4.]]),(4,1))
    dist_rel = relative_rbf_dists(Dhd_quartet)

    assert np.any(dist_rel[np.triu_indices(4)])
    assert not np.any(dist_rel[np.tril_indices(4)])



