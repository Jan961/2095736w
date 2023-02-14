import numpy as np
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists
from hdimvis.data_fetchers.Dataset import Dataset


dataset = DataFetcher().fetch_data('coil20', size=50)
mock_data= np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
mock_dataset = Dataset(mock_data, None, "mock data")

mock_data_2 = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
initial_positions_2 = 20*np.random.rand(4,2)
mock_dataset_2 = Dataset(mock_data_2, np.array([0,1,2,3]), 'mock data')

def test_one_iteration_correctly_performed():
    algo = SQuaD(dataset=mock_dataset, initial_layout=initial_positions, distance_fn= relative_rbf_dists)
    assert np.allclose(initial_positions, algo.get_positions())
    algo.one_iteration()
    assert np.allclose(initial_positions, algo.get_positions())
    algo.one_iteration()
    assert np.allclose(initial_positions, algo.get_positions()) #tests if the correspondence between low D and high D
                                                                # is maintained and the points are not shuffled
    assert np.allclose(initial_positions, algo.data)


# to run the below test uncomment the relevant sections of the Squad code - they take up a lot of space
# duplicating many calculations, therefore making the code hard to read and thus are commented out

def test_vectorised_calculations_produce_the_same_results_as_original():
    algo = SQuaD(dataset=dataset, test=True)
    for i in range(20):
        algo.one_iteration()


def test_nesterovs_momentum_v_increases_as_expected():
    algo = SQuaD(dataset=mock_dataset_2, initial_layout=initial_positions_2, nesterovs_momentum=True, momentum=0.9)
    assert not np.any(algo.nesterovs_v)  # check if all initial Nesterov's momenutm "changes or v are 0
    previous_v = algo.nesterovs_v

    for i in range(5):
        algo.one_iteration()
        assert np.sum(np.abs(previous_v) < np.sum(np.abs(algo.nesterovs_v)))
        previous_v = algo.nesterovs_v




