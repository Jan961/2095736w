import numpy as np
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.relative_rbf_dists import relative_rbf_dists
from hdimvis.data_fetchers.Dataset import Dataset


dataset = DataFetcher().fetch_data('coil20')
mock_data= np.array([[0,0],[0,1],[1,1],[1,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,1],[1,1],[1,0]], dtype='float64')
mock_dataset = Dataset(mock_data, None, "mock data")
def test_one_iteration_correctly_performed():
    algo = SQuaD(dataset=mock_dataset, initial_layout=initial_positions, distance_fn= relative_rbf_dists)
    algo.one_iteration()
    assert np.allclose(initial_positions, algo.get_positions())

