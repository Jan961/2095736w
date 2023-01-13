from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from experiments.basic_comparison.BasicComparison import BasicComparison
import numpy as np
from hdimvis.data_fetchers.Dataset import Dataset


mock_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
dataset = Dataset(mock_data, np.array([0,1,2,3]), 'mock data')


def test_basic_comparison():
    algo1 = Chalmers96(dataset=dataset, neighbour_set_size=1, sample_set_size=2)
    algo2 = SQuaD(dataset=dataset)

    metric_collection = {'stress': 10, 'average speed': 10}

    expr = BasicComparison("test experiment", algo1, algo2, iterations=2, num_repeats=2,
                           metric_collection=metric_collection)
