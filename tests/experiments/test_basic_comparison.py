from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from experiments.basic_comparison.BasicComparison import BasicComparison
import numpy as np
from hdimvis.data_fetchers.Dataset import Dataset


mock_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
dataset = Dataset(mock_data, np.array([0,1,2,3]), 'mock data')


def test_basic_comparison():
    algos_input = dict()
    algo1, name1 = Chalmers96(None, neighbour_set_size=1, sample_set_size=2), "simple 96"
    algo2, name2 = SQuaD(None), "basic squad"
    algos_input[algo1] = name1
    algos_input[algo2] = name2

    metric_collection = {'stress': 2, 'average speed': 1}

    expr = BasicComparison( algos_input, experiment_name="test experiment", iterations=2, num_repeats=2,
                            dataset_names=['mock data', 'mock data'], metric_collection = metric_collection)
    expr.run()

    print(expr.results)
    #test basic metrics
    assert len(expr.results['mock data']) == 2 # num algorithms
    assert expr.results['mock data']['simple 96'].shape[0] == 2 # num repeats
    assert expr.results['mock data']['simple 96'].shape[1] == 4 # num available metrics

    assert expr.results['mock data']['basic squad'].shape[0] == 2
    assert expr.results['mock data']['basic squad'].shape[1] == 4

    #test generation metric collection for chalmers' 96
    assert len(expr.layouts['mock data']['basic squad']) == 2 # num repeats
    assert len(expr.layouts['mock data']['simple 96']) == 2  # num repeats






