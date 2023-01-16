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

    metric_collection = {'stress': 2, 'average speed': 1}

    expr = BasicComparison( algo1, algo2, experiment_name="test experiment", iterations=3, num_repeats=2,
                           metric_collection = metric_collection)
    expr.run()

    print(expr.optional_generation_metrics)
    #test basic metrics
    assert len(expr.basic_metrics) == 2
    assert expr.basic_metrics[0].shape[0] == 2
    assert expr.basic_metrics[0].shape[1] == 4

    assert len(expr.optional_generation_metrics) == 2  # no algorithms

    #test generation metric collection for chalmers' 96
    assert expr.optional_generation_metrics[0]['stress'].shape[0] == 2 #no repeats
    assert expr.optional_generation_metrics[0]['stress'].shape[1] == 3 # no stress measurements during one run

    assert expr.optional_generation_metrics[0]['average speed'].shape[0] == 2  # no repeats
    assert expr.optional_generation_metrics[0]['average speed'].shape[1] == 4  # as for stress

    # test generation metric collection for Squad
    assert len(expr.optional_generation_metrics[1]) == 1
    assert expr.optional_generation_metrics[1]['stress'].shape[0] == 2  # no repeats
    assert expr.optional_generation_metrics[1]['stress'].shape[1] == 3  # no stress measurements during one run




