from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from experiments.utils.SimpleComparison import SimpleComparison
import numpy as np
from hdimvis.data_fetchers.Dataset import Dataset


mock_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
dataset = Dataset(mock_data, np.array([0,1,2,3]), 'mock data')


def test_correct_number_of_metrics_collected():
    algos_input = dict()
    algo1, name1 = Chalmers96(None, neighbour_set_size=1, sample_set_size=2), "simple 96"
    algo2, name2 = SNeD(None), "basic squad"
    algos_input[algo1] = name1
    algos_input[algo2] = name2

    metric_collection = {'Stress': 2, 'Average speed': 1, 'Average quartet stress':1}

    expr = SimpleComparison(algos_input, experiment_name="test experiment", iterations=2, num_repeats=2,
                            dataset_names=['mock data', 'mock data'],
                            metric_collection_during_layout_creation = metric_collection)
    expr.run()

    print(expr.results)
    #test basic metrics
    assert len(expr.results['mock data']) == 2 # num algorithms
    assert len(expr.results['mock data']['simple 96']['final stress']) == 2 # num repeats
    assert len(expr.results['mock data']['simple 96']) == 3 # num available metrics

    assert len(expr.results['mock data']['basic squad']['final squad stress']) == 2
    assert len(expr.results['mock data']['basic squad']) == 4

    #test generation metric collection for chalmers' 96
    assert len(expr.layouts['mock data']['basic squad']) == 2 # num repeats
    assert len(expr.layouts['mock data']['simple 96']) == 2  # num repeats


def test_correct_stress_calculated():
    algos_input = dict()
    algo1, name1 = Chalmers96(None, neighbour_set_size=1, sample_set_size=2), "simple 96"
    algo2, name2 = SNeD(None), "basic squad"
    algos_input[algo1] = name1
    algos_input[algo2] = name2


    expr = SimpleComparison(algos_input, experiment_name="test experiment", iterations=2, num_repeats=2,
                            dataset_names=['mock data', 'mock data'],
                            metric_collection_during_layout_creation = None)
    expr.run()

    print(expr.results)

    assert algo1.get_stress() == expr.results['mock data']['simple 96']['final stress'][-1]
    assert algo2.get_stress() == expr.results['mock data']['basic squad']['final stress'][-1]





