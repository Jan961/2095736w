from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.Chalmers96Layout import Chalmers96Layout
from hdimvis.create_low_d_layout.SQuaDLayout import SQuaDLayout
from hdimvis.data_fetchers.Dataset import Dataset
import numpy as np

mock_data = np.random.randint(0,10, (40,3))
dataset = Dataset(mock_data, None, 'mock data')
algorithms = [Chalmers96(dataset=dataset), SQuaD(dataset=dataset)]
layout_classes = [Chalmers96Layout, SQuaDLayout]

mock_data_2= np.array([[0,0],[0,1],[1,1],[1,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,1],[1,1],[1,0]], dtype='float64')
mock_dataset_2 = Dataset(mock_data, None, "mock data")


# noinspection PyTypeHints
def test_low_lvl_layout_created_correctly_for_chalmers96():
        algo =algorithms[0]
        layout_class = layout_classes[0]
        layout = LowDLayoutCreation().create_layout(algo, dataset, {'stress': 2, 'average speed': 1}, no_iters=4)
        assert isinstance(layout, layout_class)
        assert layout.optional_metric_collection['stress'] == 2
        assert layout.optional_metric_collection['average speed'] == 1




def test_stress_collected_correctly():

    for algo in algorithms:
        for i in [1,2]:
            layout = LowDLayoutCreation().create_layout(algo, dataset, {'stress': i}, no_iters=4)
            assert len(layout.collected_metrics['stress'][0]) == 4//i + 1
            assert len(layout.collected_metrics['stress'][1]) == 4 // i + 1


def test_low_lvl_layout_created_correctly_for_squad():
    algo = SQuaD(dataset=mock_dataset_2, initial_layout=initial_positions)
    layout = LowDLayoutCreation().create_layout(algo, i)




