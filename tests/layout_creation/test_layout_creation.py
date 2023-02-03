from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.Chalmers96Layout import Chalmers96Layout
from hdimvis.create_low_d_layout.SQuaDLayout import SQuaDLayout
from hdimvis.data_fetchers.Dataset import Dataset
from hdimvis.data_fetchers.DataFetcher import DataFetcher
import numpy as np

mock_data = np.random.randint(0,10, (40,3))
dataset = Dataset(mock_data, None, 'mock data')
algorithms = [Chalmers96(dataset=dataset), SQuaD(dataset=dataset)]
layout_classes = [Chalmers96Layout, SQuaDLayout]

mock_data_2= np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
mock_dataset_2 = Dataset(mock_data_2, None, "mock data")
mock_dataset_3 = DataFetcher().fetch_data("mock data")



# noinspection PyTypeHints
def test_low_lvl_layout_created_correctly_for_chalmers96():
        algo =algorithms[0]
        layout_class = layout_classes[0]
        layout = LowDLayoutCreation().create_layout(algo, optional_metric_collection={'Stress': 2, 'Average speed': 1},
                                                    no_iters=4)
        assert isinstance(layout, layout_class)
        assert layout.optional_metric_collection['Stress'] == 2
        assert layout.optional_metric_collection['Average speed'] == 1



def test_stress_collected_correctly():

    for algo in algorithms:
        for i in [1,2,3]:
            layout = LowDLayoutCreation().create_layout(algo, no_iters=4, optional_metric_collection={'Stress': i})
            if i != 3:
                assert len(layout.collected_metrics['Stress'][0]) == 4//i + 1
                assert len(layout.collected_metrics['Stress'][1]) == 4 // i + 1
            else:
                assert len(layout.collected_metrics['Stress'][0]) == 4 // i + 2
                assert len(layout.collected_metrics['Stress'][1]) == 4 // i + 2

            assert np.allclose(algo.get_vectorised_euclidian_stress(), algo.get_unvectorised_euclidian_stress())


def test_low_lvl_layout_created_correctly_for_squad():
    algo = SQuaD(dataset=mock_dataset_2, initial_layout=initial_positions)
    layout = LowDLayoutCreation().create_layout(algo, no_iters=2)

    assert np.allclose(initial_positions, layout.get_final_positions()) #tests if the correspondence between low D and high D
    assert np.allclose(initial_positions, layout.data)                                                          # maintained and the points are not shuffled


#surprising result for stress
def test_stress_decreases_as_expected():


    algo = SQuaD(dataset=mock_dataset_3, initial_layout=initial_positions)
    stress_normal_1 = algo.get_vectorised_euclidian_stress()
    measurements = {'Stress': 1, "Average quartet stress": 1}
    layout1 = LowDLayoutCreation().create_layout(algo, optional_metric_collection=measurements, no_iters=1,)
    stress_quartet_1 = algo.get_average_quartet_stress()
    layout2 = LowDLayoutCreation().create_layout(algo, optional_metric_collection=measurements, no_iters=4, )
    stress_quartet_2 = algo.get_average_quartet_stress()
    stress_normal_2 = algo.get_vectorised_euclidian_stress()
    print(layout2.collected_metrics)

    assert stress_normal_1 > stress_normal_2
    assert stress_quartet_1 > stress_quartet_2

