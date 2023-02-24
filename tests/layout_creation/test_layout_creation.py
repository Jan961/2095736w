from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.algorithms.stochastic_ntet_algo.SNeD import SNeD
from hdimvis.create_low_d_layout.Chalmers96Layout import Chalmers96Layout
from hdimvis.create_low_d_layout.SQuaDLayout import SQuaDLayout
from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from hdimvis.data_fetchers.Dataset import Dataset
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean,manhattan
import numpy as np


mock_data = np.random.randint(0,10, (40,3))
mock_data_initial_positions = np.random.randint(0,5, (40,2))
dataset = Dataset(mock_data, None, 'mock data')
algorithms = [Chalmers96(dataset=dataset), SNeD(dataset=dataset), Hybrid(dataset=dataset)]
layout_classes = [Chalmers96Layout, SQuaDLayout]

mock_data_2= np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
initial_positions = np.array([[0,0],[0,10],[10,10],[10,0]], dtype='float64')
mock_dataset_2 = Dataset(mock_data_2, None, "mock data")
mock_dataset_3 = DataFetcher().fetch_data("mock data")


# the below removes an annoying pycharm warning
# noinspection PyTypeHints
def test_low_lvl_layout_created_correctly_for_chalmers96():
        algo =algorithms[0]
        layout_class = layout_classes[0]
        layout = LowDLayoutCreation().create_layout(algo, optional_metric_collection={'Stress': 2, 'Average speed': 1},
                                                    no_iters=4)
        assert isinstance(layout, layout_class)
        assert layout.optional_metric_collection['Stress'] == 2
        assert layout.optional_metric_collection['Average speed'] == 1

def test_low_lvl_layout_created_correctly_for_squad():
    algo = SNeD(dataset=mock_dataset_2, initial_layout=initial_positions)
    layout = LowDLayoutCreation().create_layout(algo, no_iters=2)

    assert np.allclose(initial_positions, layout.get_final_positions()) #tests if the correspondence between low D and high D
    # in the internal representation is maintained and the points are not shuffled
    assert np.allclose(initial_positions, layout.data)


def test_low_lvl_layout_created_correctly_for_hybrid():
    algo = Hybrid(dataset=dataset, initial_layout=mock_data_initial_positions,
                  sample_set_size=1, neighbour_set_size=1,
                  interpolation_adjustment_sample_size = 1)
    layout = LowDLayoutCreation().create_layout(algo)

    assert not np.allclose(mock_data_initial_positions, layout.get_final_positions()) #tests if the correspondence between low D and high D
    assert np.allclose(mock_data, layout.data)




def test_stress_collected_correctly():
    norms = ['euclidian', "manhattan"]
    norm_fns = [euclidean, manhattan]

    for algo in algorithms[:-1]: #exclude hybrid as sample and neighbour set size parameter would have to be set
                                # and we don't collect stress during generation for hybrid anyway
        for j in range(2):
            norm_name = norms[j]
            norm_fn = norm_fns[j]
            for i in [1,2,3]:
                layout = LowDLayoutCreation().create_layout(algo, no_iters=4,
                                                            optional_metric_collection={'Stress': i, "norm": norm_name})
                if i != 3:
                    assert len(layout.collected_metrics['Stress'][0]) == 4//i + 1
                    assert len(layout.collected_metrics['Stress'][1]) == 4 // i + 1
                else:
                    assert len(layout.collected_metrics['Stress'][0]) == 4 // i + 2
                    assert len(layout.collected_metrics['Stress'][1]) == 4 // i + 2

                assert np.allclose(algo.get_vectorised_stress(norm_fn), algo.get_unvectorised_stress(norm_fn))





#surprising result for stress
def test_stress_decreases_as_expected():


    algo = SNeD(dataset=mock_dataset_3, initial_layout=initial_positions)
    stress_normal_1 = algo.get_vectorised_stress(euclidean)
    measurements = {'Stress': 1, "Average quartet stress": 1}
    layout1 = LowDLayoutCreation().create_layout(algo, optional_metric_collection=measurements, no_iters=1,)
    stress_quartet_1 = algo.get_average_quartet_stress()
    layout2 = LowDLayoutCreation().create_layout(algo, optional_metric_collection=measurements, no_iters=4, )
    stress_quartet_2 = algo.get_average_quartet_stress()
    stress_normal_2 = algo.get_vectorised_stress(euclidean)
    print(layout2.collected_metrics)

    assert stress_normal_1 > stress_normal_2 # "standard" stress
    assert stress_quartet_1 > stress_quartet_2 # average quartet stress

