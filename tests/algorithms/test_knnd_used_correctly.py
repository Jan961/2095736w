import numpy as np
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.Dataset import Dataset
from hdimvis.metrics.distance_measures.euclidian_and_manhattan import euclidean

mock_data = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [40, 39, 39, 39], [40, 40, 40, 40]])
dataset = Dataset(mock_data, np.array([0,1,2,3]), 'mock data')

def test_knnd_created_correctly():
    algo96 = Chalmers96(dataset=dataset, alpha=0.7, distance_fn=euclidean, use_knnd=True,
                        sample_set_size=0, neighbour_set_size=1)

    knnd_indx = algo96.knnd_index
    # check if datapoint 0 is the nn of datapoint 1 and vice versa
    for node in algo96.nodes:
        print(node.datapoint)
    assert knnd_indx.neighbor_graph[0][0][1] == 1
    assert knnd_indx.neighbor_graph[0][1][1] == 0

    # check the order of nodes and datapoints in the index is the same and correct
    assert algo96.nodes[2].datapoint[0] == 40 and algo96.nodes[2].datapoint[1] == 39
    assert algo96._get_neighbours(2)[0] == 3
    assert algo96._get_neighbours(2)[0] == knnd_indx.neighbor_graph[0][2][1]


def test_knnd_used_correctly():
    initial_layout = np.array([[0,0],[6,6],[20,20],[40,40]])
    algo96 = Chalmers96(dataset=dataset, spring_constant=0.7, damping_constant=0.3, initial_layout=initial_layout,
                        distance_fn=euclidean, use_knnd=True,
                        sample_set_size=0, neighbour_set_size=1)

    layout = LowDLayoutCreation().create_layout(algo96, no_iters=1)
    lowd_layout = layout.get_final_positions()

    # check if similar points move closer and dissimilar farther away
    assert euclidean(initial_layout[0] - initial_layout[1]) > euclidean(lowd_layout[0] - lowd_layout[1])
    assert euclidean(initial_layout[2] - initial_layout[3]) > euclidean(lowd_layout[2] - lowd_layout[3])
    assert euclidean(initial_layout[1] - initial_layout[2]) < euclidean(lowd_layout[1] - lowd_layout[2])
