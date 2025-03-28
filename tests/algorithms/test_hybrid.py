from hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid import Hybrid
from numpy.testing import assert_almost_equal
from hdimvis.data_fetchers.Dataset import Dataset
import numpy as np
import math

# code adapted and modified from 2019 Project by Iain Cattermole

dataset = Dataset(np.array([
    [0, 0],
    [math.sqrt(2)/2, math.sqrt(2)/2],
    [10, 10],
    [20, 28],
    [0, 20],
]), None, "test data")

initial_layout = np.array([[0,0],[math.sqrt(2), math.sqrt(2)], [1,1],[2,2.8],[0,2]])

def set_node_positions(algorithm):
    for node in algorithm.nodes:
        node.x, node.y = node.datapoint


def test_find_node_gets_the_parent_with_minimum_distance():
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=np.array([1, 2, 3, 4]))
    test_node = algorithm.nodes[0]
    parent_index = algorithm._find_parent(test_node)[0]

    assert parent_index == 0  # the closest node has index 0 in the sample (1 global index)
    assert np.allclose(algorithm.sample[parent_index].datapoint, np.array([math.sqrt(2)/2, math.sqrt(2)/2]))


def test_create_error_fn_returns_function_that_returns_expected():

    sample_indexes = np.array([1, 2, 3, 4])
    algorithm = Hybrid(dataset=dataset, preset_sample=sample_indexes,
                       initial_layout=initial_layout,interpolation_adjustment_sample_size=1,
                       use_correct_interpolation_error=False)


    source = algorithm.nodes[0]
    hd_distances = [algorithm.hd_distance(source, algorithm.nodes[t]) for t in sample_indexes]
    print(f"hd distances {hd_distances}")
    error_fn = algorithm._create_error_fn(0, hd_distances)

    # radius of the circle used by the fn is 1 ( distance between [0,0] and [math.sqrt(2)/2, math.sqrt(2)/2]
    # thus the point at
    # 0 angle has coords [2,0]
    # 90 angle [1,1]
    # 180  [0,0]
    # and 270 [1,-1] therefore:
    potential_positions = [[math.sqrt(2)+1,math.sqrt(2)],[math.sqrt(2),math.sqrt(2)+1],
                          [math.sqrt(2)-1,math.sqrt(2)],[math.sqrt(2),math.sqrt(2)-1]]

    for i, angle in enumerate([0, 90, 180, 270]):
        ld_distances = [ np.linalg.norm(algorithm.get_positions()[j] - potential_positions[i]) for j in sample_indexes]
        print(f"ld dist manual: {ld_distances}")

        assert error_fn(angle, correct_error_calc=True) == np.sum((np.array(ld_distances) - np.array(hd_distances))**2)

        assert error_fn(angle, correct_error_calc=False) == np.abs(np.sum(ld_distances) - np.sum(hd_distances))


# failed in the original
def test_find_circle_quadrant_finds_expected_angles():

    # expected does not mean angles we would want but those that minimise the error function as expected
    sample_indexes = np.array([1, 2, 3])
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=sample_indexes,
                       initial_layout=initial_layout,
                       interpolation_adjustment_sample_size=1,
                       use_correct_interpolation_error=True)


    # source: [0, 0] with parent [math.sqrt(2)/2, math.sqrt(2)/2],
    source = algorithm.nodes[0]
    distances_hd = [algorithm.hd_distance(source, algorithm.nodes[t]) for t in sample_indexes]
    print(f" hd distances inside test fn {distances_hd} ")
    error_fn = algorithm._create_error_fn(0, distances_hd)

    lower_angle, upper_angle = algorithm._find_circle_quadrant(error_fn)
    assert lower_angle == 180 # i.e we get the opposite quadrant to that we would want bc
    assert upper_angle == 270 # in this special case where HD distances are much larger the error function is
    # minimised by increasing the LD distances - even if that does not improve the layout

# def test_binary_search_angle_finds_best_angle_with_original_error():
#     sample_indexes = [1, 2, 3]
#     algorithm = Hybrid(dataset=dataset,
#                        preset_sample=np.array(sample_indexes),
#                        random_sample_size=1)
#     set_node_positions(algorithm)
#
#     # source: [0, 1] with parent [0, 0]
#     source = algorithm.nodes[0]
#     distances = [algorithm.hd_distance(source, algorithm.nodes[t]) for t in sample_indexes]
#     error_fn = algorithm._create_error_fn(0, distances)
#     lower_angle, upper_angle = (90, 180)
#
#     best_angle = algorithm._binary_search_angle(lower_angle, upper_angle, error_fn)
#     assert abs(best_angle - 90) <= 4  # must be within 4 degrees of best angle
#
#     # source: [1, 1] with parent [0, 0]
#     source = algorithm.nodes[4]
#     distances = [algorithm.hd_distance(source, algorithm.nodes[t]) for t in sample_indexes]
#     error_fn = algorithm._create_error_fn(0, distances)
#     lower_angle, upper_angle = (0, 90)
#
#     best_angle = algorithm._binary_search_angle(lower_angle, upper_angle, error_fn)
#     assert abs(best_angle - 45) <= 4  # must be within 4 degrees of best angle


def test_sample_distances_sum_returns_correct_sum():
    sample_indexes = [1, 2, 3]
    data = dataset.data
    algorithm = Hybrid(dataset=dataset,
                       preset_sample=np.array(sample_indexes),
                       random_sample_size=1)
    set_node_positions(algorithm)

    assert algorithm._sample_distances_sum(3, 0) == np.sum(np.linalg.norm(data[1:4] - np.array([3,0]), axis=1))
    assert algorithm._sample_distances_sum(-1, 0) == np.sum(np.linalg.norm(data[1:4] - np.array([-1,0]), axis=1))
    assert algorithm._sample_distances_sum(1, 0) == np.sum(np.linalg.norm(data[1:4] - np.array([1,0]), axis=1))

