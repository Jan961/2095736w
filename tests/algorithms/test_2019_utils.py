import numpy as np

from hdimvis.algorithms.spring_force_algos.utils import point_on_circle


def test_point_on_circle():

    x =1
    y = 1

    angles = [0, 90, 180, 270]
    expected_positions = np.array([[2,1], [1,2], [0,1], [1,0] ],dtype='float64')
    print(expected_positions)
    for i, a in enumerate(angles):
        assert np.allclose(np.array(point_on_circle(x,y, angle=a, radius=1),dtype='float64'), expected_positions[i])