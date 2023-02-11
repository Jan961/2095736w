import numpy as np

from experiments.cube.Cube import Cube



def test_cube_has_no_overlapping_points():
    for i in [4, 10 ,100]:
        for a in [0, 1/3, 1/6]:
            cube = Cube(side=4, num_points=i, angle=a)
            points = cube.get_all_points()
            assert np.unique(points, axis=0).shape == points.shape
