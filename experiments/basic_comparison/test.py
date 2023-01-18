from .BasicComparison import BasicComparison
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96


algo1 = Chalmers96(use_knnd=True)

datasets = ['mnist', 'coil20', 'rna N3k', 'airfoil']