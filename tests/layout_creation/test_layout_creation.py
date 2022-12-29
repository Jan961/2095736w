from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.algorithms.stochastic_quartet_algo.SQuaD import SQuaD
from hdimvis.create_low_d_layout.Chalmers96Layout import Chalmers96Layout
from hdimvis.create_low_d_layout.SQuaDLayout import SQuaDLayout


data, labels = DataFetcher().fetch_data('rna N3k')
algorithms = [Chalmers96(data), SQuaD(data)]
layout_classes = [Chalmers96Layout, SQuaDLayout]


# noinspection PyTypeHints
def test_correct_low_lvl_layout_created():
    for i, algo in enumerate(algorithms):
        layout = LowDLayoutCreation().create_layout(algo, data, labels, no_iters=1)
        assert isinstance(layout, layout_classes[i])





