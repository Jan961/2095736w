from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.distance_measures.euclidian_and_manhattan import euclidean
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.visualise_low_d_layout.show_layout import show_layout
import matplotlib.pyplot as plt
import numpy as np


data, labels = DataFetcher().fetch_data('rna N3k')
algo96 = Chalmers96(dataset=data, alpha=0.7, distance_fn=euclidean)

layout = LowDLayoutCreation().create_layout(algo96, data, labels, no_iters=100)

show_layout(layout, dataset=data, use_labels=True)
