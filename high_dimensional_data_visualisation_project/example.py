from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.distance_measures.poker_distance import poker_distance
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
from hdimvis.visualise_low_d_layout.visualise import show_layout
import matplotlib.pyplot as plt
import numpy as np


data = DataFetcher().fetch_data('poker', size=500)
algo96 = Chalmers96(dataset=data, alpha=0.7, distance_fn=poker_distance)
layout = LowDLayoutCreation().create_layout(algo96, return_after=100)

show_layout(layout, dataset=data, color_by=lambda d:d[10])