from hdimvis.data_fetchers.DataFetcher import DataFetcher
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96Algo import Chalmers96
from hdimvis.create_low_d_layout.LowDLayoutCreation import LowDLayoutCreation
import matplotlib.pyplot as plt
import numpy as np


data = DataFetcher().fetch_data('poker', size=500)
algo96 = Chalmers96(dataset=data, alpha=0.7)
layout = LowDLayoutCreation().create_layout(algo96, return_after=100)


pos = layout.get_final_positions()
x = pos[:, 0]
y = pos[:, 1]

# Draw plot
plt.scatter(x, y, cmap='viridis')
plt.axis('off')
plt.show()