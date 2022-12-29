from typing import Tuple, Optional, Callable
from ..create_low_d_layout.LowDLayoutBase import LowDLayoutBase
import numpy as np
import  matplotlib.pyplot as plt


def show_layout(layout:LowDLayoutBase, dataset: np.ndarray, use_labels: bool=False, alpha: float = None, color_by: Callable[[np.ndarray],
            float] = None, color_map: str = 'viridis', size: float = 1) -> None:
    """


    Draw the spring layout graph.
    dataset: this can be used to colour the datapoints on the layout by for example
        one of the dimensions of the original
    high-d dataset
    alpha: float in range 0 - 1 for the opacity of points
    color_by: function to represent a node as a single float which will be used to color it
    color_map: string name of matplotlib.pyplot.cm to take colors from when coloring
               using color_by
    """
    # Get positions of nodes
    pos: np.ndarray = layout.get_final_positions()
    x = pos[:, 0]
    y = pos[:, 1]

    # Color nodes
    colors = 'b'
    cmap = None

    if use_labels:
        colors = layout.labels
    elif color_by is not None and dataset is not None:
        colors = np.apply_along_axis(color_by, axis=1, arr=dataset)
        cmap = plt.cm.get_cmap(color_map)


    # Draw plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=alpha, s=1, c=colors, cmap=cmap)
    plt.axis('off')
    plt.show()


