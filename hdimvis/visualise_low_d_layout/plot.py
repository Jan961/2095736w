from typing import Tuple, Optional, Callable
from ..create_low_d_layout.LowDLayoutBase import LowDLayoutBase
import numpy as np
import  matplotlib.pyplot as plt


def show_layout(layout:LowDLayoutBase, use_labels: bool=False, alpha: float = None, color_by: Callable[[np.ndarray],
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
    elif color_by is not None and layout.data is not None:
        colors = np.apply_along_axis(color_by, axis=1, arr=layout.data)
        cmap = plt.cm.get_cmap(color_map)


    # Draw plot
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=alpha, s=1, c=colors, cmap=cmap)
    plt.axis('off')
    plt.show()


def show_generation_metrics(layout:LowDLayoutBase, stress: bool = True, average_speed: bool =False):
    fig, ax1 = plt.subplots()

    line1, line2, line3 = [], [], []

    if stress:
        x1 = layout.collected_metrics['stress'][0]
        y1 = layout.collected_metrics['stress'][1]
        line1 = ax1.plot(x1, y1, c='r', label="Stress")
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("Stress")
        print(line1)


    if average_speed:
        ax2 = ax1.twinx()
        x2 = layout.collected_metrics['average speed'][0]
        y2 = layout.collected_metrics['average speed'][1]
        line2 = ax2.plot(x2, y2, c='b', label="Average Speed")
        ax2.set_ylabel("Average Speed")
        print(line2)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines,labels)
    plt.show()



