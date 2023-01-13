from typing import Tuple, Optional, Callable, List
from ..create_low_d_layout.LowDLayoutBase import LowDLayoutBase
import numpy as np
import  matplotlib.pyplot as plt
import math


def show_layouts(*layouts: LowDLayoutBase, use_labels: bool = False, alpha: float = None,
                color_by: Callable[[np.ndarray],float] = None,
                color_map: str = 'viridis', size: float = 1, title: str = None,
                sub_titles: List[str] = None) -> None:


    """

    Draw the spring layout graph.
    alpha: float in range 0 - 1 for the opacity of points
    color_by: function to represent a node as a single float which will be used to color it
    this can be used to colour the datapoints on the layout by for example
    one of the dimensions of the original high-d dataset
    color_map: string name of matplotlib.pyplot.cm to take colors from when coloring
               using color_by
    """

    no_layouts = len(layouts)
    assert no_layouts > 0

    if no_layouts == 1:
        fig, axes = plt.subplots()
    else:
        r = math.floor(no_layouts/2)
        c = math.ceil(no_layouts/2)
        fig, axes = plt.subplots(r,c)

    # Get positions of nodes

    idx_r = 0
    idx_c = 0
    for i, layout in enumerate(layouts):
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

        if no_layouts == 1:
            axis = axes
        else:
            if idx_c == c-1:
                idx_r += 1
            axis = axes[idx_r, idx_c]
            idx_c += 1

        axis.scatter(x, y, alpha=alpha, s=size, c=colors, cmap=cmap)

        if sub_titles is not None and len(sub_titles) >= i+1:
            axis.title.set_text(sub_titles[i])

    if title:
        plt.title(title)


    plt.show()


def show_generation_metrics(layout, stress: bool = True, average_speed: bool = False,
                            title: str = None):

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
    if title:
        plt.title(title)
    ax1.legend(lines,labels)
    plt.show()



