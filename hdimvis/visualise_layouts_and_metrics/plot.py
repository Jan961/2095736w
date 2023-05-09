from pathlib import Path
from typing import Tuple, Optional, Callable, List
from ..create_low_d_layout.LayoutBase import LowDLayoutBase
import numpy as np
import  matplotlib.pyplot as plt
import math
import os
import pathlib


def show_layout(layout: LowDLayoutBase =None, use_labels: bool = False, alpha: float = None,
                color_by: Callable[[np.ndarray],float] = None,
                color_map: str = 'rainbow', size: float = 3, title: str = None,
                save_to: Path  =None,
                positions: np.ndarray =None,
                labels: np.ndarray= None) -> None:

    """

    Draw the spring layout graph.
    alpha: float in range 0 - 1 for the opacity of points
    color_by: function to represent a node as a single float which will be used to color it
    this can be used to colour the datapoints on the layout by for example
    one of the dimensions of the original high-d dataset
    """
    assert layout is not None or positions is not None

    fig, ax = plt.subplots()

    if layout:
        pos= layout.get_final_positions()
    else:
        pos = positions

    x = pos[:, 0]
    y = pos[:, 1]

    # Color nodes
    colors = 'b'
    cmap = None

    if labels is not None:
        colors = labels
    elif layout and use_labels:
        colors = layout.labels
    elif layout and color_by and layout.data is not None:
        colors = np.apply_along_axis(color_by, axis=1, arr=layout.data)

    if color_map is not None:
        cmap = plt.cm.get_cmap(color_map)

    ax.scatter(x, y, alpha=alpha, s=size, c=colors, cmap=cmap)
    if title:
        plt.title(title)

    plt.axis('off')
    plt.axis('equal')

    if save_to:
        plt.savefig((Path(save_to).joinpath(Path(f"{title}.png"))).resolve())

    plt.show()



def show_generation_metrics(layout, stress: bool = True, average_speed: bool = False, quartet_stress: bool = False,
                            title: str = None, save_to: Path = None, log_scale: bool = False):
    assert not average_speed or not quartet_stress # those are for different alog so can't both be used
    fig, ax1 = plt.subplots()

    line1, line2, line3 = [], [], []


    if stress:
        x1 = layout.collected_metrics['Stress'][0]
        y1 = layout.collected_metrics['Stress'][1]
        line1 = ax1.plot(x1, y1, c='r', label="Stress")
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("Stress")

        if average_speed or quartet_stress: # add second axis for speed/quartet stress
            if average_speed:
                label = 'Average speed'
            else:
                label = 'Average quartet stress'

            ax2 = ax1.twinx()
            x2 = layout.collected_metrics[label][0]
            y2 = layout.collected_metrics[label][1]
            line2 = ax2.plot(x2, y2, c='b', label=label)
            ax2.set_ylabel(label)
            if log_scale:
                ax2.yscale("log")


    elif quartet_stress: # quartet stress alone
        x1 = layout.collected_metrics['Average quartet stress'][0]
        y1 = layout.collected_metrics['Average quartet stress'][1]
        line1 = ax1.plot(x1, y1, c='r', label="Average quartet stress")
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("Average quartet stress")

    elif average_speed: # velocity alone
        x1 = layout.collected_metrics['Average speed'][0]
        y1 = layout.collected_metrics['Average speed'][1]
        line1 = ax1.plot(x1, y1, c='r', label="Average speed")
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("Average speed")

    if log_scale:
        ax1.yscale("log")

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    if title:
        plt.title(title)
    else:
        plt.title(f"{layout.algorithm.dataset.name} - {layout.algorithm.get_name()}" )

    ax1.legend(lines,labels)
    plt.tight_layout()

    if save_to:
        plt.savefig((Path(save_to).joinpath(Path(f"{title}.png"))).resolve())

    plt.show()








