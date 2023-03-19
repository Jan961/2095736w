import numpy as np
import matplotlib.pyplot as plt
from hdimvis.data_fetchers.Dataset import Dataset
from pathlib import Path


def show_globe_embedding(dataset: Dataset, points_2d: np.ndarray,
                         color_map = None, title: str = None,
                         alpha: float = 1,
                         size: float = 1,
                         save_to : Path = None):




    x = points_2d[:, 0]
    y = points_2d[:, 1]
    colors = dataset.labels

    # cmap = None
    #
    # if use_labels:
    #     colors = layout.labels
    # elif color_by is not None and layout.data is not None:
    #     colors = np.apply_along_axis(color_by, axis=1, arr=layout.data)
    #
    if color_map is not None:
        cmap = plt.cm.get_cmap(color_map)
    else:
        cmap = 'rainbow'

    fig, axes = plt.subplots()
    axes.scatter(x, y, alpha=alpha, s=size, c=colors, cmap=cmap)


    if title:
        plt.title(title)

    # plt.axis('off')
    plt.axis('equal')

    if save_to:
        plt.savefig((Path(save_to).joinpath(Path(f"{title}.png"))).resolve())

    plt.show()