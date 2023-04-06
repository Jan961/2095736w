import numpy as np
import matplotlib.pyplot as plt
from hdimvis.data_fetchers.Dataset import Dataset
from pathlib import Path

colours = ['yellow', 'limegreen', 'r', 'b', 'fuchsia', 'orange']
label_types = [1, 2, 3, 4, 5, 6]

def show_original_globe(globe : Dataset, show_oceans: bool = True):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in label_types:
        p = globe.data[globe.labels == i]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c=colours[i - 1])


    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.invert_yaxis()
    ax.set_zlabel("z")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    plt.show()


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