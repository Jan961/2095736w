import numpy as np
import matplotlib.pyplot as plt
from hdimvis.data_fetchers.Dataset import Dataset
from pathlib import Path

colours = ['yellow', 'limegreen', 'r', 'b', 'fuchsia', 'orange']
label_types = [1, 2, 3, 4, 5, 6]

def show_original_globe(globe : Dataset):
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


    fig, ax = plt.subplots()
    for i in label_types:
        p = points_2d[dataset.labels == i]
        ax.scatter(p[:, 0], p[:, 1], s=size, c=colours[i - 1], alpha=alpha)


    if title:
        plt.title(title)

    # plt.axis('off')
    plt.axis('equal')

    if save_to:
        plt.savefig((Path(save_to).joinpath(Path(f"{title}.png"))).resolve())

    plt.show()