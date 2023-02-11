from dataclasses import dataclass
import numpy as np


class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray = None, name: str = 'no name'):
        self.data = data
        self.labels = labels
        self.name = name




