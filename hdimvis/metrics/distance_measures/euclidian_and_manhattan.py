import numpy as np

def euclidean(arr: np.ndarray,  axis: int = 0) -> float | np.ndarray:
    return np.linalg.norm(arr, axis=axis)


def manhattan(arr: np.ndarray, axis : int = 0) -> float | np.ndarray:
    """
    Calculate the Manhattan distance - the sum of
    the distances along every dimension
    """
    return np.sum(np.abs(arr), axis=axis)