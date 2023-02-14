import numpy as np

def wiggle_index_vector_pair(v1 : np.ndarray, v2 : np.ndarray):

    assert v1.ndim == 1 and v2.ndim == 1, "Vectors have to be 1-dimensional np arrays"

    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    cos = np.dot(v1, v2)/(l1*l2)

    scaling_f = calculate_scaling_factor()

    transformed_l1 = vector_magnitude_map_function(l1, scaling_f)
    transformed_l2 = vector_magnitude_map_function(l1, scaling_f)
    transformed_cos = cosine_magnitude_map_function(l1)


def vector_magnitude_map_function(input: float, cutoff: float):

    assert cutoff > 0, "Cutoff has to be greater than 0"
    if input == 0:
        input += 1e-12



def cosine_magnitude_map_function(input: float):




def calculate_scaling_factor():
    pass



