import numpy as np


# running mean with smoothed-out boundary effects of convolution
# takes 1d numpy arr as input

def smooth_running_mean(input: np.ndarray, window_size: int):

    conv_arr = np.ones(window_size)/window_size
    convolved = np.convolve(input,conv_arr, mode='same')

    for i in range(window_size//2 + 1):
        upper_bound = window_size//2 + i
        convolved[i] = np.average(input[0:upper_bound])

        lower_bound = -window_size//2 - i
        convolved[-1 - i] = np.average(input[lower_bound:-1])

    return convolved