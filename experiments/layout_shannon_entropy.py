import numpy as np

from hdimvis.create_low_d_layout.LowDLayoutBase import LowDLayoutBase
from .LayoutHistogram import LayoutHistogram


def calculate_entropy(layout: LowDLayoutBase, **kwargs ):
    histogram = LayoutHistogram(layout=layout, pdf=True, **kwargs).histogram
    non_zero_entries = histogram[np.nonzero(histogram)]
    p_times_log_p  = non_zero_entries*np.log2(non_zero_entries)
    return -np.sum(p_times_log_p)

