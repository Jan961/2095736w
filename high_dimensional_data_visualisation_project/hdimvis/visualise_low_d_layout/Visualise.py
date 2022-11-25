from ..create_low_d_layout.LowDLayoutBase import LowDLayoutBase
import numpy as np
import  matplotlib.pyplot as plt

class Visualise:

    def __init__(self, layout: LowDLayoutBase):
        self.layout = layout

    def draw_layout(self):
        pos: np.ndarray = self.layout.get_final_positions()
        x = pos[:, 0]
        y = pos[:, 1]



        # Draw plot
        sc = plt.scatter(x, y,  cmap='viridis')
        plt.axis('off')

