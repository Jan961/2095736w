from hdimvis.create_low_d_layout.LowDLayoutBase import LowDLayoutBase
import numpy as np
import matplotlib.pyplot as plt


class LayoutHistogram():

    def __init__(self, bins: int = 50, layout: LowDLayoutBase = None, positions: np.ndarray = None,
                 range: np.ndarray = None, pdf: bool = False):

        assert layout is not None or positions is not None, "Provide 2D positions or a layout object"
        assert not (layout is not None and positions is not None), "Provide only either a layout object or 2D positions"
        if range is not None:
            assert range.shape == (2,2)

        self.bins =bins
        self.range = range # range  [[xmin, xmax], [ymin, ymax]]
        self.positions = positions
        self.layout = layout
        self.pdf = pdf
        self.histogram, self.xedges, self.yedges = self.create_histogram()



    def create_histogram(self):
        if self.positions is not None:
            x= self.positions[:,0]
            y = self.positions[:,1]
        else:
            x= self.layout.get_final_positions()[:,0]
            y = self.layout.get_final_positions()[:, 1]

        # the parameter "density" converts the histogram into a PDF
        return np.histogram2d(x, y, self.bins, self.range, density=self.pdf)

    def show_histogram(self ,title: str =None):

        # Histogram does not follow Cartesian convention,
        # therefore transpose H for visualization purposes.
        H = self.histogram
        #H = H.T
        fig = plt.figure()
        ax = fig.add_subplot(title= title)
        plt.imshow(H, interpolation='nearest', origin='lower',
                   extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])










