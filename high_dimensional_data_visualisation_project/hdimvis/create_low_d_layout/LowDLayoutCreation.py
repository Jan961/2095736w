from ..algorithms import BaseAlgorithm
import Chalmers96Layout

class LowDLayoutCreation:


    def create_layout(self, algorithm: BaseAlgorithm):

        if algorithm.name == 'chalmers96':
            layout = Chalmers96Layout.create(algorithm)
            return layout
