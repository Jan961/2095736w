from ..algorithms import BaseAlgorithm
from abc import ABCMeta, abstractmethod

class LowDLayoutBase:

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_final_positions(self):
        pass