from enum import Enum


class HybridStage(Enum):
    PLACE_SAMPLE = 0
    INTERPOLATE = 1
    REFINE = 2