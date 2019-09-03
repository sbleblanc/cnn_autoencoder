from enum import Enum


class Regularization(Enum):
    NONE = 0
    DROPOUT = 1
    BN_RELU = 2
    RELU_BN = 3

class ResArchitecture(Enum):
    SPLIT_LAST = 0
    FULL_INNER = 1
    FULL_PA = 2