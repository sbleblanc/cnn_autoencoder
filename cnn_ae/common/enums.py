from enum import Enum


class Regularization(Enum):
    NONE = 0
    DROPOUT = 1
    BN_RELU = 2
    RELU_BN = 3
