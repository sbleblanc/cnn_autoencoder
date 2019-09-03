from enum import Enum


class Regularization(Enum):
    NONE = 0
    DROPOUT = 1
    BN_RELU = 2
    RELU_BN = 3

class ResArchitecture(Enum):
    ORIGINAL = 0
    BN_ADD = 1
    RELU_ADD = 2
    RELU_PA = 3
    FULL_PA = 4