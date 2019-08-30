import math


def compute_lout(lin: int, padding: int, dilation: int, kernel_size: int, stride: int):
    numerator = lin + 2 * padding - dilation * (kernel_size - 1) - 1
    return math.floor(numerator / stride + 1)


def get_expected_conv_1d_lout(lin, conv_module):
    padding = conv_module.padding[0]
    dilation = conv_module.dilation[0]
    kernel_size = conv_module.kernel_size[0]
    stride = conv_module.stride[0]

    return compute_lout(lin, padding, dilation, kernel_size, stride)


def get_expected_mp_1d_lout(lin, mp_module):
    padding = mp_module.padding
    dilation = mp_module.dilation
    kernel_size = mp_module.kernel_size
    stride = mp_module.stride

    return compute_lout(lin, padding, dilation, kernel_size, stride)




