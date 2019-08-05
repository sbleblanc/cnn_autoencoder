import torch.nn as nn

def build_cnn1d_block(input_channels, output_channels, n_conv_layers, kernel_size, transposed=False, padding=None, norm_before_activation=True, forward=True):
    block = nn.Sequential()
    if padding is None:
        padding = (kernel_size - 1) // 2
    for i in range(n_conv_layers):
        cnn_input = input_channels
        cnn_output = output_channels
        if forward:
            if i > 0:
                cnn_input = output_channels
        else:
            if i < n_conv_layers - 1:
                cnn_output = input_channels
        if transposed:
            block.add_module('tconv_{}'.format(i), nn.ConvTranspose1d(cnn_input, cnn_output, kernel_size, padding=padding))
        else:
            block.add_module('conv_{}'.format(i), nn.Conv1d(cnn_input, cnn_output, kernel_size, padding=padding))
        if norm_before_activation:
            block.add_module('b_norm_{}'.format(i), nn.BatchNorm1d(cnn_output))
            block.add_module('relu_{}'.format(i), nn.ReLU())
        else:
            block.add_module('relu_{}'.format(i), nn.ReLU())
            block.add_module('b_norm_{}'.format(i), nn.BatchNorm1d(cnn_output))

    return block
