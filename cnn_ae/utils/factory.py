import torch.nn as nn
from cnn_ae.common.enums import Regularization, ResArchitecture
from cnn_ae.models.res import ResBlock
from collections import OrderedDict

def build_cnn1d_block(input_channels, output_channels, n_conv_layers, kernel_size, transposed=False, padding=None,
                      norm_before_activation=True, forward=True):
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


def build_regularized_relu_block(reg: Regularization, dropout:float = 0.5, num_elem:int = 0):
    if reg == Regularization.DROPOUT:
        return nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU()
        )
    elif reg == Regularization.BN_RELU:
        return nn.Sequential(
            nn.BatchNorm1d(num_elem),
            nn.ReLU()
        )
    elif reg == Regularization.RELU_BN:
        return nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(num_elem)
        )


def build_regularized_relu_block_(module_block: nn.Module, suffix: str, reg: Regularization, dropout:float = 0.5,
                                  num_elem:int = 0):
    if reg == Regularization.DROPOUT:
        module_block.add_module('Dropout_{}'.format(suffix), nn.Dropout(p=dropout))
        module_block.add_module('ReLU_{}'.format(suffix), nn.ReLU())
    elif reg == Regularization.BN_RELU:
        module_block.add_module('BN_{}'.format(suffix), nn.BatchNorm1d(num_elem))
        module_block.add_module('ReLU_{}'.format(suffix), nn.ReLU())
    elif reg == Regularization.RELU_BN:
        module_block.add_module('ReLU_{}'.format(suffix), nn.ReLU())
        module_block.add_module('BN_{}'.format(suffix), nn.BatchNorm1d(num_elem))


def build_from_toml_dict(values: OrderedDict, input_dim: int, final_dim: int = 1, pre_activation_reg: bool = False):
    final_block = nn.Sequential()
    last_output_size = input_dim

    for i, (section_name, section_values) in enumerate(values.items()):

        if section_name.startswith('res'):
            arch = ResArchitecture[section_values.get('format', "FULL_INNER")]
            use_projection = section_values.get('use_projection', False)
            inner_sections = OrderedDict([(k, v) for (k, v) in section_values.items() if isinstance(v, OrderedDict)])
            initial_channel = last_output_size
            inner_block, last_output_size = build_from_toml_dict(inner_sections, last_output_size, final_dim,
                                                                 arch == ResArchitecture.FULL_PA)
            if arch == ResArchitecture.SPLIT_LAST:
                final_block.add_module('{}_{}'.format(section_name, i), ResBlock(inner_block[:-1], use_projection,
                                                                                 initial_channel, last_output_size))
                final_block.add_module('ReLU_{}'.format(i), nn.ReLU())
            else:
                final_block.add_module('{}_{}'.format(section_name, i), ResBlock(inner_block, use_projection,
                                                                                 initial_channel, last_output_size))

        elif section_name.startswith('fc'):
            output_size = section_values.get('output_size', 1)
            if output_size == -1:
                output_size = final_dim
            dropout = section_values.get('dropout', 0.0)
            reg = Regularization[section_values.get('reg', 'NONE')]
            temp_fc = nn.Linear(
                in_features=last_output_size,
                out_features=output_size
            )
            if pre_activation_reg:
                build_regularized_relu_block_(final_block, i, reg, dropout=dropout, num_elem=output_size)
                final_block.add_module('{}_{}'.format(section_name, i), temp_fc)
            else:
                final_block.add_module('{}_{}'.format(section_name, i), temp_fc)
                build_regularized_relu_block_(final_block, i, reg, dropout=dropout, num_elem=output_size)
            last_output_size = output_size

    return final_block, last_output_size


def build_fc_from_toml_dict(values: OrderedDict, input_dim: int, last_output_size: int = 1, prefix: str = 'fc'):
    temp_block = nn.Sequential()
    last_output = input_dim
    for section_name, section_values in values.items():
        if not section_name.startswith(prefix):
            break
        output_dim = section_values.get('output_size', 1)
        if output_dim == -1:
            output_dim = last_output_size
        reg = Regularization[values]
    return temp_block