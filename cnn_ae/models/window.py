import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_ae.models.res import ResBlock
from cnn_ae.common.enums import Regularization, ResArchitecture
from cnn_ae.utils.factory import build_regularized_relu_block, build_from_toml_dict
from cnn_ae.utils.math import get_expected_conv_1d_lout, get_expected_mp_1d_lout
from collections import OrderedDict
import toml

class MLP(nn.Module):

    def __init__(self, window_size: int, vocab_size: int, hid_size: int, depth: int = 1, dropout: float = 0.5,
                 regularization: Regularization = Regularization.DROPOUT):
        super(MLP, self).__init__()

        self.oh = nn.Embedding.from_pretrained(torch.eye(vocab_size, dtype=torch.float))

        self.lin = nn.Sequential()

        self.lin.add_module('input', nn.Sequential(
            nn.Linear((window_size - 1) * vocab_size, hid_size),
            build_regularized_relu_block(regularization, dropout, hid_size)
        ))
        for i in range(depth):
            self.lin.add_module('inner_{}'.format(i), nn.Sequential(
                nn.Linear(hid_size, hid_size),
                build_regularized_relu_block(regularization, dropout, hid_size)
            ))
        self.lin.add_module('output', nn.Linear(hid_size, vocab_size))


    def forward(self, noised):
        oh_encoded = self.oh(noised).view(noised.shape[0], -1)
        return self.lin(oh_encoded)


class CNN(nn.Module):

    @classmethod
    def from_conf(cls, conf_fn: str, window_size: int, vocab_size: int):
        window_size -= 1
        model = cls(window_size, vocab_size, 0, False)
        model_conf = toml.load(conf_fn, _dict=OrderedDict)
        section_iter = iter(model_conf.items())

        current_section_name, current_section_values = next(section_iter)
        model.emb = nn.Embedding(vocab_size, current_section_values.get('emb_size', 1))

        model.cnn = nn.Sequential()
        current_section_name, current_section_values = next(section_iter)
        last_output_size = model.emb.embedding_dim
        current_lin = window_size
        while current_section_name.startswith('cnn') or current_section_name.startswith(
                'pooling') or current_section_name.startswith('res'):
            if current_section_name.startswith('res'):
                inner_block = nn.Sequential()
                arch = ResArchitecture[current_section_values.pop('format', "FULL_INNER")]
                use_projection = current_section_values.pop('use_projection', False)
                initial_channel = last_output_size
                for i, (inner_block_name, inner_block_values) in enumerate(current_section_values.items()):
                    out_channels = inner_block_values.get('output_channel', 50)
                    temp_block = nn.Sequential()
                    temp_cnn = nn.Conv1d(
                        in_channels=last_output_size,
                        out_channels=out_channels,
                        kernel_size=inner_block_values.get('kernel_size', 3),
                        padding=inner_block_values.get('padding', 1),
                        stride=inner_block_values.get('stride', 1),
                        dilation=inner_block_values.get('dilation', 1),
                    )
                    if arch == ResArchitecture.FULL_PA:
                        temp_reg = build_regularized_relu_block(
                            reg=Regularization[inner_block_values.get('reg', 'BN_RELU')],
                            num_elem=last_output_size
                        )
                    else:
                        temp_reg = build_regularized_relu_block(
                            reg=Regularization[inner_block_values.get('reg', 'BN_RELU')],
                            num_elem=out_channels
                        )
                    if arch == ResArchitecture.SPLIT_LAST :
                        temp_block.add_module('Convolution', temp_cnn)
                        if i == len(current_section_values.items()) - 1:
                            temp_block.add_module('ReLU Block', temp_reg[0])
                        else:
                            temp_block.add_module('ReLU Block', temp_reg)
                    elif arch == ResArchitecture.FULL_INNER:
                        temp_block.add_module('Convolution', temp_cnn)
                        temp_block.add_module('ReLU Block', temp_reg)
                    elif arch == ResArchitecture.FULL_PA:
                        temp_block.add_module('ReLU Block', temp_reg)
                        temp_block.add_module('Convolution', temp_cnn)
                    inner_block.add_module(inner_block_name, temp_block)
                    last_output_size = out_channels
                    current_lin = get_expected_conv_1d_lout(current_lin, temp_cnn)
                if arch == ResArchitecture.SPLIT_LAST:
                    model.cnn.add_module(current_section_name, nn.Sequential(
                        ResBlock(inner_block, use_projection=use_projection, source_size=initial_channel,
                                 target_size=last_output_size),
                        nn.ReLU()
                    ))
                else:
                    model.cnn.add_module(current_section_name, ResBlock(inner_block, use_projection=use_projection,
                                                                        source_size=initial_channel,
                                                                        target_size=last_output_size))

            if current_section_name.startswith('cnn'):
                out_channels = current_section_values.get('output_channel', 50)
                temp_block = nn.Sequential()
                temp_block.add_module('Convolution', nn.Conv1d(
                    in_channels=last_output_size,
                    out_channels=out_channels,
                    kernel_size=current_section_values.get('kernel_size', 3),
                    padding=current_section_values.get('padding', 1),
                    stride=current_section_values.get('stride', 1),
                    dilation=current_section_values.get('dilation', 1),
                ))
                temp_block.add_module('ReLU Block', build_regularized_relu_block(
                    reg=Regularization[current_section_values.get('reg', 'RELU_BN')],
                    num_elem=out_channels
                ))
                model.cnn.add_module(current_section_name, temp_block)
                last_output_size = out_channels
                current_lin = get_expected_conv_1d_lout(current_lin, temp_block[0])

            if current_section_name.startswith('pool'):
                model.cnn.add_module(current_section_name, nn.MaxPool1d(
                    kernel_size=current_section_values.get('kernel_size', 2),
                    stride=current_section_values.get('stride', 2)
                ))
                current_lin = get_expected_mp_1d_lout(current_lin, model.cnn[-1])

            current_section_name, current_section_values = next(section_iter)

        last_output_size = last_output_size * current_lin
        model.fc = nn.Sequential()
        while True:
            reg = Regularization[current_section_values.get('reg', "NONE")]

            out_size = current_section_values.get('output_size', 512)
            if out_size == -1:
                out_size = vocab_size

            temp_block = nn.Sequential()
            temp_block.add_module("Linear", nn.Linear(
                in_features=last_output_size,
                out_features=out_size
            ))
            if reg != Regularization.NONE:
                temp_block.add_module('ReLU Block', build_regularized_relu_block(
                    reg=reg,
                    dropout=current_section_values.get('dropout', 0.0),
                    num_elem=out_size
                ))
            model.fc.add_module(current_section_name, temp_block)
            last_output_size = out_size

            try:
                current_section_name, current_section_values = next(section_iter)
            except StopIteration:
                break

        return model

    def __init__(self, window_size: int, vocab_size: int, emb_size: int, default_build: bool = True):
        super(CNN, self).__init__()

        if default_buid:
            output_channels = 100

            self.emb = nn.Embedding(vocab_size, emb_size)

            self.cnn = nn.Sequential(
                nn.Conv1d(emb_size, output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(output_channels),
                nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(output_channels),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(output_channels, 2 * output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(2 * output_channels),
                nn.Conv1d(2 * output_channels, 2 * output_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(2 * output_channels)
            )

            hid_size = 1024
            self.fc = nn.Sequential(
                nn.Linear(2 * output_channels * (window_size // 2), hid_size),
                nn.ReLU(),
                nn.BatchNorm1d(hid_size),
                nn.Linear(hid_size, hid_size),
                nn.ReLU(),
                nn.BatchNorm1d(hid_size),
                nn.Linear(hid_size, vocab_size)
            )

    def forward(self, input):
        embedded = self.emb(input).permute(0, 2, 1)
        output = self.cnn(embedded)
        output = self.fc(output.view(input.shape[0], -1))
        return output


class ResMLP(nn.Module):

    def __init__(self, conf_fn: str, window_size: int, vocab_size: int,):
        super(ResMLP, self).__init__()
        model_conf = toml.load(conf_fn, _dict=OrderedDict)

        self.oh = nn.Embedding.from_pretrained(torch.eye(vocab_size, dtype=torch.float))

        self.lin, _ = build_from_toml_dict(model_conf, (window_size - 1) * vocab_size, vocab_size)

    def forward(self, noised):
        oh_encoded = self.oh(noised).view(noised.shape[0], -1)
        return self.lin(oh_encoded)




