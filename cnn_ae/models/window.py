import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_ae.common.enums import Regularization
from cnn_ae.utils.factory import build_regularized_relu_block
from cnn_ae.utils.math import get_expected_conv_1d_lout, get_expected_mp_1d_lout
import configparser

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
        model_conf = configparser.ConfigParser()
        model_conf.read(conf_fn)
        section_iter = iter(model_conf.sections())

        current_section = model_conf[next(section_iter)]
        model.emb = nn.Embedding(vocab_size, current_section.getint('emb_size', 1))

        model.cnn = nn.Sequential()
        current_section = model_conf[next(section_iter)]
        last_output_size = model.emb.embedding_dim
        current_lin = window_size
        while current_section.name.startswith('CNN') or current_section.name.startswith('Pooling'):
            if current_section.name.startswith('CNN'):
                out_channels = current_section.getint('output_channel', 50)
                temp_block = nn.Sequential()
                temp_block.add_module('Convolution', nn.Conv1d(
                    in_channels=last_output_size,
                    out_channels=out_channels,
                    kernel_size=current_section.getint('kernel_size', 3),
                    padding=current_section.getint('padding', 1),
                    stride=current_section.getint('stride', 1),
                    dilation=current_section.getint('dilation', 1),
                ))
                temp_block.add_module('ReLU Block', build_regularized_relu_block(
                    reg=Regularization[current_section.get('reg', 'RELU_BN')],
                    num_elem=out_channels
                ))
                model.cnn.add_module(current_section.name, temp_block)
                last_output_size = out_channels
                current_lin = get_expected_conv_1d_lout(current_lin, temp_block[0])
            else:
                model.cnn.add_module(current_section.name, nn.MaxPool1d(
                    kernel_size=current_section.getint('kernel_size', 2),
                    stride=current_section.getint('stride', 2)
                ))
                current_lin = get_expected_mp_1d_lout(current_lin, model.cnn[-1])

            current_section = model_conf[next(section_iter)]

        last_output_size = last_output_size * current_lin
        model.fc = nn.Sequential()
        while True:
            reg = Regularization[current_section.get('reg', "NONE")]

            out_size = current_section.getint('output_size', 512)
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
                    dropout=current_section.getfloat('dropout', 0.0),
                    num_elem=out_size
                ))
            model.fc.add_module(current_section.name, temp_block)
            last_output_size = out_size

            try:
                current_section = model_conf[next(section_iter)]
            except StopIteration:
                break

        return model

    def __init__(self, window_size: int, vocab_size: int, emb_size: int, default_build: bool = True):
        super(CNN, self).__init__()

        if default_build:
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


