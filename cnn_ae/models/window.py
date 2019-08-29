import torch
import torch.nn as nn
import torch.nn.functional as F
from cnn_ae.common.enums import Regularization
from cnn_ae.utils.factory import build_regularized_relu_block

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

    def __init__(self, window_size: int, vocab_size: int, emb_size: int):
        super(CNN, self).__init__()

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


