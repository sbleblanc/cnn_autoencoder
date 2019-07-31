import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DownsamplingCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, z_size, kernel_size=3, out_channels=64):
        super(DownsamplingCNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_size, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2, 2)
        )

    def forward(self, input):
        # input: [N, L]
        # emb: [N, L, E] => [N, E, L]
        embedded = self.emb(input).permute(0, 2, 1)
        # cnn: [N, out_channels, L]
        output = self.cnn(embedded)
        return output


class UpsamplingCNNDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, in_channels, kernel_size=3, out_channels=64):
        super(UpsamplingCNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),
        )

        self.lin = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.Linear(hid_size, hid_size),
            nn.Linear(hid_size, vocab_size)
        )

    def forward(self, input):
        output = self.cnn(input).permute(0, 2, 1)
        output = F.softmax(self.lin(output), 2)
        return output


class DeepCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, num_layers=1):
        super(DeepCNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.cnns = nn.ModuleList()
        num_filters = 64
        cnn = nn.Sequential(
            nn.Conv1d(emb_size, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_filters),
        )
        self.cnns.append(cnn)
        for i in range(1, num_layers):
            cnn = nn.Sequential(
                nn.Conv1d(num_filters, num_filters, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters),
            )
            self.cnns.append(cnn)

    def forward(self, input):
        output = self.emb(input).permute(0, 2, 1)
        for cnn in self.cnns:
            output = cnn(output)
        return output


class CNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, kernel_size=3, out_channels=5):
        super(CNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.cnn = nn.Sequential(
            nn.Conv1d(emb_size, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, input):
        # input: [N, L]
        # emb: [N, L, E] => [N, E, L]
        embedded = self.emb(input).permute(0, 2, 1)
        # cnn: [N, out_channels, L]
        output = self.cnn(embedded)
        return output


class CNNDecoder(nn.Module):

    def __init__(self, vocab_size, in_channels, kernel_size=3, out_channels=5):
        super(CNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

        self.lin = nn.Linear(out_channels, vocab_size)

    def forward(self, input):
        # input: [N, in_channels, L]
        output = self.cnn(input).permute(0, 2, 1)
        # cnn: [N, out_channels, L] => [N, L, out_channels]
        output = F.softmax(self.lin(output), 2)
        # output: [N, L, vocab_size]
        return output


class CNNAE(nn.Module):

    def __init__(self, enc, dec):
        super(CNNAE, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, input):
        encoded = self.enc(input)
        decoded = self.dec(encoded)
        return decoded
