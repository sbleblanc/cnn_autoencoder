import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DownsamplingCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, kernel_size=3):
        super(DownsamplingCNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernel_size = kernel_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.outer_cnn = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size)
        )
        self.inner_cnn = nn.Sequential(
            nn.Conv1d(emb_size, emb_size*2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size*2),
            nn.Conv1d(emb_size*2, emb_size*2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size*2)
        )

    def forward(self, input):
        size_history = []
        indices_history = []
        kernel_history = []
        embedded = self.emb(input).permute(0, 2, 1)

        size_history.append(embedded.size())
        output = self.outer_cnn(embedded)
        kernel_history.append(2)
        output, indices = F.max_pool1d(output, kernel_size=2, stride=2, return_indices=True)
        indices_history.append(indices)

        size_history.append(output.size())
        output = self.inner_cnn(output)
        kernel_history.append(output.shape[-1])
        output, indices = F.max_pool1d(output, kernel_size=output.shape[-1], return_indices=True)
        indices_history.append(indices)

        return output, size_history, indices_history, kernel_history


class UpsamplingCNNDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, in_channels, kernel_size=3):
        super(UpsamplingCNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.outer_cnn = nn.Sequential(
            nn.Conv1d(emb_size* 2, emb_size * 2, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size * 2),
            nn.Conv1d(emb_size * 2, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size)
        )
        self.inner_cnn = nn.Sequential(
            nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),
            nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size)
        )

        self.lin = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.Linear(hid_size, hid_size),
            nn.Linear(hid_size, vocab_size)
        )

    def forward(self, input, size_history, indices_history, kernel_history):

        output = F.max_unpool1d(input, indices_history.pop(), kernel_history.pop(), output_size=size_history.pop())
        output = self.outer_cnn(output)
        output = F.max_unpool1d(output, indices_history.pop(), kernel_history.pop(), output_size=size_history.pop())
        output = self.inner_cnn(output).permute(0, 2, 1)

        output = self.lin(output)

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
        encoded, size_history, indices_history, kernel_history = self.enc(input)
        decoded = self.dec(encoded, size_history, indices_history, kernel_history)
        return decoded
