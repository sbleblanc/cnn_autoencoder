import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cnn_ae.utils.factory as factory
from cnn_ae.common.exceptions import InconsistentPoolingLayersException


class DeepMOTDSCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, z_size, kernels_channels):
        super(DeepMOTDSCNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(vocab_size, emb_size)

        lin_input_size = 0
        self.cnn_blocks = nn.ModuleList()
        for kernel, channel in kernels_channels:
            self.cnn_blocks.append(factory.build_cnn1d_block(emb_size, channel, 1, kernel, padding=0))
            lin_input_size += channel

        self.lin = nn.Sequential(
            nn.Linear(lin_input_size, hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hid_size, z_size)
        )

    def get_cnn_weights(self):
        return [cnn[0].weight for cnn in self.cnn_blocks]

    def forward(self, input):
        indices_history = []
        size_history = []
        max_pooled = []
        embedded = self.emb(input).permute(0, 2, 1)
        for cnn in self.cnn_blocks:
            output = cnn(embedded)
            size_history.append(output.size())
            output, indices = F.max_pool1d(output, kernel_size=output.shape[-1], return_indices=True)
            max_pooled.append(output)
            indices_history.append(indices)
        combined = torch.cat(max_pooled, dim=1).squeeze(2)
        output = self.lin(combined)
        return output, size_history, indices_history, embedded.size()

class DeepMOTUSCNNDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, z_size, us_hid_size, fc_hid_size, kernels_channels, device):
        super(DeepMOTUSCNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.kernels_channels = kernels_channels
        self.device = device

        lin_input_size = 0
        self.cnn_blocks = nn.ModuleList()
        for kernel, channel in kernels_channels:
            self.cnn_blocks.append(factory.build_cnn1d_block(channel, emb_size, 1, kernel, transposed=True, padding=0))
            lin_input_size += channel

        self.z_lin = nn.Sequential(
            nn.Linear(z_size, us_hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(us_hid_size, us_hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(us_hid_size, lin_input_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(emb_size, fc_hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc_hid_size, fc_hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(fc_hid_size, vocab_size)
        )

    def tie_weights(self, encoder_weights):
        for cnn, tied_weight in zip(self.cnn_blocks, encoder_weights):
            cnn.tied_weight = tied_weight

    def forward(self, input, size_history, indices_history, embedded_size):

        output = self.z_lin(input)

        outputs = []
        current_index = 0
        for _, channels in self.kernels_channels:
            outputs.append(output[:, current_index:current_index + channels].unsqueeze(2))
            current_index += channels

        combined = torch.zeros(embedded_size, device=self.device)
        for input, cnn, size, indices in zip(outputs, self.cnn_blocks, size_history, indices_history):
            output = F.max_unpool1d(input, indices, size[-1], output_size=size)
            output = cnn(output)
            combined += output

        output = self.fc(combined.permute(0, 2, 1))

        return output

class ShallowDSCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, kernels_channels, num_conv=1):
        super(ShallowDSCNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.cnn_blocks = nn.ModuleList()
        for kernel, channel in kernels_channels:
            self.cnn_blocks.append(factory.build_cnn1d_block(emb_size, channel, num_conv, kernel, padding=0))

    def get_cnn_weights(self):
        return [cnn[0].weight for cnn in self.cnn_blocks]

    def forward(self, input):
        indices_history = []
        size_history = []
        max_pooled = []
        embedded = self.emb(input).permute(0, 2, 1)
        for cnn in self.cnn_blocks:
            output = cnn(embedded)
            size_history.append(output.size())
            output, indices = F.max_pool1d(output, kernel_size=output.shape[-1], return_indices=True)
            max_pooled.append(output)
            indices_history.append(indices)
        return max_pooled, size_history, indices_history, embedded.size()


class ShallowUSCNNDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, kernels_channels, device, num_conv=1):
        super(ShallowUSCNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.device = device

        self.cnn_blocks = nn.ModuleList()
        for kernel, channel in kernels_channels:
            self.cnn_blocks.append(factory.build_cnn1d_block(channel, emb_size, num_conv, kernel, transposed=True, padding=0))

        self.lin = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hid_size, vocab_size)
        )

    def tie_weights(self, encoder_weights):
        for cnn, tied_weight in zip(self.cnn_blocks, encoder_weights):
            cnn.tied_weight = tied_weight

    def forward(self, inputs, size_history, indices_history, embedded_size):

        combined = torch.zeros(embedded_size, device=self.device)
        for input, cnn, size, indices in zip(inputs, self.cnn_blocks, size_history, indices_history):
            output = F.max_unpool1d(input, indices, size[-1], output_size=size)
            output = cnn(output)
            combined += output

        output = self.lin(combined.permute(0, 2, 1))

        return output


class DeepDSCNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_size, layers_kcn, pooling_ks):
        super(DeepDSCNNEncoder, self).__init__()

        if len(layers_kcn) != len(pooling_ks):
            raise InconsistentPoolingLayersException("Expected {} pooling parameters".format(len(layers_kcn)))

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.pooling_ks = pooling_ks

        self.emb = nn.Embedding(vocab_size, emb_size)

        self.cnn_blocks = nn.ModuleList()
        for i, (kernel_size, channels, n) in enumerate(layers_kcn):
            if i == 0:
                self.cnn_blocks.append(factory.build_cnn1d_block(emb_size, channels, n, kernel_size))
            else:
                self.cnn_blocks.append(factory.build_cnn1d_block(layers_kcn[i-1][1], channels, n, kernel_size))

    def forward(self, input):
        size_history = []
        indices_history = []
        kernel_history = []
        embedded = self.emb(input).permute(0, 2, 1)

        for i, cnn in enumerate(self.cnn_blocks):
            if i == 0:
                size_history.append(embedded.size())
                cnn_input = embedded
            else:
                size_history.append(output.size())
                cnn_input = output
            output = cnn(cnn_input)
            if self.pooling_ks[i][0] == -1:
                pooling_kernel_size = output.shape[-1]
            else:
                pooling_kernel_size = self.pooling_ks[i][0]

            kernel_history.append(pooling_kernel_size)
            output, indices = F.max_pool1d(output, kernel_size=pooling_kernel_size, stride=self.pooling_ks[i][1],
                                           return_indices=True)
            indices_history.append(indices)

        return output, size_history, indices_history, kernel_history


class DeepUSCNNDecoder(nn.Module):

    def __init__(self, vocab_size, emb_size, hid_size, layers_kcn):
        super(DeepUSCNNDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size

        self.cnn_blocks = nn.ModuleList()
        for i in range(len(layers_kcn)):
            if i < len(layers_kcn) - 1:
                self.cnn_blocks.append(
                    factory.build_cnn1d_block(layers_kcn[i][1],
                                              layers_kcn[i+1][1],
                                              layers_kcn[i][2],
                                              layers_kcn[i][0], forward=False))
            else:
                self.cnn_blocks.append(
                    factory.build_cnn1d_block(layers_kcn[i][1],
                                              emb_size,
                                              layers_kcn[i][2],
                                              layers_kcn[i][0], forward=False))
        # for i, (kernel_size, channels, n) in enumerate(layers_kcn):
        #     if i == 0:
        #         self.cnn_blocks.append(factory.build_cnn1d_block(emb_size, channels, n, kernel_size, forward=False))
        #     else:
        #         self.cnn_blocks.append(factory.build_cnn1d_block(layers_kcn[i - 1][1], channels, n, kernel_size , forward=False))
        #
        # self.outer_cnn = nn.Sequential(
        #     nn.Conv1d(emb_size* 2, emb_size * 2, kernel_size, padding=(kernel_size - 1) // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(emb_size * 2),
        #     nn.Conv1d(emb_size * 2, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(emb_size)
        # )
        # self.inner_cnn = nn.Sequential(
        #     nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(emb_size),
        #     nn.Conv1d(emb_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(emb_size)
        # )

        self.lin = nn.Sequential(
            nn.Linear(emb_size, hid_size),
            nn.Dropout(),
            nn.Linear(hid_size, hid_size),
            nn.Dropout(),
            nn.Linear(hid_size, vocab_size)
        )

    def forward(self, input, size_history, indices_history, kernel_history):

        output = input
        for cnn in self.cnn_blocks:
            output = F.max_unpool1d(output, indices_history.pop(), kernel_history.pop(), output_size=size_history.pop())
            output = cnn(output)

        output = self.lin(output.permute(0, 2, 1))

        return output



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

    def __init__(self, vocab_size, emb_size, hid_size, kernel_size=3):
        super(UpsamplingCNNDecoder, self).__init__()
        self.vocab_size = vocab_size
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
            nn.Dropout(),
            nn.Linear(hid_size, hid_size),
            nn.Dropout(),
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
        max_pooled, size_history, indices_history, embedded_size = self.enc(input)
        decoded = self.dec(max_pooled, size_history, indices_history, embedded_size)
        return decoded
