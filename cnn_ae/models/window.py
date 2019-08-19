import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, window_size: int, vocab_size: int, hid_size: int, depth: int = 1):
        super(MLP, self).__init__()

        self.oh = nn.Embedding.from_pretrained(torch.eye(vocab_size, dtype=torch.float))

        self.lin = nn.Sequential()
        self.lin.add_module('input', nn.Sequential(
            nn.Linear((window_size - 1) * vocab_size, hid_size),
            nn.Dropout(),
            nn.ReLU()
        ))
        for i in range(depth):
            self.lin.add_module('inner_{}'.format(i), nn.Sequential(
                nn.Linear(hid_size, hid_size),
                nn.Dropout(),
                nn.ReLU()
            ))
        self.lin.add_module('output', nn.Linear(hid_size, vocab_size))

    def forward(self, noised):
        oh_encoded = self.oh(noised).view(noised.shape[0], -1)
        return self.lin(oh_encoded)
