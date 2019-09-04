import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, inner_network: nn.Module, use_projection: bool = False, source_size: int = 0,
                 target_size: int = 0):
        super(ResBlock, self).__init__()
        self.inner_network = inner_network
        self.use_projection = use_projection
        if use_projection:
            self.proj = nn.Linear(source_size, target_size, bias=False)

    def forward(self, input):
        identity = input
        output = self.inner_network(input)
        if self.use_projection:
            return output + self.proj(identity.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return output + identity
