import math
import torch
import torch.nn.functional as F


def batch_attention(queries, keys, values):
    d_k = keys.shape[-1]
    scores = torch.bmm(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
    attention_weights = F.softmax(scores, dim=-1)
    return torch.bmm(attention_weights, values)
