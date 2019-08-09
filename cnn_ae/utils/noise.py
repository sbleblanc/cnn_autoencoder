import torch


def noise_char_input_slow(data, ratio=0.01):
    flipped_buffer = data.clone()
    to_flip = (torch.zeros(data.size()).uniform_(0, 1) < ratio).nonzero()
    for i in range(to_flip.shape[0]):
        y, x = to_flip[i, 0], to_flip[i, 1]
        if x == 0:
            continue

        buffer = flipped_buffer[y, x - 1].item()
        flipped_buffer[y, x - 1] = flipped_buffer[y, x].item()
        flipped_buffer[y, x] = buffer

    return flipped_buffer


def noise_char_input(data, ratio=0.01):
    flipped_buffer = data.clone()
    to_flip = (torch.zeros(data.size()).uniform_() < ratio).nonzero()
    to_flip = to_flip.index_select(0, (to_flip[:, 1] % 2 == 1).nonzero().squeeze(1))
    to_flip_offset = to_flip.clone()
    to_flip_offset[:, 1] -= 1
    flipped_buffer[tuple(to_flip.t())] = data[tuple(to_flip_offset.t())]
    flipped_buffer[tuple(to_flip_offset.t())] = data[tuple(to_flip.t())]
    return flipped_buffer
