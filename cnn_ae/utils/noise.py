import torch


def noise_char_input(data, ratio=0.01):
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
